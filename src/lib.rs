#![feature(temporary_niche_types)]
#![feature(sized_type_properties)]
#![feature(ptr_as_uninit)]
#![feature(maybe_uninit_slice)]

// TODO: append front, etc.
mod mirrored;

use crate::mirrored::MAX_PHYSICAL_BUF_SIZE;
use mirrored::{MirroredBuffer, mirrored_allocation_unit};
use num::Zero;
use std::{
    cmp::Ordering,
    iter::{FromIterator, FusedIterator},
    marker::PhantomData,
    mem::{MaybeUninit, SizedTypeProperties, needs_drop, replace, zeroed},
    ops::{Bound, Deref, DerefMut, Neg, RangeBounds},
    ptr::{NonNull, copy, copy_nonoverlapping, drop_in_place, read, slice_from_raw_parts_mut, write},
};

#[derive(Debug, Clone)]
pub struct SliceRingBuffer<T> {
    buf: MirroredBuffer<T>,
    head: usize,
    len: usize,
}

impl<T> SliceRingBuffer<T> {
    const GROW_FACTOR: usize = 2;
    pub const MAX_CAPACITY: usize = MAX_PHYSICAL_BUF_SIZE;

    #[inline(always)]
    pub fn new() -> Self { Self::with_capacity(if T::IS_ZST { MAX_PHYSICAL_BUF_SIZE } else { 0 }) }

    /// 分配多少个物理元素的容量
    #[inline(always)]
    pub fn with_capacity(cap: usize) -> Self {
        assert!(cap <= MAX_PHYSICAL_BUF_SIZE);
        Self { buf: MirroredBuffer::with_capacity(cap), head: 0, len: 0 }
    }

    /// 物理容量
    #[inline(always)]
    pub fn capacity(&self) -> usize { self.buf.physical_capacity() }

    /// 物理长度
    #[inline(always)]
    pub fn len(&self) -> usize {
        let len = self.len;
        debug_assert!(len <= self.capacity(), "len:{} > capacity:{}", len, self.capacity());
        len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.len().is_zero() }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let len = self.len();
        let cap = self.capacity();
        debug_assert!(len <= cap);
        len == cap
    }

    /// 获取头的位置，头只会在 0..v_cap / 2 间
    #[inline(always)]
    pub fn head(&self) -> usize {
        let head = self.head;
        debug_assert!(head.is_zero() || head < self.capacity());
        head
    }

    /// 获取尾巴的位置，尾巴会在 0..v_cap 间，但是返回的是在第一段虚拟地址的位置
    #[inline(always)]
    pub fn tail(&self) -> usize {
        let cap = self.capacity();
        let head = self.head();
        let len = self.len();
        debug_assert!(head.checked_add(len) <= Some(self.buf.virtual_size()));
        (head + len) % cap
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        let head = self.head();
        unsafe { self.buf.virtual_uninit_slice_at(head, len).assume_init_ref() }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let head = self.head();
        let len = self.len();
        unsafe { self.buf.virtual_uninit_slice_mut_at(head, len).assume_init_mut() }
    }

    #[inline(always)]
    pub fn as_uninit_slice(&self) -> &[MaybeUninit<T>] {
        let len = self.len();
        let head = self.head();
        self.buf.virtual_uninit_slice_at(head, len)
    }

    #[inline(always)]
    pub fn as_uninit_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        let len = self.len();
        let head = self.head();
        self.buf.virtual_uninit_slice_mut_at(head, len)
    }

    #[inline(always)]
    pub fn uninit_slice(&mut self) -> &mut [MaybeUninit<T>] {
        let cap = self.capacity();
        let len = self.len();
        let head = self.head();
        debug_assert!(len.checked_add(head) < Some(self.buf.virtual_size()));
        let uninit_start = self.tail();
        let uninit_len = cap - len;
        self.buf.virtual_uninit_slice_mut_at(uninit_start, uninit_len)
    }

    /// 在第一段虚拟地址(0..p_cap)内环绕移动头指针，副作用是会更新长度
    #[inline(always)]
    pub fn move_head(&mut self, mv: isize) {
        let cap = self.capacity();
        let head = &mut self.head();
        let cap = cap as isize;
        assert!(cap > 0);
        let new_head = (*head as isize + mv).rem_euclid(cap);
        self.head = new_head as usize;
        let new_len = self.len.strict_sub_signed(mv);
        self.len = new_len;
    }

    /// 仅仅增加长度
    #[inline(always)]
    pub fn move_tail(&mut self, mv: isize) {
        let len = self.len() as isize;
        let cap = self.capacity() as isize;
        assert!(len.checked_add(mv) <= Some(cap) && len.checked_add(mv) >= Some(0));
        self.len = (len + mv) as usize;
    }

    #[inline(always)]
    pub fn append(&mut self, other: &mut Self) {
        self.reserve(other.len());
        if T::IS_ZST {
            self.len += other.len();
            unsafe {
                self.buf.set_size_unchecked(self.buf.virtual_size() + other.buf.virtual_size());
            }
            return;
        }
        unsafe {
            let uninit = self.uninit_slice();
            let src = other.as_slice().as_ptr();
            let dst = uninit.as_mut_ptr().cast();
            copy_nonoverlapping(src, dst, other.len);
            self.len += other.len;
            other.len = 0; // 清空 other，避免旧容器的元素（已经被拷贝到本容器）被析构，但是他们原本占用的内存会被释放
        }
    }

    #[inline(always)]
    pub fn front(&self) -> Option<&T> { self.as_slice().first() }

    #[inline(always)]
    pub fn front_mut(&mut self) -> Option<&mut T> { self.as_mut_slice().first_mut() }

    #[inline(always)]
    pub fn back(&self) -> Option<&T> { self.as_slice().last() }

    #[inline(always)]
    pub fn back_mut(&mut self) -> Option<&mut T> { self.as_mut_slice().last_mut() }

    /// Returns a reference to the element at the given index.
    ///
    /// Element at index 0 is the front of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use slice_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buf = SliceRingBuffer::new();
    /// buf.push_back(1);
    /// buf.push_back(2);
    /// buf.push_back(3);
    ///
    /// assert_eq!(buf.get(0), Some(&1));
    /// assert_eq!(buf.get(1), Some(&2));
    /// assert_eq!(buf.get(2), Some(&3));
    /// assert_eq!(buf.get(3), None);
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            let head = self.head();
            let capacity = self.capacity();
            let actual_index = (head + index) % capacity;
            unsafe { Some(self.buf.virtual_uninit_slice_at(actual_index, 1).get_unchecked(0).assume_init_ref()) }
        }
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// Element at index 0 is the front of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use slice_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buf = SliceRingBuffer::new();
    /// buf.push_back(1);
    /// buf.push_back(2);
    /// buf.push_back(3);
    ///
    /// if let Some(elem) = buf.get_mut(1) {
    ///     *elem = 7;
    /// }
    ///
    /// assert_eq!(buf.as_slice(), &[1, 7, 3]);
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len() {
            None
        } else {
            let head = self.head();
            let capacity = self.capacity();
            let actual_index = (head + index) % capacity;
            unsafe {
                Some(self.buf.virtual_uninit_slice_mut_at(actual_index, 1).get_unchecked_mut(0).assume_init_mut())
            }
        }
    }

    /// 禁止 ZST 调用此函数
    fn realloc_and_restore_part(&mut self, new_p_cap: usize) {
        debug_assert!(new_p_cap <= MAX_PHYSICAL_BUF_SIZE);
        debug_assert!(!T::IS_ZST);
        unsafe {
            let len = self.len();
            let obsolete_len = len.saturating_sub(new_p_cap);
            let reserve_len = len - obsolete_len;
            let new_buf = MirroredBuffer::<T>::with_capacity(new_p_cap);
            let old_buf = &mut self.buf;
            if obsolete_len.is_zero() {
                let dst = new_buf.as_ptr();
                let src = old_buf.as_ptr().add(self.head());
                // 拷贝要保留的元素
                copy_nonoverlapping(src, dst, reserve_len);
                // 确保要丢掉的元素都能正确析构
                let drop_start = src.add(reserve_len);
                let drop_slice = slice_from_raw_parts_mut(drop_start, obsolete_len);
                drop_in_place(drop_slice);
            }
            let old_buf = replace(&mut self.buf, new_buf);
            drop(old_buf); // 释放旧的缓冲区
            self.head = 0;
            self.len = reserve_len;
        }
    }

    /// Creates a draining iterator that removes the specified range in the `SliceRingBuffer`
    /// and yields the removed items.
    ///
    /// When the iterator **is** dropped, all elements in the range are removed
    /// from the deque, even if the iterator was not fully consumed. If the
    /// iterator **is not** dropped (with [`mem::forget`] for example), the deque will be left
    /// in an inconsistent state.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the deque.
    ///
    /// # Examples
    ///
    /// ```
    /// use slice_ring_buffer::SliceRingBuffer;
    ///
    /// let mut v = SliceRingBuffer::new();
    /// v.push_back(1);
    /// v.push_back(2);
    /// v.push_back(3);
    ///
    /// let drained: SliceRingBuffer<_> = v.drain(1..).collect();
    /// assert_eq!(drained.as_slice(), &[2, 3]);
    /// assert_eq!(v.as_slice(), &[1]);
    /// ```
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        // Both `start` and `end` are relative to the front of the deque
        use Bound::*;
        let len = self.len();
        let start = match range.start_bound() {
            Included(&n) => n,
            Excluded(&n) => n + 1,
            Unbounded => 0,
        };
        let end = match range.end_bound() {
            Included(&n) => n + 1,
            Excluded(&n) => n,
            Unbounded => len,
        };
        assert!(start <= end, "drain lower bound was too large");
        assert!(end <= len, "drain upper bound was too large");
        let drain_len = end - start;
        let new_len = self.len - drain_len;
        let drain_start = (self.head() + start) % self.capacity();
        self.len = drain_start;
        Drain { drain_len, head: 0, remaining: drain_len, new_len, inner: NonNull::from(self), _marker: PhantomData }
    }

    #[inline(always)]
    pub fn try_push_front(&mut self, value: T) -> Option<&T> {
        if self.is_full() {
            return None;
        }
        self.move_head(-1);
        unsafe {
            if T::IS_ZST {
                return Some(NonNull::dangling().as_ref());
            }
            let uninit = self.as_uninit_mut_slice().first_mut().unwrap_unchecked();
            let init = uninit.write(value);
            Some(init)
        }
    }

    #[inline(always)]
    pub fn push_front(&mut self, value: T) -> &T {
        unsafe {
            assert!(self.len().checked_add(1) <= Some(MAX_PHYSICAL_BUF_SIZE));
            let required_cap = self.len() + 1;

            if T::IS_ZST && self.capacity() < required_cap {
                self.buf.set_size_unchecked(required_cap * 2);
                return self.try_push_front(value).unwrap_unchecked();
            }
            if self.capacity() < required_cap {
                self.realloc_and_restore_part(required_cap.strict_mul(Self::GROW_FACTOR));
            }
            self.try_push_front(value).unwrap_unchecked()
        }
    }

    #[inline(always)]
    pub fn try_push_back(&mut self, value: T) -> Option<&T> {
        if self.is_full() {
            return None;
        }
        self.move_tail(1);
        unsafe {
            if T::IS_ZST {
                return Some(NonNull::dangling().as_ref());
            }
            let uninit = self.as_uninit_mut_slice().last_mut().unwrap_unchecked();
            let init = uninit.write(value);
            Some(init)
        }
    }

    #[inline(always)]
    pub fn push_back(&mut self, value: T) -> &T {
        unsafe {
            assert!(self.len().checked_add(1) <= Some(MAX_PHYSICAL_BUF_SIZE));
            let required_cap = self.len() + 1;
            if T::IS_ZST && self.capacity() < required_cap {
                self.buf.set_size_unchecked(required_cap * 2);
                return self.try_push_back(value).unwrap_unchecked();
            }
            if self.capacity() < required_cap {
                self.realloc_and_restore_part(required_cap.strict_mul(Self::GROW_FACTOR));
            }
            self.try_push_back(value).unwrap_unchecked()
        }
    }

    #[inline(always)]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        if T::IS_ZST {
            self.move_tail(-1);
            return Some(unsafe { zeroed() });
        }
        unsafe {
            let removed = self.last_mut().map(|p| read(p as *mut _))?;
            self.move_tail(-1);
            Some(removed)
        }
    }

    #[inline(always)]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        if T::IS_ZST {
            self.move_head(1);
            return Some(unsafe { zeroed() });
        }
        unsafe {
            let removed = self.first_mut().map(|p| read(p as *mut _))?;
            self.move_head(1);
            Some(removed)
        }
    }

    //todo: new api: truncate_front

    #[inline(always)]
    pub fn truncate_back(&mut self, n: usize) {
        let current_len = self.len();
        if n >= current_len {
            return;
        }
        // 上面的 guard 保证了 current 不小于 target
        let mv = ((current_len - n) as isize).neg();
        if T::IS_ZST {
            self.move_tail(mv);
            debug_assert_eq!(self.len(), n);
            return;
        }
        let s = unsafe { self.get_unchecked_mut(n..current_len) } as *mut [_];
        unsafe {
            drop_in_place(s);
            self.move_tail(mv);
        }
        debug_assert_eq!(self.len(), n)
    }

    // 默认是从尾部截断
    #[inline(always)]
    pub fn truncate(&mut self, n: usize) { self.truncate_back(n); }

    // 包含 at
    #[inline(always)]
    pub fn split_off(&mut self, at: usize) -> Self {
        let tail_len = self.len().strict_sub(at);
        let mut another = Self::with_capacity(tail_len);
        if tail_len > 0 {
            if T::IS_ZST {
                another.len = tail_len;
                self.len = at;
                return another;
            }
            let tail_slice = unsafe { self.get_unchecked(at..) };
            unsafe {
                copy_nonoverlapping(tail_slice.as_ptr(), another.as_mut_ptr(), tail_len);
                another.len = tail_len;
            }
        }
        self.len = at;
        another
    }

    // 劈走 at 之前的元素
    #[inline(always)]
    pub fn split_to(&mut self, at: usize) -> Self {
        let prev_len = at;
        let mut another = Self::with_capacity(prev_len);
        if prev_len > 0 {
            if T::IS_ZST {
                self.len -= prev_len;
                another.len = prev_len;
                return another;
            }
            let prev_slice = unsafe { self.get_unchecked(..at) };
            unsafe {
                copy_nonoverlapping(prev_slice.as_ptr(), another.as_mut_ptr(), prev_len);
                another.len = prev_len;
            }
        }
        self.move_head(prev_len as isize);
        another
    }

    #[inline(always)]
    pub fn clear(&mut self) { self.truncate(0); }

    #[inline(always)]
    pub fn swap_remove(&mut self, idx: usize) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        if T::IS_ZST {
            return self.pop_back();
        }
        let len = self.len();
        // 若索引指向最后一个元素，直接弹出
        // is_empty() 保证了 len > 0
        if idx != len - 1 {
            self.swap(idx, len - 1);
        }
        self.pop_back()
    }

    #[inline(always)]
    pub fn try_insert(&mut self, idx: usize, elem: T) -> Option<&T> {
        let len = self.len();
        match len {
            _ if idx > len || self.is_full() => {
                return None;
            }
            _ if idx.is_zero() => {
                return self.try_push_front(elem);
            }
            _ if idx == len => {
                return self.try_push_back(elem);
            }
            _ => {}
        }
        self.move_tail(1);
        if T::IS_ZST {
            return Some(unsafe { NonNull::dangling().as_ref() });
        }
        unsafe {
            // 增加长度并获取可变切片
            let slice = self.as_mut_slice();
            // 计算需要移动的元素数量
            // 上面的两次判断保证了 len 大于 idx
            let count = len - idx;
            let src = slice.as_mut_ptr().add(idx);
            // cap > len > idx
            let dst = src.add(1);
            copy(src, dst, count);
            write(slice.as_mut_ptr().add(idx), elem);
            Some(&*src)
        }
    }

    pub fn insert(&mut self, idx: usize, elem: T) -> &T {
        let len = self.len();
        assert!(idx <= len, "index outbound");
        if !self.is_full() {
            return unsafe { self.try_insert(idx, elem).unwrap_unchecked() };
        }
        match idx {
            0 => return self.push_front(elem),
            _ if idx == len => return self.push_back(elem),
            _ => {}
        }
        if T::IS_ZST {
            return unsafe { NonNull::dangling().as_ref() };
        }
        let new_buf: MirroredBuffer<T> =
            MirroredBuffer::with_capacity(self.capacity().checked_add(1).expect("cap overflow"));
        let src = self.as_mut_slice().as_ptr();
        let dst = new_buf.as_ptr();
        unsafe {
            copy(src, dst, idx);
            let src = src.add(idx);
            let target = dst.add(idx);
            copy(&elem, target, 1);
            let dst = target.add(1);
            copy(src, dst, len - idx);
            self.buf = new_buf;
            self.head = 0;
            self.len = len + 1;
            &*target
        }
    }

    #[inline(always)]
    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len(), "index out of bounds: the len is {} but the index is {}", idx, self.len());
        unsafe {
            if T::IS_ZST {
                self.move_tail(-1);
                return zeroed();
            }
            let len = self.len();
            let ptr = self.as_mut_ptr();
            let target = ptr.add(idx);
            let removed = read(target);
            copy(ptr.add(idx + 1), target, len - idx - 1);
            self.move_tail(-1);
            removed
        }
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        let required_cap = self.len().checked_add(additional);
        assert!(required_cap <= Some(MAX_PHYSICAL_BUF_SIZE));
        let required_cap = unsafe { required_cap.unwrap_unchecked() };

        let cap = self.capacity();
        if cap >= required_cap {
            return;
        }
        if T::IS_ZST {
            unsafe {
                self.buf.set_size_unchecked(required_cap * 2);
            }
            return;
        }
        let new_cap = required_cap.max(cap.saturating_mul(Self::GROW_FACTOR));
        self.realloc_and_restore_part(new_cap);
    }

    // 缩到指定的容量，如果小于当前长度则 panic
    #[inline(always)]
    pub fn shrink_to(&mut self, min_cap: usize) {
        if T::IS_ZST {
            return;
        }
        let len = self.len();
        assert!(min_cap >= len, "min_capacity ({}) cannot be less than current length ({})", min_cap, len);
        // 如果请求的容量已经大于或等于当前容量，不需要操作
        if min_cap >= self.capacity() {
            return;
        }
        let ideal_virtual_size = mirrored_allocation_unit::<T>(min_cap);
        if ideal_virtual_size < self.buf.virtual_size() {
            self.realloc_and_restore_part(min_cap);
        }
    }

    /// 由于对齐问题，并不会完全fit
    #[inline(always)]
    pub fn shrink_to_fit(&mut self) {
        if T::IS_ZST {
            unsafe {
                self.buf.set_size_unchecked(self.len() * 2);
            }
            return;
        }
        let len = self.len();
        let ideal_virtual_size = mirrored_allocation_unit::<T>(len);
        if ideal_virtual_size >= self.buf.virtual_size() {
            return;
        }
        self.realloc_and_restore_part(len);
    }

    /// 将 [0,mid)元素移动到末尾，正为左移，负值为右移
    #[inline(always)]
    pub fn rotate(&mut self, mv: isize) {
        if T::IS_ZST {
            return;
        }
        assert!(!self.is_empty(), "call rotate while this is empty");
        // 有一个特例，当元素刚好填满的时候不需要拷贝就可以衔接到被循环的部分
        // 1 2 3 4 5 1 2 3 4 5
        //
        if self.is_full() {
            self.head = (self.head() as isize + mv).rem_euclid(self.capacity() as isize) as usize;
            return;
        }
        let idx = mv.rem_euclid(self.len() as isize) as usize;
        // 比如 刚好要移动 len 或 -len 就相当于没有移动
        if idx.is_zero() {
            return;
        }
        self[..idx].reverse();
        self[idx..].reverse();
        self.reverse();
    }
}

impl<T> Deref for SliceRingBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target { self.as_slice() }
}

impl<T> DerefMut for SliceRingBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { self.as_mut_slice() }
}

impl<T> Default for SliceRingBuffer<T> {
    #[inline(always)]
    fn default() -> Self { Self::new() }
}

impl<T> AsRef<[T]> for SliceRingBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] { self }
}

impl<T> AsMut<[T]> for SliceRingBuffer<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] { &mut *self }
}

impl<T: PartialEq> PartialEq for SliceRingBuffer<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool { self.as_slice() == other.as_slice() }
}

impl<T: Eq> Eq for SliceRingBuffer<T> {}

impl<T: PartialOrd> PartialOrd for SliceRingBuffer<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.as_slice().partial_cmp(other.as_slice()) }
}

impl<T: Ord> Ord for SliceRingBuffer<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering { self.as_slice().cmp(other.as_slice()) }
}

#[derive(Debug)]
pub struct IntoIter<T> {
    inner: SliceRingBuffer<T>,
}

impl<T> ExactSizeIterator for IntoIter<T> {
    #[inline(always)]
    fn len(&self) -> usize { self.inner.len() }
}

impl<T> IntoIterator for SliceRingBuffer<T> {
    type IntoIter = IntoIter<T>;
    type Item = T;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter { IntoIter { inner: self } }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> { self.inner.pop_front() }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> { self.inner.pop_back() }
}

impl<T> FusedIterator for IntoIter<T> {}

unsafe impl<T: Send> Send for IntoIter<T> {}

unsafe impl<T: Sync> Sync for IntoIter<T> {}

#[derive(Debug)]
pub struct Drain<'a, T> {
    // drain_start is stored as inner.len, restore when drain was dropped
    drain_len: usize,
    head: usize,
    remaining: usize,
    new_len: usize,
    inner: NonNull<SliceRingBuffer<T>>,
    _marker: PhantomData<&'a T>, // Needed to make Drain covariant over T
}
// origin_len = drain_len + new_len
impl<T> Drain<'_, T> {
    #[inline(always)]
    fn move_head_forward(&mut self, mv: usize) {
        self.head += mv;
        self.remaining -= mv;
    }

    #[inline(always)]
    fn move_tail_backward(&mut self, mv: usize) { self.remaining -= mv; }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        let inner = unsafe { self.inner.as_mut() };
        if T::IS_ZST || self.drain_len.is_zero() {
            inner.len = self.new_len;
            return;
        }
        let drain_start = inner.len();
        let drop_start = drain_start + self.head;
        let drop_len = self.remaining;
        if needs_drop::<T>() && !drop_len.is_zero() {
            unsafe {
                drop_in_place(self.inner.as_mut().as_mut_slice().get_unchecked_mut(drop_start..drop_start + drop_len))
            }
        }
        use Ordering::*;
        // 相对 head 位置
        let front_len = drain_start;
        let back_len = self.new_len - drain_start; // origin_len(drain_len + new_len) - drain_len - drain_start
        let ptr = unsafe { inner.buf.as_ptr().add(inner.head()) };
        match back_len.cmp(&front_len) {
            Less | Equal => {
                unsafe {
                    let dst = ptr.add(drain_start);
                    let src = dst.add(self.drain_len) as *const _;
                    copy_nonoverlapping(src, dst, back_len);
                    inner.len = self.new_len;
                };
            }
            Greater => unsafe {
                let src = ptr as *const _;
                let dst = ptr.add(self.drain_len);
                copy_nonoverlapping(src, dst, front_len);
                inner.head += self.drain_len;
                inner.len = self.new_len;
            },
        }
    }
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.remaining.is_zero() {
            return None;
        }
        if T::IS_ZST {
            self.move_head_forward(1);
            return Some(unsafe { zeroed() });
        }
        let drain_start = unsafe { self.inner.as_ref().len() };
        let pos = unsafe { self.inner.as_ref().head() } + drain_start + self.head;
        let target = unsafe { self.inner.as_mut().buf.get_unchecked_mut(pos) };
        self.move_head_forward(1);
        Some(unsafe { target.assume_init_read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) { (self.remaining, Some(self.remaining)) }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining.is_zero() {
            return None;
        }
        if T::IS_ZST {
            self.move_tail_backward(1);
            return Some(unsafe { zeroed() });
        }
        let drain_start = unsafe { self.inner.as_ref().len() };
        let pos = unsafe { self.inner.as_ref().head() } + drain_start + self.head + self.remaining - 1;
        let target = unsafe { self.inner.as_mut().buf.get_unchecked_mut(pos) };
        self.move_tail_backward(1);
        Some(unsafe { target.assume_init_read() })
    }
}
impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T> FusedIterator for Drain<'a, T> {}

impl<T> From<Vec<T>> for SliceRingBuffer<T> {
    #[inline(always)]
    fn from(vec: Vec<T>) -> Self { Self::from_iter(vec) }
}

impl<T> FromIterator<T> for SliceRingBuffer<T> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iterator = iter.into_iter();
        let (lower, upper) = iterator.size_hint();
        // Use the upper bound if available, otherwise use lower bound
        let cap = upper.unwrap_or(lower);
        let mut rb = SliceRingBuffer::with_capacity(cap);
        rb.extend(iterator);
        rb
    }
}

impl<T> Extend<T> for SliceRingBuffer<T> {
    #[inline(always)]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let (lower, upper) = iterator.size_hint();
        // Use the upper bound if available, otherwise use lower bound
        let reserve_size = upper.unwrap_or(lower);
        self.reserve(reserve_size);
        for item in iterator {
            self.push_back(item);
        }
    }
}

impl<T: Clone> From<&[T]> for SliceRingBuffer<T> {
    #[inline(always)]
    fn from(slice: &[T]) -> Self {
        let mut buffer = SliceRingBuffer::with_capacity(slice.len());
        buffer.extend(slice.iter().cloned());
        buffer
    }
}

impl<T: Clone, const N: usize> From<[T; N]> for SliceRingBuffer<T> {
    #[inline(always)]
    fn from(arr: [T; N]) -> Self {
        let mut buffer = SliceRingBuffer::with_capacity(N);
        buffer.extend(arr.iter().cloned());
        buffer
    }
}

unsafe impl<T> Send for SliceRingBuffer<T> where T: Send {}
unsafe impl<T> Sync for SliceRingBuffer<T> where T: Sync {}

impl<T> Drop for SliceRingBuffer<T> {
    fn drop(&mut self) {
        let p = self.as_mut_slice() as *mut [_];
        unsafe { drop_in_place(p) };
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::mirrored::allocation_granularity;

    use super::*;
    use core::{iter, ops::Not, sync::atomic::AtomicUsize};
    use num::Integer;
    use std::rc::Rc;

    #[test]
    fn test_new() {
        let rb: SliceRingBuffer<i32> = SliceRingBuffer::new();
        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
        assert_eq!(rb.capacity(), 0);
    }

    #[test]
    fn test_new_zst() {
        let zst_rb: SliceRingBuffer<()> = SliceRingBuffer::new();
        assert_eq!(zst_rb.capacity(), MAX_PHYSICAL_BUF_SIZE);
        assert_eq!(zst_rb.len(), 0);
        assert!(zst_rb.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let rb: SliceRingBuffer<i32> = SliceRingBuffer::with_capacity(1);
        assert_eq!(rb.len(), 0);
        assert!(!rb.is_full());
        assert!(rb.is_empty());
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>());
    }

    #[test]
    fn test_with_capacity_zst() {
        let zst_rb: SliceRingBuffer<()> = SliceRingBuffer::with_capacity(1);
        assert!(zst_rb.is_empty());
        assert!(!zst_rb.is_full());
        assert_eq!(zst_rb.len(), 0);
        assert_eq!(zst_rb.capacity(), 1);
    }

    #[test]
    fn test_push_back_and_pop_front() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(1i32);
        rb.push_back(2);
        rb.push_back(3);

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>()); // 16KiB ag

        assert_eq!(rb.pop_front(), Some(1));
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_back_and_pop_back() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(1i32);
        rb.push_back(2);
        rb.push_back(3);

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>()); // 16KiB ag

        assert_eq!(rb.pop_back(), Some(3));
        assert_eq!(rb.pop_back(), Some(2));
        assert_eq!(rb.pop_back(), Some(1));
        assert_eq!(rb.pop_back(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_front_and_pop_front() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_front(1i32);
        rb.push_front(2);
        rb.push_front(3);

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>()); // 16KiB ag

        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(1));
        assert_eq!(rb.pop_front(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_front_and_pop_back() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_front(1i32);
        rb.push_front(2);
        rb.push_front(3);

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.as_slice(), &[3, 2, 1]);

        assert_eq!(rb.pop_back(), Some(1));
        assert_eq!(rb.pop_back(), Some(2));
        assert_eq!(rb.pop_back(), Some(3));
        assert_eq!(rb.pop_back(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_back_and_pop_front_zst() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(());
        rb.push_back(());
        rb.push_back(());

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), 3);
        assert_eq!(rb.as_slice(), &[(), (), ()]);

        assert_eq!(rb.pop_front(), Some(()));
        assert_eq!(rb.pop_front(), Some(()));
        assert_eq!(rb.pop_front(), Some(()));
        assert_eq!(rb.pop_front(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_back_and_pop_back_zst() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_front(());
        rb.push_front(());
        rb.push_front(());

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), 3);
        assert_eq!(rb.as_slice(), &[(), (), ()]);

        assert_eq!(rb.pop_back(), Some(()));
        assert_eq!(rb.pop_back(), Some(()));
        assert_eq!(rb.pop_back(), Some(()));
        assert_eq!(rb.pop_back(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_front_and_pop_back_zst() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(());
        rb.push_back(());
        rb.push_back(());

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), 3);
        assert_eq!(rb.as_slice(), &[(), (), ()]);

        assert_eq!(rb.pop_back(), Some(()));
        assert_eq!(rb.pop_back(), Some(()));
        assert_eq!(rb.pop_back(), Some(()));
        assert_eq!(rb.pop_back(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_front_and_pop_front_zst() {
        let mut rb: SliceRingBuffer<()> = SliceRingBuffer::with_capacity(3);
        rb.push_front(());
        rb.push_front(());
        rb.push_front(());

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), 3);
        assert_eq!(rb.as_slice(), &[(), (), ()]);

        assert_eq!(rb.pop_front(), Some(()));
        assert_eq!(rb.pop_front(), Some(()));
        assert_eq!(rb.pop_front(), Some(()));
        assert_eq!(rb.pop_front(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_wrap_push_front_and_pop_back() {
        let ag = allocation_granularity();
        let mut rb = SliceRingBuffer::with_capacity(ag + 1);
        for _ in 0..ag - 1 {
            rb.push_front(false);
        }
        rb.push_front(true);

        assert_eq!(rb.len(), ag);
        assert!(!rb.is_empty());

        for _ in 0..ag - 1 {
            assert_eq!(rb.pop_back(), Some(false));
        }
        assert_eq!(rb.pop_back(), Some(true));
        assert_eq!(rb.pop_back(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_wrap_push_back_and_pop_front() {
        let ag = allocation_granularity();
        let mut rb = SliceRingBuffer::with_capacity(ag + 1);
        for _ in 0..ag - 1 {
            rb.push_back(false);
        }
        rb.push_back(true);

        assert_eq!(rb.len(), ag);
        assert!(!rb.is_empty());

        for _ in 0..ag - 1 {
            assert_eq!(rb.pop_front(), Some(false));
        }
        assert_eq!(rb.pop_front(), Some(true));
        assert_eq!(rb.pop_front(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_full().not());
        assert!(rb.is_empty());
    }

    #[test]
    fn test_wrap_push_back_and_pop_front_zst() {
        let mut zst_rb = SliceRingBuffer::new();
        zst_rb.push_back(());
        assert!(!zst_rb.is_empty());
        assert!(!zst_rb.is_full());
        assert_eq!(zst_rb.len(), 1);
        assert_eq!(zst_rb.capacity(), MAX_PHYSICAL_BUF_SIZE);
        zst_rb.pop_back();
        assert!(zst_rb.is_empty());
        assert_eq!(zst_rb.capacity(), MAX_PHYSICAL_BUF_SIZE);
        assert!(!zst_rb.is_full());
    }

    #[test]
    fn test_wrap_around() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(3);
        assert_eq!(rb.pop_front(), Some(1));
        rb.push_back(4);

        assert_eq!(rb.as_slice(), &[2, 3, 4]);
        assert_eq!(rb.len(), 3);

        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), Some(4));
        assert_eq!(rb.pop_front(), None);
    }

    #[test]
    fn test_deref() {
        let mut rb = SliceRingBuffer::with_capacity(4);
        rb.push_back(10);
        rb.push_back(20);
        rb.push_back(30);

        assert_eq!(&[10, 20, 30], &*rb);
    }

    #[test]
    fn test_deref_zst() {
        let rb = SliceRingBuffer::from_iter(iter::repeat_n((), 8));
        assert_eq!(&*rb, &[(); 8]);
    }

    #[test]
    fn test_front_back() {
        let mut rb = SliceRingBuffer::with_capacity(5);
        assert!(rb.front().is_none());
        assert!(rb.back().is_none());

        rb.push_back(1);
        assert_eq!(rb.front(), Some(&1));
        assert_eq!(rb.back(), Some(&1));

        rb.push_back(2);
        assert_eq!(rb.front(), Some(&1));
        assert_eq!(rb.back(), Some(&2));

        rb.push_front(0);
        assert_eq!(rb.front(), Some(&0));
        assert_eq!(rb.back(), Some(&2));
    }

    #[test]
    fn test_front_back_zst() {
        let mut rb = SliceRingBuffer::new();
        rb.push_back(());
        rb.pop_back();
        assert!(rb.front().is_none());
        assert!(rb.back().is_none());
        rb.push_back(());
        assert!(rb.front().is_some());
        assert!(rb.back().is_some());
        assert_eq!(rb.pop_back(), Some(()));
    }

    #[test]
    fn test_clear() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(1);
        rb.push_back(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_clear_zst() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(());
        rb.push_back(());
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_into_iter() {
        let mut rb = SliceRingBuffer::with_capacity(4);
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(3);

        let mut iter = rb.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_zst() {
        let mut rb = SliceRingBuffer::with_capacity(4);
        rb.push_back(());
        rb.push_back(());
        rb.push_back(());

        let mut iter = rb.into_iter();
        assert_eq!(iter.next(), Some(()));
        assert_eq!(iter.next_back(), Some(()));
        assert_eq!(iter.next(), Some(()));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_from_iter() {
        let data = vec![1, 2, 3, 4, 5];
        let rb: SliceRingBuffer<i32> = data.into_iter().collect();
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_from_iter_zst() {
        let data = vec![(); 5];
        let rb = data.into_iter().collect::<SliceRingBuffer<_>>();
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.as_slice(), &[(); 5]);
    }

    #[test]
    fn test_from_array() {
        let arr = [1, 2, 3, 4, 5];
        let rb = SliceRingBuffer::from(arr);
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.as_slice(), &arr);
    }

    #[test]
    fn test_from_array_zst() {
        let arr = [(); 5];
        let rb = SliceRingBuffer::from(arr);
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.as_slice(), &arr);
    }

    #[test]
    fn test_from_slice() {
        let slice = [1, 1, 4, 5, 1, 5].as_slice();
        let rb = SliceRingBuffer::from(slice);
        assert_eq!(rb.len(), 6);
        assert_eq!(rb.as_slice(), slice);
    }

    #[test]
    fn test_from_slice_zst() {
        let slice = [(); 6].as_slice();
        let rb = SliceRingBuffer::from(slice);
        assert_eq!(rb.len(), 6);
        assert_eq!(rb.as_slice(), slice);
    }

    #[test]
    fn test_extend() {
        let mut rb = SliceRingBuffer::from(vec![1, 2]);
        rb.extend(vec![3, 4, 5]);
        assert_eq!(rb.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_extend_zst() {
        let mut rb = SliceRingBuffer::from(vec![(); 2]);
        rb.extend(vec![(); 4]);
        assert_eq!(rb.as_slice(), &[(); 6]);
    }

    #[test]
    fn test_realloc() {
        let ag = allocation_granularity();
        let mut rb = SliceRingBuffer::with_capacity(ag);
        for i in 0..ag * 2 {
            if i.is_even() {
                rb.push_back(i);
            } else {
                rb.push_front(i);
            }
        }
        rb.push_back(114514);
        assert!(rb.capacity() > ag * 3 / size_of::<usize>());
        assert_eq!(rb.len(), ag * 2 + 1);
    }

    #[test]
    fn test_drop() {
        use core::sync::atomic::Ordering;
        let drop_counter = Rc::new(AtomicUsize::new(0));

        struct DropTracker {
            counter: Rc<AtomicUsize>,
        }

        impl Drop for DropTracker {
            fn drop(&mut self) {
                self.counter
                    .fetch_update(Ordering::Relaxed, Ordering::Acquire, |x| Some(x + 1))
                    .expect("atomic update failed");
            }
        }

        let clone_tracker = || DropTracker { counter: drop_counter.clone() };
        let mut rb = SliceRingBuffer::new();
        rb.push_back(clone_tracker());
        let _ = rb.pop_front().unwrap();
        assert_eq!(drop_counter.load(Ordering::Acquire), 1);

        let ag = allocation_granularity();
        let count = ag / size_of::<DropTracker>();
        for _ in 0..count {
            rb.push_back(clone_tracker());
        }
        assert_eq!(rb.len(), count);
        let _ = rb.pop_back();
        assert_eq!(rb.len(), count - 1);
        assert_eq!(drop_counter.load(Ordering::Acquire), 2);

        rb.push_back(clone_tracker());
        rb.push_back(clone_tracker());
        // realloc here

        assert_eq!(rb.len(), count + 1);
        drop(rb);
        assert_eq!(drop_counter.load(Ordering::Acquire), count + 3);
    }

    #[test]
    fn test_rotate() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        rb.rotate(2);
        assert_eq!(rb.as_slice(), &[3, 4, 5, 1, 2]);

        rb.rotate(-1);
        assert_eq!(rb.as_slice(), &[2, 3, 4, 5, 1]);

        rb.rotate(5);
        assert_eq!(rb.as_slice(), &[2, 3, 4, 5, 1]);
        rb.rotate(0);
        assert_eq!(rb.as_slice(), &[2, 3, 4, 5, 1]);
        rb.rotate(-5);
        assert_eq!(rb.as_slice(), &[2, 3, 4, 5, 1]);

        let mut rb = SliceRingBuffer::from_iter(iter::repeat_n(false, allocation_granularity()));
        *rb.last_mut().unwrap() = true;
        rb.rotate(allocation_granularity() as isize);
        assert_eq!(rb.last(), Some(&true));
        rb.rotate((allocation_granularity() as isize).neg());
        assert_eq!(rb.last(), Some(&true));
        rb.rotate(-1);
        assert_eq!(rb.first(), Some(&true));
    }

    #[test]
    fn test_rotate_zst() {
        let mut rb = SliceRingBuffer::from([(); 5]);
        rb.rotate(1);
        assert_eq!(rb.first(), Some(&()));
        rb.rotate(-1);
        assert_eq!(rb.first(), Some(&()));
    }

    #[test]
    fn test_split_off() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        let rb2 = rb.split_off(3);

        assert_eq!(rb.as_slice(), &[1, 2, 3]);
        assert_eq!(rb2.as_slice(), &[4, 5]);

        let ag = allocation_granularity();
        let mut rb = SliceRingBuffer::from_iter(std::iter::repeat_n(true, ag - 8));
        for _ in 0..8 {
            rb.push_back(false);
        }
        let right = rb.split_off(ag - 8);
        assert_eq!(right.as_slice(), &[false; 8]);
        assert_eq!(rb.len(), ag - 8);
        for item in rb.as_slice() {
            assert!(*item);
        }
    }

    #[test]
    fn test_split_off_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 8]);
        let right = zst_rb.split_off(4);
        assert_eq!(right.len(), 4);
        assert_eq!(zst_rb.len(), 4);
    }

    #[test]
    fn test_append() {
        let mut rb1 = SliceRingBuffer::from_iter(iter::repeat_n(true, allocation_granularity()));
        let mut rb2 = SliceRingBuffer::from_iter(iter::repeat_n(false, allocation_granularity()));

        rb1.append(&mut rb2);

        for _ in 0..allocation_granularity() {
            assert!(rb1.pop_front().unwrap());
        }
        for _ in 0..allocation_granularity() {
            assert!(!rb1.pop_back().unwrap());
        }
        assert!(rb2.is_empty());
    }

    #[test]
    fn test_append_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 8]);
        zst_rb.append(&mut zst_rb.clone());
        assert!(!zst_rb.is_empty());
        assert_eq!(zst_rb.len(), 16);
    }

    #[test]
    fn test_truncate_back() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        rb.truncate_back(3);
        assert_eq!(rb.as_slice(), &[1, 2, 3]);
        rb.truncate_back(5);
        assert_eq!(rb.as_slice(), &[1, 2, 3]);

        let mut rb = SliceRingBuffer::from_iter(0usize..allocation_granularity());
        rb.extend(0..allocation_granularity());
        for i in 0..allocation_granularity() {
            assert_eq!(i, rb[i])
        }

        for (idx, val) in (allocation_granularity()..allocation_granularity() * 2).enumerate() {
            assert_eq!(idx, rb[val])
        }
    }

    #[test]
    fn test_truncate_back_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 96]);
        zst_rb.truncate_back(3);
        assert_eq!(zst_rb.as_slice(), &[(); 3])
    }

    #[test]
    fn test_remove() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(rb.remove(2), 3);
        assert_eq!(rb.as_slice(), &[1, 2, 4, 5]);
    }

    #[test]
    fn test_remove_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 3]);
        assert_eq!(zst_rb.remove(0), ());
        assert_eq!(zst_rb.as_slice(), &[(), ()]);
        zst_rb.remove(1);
        assert_eq!(zst_rb.len(), 1);
    }

    #[test]
    #[should_panic]
    fn test_remove_out_of_bounds() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3]);
        rb.remove(3);
    }

    #[test]
    #[should_panic]
    fn test_remove_out_of_bounds_zst() {
        let mut rb = SliceRingBuffer::from([(); 8]);
        rb.remove(10);
    }

    #[test]
    fn test_swap_remove() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(rb.swap_remove(1), Some(2));
        assert_eq!(rb.as_slice(), &[1, 5, 3, 4]);
    }

    #[test]
    fn test_swap_remove_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 8]);
        zst_rb.swap_remove(3);
        assert_eq!(zst_rb.len(), 7);
    }

    #[test]
    fn test_try_insert() {
        let mut rb = SliceRingBuffer::with_capacity(5);
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(4);
        rb.push_back(5);

        assert!(rb.try_insert(2, 3).is_some());
        assert_eq!(rb.as_slice(), &[1, 2, 3, 4, 5]);
    }
    #[test]
    fn test_try_insert_zst() {
        let mut rb = SliceRingBuffer::from([(); 5]);
        assert!(rb.try_insert(2, ()).is_none());
        assert_eq!(rb.len(), 5);
    }

    #[test]
    fn test_insert() {
        let ag = allocation_granularity();
        let mut rb = SliceRingBuffer::from(vec![false; ag]);
        rb.insert(2, true);
        assert!(rb.is_full().not());
        assert_eq!(rb.swap_remove(2), Some(true));
        assert_eq!(rb.swap_remove(0), Some(false));
        assert_eq!(rb.swap_remove(rb.len() - 1), Some(false));
        assert_eq!(rb.capacity(), 2 * ag);
    }

    #[test]
    fn test_insert_zst() {
        let ag = allocation_granularity();
        let mut rb = SliceRingBuffer::from(vec![(); ag]);
        rb.insert(2, ());
        assert!(rb.is_full());
        assert_eq!(rb.swap_remove(2), Some(()));
        assert_eq!(rb.capacity(), ag);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut rb = SliceRingBuffer::with_capacity(20);
        rb.extend(0..5);
        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>());
        assert_eq!(rb.as_slice(), &[0, 1, 2, 3, 4]);
        let mut rb = SliceRingBuffer::with_capacity(2 * allocation_granularity());
        rb.push_back(1);
        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>());
    }

    #[test]
    fn test_shrink_to_fit_zst() {
        let mut rb = SliceRingBuffer::with_capacity(20);
        rb.extend([(); 5]);
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.capacity(), 20);
        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), 5);
    }

    #[test]
    fn test_slice_ring_buffer_empty() {
        let rb: SliceRingBuffer<i32> = SliceRingBuffer::new();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 0);
        assert_eq!(rb.as_slice(), &[]);
    }

    #[test]
    fn test_slice_ring_buffer_empty_zst() {
        let rb: SliceRingBuffer<()> = SliceRingBuffer::new();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), MAX_PHYSICAL_BUF_SIZE);
        assert_eq!(rb.as_slice(), &[]);
    }

    #[test]
    fn test_index_access() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3]);
        assert_eq!(rb[0], 1);
        assert_eq!(rb[1], 2);
        assert_eq!(rb[2], 3);

        rb[1] = 5;
        assert_eq!(rb[1], 5);
        assert_eq!(rb.as_slice(), &[1, 5, 3]);
    }

    #[test]
    fn test_reserve_and_shrink() {
        let mut rb = SliceRingBuffer::with_capacity(5);
        rb.extend(0..5);
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>());

        rb.reserve(10);
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>());

        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), allocation_granularity() / size_of::<i32>());
    }

    #[test]
    fn test_reserve_and_shrink_zst() {
        let mut rb: SliceRingBuffer<()> = SliceRingBuffer::new();
        assert_eq!(rb.capacity(), MAX_PHYSICAL_BUF_SIZE);

        rb.reserve(10);
        assert_eq!(rb.capacity(), MAX_PHYSICAL_BUF_SIZE);

        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), 0);
    }

    #[test]
    fn test_reserve() {
        let mut rb = SliceRingBuffer::from([false; 10]);
        rb.reserve(allocation_granularity() * 2);
        assert_eq!(rb.capacity(), allocation_granularity() * 3);
        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), allocation_granularity());

        let mut rb = SliceRingBuffer::from(vec![false; allocation_granularity()]);
        rb.reserve(allocation_granularity() / 2);
        assert_eq!(rb.capacity(), allocation_granularity() * 2);
    }

    #[test]
    fn test_reserve_zst() {
        let mut rb = SliceRingBuffer::from([(); 10]);
        rb.reserve(32);
        assert_eq!(rb.capacity(), 42);
        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), 10);
        rb.reserve(5);
        assert_eq!(rb.capacity(), 15);
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut rb = SliceRingBuffer::new();
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(3);

        // Test get
        assert_eq!(rb.get(0), Some(&1));
        assert_eq!(rb.get(1), Some(&2));
        assert_eq!(rb.get(2), Some(&3));
        assert_eq!(rb.get(3), None);

        // Test get_mut
        if let Some(elem) = rb.get_mut(1) {
            *elem = 7;
        }
        assert_eq!(rb.as_slice(), &[1, 7, 3]);

        // Test out of bounds
        assert_eq!(rb.get_mut(3), None);
    }

    #[test]
    fn test_get_and_get_mut_zst() {
        let rb = SliceRingBuffer::from([(); 10]);
        assert!(rb.get(3).is_some());
        assert!(rb.get(0).is_some());
        assert!(rb.get(10).is_none())
    }

    #[test]
    fn test_drain_middle() {
        let mut buf = SliceRingBuffer::from_iter(0..6); // [0, 1, 2, 3, 4, 5]
        let drained: Vec<_> = buf.drain(2..4).collect(); // Drain [2, 3]

        assert_eq!(drained, vec![2, 3]);
        assert_eq!(buf.as_slice(), &[0, 1, 4, 5]);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_drain_all() {
        let mut buf = SliceRingBuffer::from_iter(0..5);
        let drained: Vec<_> = buf.drain(..).collect();

        assert_eq!(drained, vec![0, 1, 2, 3, 4]);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_drain_from_start() {
        let mut buf = SliceRingBuffer::from_iter(0..5);
        let drained: Vec<_> = buf.drain(..2).collect(); // Drain [0, 1]

        assert_eq!(drained, vec![0, 1]);
        assert_eq!(buf.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_drain_to_end() {
        let mut buf = SliceRingBuffer::from_iter(0..5);
        let drained: Vec<_> = buf.drain(3..).collect(); // Drain [3, 4]

        assert_eq!(drained, vec![3, 4]);
        assert_eq!(buf.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_drain_empty_range() {
        let mut buf = SliceRingBuffer::from_iter(0..3);
        let drained: Vec<_> = buf.drain(1..1).collect();

        assert!(drained.is_empty());
        assert_eq!(buf.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_drain_drop_incomplete_move_back() {
        // 测试 back_len <= front_len 的情况，移动后面的元素
        let mut buf = SliceRingBuffer::from_iter(0..10); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        // Drain 2..5 -> [2, 3, 4]. front_len = 2, back_len = 5. back_len > front_len
        // 这里应该是移动前面的元素
        {
            let mut drainer = buf.drain(2..5);
            assert_eq!(drainer.next(), Some(2));
            // 迭代器在这里被 drop，剩余的 [3, 4] 会被清理
        }

        // 最终结果应该是移除了 [2, 3, 4]
        assert_eq!(buf.as_slice(), &[0, 1, 5, 6, 7, 8, 9]);
        assert_eq!(buf.len(), 7);
    }

    #[test]
    fn test_drain_drop_incomplete_move_front() {
        // 测试 back_len > front_len 的情况，移动前面的元素
        let mut buf = SliceRingBuffer::from_iter(0..10); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        // Drain 7..9 -> [7, 8]. front_len = 7, back_len = 1. front_len > back_len
        {
            let mut drainer = buf.drain(7..9);
            assert_eq!(drainer.next(), Some(7));
            // 迭代器在这里被 drop，剩余的 [8] 会被清理
        }

        // 最终结果应该是移除了 [7, 8]
        assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 9]);
        assert_eq!(buf.len(), 8);
    }

    #[test]
    fn test_drain_zst() {
        let mut buf = SliceRingBuffer::<()>::new();
        buf.push_back(());
        buf.push_back(());
        buf.push_back(());
        buf.push_back(());
        buf.push_back(());
        assert_eq!(buf.len(), 5);

        let mut drained = buf.drain(1..4);

        assert_eq!(drained.len(), 3);
        assert_eq!(drained.next(), Some(()));
        assert_eq!(drained.next(), Some(()));
        assert_eq!(drained.next(), Some(()));
        assert_eq!(drained.next(), None);

        // Drop drainer
        drop(drained);

        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_drain_zst_drop_incomplete() {
        let mut buf = SliceRingBuffer::<()>::new();
        buf.extend(iter::repeat_n((), 10));
        assert_eq!(buf.len(), 10);

        {
            let mut drainer = buf.drain(2..8); // Drain 6 elements
            assert_eq!(drainer.next(), Some(()));
            assert_eq!(drainer.next(), Some(()));
            // Drainer is dropped here
        }

        // 即使只迭代了2个，也应该移除全部6个元素
        assert_eq!(buf.len(), 4);
    }

    #[test]
    #[should_panic]
    fn test_drain_panic_end_out_of_bounds() {
        let mut buf = SliceRingBuffer::from_iter(0..5);
        let _ = buf.drain(..6);
    }

    #[test]
    #[should_panic]
    fn test_drain_panic_start_out_of_bounds() {
        // start > len 会导致 end > len (因为 end >= start)
        let mut buf = SliceRingBuffer::from_iter(0..5);
        let _ = buf.drain(6..);
    }

    #[test]
    fn test_drain_next_back_only() {
        let mut buf = SliceRingBuffer::from_iter(0..10);
        // Drain [2, 3, 4, 5, 6, 7]
        let mut drainer = buf.drain(2..8);

        assert_eq!(drainer.next_back(), Some(7));
        assert_eq!(drainer.next_back(), Some(6));
        assert_eq!(drainer.next_back(), Some(5));
        assert_eq!(drainer.next_back(), Some(4));
        assert_eq!(drainer.next_back(), Some(3));
        assert_eq!(drainer.next_back(), Some(2));
        assert_eq!(drainer.next_back(), None);

        // Drop the now-empty drainer
        drop(drainer);
        assert_eq!(buf.as_slice(), &[0, 1, 8, 9]);
    }

    #[test]
    fn test_drain_mixed_next_and_next_back() {
        let mut buf = SliceRingBuffer::from_iter(0..10);
        // Drain [2, 3, 4, 5, 6, 7]
        let mut drainer = buf.drain(2..8);

        assert_eq!(drainer.next(), Some(2));
        assert_eq!(drainer.next_back(), Some(7));
        assert_eq!(drainer.len(), 4); // Remaining: [3, 4, 5, 6]

        assert_eq!(drainer.next(), Some(3));
        assert_eq!(drainer.next_back(), Some(6));
        assert_eq!(drainer.len(), 2); // Remaining: [4, 5]

        let remaining: Vec<_> = drainer.collect();
        assert_eq!(remaining, vec![4, 5]);

        assert_eq!(buf.as_slice(), &[0, 1, 8, 9]);
    }

    #[test]
    fn test_drain_next_back_until_empty() {
        let mut buf = SliceRingBuffer::from_iter(0..5);
        let mut drainer = buf.drain(1..4); // [1, 2, 3]

        assert_eq!(drainer.next(), Some(1));
        assert_eq!(drainer.next_back(), Some(3));
        assert_eq!(drainer.next(), Some(2)); // Last element

        assert_eq!(drainer.next(), None);
        assert_eq!(drainer.next_back(), None);
    }

    #[test]
    fn test_drain_next_back_zst() {
        let mut buf = SliceRingBuffer::<()>::new();
        buf.extend(iter::repeat_n((), 10));

        let mut drainer = buf.drain(3..7); // Drain 4 elements
        assert_eq!(drainer.len(), 4);

        assert_eq!(drainer.next_back(), Some(()));
        assert_eq!(drainer.len(), 3);

        assert_eq!(drainer.next(), Some(()));
        assert_eq!(drainer.len(), 2);

        assert_eq!(drainer.next_back(), Some(()));
        assert_eq!(drainer.len(), 1);

        assert_eq!(drainer.next_back(), Some(()));
        assert_eq!(drainer.len(), 0);

        assert_eq!(drainer.next_back(), None);

        drop(drainer);
        assert_eq!(buf.len(), 6);
    }

    #[test]
    fn test_split_to() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);

        // 从中间分割
        let front = rb.split_to(3);
        assert_eq!(front.as_slice(), &[1, 2, 3]);
        assert_eq!(rb.as_slice(), &[4, 5]);
        assert_eq!(rb.len(), 2);

        // 从头部分割 (结果为空)
        let mut rb2 = SliceRingBuffer::from(vec![10, 20]);
        let front2 = rb2.split_to(0);
        assert!(front2.is_empty());
        assert_eq!(rb2.as_slice(), &[10, 20]);

        // 分割所有元素
        let mut rb3 = SliceRingBuffer::from(vec![10, 20]);
        let front3 = rb3.split_to(2);
        assert_eq!(front3.as_slice(), &[10, 20]);
        assert!(rb3.is_empty());
    }

    #[test]
    fn test_split_to_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 8]);
        let front = zst_rb.split_to(5);
        assert_eq!(front.len(), 5);
        assert_eq!(zst_rb.len(), 3);
    }

    #[test]
    fn test_clone() {
        let mut rb1 = SliceRingBuffer::with_capacity(5);
        rb1.extend(0..5); // [0, 1, 2, 3, 4]
        assert_eq!(rb1.pop_front(), Some(0));
        assert_eq!(rb1.pop_front(), Some(1));
        rb1.push_back(5); // 此时缓冲区是环绕状态: [2, 3, 4, 5]，head 在索引 2

        let rb2 = rb1.clone();

        // 确保它们相等但相互独立
        assert_eq!(rb1.as_slice(), rb2.as_slice());
        assert_eq!(rb1.len(), rb2.len());
        assert_eq!(rb1.head(), rb2.head());

        // 修改一个，另一个不应受影响
        rb1.push_back(100);
        assert_ne!(rb1.as_slice(), rb2.as_slice());
        assert_eq!(rb2.as_slice(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_clone_zst() {
        let mut rb1: SliceRingBuffer<()> = SliceRingBuffer::with_capacity(10);
        rb1.extend(iter::repeat_n((), 5));
        let rb2 = rb1.clone();

        assert_eq!(rb1.len(), 5);
        assert_eq!(rb2.len(), 5);

        rb1.push_back(());
        assert_eq!(rb1.len(), 6);
        assert_eq!(rb2.len(), 5);
    }

    #[test]
    fn test_shrink_to() {
        let mut rb = SliceRingBuffer::with_capacity(allocation_granularity() * 2);
        rb.extend(0..10);

        // 缩容到比当前长度大的容量
        rb.shrink_to(allocation_granularity());
        assert_eq!(rb.capacity(), allocation_granularity());
        assert_eq!(rb.as_slice(), &(0..10).collect::<Vec<_>>());

        // 缩容到比当前容量还大，应该什么都不做
        rb.shrink_to(allocation_granularity() * 2);
        assert_eq!(rb.capacity(), allocation_granularity());
    }

    #[test]
    #[should_panic(expected = "min_capacity (5) cannot be less than current length (10)")]
    fn test_shrink_to_panic() {
        let mut rb = SliceRingBuffer::from(vec![0; 10]);
        rb.shrink_to(5); // 这里应该 panic
    }

    #[test]
    fn test_try_push_full() {
        let mut rb = SliceRingBuffer::from(vec![false; allocation_granularity()]);
        assert_eq!(rb.len(), allocation_granularity());
        assert_eq!(rb.capacity(), allocation_granularity());
        assert!(rb.is_full());

        // 尝试 push 应该失败并返回 None
        assert!(rb.try_push_back(true).is_none());
        assert!(rb.try_push_front(true).is_none());

        // 缓冲区内容应保持不变
        assert_eq!(rb.len(), allocation_granularity());
    }
    //todo truncate
    #[test]
    fn test_rotate_full() {
        let mut rb = SliceRingBuffer::from(vec![false; allocation_granularity()]);
        unsafe {
            *rb.get_unchecked_mut(0) = true;
        }
        assert_eq!(rb.get(0), Some(&true));
        assert_eq!(rb.head(), 0);
        // 对一个满的缓冲区进行 rotate 应该只移动 head
        rb.rotate(-2);
        assert_eq!(rb.head(), allocation_granularity() - 2);
        // 因为 head 移动了，所以切片视图会改变
        assert_eq!(rb.get(2), Some(&true));
    }

    #[test]
    fn test_eq_and_ord() {
        let rb1 = SliceRingBuffer::from(vec![1, 2, 3]);
        let mut rb2 = SliceRingBuffer::from(vec![1, 2, 3]);
        let rb3 = SliceRingBuffer::from(vec![1, 2, 4]);
        let rb4 = SliceRingBuffer::from(vec![1, 2]);

        // 测试相等性
        assert_eq!(rb1, rb2);
        assert_ne!(rb1, rb3);
        assert_ne!(rb1, rb4);

        // 测试顺序
        assert!(rb1 < rb3);
        assert!(rb4 < rb1);

        // 测试环绕状态下的缓冲区
        rb2.pop_front();
        rb2.push_back(4); // 变成 [2, 3, 4]，但 head != 0
        let rb5 = SliceRingBuffer::from(vec![2, 3, 4]); // head == 0
        assert_eq!(rb2, rb5);
    }
}
