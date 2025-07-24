#![feature(temporary_niche_types)]
#![feature(sized_type_properties)]
#![feature(ptr_as_uninit)]
#![feature(strict_overflow_ops)]
#![cfg_attr(not(any(feature = "use_std", test)), no_std)]

mod mirrored;

use core::{
    cmp::Ordering,
    iter::FromIterator,
    mem::{replace, MaybeUninit, SizedTypeProperties},
    ops::{Deref, DerefMut, Neg},
    ptr::{copy, copy_nonoverlapping, drop_in_place, read, slice_from_raw_parts_mut, write},
};
use mirrored::{mirrored_allocation_unit, MirroredBuffer, MAX_USIZE_WITHOUT_HIGHEST_BIT};
use num::Zero;
//todo 处理 zst

// 只需要保证三件事：
// head 被限制在0..cap，因为前半部分和后半部分是镜像的
// head + len <= cap， 因为不能越界
// cap <= isize::max //类型要求
pub struct SliceRingBuffer<T> {
    buf: MirroredBuffer<T>,
    head: usize, //was jailed in 0..capacity
    len: usize,  //0..=capacity
}

impl<T> SliceRingBuffer<T> {
    #[inline]
    pub fn new() -> Self { Self { buf: MirroredBuffer::new(), head: 0, len: 0 } }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self { Self { buf: MirroredBuffer::with_capacity(cap), head: 0, len: 0 } }

    #[inline]
    pub fn capacity(&self) -> usize {
        let cap = self.buf.capacity();
        debug_assert!(cap <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        cap
    }

    #[inline]
    pub fn len(&self) -> usize {
        let len = self.len;
        debug_assert!(len <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        len
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        let len = self.len;
        let cap = self.capacity();
        debug_assert!(len <= cap);
        len == cap
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let head = self.head;
        let len = self.len;
        debug_assert!(self.capacity() <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        debug_assert!(head <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        debug_assert!(head < self.capacity());
        debug_assert!(len <= self.capacity());
        unsafe { self.buf.virtual_slice_at_unchecked(head, len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let head = self.head;
        let len = self.len;
        debug_assert!(head <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        debug_assert!(head < self.capacity());
        debug_assert!(len <= self.capacity());
        unsafe { self.buf.virtual_slice_mut_at_unchecked(head, len) }
    }

    #[inline]
    pub unsafe fn uninit_slice(&mut self) -> &mut [MaybeUninit<T>] {
        let cap = self.capacity();
        let len = self.len;
        let head = self.head;
        debug_assert!(len.checked_add(head) < Some(cap * 2));
        // 安全性，为了预防溢出：只需要通过上面就可以确保它们的和在 isize::MAX 以内
        let uninit_start = (head + len) % cap;
        debug_assert!(len <= cap);
        let uninit_len = cap - len;
        self.buf.uninit_virtual_slice_mut_at(uninit_start, uninit_len)
    }

    #[inline]
    pub fn move_head(&mut self, mv: isize) {
        // cap 为 0 直接panic
        assert!(self.head < self.capacity());
        unsafe { self.move_head_unchecked(mv) }
    }

    // 移动头指针并长度 + 1
    #[inline]
    pub unsafe fn move_head_unchecked(&mut self, mv: isize) {
        let cap = self.capacity();
        let head = &mut self.head;
        // 安全性：cap 已经在 mirroredbuffer 的成员类型中被保证了在 0..=isize::MAX 范围内
        let cap = cap as isize;
        // 安全性，需要其他入口函数保证
        debug_assert!(*head <= MAX_USIZE_WITHOUT_HIGHEST_BIT && cap > 0);
        // 安全性，经过模运算后不可能超出 cap
        let mut new_head = (*head as isize + mv) % cap;
        if new_head.is_negative() {
            // 安全性：如果new_head 是负数，加上 cap 不会溢出，因为 cap 已经被保证在 0..=isize::MAX
            // 范围内，并且也不可能超出容量
            new_head += cap;
        }
        *head = new_head as usize;
        let new_len = self.len.strict_sub_signed(mv);
        self.len = new_len;
    }

    // 仅仅增加长度
    #[inline]
    pub fn move_tail(&mut self, mv: isize) {
        let len = self.len as isize;
        debug_assert!(len >= 0);
        let cap = self.capacity() as isize;
        // 安全性，防止溢出顺便防止超出容量
        assert!(len.checked_add(mv) <= Some(cap));
        unsafe { self.move_tail_unchecked(mv) };
    }

    #[inline]
    pub unsafe fn move_tail_unchecked(&mut self, mv: isize) {
        let len = self.len as isize;
        debug_assert!(len >= 0);
        let cap = self.capacity() as isize;
        debug_assert!(len.checked_add(mv) <= Some(cap) && len.checked_add(mv) >= Some(0));
        self.len = (len + mv) as usize;
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        assert!(self.len.checked_add(other.len) <= Some(self.capacity()));
        unsafe {
            let uninit = self.uninit_slice();
            let src = other.as_slice().as_ptr();
            let dst = uninit.as_mut_ptr() as *mut T;
            copy_nonoverlapping(src, dst, other.len);
            self.len += other.len;
            other.len = 0; // 清空 other，避免旧容器的元素（已经被拷贝到本容器）被析构
        }
    }

    #[inline]
    pub fn front(&self) -> Option<&T> { self.as_slice().first() }

    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> { self.as_mut_slice().first_mut() }

    #[inline]
    pub fn back(&self) -> Option<&T> { self.as_slice().last() }

    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> { self.as_mut_slice().last_mut() }

    // 默认行为是拷贝前面的对象，新容量装不下就drop了
    // 默认不处理0容量
    fn realloc_and_restore_part(&mut self, cap: usize) {
        unsafe {
            let len = self.len;
            let obsolete = len.saturating_sub(cap);
            let reserve = len - obsolete;
            let mut new_buf = MirroredBuffer::<T>::with_capacity(cap);
            debug_assert!(self.head < self.capacity());
            let old_buf = &mut self.buf;
            let dst = new_buf.as_mut_ptr().cast::<T>();
            let src = old_buf.as_mut_ptr().cast::<T>().add(self.head);
            // 拷贝要保留的元素
            copy_nonoverlapping(src, dst, reserve);
            // 要丢掉的起始位置
            let to_be_drop = src.add(reserve);
            // 确保要丢掉的元素都能正确析构
            if !obsolete.is_zero() {
                let d = slice_from_raw_parts_mut(to_be_drop, obsolete);
                drop_in_place(d);
            }
            let old_buf = replace(&mut self.buf, new_buf);
            drop(old_buf); // 释放旧的缓冲区
            self.head = 0;
            self.len = reserve;
        }
    }

    #[inline]
    pub fn try_push_front(&mut self, value: T) -> Option<&T> {
        if self.is_full() {
            return None;
        }
        // 上面保证了缓冲区不仅没满还非空
        unsafe {
            // move head forward
            self.move_head_unchecked(-1);
            let p = self.as_mut_slice().first_mut().unwrap_unchecked();
            *p = value;
            Some(&*p)
        }
    }

    #[inline]
    pub fn push_front(&mut self, value: T) -> &T {
        // even just a little bit more, it could allocate a huge page
        unsafe {
            let min_cap = self.len.strict_add(1);
            if self.capacity() < min_cap {
                self.realloc_and_restore_part(min_cap);
            }
            self.try_push_front(value).unwrap_unchecked()
        }
    }

    #[inline]
    pub fn try_push_back(&mut self, value: T) -> Option<&T> {
        if self.is_full() {
            return None;
        }
        unsafe {
            // move tail backward
            self.move_tail_unchecked(1);
            let p = self.as_mut_slice().last_mut()?;
            *p = value;
            Some(&*p)
        }
    }

    #[inline]
    pub fn push_back(&mut self, value: T) -> &T {
        unsafe {
            let min_cap = self.len.strict_add(1);
            if self.capacity() < min_cap {
                self.realloc_and_restore_part(min_cap);
            }
            self.try_push_back(value).unwrap_unchecked()
        }
    }

    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        unsafe {
            let removed = self.last_mut().and_then(|x| Some(read(x as *mut T)))?;
            self.move_tail_unchecked(-1);
            Some(removed)
        }
    }

    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        unsafe {
            let removed = self.first_mut().and_then(|x| Some(read(x as *mut T)))?;
            self.move_head_unchecked(1);
            Some(removed)
        }
    }

    #[inline]
    pub fn truncate_back(&mut self, n: usize) {
        let current = self.len;
        if n >= current {
            return;
        }
        // 上面的 guard 保证了 current 不小于 target
        let mv = (current - n) as isize;
        let s = &mut self[n..] as *mut [_];
        unsafe {
            drop_in_place(s);
            self.move_tail_unchecked(mv.neg());
        }
        debug_assert_eq!(self.len(), n)
    }

    // 默认是从尾部裁切
    #[inline]
    pub fn truncate(&mut self, n: usize) { self.truncate_back(n); }

    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "split_off at is out of bounds");
        let tail_len = self.len() - at;
        let mut other = Self::with_capacity(tail_len);
        if tail_len > 0 {
            let self_slice = self.as_slice();
            let tail_slice = &self_slice[at..];
            unsafe {
                copy_nonoverlapping(tail_slice.as_ptr(), other.as_mut_ptr(), tail_len);
                other.len = tail_len;
            }
        }
        self.len = at;
        other
    }

    #[inline]
    pub fn clear(&mut self) { self.truncate_back(0); }

    #[inline]
    pub fn swap_remove(&mut self, idx: usize) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let len = self.len;
        // 若索引指向最后一个元素，直接弹出
        // is_empty() 保证了 len > 0
        if idx != len - 1 {
            self.swap(idx, len - 1);
        }
        self.pop_back()
    }

    #[inline]
    pub fn try_insert(&mut self, idx: usize, elem: T) -> Option<&T> {
        let len = self.len();
        if idx > len || self.is_full() {
            return None;
        }
        if idx == len {
            self.try_push_back(elem);
            return self.as_slice().last();
        }
        unsafe {
            // 增加长度并获取可变切片
            self.move_tail_unchecked(1);
            let slice = self.as_mut_slice();

            // 计算需要移动的元素数量
            // 上面的两次判断保证了 len 大于 idx
            let count = len - idx;
            let target = slice.as_mut_ptr().add(idx);
            // cap > len > idx
            let dst = target.add(1);
            copy(target, dst, count);
            write(slice.as_mut_ptr().add(idx), elem);
            Some(&*target)
        }
    }

    #[inline]
    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len());
        unsafe {
            let len = self.len();
            let ptr = self.as_mut_ptr();
            let target = ptr.add(idx);
            let removed = read(target);
            copy(ptr.add(idx + 1), target, len - 1 - idx);
            self.move_tail_unchecked(-1);
            removed
        }
    }

    // 可能会保留更多内存，由于分配需要按照分配粒度对齐，所以没有实现 reserve_exact
    pub fn reserve(&mut self, additional: usize) {
        let required_cap = self.len().strict_add(additional);
        let cap = self.capacity();
        if cap >= required_cap {
            return;
        }
        let new_cap = required_cap.max(cap.saturating_mul(2));
        self.realloc_and_restore_part(new_cap);
    }

    // 缩到指定的容量，如果小于当前长度则 panic
    pub fn shrink_to(&mut self, min_cap: usize) {
        if T::IS_ZST || self.buf.is_not_allocated() {
            return;
        }
        let len = self.len();
        assert!(min_cap < len, "min_capacity is less than current length");
        if min_cap == len {
            return;
        }
        let ideal_virtual_size = mirrored_allocation_unit::<T>(min_cap);
        if ideal_virtual_size < self.buf.virtual_size() {
            self.realloc_and_restore_part(min_cap);
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if T::IS_ZST || self.buf.is_not_allocated() {
            return;
        }
        let len = self.len();
        let ideal_virtual_size = mirrored_allocation_unit::<T>(len);
        if ideal_virtual_size >= self.buf.virtual_size() {
            return;
        }
        // ideal_virtual_size 这里不太可能为0，因为 is_not_allocated 保证了 len > 0，
        // mirrored_allocation_unit 也会返回大于0的值
        self.realloc_and_restore_part(ideal_virtual_size);
    }

    pub fn into_iter(self) -> IntoIter<T> { IntoIter { inner: self } }

    pub fn rotate_left(&mut self, mid: usize) {
        assert!(mid <= self.len(), "rotate_left mid is out of bounds");
        let cap = self.capacity();
        assert!(cap != 0, "rotate_left called on an empty buffer");
        if mid == 0 {
            return;
        }
        self.head = (self.head + mid) % cap;
    }

    pub fn rotate_right(&mut self, k: usize) {
        assert!(k <= self.len(), "rotate_right k is out of bounds");
        if k == 0 {
            return;
        }
        let len = self.len();
        self.rotate_left(len - k);
    }
}

impl<T> Deref for SliceRingBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target { self.as_slice() }
}

impl<T> DerefMut for SliceRingBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target { self.as_mut_slice() }
}

impl<T> Default for SliceRingBuffer<T> {
    #[inline]
    fn default() -> Self { Self::new() }
}

impl<T> AsRef<[T]> for SliceRingBuffer<T> {
    fn as_ref(&self) -> &[T] { &*self }
}

impl<T> AsMut<[T]> for SliceRingBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] { &mut *self }
}

impl<T: PartialEq> PartialEq for SliceRingBuffer<T> {
    fn eq(&self, other: &Self) -> bool { self.as_slice() == other.as_slice() }
}

impl<T: Eq> Eq for SliceRingBuffer<T> {}

impl<T: PartialOrd> PartialOrd for SliceRingBuffer<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.as_slice().partial_cmp(other.as_slice()) }
}

impl<T: Ord> Ord for SliceRingBuffer<T> {
    fn cmp(&self, other: &Self) -> Ordering { self.as_slice().cmp(other.as_slice()) }
}

pub struct IntoIter<T> {
    inner: SliceRingBuffer<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> { self.inner.pop_front() }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }
}

impl<T> From<Vec<T>> for SliceRingBuffer<T> {
    fn from(vec: Vec<T>) -> Self { Self::from_iter(vec.into_iter()) }
}

impl<'a, T: Clone> From<&'a [T]> for SliceRingBuffer<T> {
    fn from(slice: &'a [T]) -> Self {
        let len = slice.len();
        let mut rb = SliceRingBuffer::with_capacity(len);
        for item in slice {
            unsafe {
                rb.try_push_back(item.clone()).unwrap_unchecked();
            }
        }
        rb
    }
}

impl<T> FromIterator<T> for SliceRingBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut rb = SliceRingBuffer::with_capacity(lower);
        rb.extend(iterator);
        rb
    }
}

impl<T> Extend<T> for SliceRingBuffer<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        self.reserve(lower);
        for item in iterator {
            self.push_back(item);
        }
    }
}

unsafe impl<T> Send for SliceRingBuffer<T> where T: Send {}
unsafe impl<T> Sync for SliceRingBuffer<T> where T: Sync {}
