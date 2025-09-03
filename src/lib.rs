#![feature(temporary_niche_types)]
#![feature(sized_type_properties)]
#![feature(ptr_as_uninit)]
#![feature(maybe_uninit_slice)]
#![cfg_attr(not(any(feature = "use_std", test)), no_std)]

mod mirrored;

use core::{
    cmp::Ordering,
    iter::FromIterator,
    mem::{self, MaybeUninit, SizedTypeProperties, replace},
    ops::{Deref, DerefMut, Neg},
    ptr::{NonNull, copy, copy_nonoverlapping, drop_in_place, read, slice_from_raw_parts_mut, write},
};
use mirrored::{MAX_VIRTUAL_BUF_SIZE, MirroredBuffer, mirrored_allocation_unit};
use num::Zero;

use crate::mirrored::MAX_PHYSICAL_BUF_SIZE;

pub struct SliceRingBuffer<T> {
    buf: MirroredBuffer<T>,
    head: usize,
    len: usize,
}

impl<T> SliceRingBuffer<T> {
    /// 创建一个容量为0的容器
    /// 对于 ZST 类型，容量则为 isize::MAX / 2
    #[inline(always)]
    pub fn new() -> Self {
        if T::IS_ZST {
            return Self::with_capacity(MAX_VIRTUAL_BUF_SIZE);
        }
        Self::with_capacity(0)
    }

    /// 分配多少个物理元素的容量
    /// 由于镜像缓冲区的虚拟长度只能达到 isize::MAX ，所以最大物理元素数量只能分配到 isize::MAX / 2
    #[inline(always)]
    pub fn with_capacity(cap: usize) -> Self {
        let v_cap = cap * 2;
        assert!(v_cap <= MAX_VIRTUAL_BUF_SIZE);
        Self { buf: MirroredBuffer::with_capacity(v_cap), head: 0, len: 0 }
    }

    /// 物理容量
    #[inline(always)]
    pub fn capacity(&self) -> usize { self.buf.physical_capacity() }

    /// 物理长度
    #[inline(always)]
    pub fn len(&self) -> usize {
        let len = self.len;
        debug_assert!(len <= self.capacity());
        len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

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
        debug_assert!(head == 0 || head < self.capacity());
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

    /// 在第一段虚拟地址(0..p_cap)内环绕移动头指针
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

    // 仅仅增加长度
    #[inline(always)]
    pub fn move_tail(&mut self, mv: isize) {
        let len = self.len() as isize;
        let cap = self.capacity() as isize;
        assert!(len.checked_add(mv) <= Some(cap) && len.checked_add(mv) >= Some(0));
        self.len = (len + mv) as usize;
    }

    #[inline(always)]
    pub fn append(&mut self, other: &mut Self) {
        assert!(self.len().checked_add(other.len()) <= Some(self.capacity()));
        if T::IS_ZST {
            self.len += other.len();
            self.buf.size =
                unsafe { mirrored::Size::new_unchecked(self.buf.virtual_size() + other.buf.virtual_size()) };
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

    /// 禁止 ZST 调用此函数
    fn realloc_and_restore_part(&mut self, new_cap: usize) {
        debug_assert!(new_cap <= MAX_VIRTUAL_BUF_SIZE);
        debug_assert!(!T::IS_ZST);
        unsafe {
            let len = self.len();
            let obsolete_len = len.saturating_sub(new_cap);
            let reserve_len = len - obsolete_len;
            let new_buf = MirroredBuffer::<T>::with_capacity(new_cap);
            let old_buf = &mut self.buf;
            if !obsolete_len.is_zero() {
                let dst = new_buf.as_ptr();
                let src = old_buf.as_ptr().add(self.head());
                // 拷贝要保留的元素
                copy_nonoverlapping(src, dst, reserve_len);
                // 确保要丢掉的元素都能正确析构
                let drop_start = src.add(reserve_len);
                let d = slice_from_raw_parts_mut(drop_start, obsolete_len);
                drop_in_place(d);
            }
            let old_buf = replace(&mut self.buf, new_buf);
            drop(old_buf); // 释放旧的缓冲区
            self.head = 0;
            self.len = reserve_len;
        }
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
            if !T::IS_ZST && self.capacity() < required_cap {
                self.realloc_and_restore_part(required_cap.strict_mul(2));
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
            if !T::IS_ZST && self.capacity() < required_cap {
                self.realloc_and_restore_part(required_cap.strict_mul(2));
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
            return Some(unsafe { mem::zeroed() });
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
            return Some(unsafe { mem::zeroed() });
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
        self.len -= prev_len;
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

    #[inline]
    pub fn try_insert(&mut self, idx: usize, elem: T) -> Option<&T> {
        let len = self.len();
        match len {
            _ if idx > len || self.is_full() => {
                return None;
            }
            0 => {
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

    #[inline]
    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len(), "index out of bounds: the len is {} but the index is {}", idx, self.len());
        unsafe {
            if T::IS_ZST {
                self.move_tail(-1);
                return mem::zeroed();
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

    // 可能会保留更多内存，由于分配需要按照分配粒度对齐，所以没有实现 reserve_exact
    pub fn reserve(&mut self, additional: usize) {
        if T::IS_ZST {
            return;
        }
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
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if T::IS_ZST {
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

    /// 将 [0,mid)元素移动到末尾，正为左移，负值为右移
    pub fn rotate(&mut self, mv: isize) {
        if T::IS_ZST {
            return;
        }
        if self.is_empty() {
            panic!("call rotate while this is empty");
        }
        if mv == 0 {
            return;
        }
        // 有一个特例，当元素刚好填满的时候不需要拷贝就可以衔接到被循环的部分
        if self.is_full() {
            self.move_head(mv);
            self.move_tail(mv);
            return;
        }
        let idx = mv.rem_euclid(self.len() as isize) as usize;
        // 比如 刚好要移动 len 或 -len 就相当于没有移动
        if idx == 0 {
            return;
        }
        self[..idx].reverse();
        self[idx..].reverse();
        self.reverse();
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
    fn as_ref(&self) -> &[T] { self }
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

impl<T> ExactSizeIterator for IntoIter<T> {
    #[inline]
    fn len(&self) -> usize { self.inner.len() }
}

impl<T> IntoIterator for SliceRingBuffer<T> {
    type IntoIter = IntoIter<T>;
    type Item = T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter { IntoIter { inner: self } }
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

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> { self.inner.pop_back() }
}

impl<T> From<Vec<T>> for SliceRingBuffer<T> {
    fn from(vec: Vec<T>) -> Self { Self::from_iter(vec) }
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

impl<T> Drop for SliceRingBuffer<T> {
    fn drop(&mut self) {
        let p = self.as_mut_slice() as *mut [_];
        unsafe { drop_in_place(p) };
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_with_capacity() {
        let rb: SliceRingBuffer<i32> = SliceRingBuffer::with_capacity(10);
        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
        assert!(rb.capacity() >= 10);
    }

    #[test]
    fn test_push_back_and_pop_front() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(3);

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
        assert_eq!(rb.capacity(), 16384); // 16KiB ag

        assert_eq!(rb.pop_front(), Some(1));
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), None);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_front_and_pop_back() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_front(1);
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
        assert!(rb.is_empty());
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
    fn test_deref_mut() {
        let mut rb = SliceRingBuffer::with_capacity(4);
        rb.push_back(10);
        rb.push_back(20);
        rb.push_back(30);

        rb[1] = 25;
        assert_eq!(&[10, 25, 30], &*rb);
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
    fn test_clear() {
        let mut rb = SliceRingBuffer::with_capacity(3);
        rb.push_back(1);
        rb.push_back(2);
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
    fn test_from_iter() {
        let data = vec![1, 2, 3, 4, 5];
        let rb: SliceRingBuffer<i32> = data.into_iter().collect();
        assert_eq!(rb.len(), 5);
        assert_eq!(rb.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_extend() {
        let mut rb = SliceRingBuffer::from(vec![1, 2]);
        rb.extend(vec![3, 4, 5]);
        assert_eq!(rb.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_realloc() {
        let mut rb = SliceRingBuffer::with_capacity(2);
        for i in 0..0x4000 {
            if i.is_even() {
                rb.push_back(i);
            } else {
                rb.push_front(i);
            }
        }
        rb.push_back(114514);
        assert!(rb.capacity() > 0x4000);
        assert_eq!(rb.len(), 0x4000 + 1);
    }

    #[test]
    fn test_drop() {
        let drop_counter = Rc::new(std::cell::Cell::new(0));

        struct DropTracker {
            counter: Rc<std::cell::Cell<i32>>,
        }

        impl Drop for DropTracker {
            fn drop(&mut self) { self.counter.set(self.counter.get() + 1); }
        }

        let mut rb: SliceRingBuffer<DropTracker> = SliceRingBuffer::with_capacity(4);
        rb.push_back(DropTracker { counter: drop_counter.clone() });
        rb.push_back(DropTracker { counter: drop_counter.clone() });
        rb.push_back(DropTracker { counter: drop_counter.clone() });

        rb.pop_front();
        assert_eq!(drop_counter.get(), 1);

        rb.clear();
        assert_eq!(drop_counter.get(), 3);

        rb.push_back(DropTracker { counter: drop_counter.clone() });
        rb.push_back(DropTracker { counter: drop_counter.clone() });
        drop(rb);
        assert_eq!(drop_counter.get(), 5);
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
    }
    // todo 测试16k元素
    #[test]
    fn test_split_off() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        let rb2 = rb.split_off(3);

        assert_eq!(rb.as_slice(), &[1, 2, 3]);
        assert_eq!(rb2.as_slice(), &[4, 5]);
    }

    #[test]
    fn test_append() {
        let mut rb1 = SliceRingBuffer::from(vec![1, 2, 3]);
        let mut rb2 = SliceRingBuffer::from(vec![4, 5]);

        rb1.append(&mut rb2);

        assert_eq!(rb1.as_slice(), &[1, 2, 3, 4, 5]);
        assert!(rb2.is_empty());
    }

    #[test]
    fn test_truncate() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        rb.truncate(3);
        assert_eq!(rb.as_slice(), &[1, 2, 3]);
        rb.truncate(5); // No debería tener efecto
        assert_eq!(rb.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_remove() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(rb.remove(2), 3);
        assert_eq!(rb.as_slice(), &[1, 2, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn test_remove_out_of_bounds() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3]);
        rb.remove(3);
    }

    #[test]
    fn test_swap_remove() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(rb.swap_remove(1), Some(2));
        assert_eq!(rb.as_slice(), &[1, 5, 3, 4]);
    }

    #[test]
    fn test_insert() {
        let mut rb = SliceRingBuffer::with_capacity(5);
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(4);
        rb.push_back(5);

        assert!(rb.try_insert(2, 3).is_some());
        assert_eq!(rb.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut rb = SliceRingBuffer::with_capacity(20);
        rb.extend(0..5);
        rb.shrink_to_fit();
        assert_eq!(rb.capacity(), 0x4000);
        assert_eq!(rb.as_slice(), &[0, 1, 2, 3, 4]);
    }
}
