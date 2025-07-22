#![cfg_attr(not(any(feature = "use_std", test)), no_std)]
#![feature(temporary_niche_types)]
#![feature(sized_type_properties)]
#![feature(ptr_as_uninit)]

#[cfg(any(feature = "use_std", test))]
extern crate core;

#[cfg(any(target_os = "macos", target_os = "ios"))]
extern crate mach2 as mach;

#[cfg(unix)]
extern crate libc;

#[cfg(windows)]
extern crate windows;

mod mirrored;
pub use mirrored::MirroredBuffer;
use num::Zero;

use core::{
    cmp, convert, fmt, hash, iter,
    mem::{self, MaybeUninit},
    ops::{self, Add, Neg},
    ptr::{self, NonNull},
    slice::{self, SlicePattern},
    str, usize,
};

use crate::mirrored::MAX_USIZE_WITHOUT_HIGHEST_BIT;

// 只需要保证三件事：
// head 被限制在0..cap，因为前半部分和后半部分是镜像的
// head + len <= cap， 因为不能越界
// cap <= isize::max //类型要求
pub struct SliceRingBuffer<T> {
    buf: MirroredBuffer<T>,
    head: usize, //was jailed in 0..capacity
    len: usize,  //0..=capacity
}

unsafe impl<T> Send for SliceRingBuffer<T> where T: Send {}
unsafe impl<T> Sync for SliceRingBuffer<T> where T: Sync {}

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
        let cap = self.capacity();
        debug_assert!(head <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        debug_assert!(head < cap);
        debug_assert!(len <= cap);
        unsafe { self.buf.virtual_slice_at_unchecked(head, len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let head = self.head;
        let len = self.len;
        let cap = self.capacity();
        debug_assert!(head <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        debug_assert!(head < cap);
        debug_assert!(len <= cap);
        unsafe { self.buf.virtual_slice_mut_at_unchecked(head, len) }
    }

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
    pub unsafe fn move_head_unchecked(&mut self, mv: isize) {
        let cap = self.capacity();
        let head = &mut self.head;
        // 安全性：cap 已经在 mirroredbuffer 的成员类型中被保证了在 0..=isize::MAX 范围内
        let cap = cap as isize;
        // 安全性，需要其他入口函数保证
        debug_assert!(*head <= MAX_USIZE_WITHOUT_HIGHEST_BIT);
        // 安全性，经过模运算后不可能超出 cap
        let mut new_head = (*head as isize + mv) % cap;
        if new_head.is_negative() {
            // 安全性：如果new_head 是负数，加上 cap 不会溢出，因为 cap 已经被保证在 0..=isize::MAX
            // 范围内，并且也不可能超出容量
            new_head += cap;
        }
        *head = new_head as usize;
        self.len += 1;
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
        self.len += mv as usize;
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        assert!(self.len + other.len <= self.capacity());
        unsafe {
            let uninit = self.uninit_slice();
            let src = other.as_slice().as_ptr();
            let dst = uninit.as_mut_ptr() as *mut T;
            ptr::copy_nonoverlapping(src, dst, other.len);
            self.len += other.len;
            other.len = 0; // 清空 other
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
    fn realloc_and_restore_part(&mut self, cap: usize) {
        unsafe {
            let len = self.len;
            let obsolete = len.saturating_sub(cap);
            let reserve = len - obsolete;
            let mut new_buf = MirroredBuffer::<T>::with_capacity(cap);
            let old_buf = &mut self.buf;
            let dst = new_buf.as_mut_ptr().cast::<T>();
            let src = old_buf.as_mut_ptr().cast::<T>().add(self.head);
            ptr::copy_nonoverlapping(src, dst, reserve);
            let to_be_drop = src.add(reserve);
            if !obsolete.is_zero() {
                let d = ptr::slice_from_raw_parts_mut(to_be_drop, obsolete);
                ptr::drop_in_place(d);
            }
            self.buf = new_buf;
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

    pub fn push_front(&mut self, value: T) -> &T {
        // even just a little bit more, it could allocate a huge page
        unsafe {
            let min_cap = self.len + 1;
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

    pub fn push_back(&mut self, value: T) -> &T {
        unsafe {
            let min_cap = self.len + 1;
            if self.capacity() < min_cap {
                self.realloc_and_restore_part(min_cap);
            }
            self.try_push_back(value).unwrap_unchecked()
        }
    }

    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        unsafe {
            let removed = self.last_mut().and_then(|x| Some(ptr::read(x as *mut T)));
            self.move_tail_unchecked(-1);
            removed
        }
    }

    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        unsafe {
            let removed = self.first_mut().and_then(|x| Some(ptr::read(x as *mut T)));
            self.move_head_unchecked(1);
            self.len -= 1;
            removed
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {}

    pub fn truncate_back(&mut self, n: usize) {
        let current = self.len;
        if n >= current {
            return;
        }
        // 上面的 guard 保证了 current 不小于 target
        let mv = (current - n) as isize;
        let s = &mut self[n..] as *mut [_];
        unsafe {
            ptr::drop_in_place(s);
            self.move_tail_unchecked(mv.neg());
        }
        debug_assert_eq!(self.len(), n)
    }

    // 默认是从尾部裁切
    pub fn truncate(&mut self, n: usize) { self.truncate_back(n); }

    #[inline]
    pub fn clear(&mut self) { self.truncate_back(0); }

    #[inline]
    pub fn swap_remove(&mut self, idx: usize) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let len = self.len();
        // 若索引指向最后一个元素，直接弹出
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
            let count = len - idx;
            let target = slice.as_mut_ptr().add(idx);
            // cap > len > idx
            let dst = target.add(1);
            ptr::copy(target, dst, count);
            ptr::write(slice.as_mut_ptr().add(idx), elem);
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
            let removed = ptr::read(target);
            ptr::copy(ptr.add(idx + 1), target, len - idx - 1);
            self.move_tail_unchecked(-1);
            removed
        }
    }
}

impl<T> SliceRingBuffer<T>
where
    T: Clone,
{
    pub fn copy_from_slice(&mut self, other: &[T]) {
        //用一下min
        self.clear(); // 清空现有数据
        assert!(other.len() <= self.capacity(), "source slice length exceeds buffer capacity");
        self.head = 0; // 重置头指针
        let uninit_slice = self.buf.as_uninit_virtual_slice_mut();
        for (i, elem) in other.iter().enumerate() {
            uninit_slice[i].write(elem.clone());
        }
        self.len = other.len();
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {}

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {}
}

impl<T> Drop for SliceRingBuffer<T> {
    #[inline]
    fn drop(&mut self) {
        // In Rust, if Drop::drop panics, the value must be leaked,
        // therefore we don't need to make sure that we handle that case
        // here:
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(self);
        }
        // Buffer handles deallocation
    }
}

impl<T> ops::Deref for SliceRingBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target { self.as_slice() }
}

impl<T> ops::DerefMut for SliceRingBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target { self.as_mut_slice() }
}

impl<T> Default for SliceRingBuffer<T> {
    #[inline]
    fn default() -> Self { Self::new() }
}

impl<T> convert::AsRef<[T]> for SliceRingBuffer<T> {
    fn as_ref(&self) -> &[T] { &*self }
}

impl<T> convert::AsMut<[T]> for SliceRingBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] { &mut *self }
}
