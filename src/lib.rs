#![cfg_attr(feature = "unstable", feature(temporary_niche_types))]
#![cfg_attr(feature = "unstable", feature(maybe_uninit_slice))]
#![cfg_attr(not(feature = "unstable"), allow(unstable_name_collisions))]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

#[cfg(feature = "bytes")]
mod bytes;

#[cfg(feature = "io")]
mod io;

#[cfg(feature = "serde")]
mod serde;

#[cfg(feature = "tokio-io")]
mod tokio;

#[cfg(not(feature = "unstable"))]
mod stable;
#[cfg(not(feature = "unstable"))]
use stable::MaybeUninitCompact;

mod mirrored;
use crate::mirrored::MAX_PHYSICAL_BUF_SIZE;
use mirrored::{MirroredBuffer, mirrored_allocation_unit};
use num::Zero;
use std::{
    cmp::Ordering,
    iter::{FromIterator, FusedIterator},
    marker::PhantomData,
    mem::{MaybeUninit, needs_drop, zeroed},
    ops::{Deref, DerefMut, Neg, RangeBounds},
    ptr,
    ptr::{NonNull, copy, copy_nonoverlapping, drop_in_place, read, write},
};

#[derive(Debug)]
pub struct SliceRingBuffer<T> {
    buf: MirroredBuffer<T>,
    head: usize,
    len: usize,
}

impl<T> SliceRingBuffer<T> {
    const ELEM_IS_ZST: bool = MirroredBuffer::<T>::ELEM_IS_ZST;
    /// Growth factor used when expanding the buffer capacity.
    const GROW_FACTOR: usize = 2;
    /// Maximum possible capacity for the buffer.
    pub const MAX_CAPACITY: usize = MAX_PHYSICAL_BUF_SIZE;

    /// Creates a new empty `SliceRingBuffer`.
    ///
    /// For ZST types, initializes with maximum capacity. For non-ZST types,
    /// initializes with zero capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer: SliceRingBuffer<i32> = SliceRingBuffer::new();
    /// assert_eq!(buffer.len(), 0);
    /// assert_eq!(buffer.capacity(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self { Self::with_capacity(if Self::ELEM_IS_ZST { MAX_PHYSICAL_BUF_SIZE } else { 0 }) }

    /// Creates a new empty `SliceRingBuffer` with the specified capacity.
    ///
    /// The actual allocated capacity may be larger than requested due to
    /// memory alignment requirements.
    ///
    /// # Parameters
    ///
    /// - `cap`: The desired capacity in number of elements
    ///
    /// # Panics
    ///
    /// Panics if `cap` exceeds `MAX_CAPACITY`.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer: SliceRingBuffer<i32> = SliceRingBuffer::with_capacity(10);
    /// assert_eq!(buffer.len(), 0);
    /// assert!(buffer.capacity() >= 10);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        assert!(cap <= MAX_PHYSICAL_BUF_SIZE);
        Self { buf: MirroredBuffer::with_capacity(cap), head: 0, len: 0 }
    }

    /// Returns the physical capacity of the buffer.
    ///
    /// This is the maximum number of elements that can be stored without
    /// reallocating memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let buffer: SliceRingBuffer<i32> = SliceRingBuffer::with_capacity(10);
    /// assert!(buffer.capacity() >= 10);
    /// ```
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize { self.buf.physical_capacity() }

    /// Returns the number of elements currently in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        let len = self.len;
        debug_assert!(len <= self.capacity(), "len:{} > capacity:{}", len, self.capacity());
        len
    }

    /// Returns `true` if the buffer contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer: SliceRingBuffer<i32> = SliceRingBuffer::new();
    /// assert!(buffer.is_empty());
    ///
    /// buffer.push_back(1);
    /// assert!(!buffer.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool { self.len().is_zero() }

    /// Returns `true` if the buffer is at maximum capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::with_capacity(2);
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    /// assert!(!buffer.is_full());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        let len = self.len();
        let cap = self.capacity();
        debug_assert!(len <= cap);
        len == cap
    }

    /// Returns the current head position in the buffer.
    ///
    /// The head position is always in the range `0..capacity()`
    #[inline]
    #[must_use]
    pub fn head(&self) -> usize {
        let head = self.head;
        debug_assert!(head.is_zero() || head < self.capacity());
        head
    }

    /// Returns the current tail position in the buffer.
    ///
    /// The tail position is calculated as `(head + len) % capacity()`.
    #[inline]
    #[must_use]
    pub fn tail(&self) -> usize {
        let cap = self.capacity();
        let head = self.head();
        let len = self.len();
        debug_assert!(head.checked_add(len) <= Some(self.buf.virtual_size()));
        (head + len) % cap
    }

    /// Returns a reference to the underlying slice of initialized elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    /// assert_eq!(buffer.as_slice(), &[1, 2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        let head = self.head();
        unsafe { self.buf.virtual_uninit_slice_at(head, len).assume_init_ref() }
    }

    /// Returns a mutable reference to the underlying slice of initialized elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    ///
    /// buffer.as_mut_slice()[0] = 10;
    /// assert_eq!(buffer.as_slice(), &[10, 2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let head = self.head();
        let len = self.len();
        unsafe { self.buf.virtual_uninit_slice_mut_at(head, len).assume_init_mut() }
    }

    /// Returns a reference to the underlying slice of potentially uninitialized elements.
    #[inline]
    #[must_use]
    pub fn as_uninit_slice(&self) -> &[MaybeUninit<T>] {
        let len = self.len();
        let head = self.head();
        self.buf.virtual_uninit_slice_at(head, len)
    }

    /// Returns a mutable reference to the underlying slice of potentially uninitialized elements.
    #[inline]
    #[must_use]
    pub fn as_uninit_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        let len = self.len();
        let head = self.head();
        self.buf.virtual_uninit_slice_mut_at(head, len)
    }

    /// Returns a mutable reference to the uninitialized portion of the buffer.
    #[inline]
    #[must_use]
    pub fn uninit_slice(&mut self) -> &mut [MaybeUninit<T>] {
        let cap = self.capacity();
        let len = self.len();
        let head = self.head();
        debug_assert!(len.checked_add(head) < Some(self.buf.virtual_size()));
        let uninit_start = self.tail();
        let uninit_len = cap - len;
        self.buf.virtual_uninit_slice_mut_at(uninit_start, uninit_len)
    }

    /// Moves the head position by the specified amount and updates the length accordingly.
    ///
    /// # Parameters
    ///
    /// - `mv`: The amount to move the head (can be negative)
    ///
    /// # Panics
    ///
    /// - `cap` == 0
    #[inline]
    pub fn move_head(&mut self, mv: isize) {
        let cap = self.capacity();
        let head = &mut self.head();
        let cap = cap.cast_signed();
        assert!(cap > 0);
        let new_head = ((*head).cast_signed() + mv).rem_euclid(cap);
        self.head = new_head.cast_unsigned();
        let new_len = self.len.strict_sub_signed(mv);
        self.len = new_len;
    }

    /// Changes the length by the specified amount without moving elements.
    ///
    /// # Parameters
    ///
    /// - `mv`: The amount to change the length by (can be negative)
    ///
    /// # Panics
    ///
    /// Panics if the new length would be negative or exceed capacity.
    #[inline]
    pub fn move_tail(&mut self, mv: isize) {
        let len = self.len().cast_signed();
        let cap = self.capacity().cast_signed();
        assert!(len.checked_add(mv) <= Some(cap) && len.checked_add(mv) >= Some(0));
        self.len = (len + mv).cast_unsigned();
    }

    /// Prepends all elements from another buffer to the beginning of this buffer.
    ///
    /// After this operation, `other` will be empty, and all its elements will
    /// be positioned before the existing elements in this buffer.
    ///
    /// # Parameters
    ///
    /// - `other`: The buffer to prepend elements from
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buf1 = SliceRingBuffer::new();
    /// buf1.push_back(3);
    /// buf1.push_back(4);
    ///
    /// let mut buf2 = SliceRingBuffer::new();
    /// buf2.push_back(1);
    /// buf2.push_back(2);
    ///
    /// buf1.append_front(&mut buf2);
    /// assert_eq!(buf1.as_slice(), &[1, 2, 3, 4]);
    /// assert!(buf2.is_empty());
    /// ```
    #[inline]
    pub fn append_front(&mut self, other: &mut Self) {
        let other_len = other.len();
        self.reserve(other_len);
        if Self::ELEM_IS_ZST {
            self.len += other.len();
            other.len = 0;
            return;
        }
        if other_len == 0 {
            return;
        }
        self.move_head(-other_len.cast_signed());
        unsafe {
            let src = other.as_slice().as_ptr();
            let dst = self.as_mut_ptr();
            copy_nonoverlapping(src, dst, other_len);
            other.len = 0; // Clear other buffer to prevent double-drop
        }
    }

    /// Appends all elements from another buffer to the end of this buffer.
    ///
    /// After this operation, `other` will be empty.
    ///
    /// # Parameters
    ///
    /// - `other`: The buffer to append elements from
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        let other_len = other.len();
        self.reserve(other_len);
        if Self::ELEM_IS_ZST {
            self.len += other_len;
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

    /// Returns a reference to the first element in the buffer, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// assert_eq!(buffer.front(), None);
    ///
    /// buffer.push_back(1);
    /// assert_eq!(buffer.front(), Some(&1));
    /// ```
    #[inline]
    #[must_use]
    pub fn front(&self) -> Option<&T> { self.as_slice().first() }

    /// Returns a mutable reference to the first element in the buffer, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// assert_eq!(buffer.front_mut(), None);
    ///
    /// buffer.push_back(1);
    /// if let Some(first) = buffer.front_mut() {
    ///     *first = 2;
    /// }
    /// assert_eq!(buffer.front(), Some(&2));
    /// ```
    #[inline]
    #[must_use]
    pub fn front_mut(&mut self) -> Option<&mut T> { self.as_mut_slice().first_mut() }

    /// Returns a reference to the last element in the buffer, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// assert_eq!(buffer.back(), None);
    ///
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    /// assert_eq!(buffer.back(), Some(&2));
    /// ```
    #[inline]
    #[must_use]
    pub fn back(&self) -> Option<&T> { self.as_slice().last() }

    /// Returns a mutable reference to the last element in the buffer, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// assert_eq!(buffer.back_mut(), None);
    ///
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    /// if let Some(last) = buffer.back_mut() {
    ///     *last = 3;
    /// }
    /// assert_eq!(buffer.back(), Some(&3));
    /// ```
    #[inline]
    #[must_use]
    pub fn back_mut(&mut self) -> Option<&mut T> { self.as_mut_slice().last_mut() }

    /// Returns a reference to the element at the given index.
    ///
    /// Element at index 0 is the front of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
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
    #[inline]
    #[must_use]
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
    /// use sliceable_ring_buffer::SliceRingBuffer;
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
    #[inline]
    #[must_use]
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

    /// Reallocates the buffer with a new capacity and restores part of the elements.
    ///
    /// This function is only for non-ZST types.
    ///
    /// # Parameters
    ///
    /// - `new_p_cap`: The new physical capacity
    fn realloc_and_restore_part(&mut self, new_p_cap: usize) {
        debug_assert!(new_p_cap <= MAX_PHYSICAL_BUF_SIZE);
        debug_assert!(!Self::ELEM_IS_ZST);
        let mut new_buf = MirroredBuffer::<T>::with_capacity(new_p_cap);
        let len = self.len();
        let reserve_len = len.min(new_p_cap);
        let old_slice = self.as_slice();
        let new_uninit_slice = new_buf.as_uninit_virtual_mut_slice();
        unsafe {
            for i in 0..reserve_len {
                let val = ptr::read(old_slice.get_unchecked(i));
                new_uninit_slice.get_unchecked_mut(i).write(val);
            }
        }
        self.len = 0;
        self.buf = new_buf;
        self.head = 0;
        self.len = reserve_len;
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
    /// use sliceable_ring_buffer::SliceRingBuffer;
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
        use std::ops::Bound::{Excluded, Included, Unbounded};
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
        Drain {
            len: drain_len,
            head: 0,
            remaining: drain_len,
            new_len,
            inner: NonNull::from(self),
            _marker: PhantomData,
        }
    }

    /// Attempts to add an element to the front of the buffer.
    ///
    /// Returns `Some(&T)` if successful, or `None` if the buffer is full.
    ///
    /// # Parameters
    ///
    /// - `value`: The value to add to the front
    #[inline]
    #[must_use]
    pub fn try_push_front(&mut self, value: T) -> Option<&T> {
        if self.is_full() {
            return None;
        }
        self.move_head(-1);
        unsafe {
            if Self::ELEM_IS_ZST {
                return Some(NonNull::dangling().as_ref());
            }
            let uninit = self.as_uninit_mut_slice().first_mut().unwrap_unchecked();
            let init = uninit.write(value);
            Some(init)
        }
    }

    /// Adds an element to the front of the buffer.
    ///
    /// If the buffer is at capacity, it will be reallocated with more space.
    ///
    /// # Parameters
    ///
    /// - `value`: The value to add to the front
    ///
    /// # Panics
    ///
    /// Panics if the new length would exceed `MAX_CAPACITY`.
    #[inline]
    pub fn push_front(&mut self, value: T) -> &T {
        unsafe {
            assert!(self.len().checked_add(1) <= Some(MAX_PHYSICAL_BUF_SIZE));
            let required_cap = self.len() + 1;

            if Self::ELEM_IS_ZST && self.capacity() < required_cap {
                self.buf.set_size_unchecked(required_cap * 2);
                return self.try_push_front(value).unwrap_unchecked();
            }
            if self.capacity() < required_cap {
                self.realloc_and_restore_part(required_cap.strict_mul(Self::GROW_FACTOR));
            }
            self.try_push_front(value).unwrap_unchecked()
        }
    }

    /// Attempts to add an element to the back of the buffer.
    ///
    /// Returns `Some(&T)` if successful, or `None` if the buffer is full.
    ///
    /// # Parameters
    ///
    /// - `value`: The value to add to the back
    #[inline]
    #[must_use]
    pub fn try_push_back(&mut self, value: T) -> Option<&T> {
        if self.is_full() {
            return None;
        }
        self.move_tail(1);
        unsafe {
            if Self::ELEM_IS_ZST {
                return Some(NonNull::dangling().as_ref());
            }
            let uninit = self.as_uninit_mut_slice().last_mut().unwrap_unchecked();
            let init = uninit.write(value);
            Some(init)
        }
    }

    /// Adds an element to the back of the buffer.
    ///
    /// If the buffer is at capacity, it will be reallocated with more space.
    ///
    /// # Parameters
    ///
    /// - `value`: The value to add to the back
    ///
    /// # Panics
    ///
    /// Panics if the new length would exceed `MAX_CAPACITY`.
    #[inline]
    pub fn push_back(&mut self, value: T) -> &T {
        unsafe {
            assert!(self.len().checked_add(1) <= Some(MAX_PHYSICAL_BUF_SIZE));
            let required_cap = self.len() + 1;
            if Self::ELEM_IS_ZST && self.capacity() < required_cap {
                self.buf.set_size_unchecked(required_cap * 2);
                return self.try_push_back(value).unwrap_unchecked();
            }
            if self.capacity() < required_cap {
                self.realloc_and_restore_part(required_cap.strict_mul(Self::GROW_FACTOR));
            }
            self.try_push_back(value).unwrap_unchecked()
        }
    }

    /// Removes and returns the last element from the buffer, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    ///
    /// assert_eq!(buffer.pop_back(), Some(2));
    /// assert_eq!(buffer.pop_back(), Some(1));
    /// assert_eq!(buffer.pop_back(), None);
    /// ```
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        if Self::ELEM_IS_ZST {
            self.move_tail(-1);
            return Some(unsafe { zeroed() });
        }
        unsafe {
            let removed = self.last_mut().map(|p| read(ptr::from_mut(p)))?;
            self.move_tail(-1);
            Some(removed)
        }
    }

    /// Removes and returns the first element from the buffer, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use sliceable_ring_buffer::SliceRingBuffer;
    ///
    /// let mut buffer = SliceRingBuffer::new();
    /// buffer.push_back(1);
    /// buffer.push_back(2);
    ///
    /// assert_eq!(buffer.pop_front(), Some(1));
    /// assert_eq!(buffer.pop_front(), Some(2));
    /// assert_eq!(buffer.pop_front(), None);
    /// ```
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        if Self::ELEM_IS_ZST {
            self.move_head(1);
            return Some(unsafe { zeroed() });
        }
        unsafe {
            let removed = self.first_mut().map(|p| read(ptr::from_mut(p)))?;
            self.move_head(1);
            Some(removed)
        }
    }

    /// Truncates the buffer from the back to contain no more than `n` elements.
    ///
    /// If `n` is greater than or equal to the current length, this has no effect.
    ///
    /// # Parameters
    ///
    /// - `n` is the maximum number of elements to keep
    #[inline]
    pub fn truncate_back(&mut self, n: usize) {
        let current_len = self.len();
        if n >= current_len {
            return;
        }
        // 上面的 guard 保证了 current 不小于 target
        let mv = (current_len - n).cast_signed().neg();
        if Self::ELEM_IS_ZST {
            self.move_tail(mv);
            debug_assert_eq!(self.len(), n);
            return;
        }
        let s = unsafe { self.get_unchecked_mut(n..current_len) } as *mut [_];
        unsafe {
            drop_in_place(s);
            self.move_tail(mv);
        }
        debug_assert_eq!(self.len(), n);
    }

    /// Truncates the buffer from the back to contain no more than `n` elements.
    ///
    /// This is equivalent to `truncate_back`.
    ///
    /// # Parameters
    ///
    /// - `n` is the maximum number of elements to keep
    #[inline]
    pub fn truncate(&mut self, n: usize) { self.truncate_back(n); }

    /// Splits the buffer into two at the given index.
    ///
    /// Returns a new buffer containing elements from index `at` onwards.
    /// Elements before index `at` remain in this buffer.
    ///
    /// # Parameters
    ///
    /// - `at`: The index to split at
    #[inline]
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self {
        let tail_len = self.len().strict_sub(at);
        let mut another = Self::with_capacity(tail_len);
        if tail_len > 0 {
            if Self::ELEM_IS_ZST {
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

    /// Splits the buffer into two at the given index.
    ///
    /// Returns a new buffer containing elements before index `at`.
    /// Elements from index `at` onwards remain in this buffer.
    ///
    /// # Parameters
    ///
    /// - `at`: The index to split at
    #[inline]
    #[must_use]
    pub fn split_to(&mut self, at: usize) -> Self {
        let prev_len = at;
        let mut another = Self::with_capacity(prev_len);
        if prev_len > 0 {
            if Self::ELEM_IS_ZST {
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
        self.move_head(prev_len.cast_signed());
        another
    }

    /// Clears the buffer, removing all elements.
    ///
    /// This does not change the capacity of the buffer.
    #[inline]
    pub fn clear(&mut self) { self.truncate(0); }

    /// Removes an element at the given index and returns it.
    ///
    /// The last element is moved to the removed position before removal.
    /// Returns `None` if the buffer is empty.
    ///
    /// # Parameters
    ///
    /// - `idx`: The index of the element to remove
    #[inline]
    pub fn swap_remove(&mut self, idx: usize) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        if Self::ELEM_IS_ZST {
            return self.pop_back();
        }
        let len = self.len();
        if idx != len - 1 {
            self.swap(idx, len - 1);
        }
        self.pop_back()
    }

    /// Attempts to insert an element at the given index.
    ///
    /// Returns `Some(&T)` if successful, or `None` if the buffer is full or index is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `idx`: The index to insert at
    /// - `elem`: The element to insert
    #[inline]
    #[must_use]
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
        if Self::ELEM_IS_ZST {
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

    /// Inserts an element at the given index.
    ///
    /// If the buffer is at capacity, it will be reallocated with more space.
    ///
    /// # Parameters
    ///
    /// - `idx`: The index to insert at
    /// - `elem`: The element to insert
    ///
    /// # Panics
    ///
    /// Panics if `idx` is greater than the current length.
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
        if Self::ELEM_IS_ZST {
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
            copy(&raw const elem, target, 1);
            let dst = target.add(1);
            copy(src, dst, len - idx);
            self.buf = new_buf;
            self.head = 0;
            self.len = len + 1;
            &*target
        }
    }

    /// Removes and returns the element at the given index.
    ///
    /// # Parameters
    ///
    /// - `idx`: The index of the element to remove
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    #[inline]
    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len(), "index out of bounds: the len is {} but the index is {}", idx, self.len());
        unsafe {
            if Self::ELEM_IS_ZST {
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

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// The buffer may reserve more space than requested.
    ///
    /// # Parameters
    ///
    /// - `additional`: The number of additional elements to reserve space for
    ///
    /// # Panics
    ///
    /// Panics if the new capacity would exceed `MAX_CAPACITY`.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let required_cap = self.len().checked_add(additional);
        assert!(required_cap <= Some(MAX_PHYSICAL_BUF_SIZE));
        let required_cap = unsafe { required_cap.unwrap_unchecked() };

        let cap = self.capacity();
        if cap >= required_cap {
            return;
        }
        if Self::ELEM_IS_ZST {
            unsafe {
                self.buf.set_size_unchecked(required_cap * 2);
            }
            return;
        }
        let new_cap = required_cap.max(cap.saturating_mul(Self::GROW_FACTOR));
        self.realloc_and_restore_part(new_cap);
    }

    /// Shrinks the capacity to the specified minimum, if possible.
    ///
    /// # Parameters
    ///
    /// - `min_cap`: The minimum capacity to shrink to
    ///
    /// # Panics
    ///
    /// Panics if `min_cap` is less than the current length.
    #[inline]
    pub fn shrink_to(&mut self, min_cap: usize) {
        if Self::ELEM_IS_ZST {
            return;
        }
        let len = self.len();
        assert!(min_cap >= len, "min_capacity ({min_cap}) cannot be less than current length ({len})");
        // 如果请求的容量已经大于或等于当前容量，不需要操作
        if min_cap >= self.capacity() {
            return;
        }
        let ideal_virtual_size = mirrored_allocation_unit::<T>(min_cap);
        if ideal_virtual_size < self.buf.virtual_size() {
            self.realloc_and_restore_part(min_cap);
        }
    }

    /// Shrinks the capacity to fit the current length.
    ///
    /// Due to alignment requirements, the actual capacity may be slightly larger.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if Self::ELEM_IS_ZST {
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

    /// Rotates the buffer by the specified amount.
    ///
    /// Positive values rotate left, negative values rotate right.
    ///
    /// # Parameters
    ///
    /// - `mv`: The amount to rotate by
    ///
    /// # Panics
    ///
    /// Panics if the buffer is empty.
    #[inline]
    pub fn rotate(&mut self, mv: isize) {
        if Self::ELEM_IS_ZST {
            return;
        }
        assert!(!self.is_empty(), "call rotate while this is empty");
        if self.is_full() {
            self.head = (self.head().cast_signed() + mv).rem_euclid(self.capacity().cast_signed()) as usize;
            return;
        }
        let idx = mv.rem_euclid(self.len().cast_signed()) as usize;
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
    #[inline]
    fn as_ref(&self) -> &[T] { self }
}

impl<T> AsMut<[T]> for SliceRingBuffer<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] { &mut *self }
}

impl<T: PartialEq> PartialEq for SliceRingBuffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool { self.as_slice() == other.as_slice() }
}

impl<T: Eq> Eq for SliceRingBuffer<T> {}

impl<T: PartialOrd> PartialOrd for SliceRingBuffer<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.as_slice().partial_cmp(other.as_slice()) }
}

impl<T: Ord> Ord for SliceRingBuffer<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering { self.as_slice().cmp(other.as_slice()) }
}

#[derive(Debug)]
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

impl<T: Clone> Clone for SliceRingBuffer<T> {
    fn clone(&self) -> Self { self.iter().cloned().collect() }

    fn clone_from(&mut self, src: &Self) {
        if ptr::eq(self, src) {
            return;
        }
        self.clear();
        self.extend(src.iter().cloned());
    }
}

impl<T> FusedIterator for IntoIter<T> {}

unsafe impl<T: Send> Send for IntoIter<T> {}

unsafe impl<T: Sync> Sync for IntoIter<T> {}

#[derive(Debug)]
pub struct Drain<'a, T> {
    // drain_start is stored as inner.len, restore when drain was dropped
    len: usize,
    head: usize,
    remaining: usize,
    new_len: usize,
    inner: NonNull<SliceRingBuffer<T>>,
    _marker: PhantomData<&'a T>, // Needed to make Drain covariant over T
}

impl<T> Drain<'_, T> {
    const ELEM_IS_ZST: bool = SliceRingBuffer::<T>::ELEM_IS_ZST;

    #[inline]
    const fn move_head_forward(&mut self, mv: usize) {
        self.head += mv;
        self.remaining -= mv;
    }

    #[inline]
    const fn move_tail_backward(&mut self, mv: usize) { self.remaining -= mv; }
}

impl<T> Drop for Drain<'_, T> {
    fn drop(&mut self) {
        use Ordering::{Equal, Greater, Less};
        let inner = unsafe { self.inner.as_mut() };
        if Self::ELEM_IS_ZST || self.len.is_zero() {
            inner.len = self.new_len;
            return;
        }
        let drain_start = inner.len();
        let drop_start = drain_start + self.head;
        let drop_len = self.remaining;
        if needs_drop::<T>() && !drop_len.is_zero() {
            unsafe {
                drop_in_place(self.inner.as_mut().as_mut_slice().get_unchecked_mut(drop_start..drop_start + drop_len));
            }
        }
        let front_len = drain_start;
        let back_len = self.new_len - drain_start; // origin_len(drain_len + new_len) - drain_len - drain_start
        let ptr = unsafe { inner.buf.as_ptr().add(inner.head()) };
        match back_len.cmp(&front_len) {
            Less | Equal => {
                unsafe {
                    let dst = ptr.add(drain_start);
                    let src = dst.add(self.len).cast_const();
                    copy_nonoverlapping(src, dst, back_len);
                    inner.len = self.new_len;
                };
            }
            Greater => unsafe {
                let src = ptr.cast_const();
                let dst = ptr.add(self.len);
                copy_nonoverlapping(src, dst, front_len);
                inner.head += self.len;
                inner.len = self.new_len;
            },
        }
    }
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.remaining.is_zero() {
            return None;
        }
        if Self::ELEM_IS_ZST {
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

impl<T> DoubleEndedIterator for Drain<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining.is_zero() {
            return None;
        }
        if Self::ELEM_IS_ZST {
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
impl<T> ExactSizeIterator for Drain<'_, T> {}

impl<T> FusedIterator for Drain<'_, T> {}

impl<T> From<Vec<T>> for SliceRingBuffer<T> {
    #[inline]
    fn from(vec: Vec<T>) -> Self { Self::from_iter(vec) }
}

impl<T> FromIterator<T> for SliceRingBuffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iterator = iter.into_iter();
        let (lower, upper) = iterator.size_hint();
        // Use the upper bound if available, otherwise use lower bound
        let cap = upper.unwrap_or(lower);
        let mut rb = Self::with_capacity(cap);
        rb.extend(iterator);
        rb
    }
}

impl<T> Extend<T> for SliceRingBuffer<T> {
    #[inline]
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
    #[inline]
    fn from(slice: &[T]) -> Self {
        let mut buffer = Self::with_capacity(slice.len());
        buffer.extend(slice.iter().cloned());
        buffer
    }
}

impl<T: Clone, const N: usize> From<[T; N]> for SliceRingBuffer<T> {
    #[inline]
    fn from(arr: [T; N]) -> Self {
        let mut buffer = Self::with_capacity(N);
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
        let rb = iter::repeat_n((), 8).collect::<SliceRingBuffer<_>>();
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
        rb.push_back(114_514);
        assert!(rb.capacity() > ag * 3 / size_of::<usize>());
        assert_eq!(rb.len(), ag * 2 + 1);
    }

    #[test]
    fn test_drop() {
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
        use core::sync::atomic::Ordering;
        let drop_counter = Rc::new(AtomicUsize::new(0));

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

        let mut rb = iter::repeat_n(false, allocation_granularity()).collect::<SliceRingBuffer<_>>();
        *rb.last_mut().unwrap() = true;
        rb.rotate(allocation_granularity().cast_signed());
        assert_eq!(rb.last(), Some(&true));
        rb.rotate(allocation_granularity().cast_signed().neg());
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
        let mut rb = std::iter::repeat_n(true, ag - 8).collect::<SliceRingBuffer<_>>();
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
        let mut rb1 = iter::repeat_n(true, allocation_granularity()).collect::<SliceRingBuffer<_>>();
        let mut rb2 = iter::repeat_n(false, allocation_granularity()).collect::<SliceRingBuffer<_>>();

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

        let mut rb = (0usize..allocation_granularity()).collect::<SliceRingBuffer<_>>();
        rb.extend(0..allocation_granularity());
        for i in 0..allocation_granularity() {
            assert_eq!(i, rb[i]);
        }

        for (idx, val) in (allocation_granularity()..allocation_granularity() * 2).enumerate() {
            assert_eq!(idx, rb[val]);
        }
    }

    #[test]
    fn test_truncate_back_zst() {
        let mut zst_rb = SliceRingBuffer::from([(); 96]);
        zst_rb.truncate_back(3);
        assert_eq!(zst_rb.as_slice(), &[(); 3]);
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
        assert_eq!(size_of_val(&zst_rb.remove(0)), 0);
        assert_eq!(zst_rb.as_slice(), &[(), ()]);
        zst_rb.remove(1);
        assert_eq!(zst_rb.len(), 1);
    }

    #[test]
    #[should_panic = "out of bounds"]
    fn test_remove_out_of_bounds() {
        let mut rb = SliceRingBuffer::from(vec![1, 2, 3]);
        rb.remove(3);
    }

    #[test]
    #[should_panic = "out of bounds"]
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
    fn test_sliceable_ring_buffer_empty() {
        let rb: SliceRingBuffer<i32> = SliceRingBuffer::new();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 0);
        assert_eq!(rb.as_slice(), &[] as &[i32]);
    }

    #[test]
    fn test_sliceable_ring_buffer_empty_zst() {
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
        assert!(rb.get(10).is_none());
    }

    #[test]
    fn test_drain_middle() {
        let mut buf = (0..6).collect::<SliceRingBuffer<_>>(); // [0, 1, 2, 3, 4, 5]
        let drained: Vec<_> = buf.drain(2..4).collect(); // Drain [2, 3]

        assert_eq!(drained, vec![2, 3]);
        assert_eq!(buf.as_slice(), &[0, 1, 4, 5]);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_drain_all() {
        let mut buf = (0..5).collect::<SliceRingBuffer<_>>();
        let drained: Vec<_> = buf.drain(..).collect();

        assert_eq!(drained, vec![0, 1, 2, 3, 4]);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_drain_from_start() {
        let mut buf = (0..5).collect::<SliceRingBuffer<_>>();
        let drained: Vec<_> = buf.drain(..2).collect(); // Drain [0, 1]

        assert_eq!(drained, vec![0, 1]);
        assert_eq!(buf.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_drain_to_end() {
        let mut buf = (0..5).collect::<SliceRingBuffer<_>>();
        let drained: Vec<_> = buf.drain(3..).collect(); // Drain [3, 4]

        assert_eq!(drained, vec![3, 4]);
        assert_eq!(buf.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_drain_empty_range() {
        let mut buf = (0..3).collect::<SliceRingBuffer<_>>();

        assert!(buf.drain(1..1).next().is_none());
        assert_eq!(buf.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_drain_drop_incomplete_move_back() {
        // 测试 back_len <= front_len 的情况，移动后面的元素
        let mut buf = (0..10).collect::<SliceRingBuffer<_>>(); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
        let mut buf = (0..10).collect::<SliceRingBuffer<_>>(); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
    #[should_panic = "drain upper bound was too large"]
    fn test_drain_panic_end_out_of_bounds() {
        let mut buf = (0..5).collect::<SliceRingBuffer<_>>();
        let _ = buf.drain(..6);
    }

    #[test]
    #[should_panic = "drain lower bound was too large"]
    fn test_drain_panic_start_out_of_bounds() {
        // start > len 会导致 end > len (因为 end >= start)
        let mut buf = (0..5).collect::<SliceRingBuffer<_>>();
        let _ = buf.drain(6..);
    }

    #[test]
    fn test_drain_next_back_only() {
        let mut buf = (0..10).collect::<SliceRingBuffer<_>>();
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
        let mut buf = (0..10).collect::<SliceRingBuffer<_>>();
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
        let mut buf = (0..5).collect::<SliceRingBuffer<_>>();
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
        rb1.push_back(5); // 此时缓冲区是 [2, 3, 4, 5]，head 在索引 2

        let rb2 = rb1.clone();

        // 确保它们相等但相互独立
        assert_eq!(rb1.as_slice(), rb2.as_slice());
        assert_eq!(rb1.len(), rb2.len());
        assert_eq!(rb1.head(), 2);
        assert_eq!(rb2.head(), 0);

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

    #[test]
    fn test_append_front() {
        // Test basic functionality with non-ZST types
        let mut rb1 = SliceRingBuffer::new();
        rb1.push_back(3);
        rb1.push_back(4);
        rb1.push_back(5);

        let mut rb2 = SliceRingBuffer::new();
        rb2.push_back(1);
        rb2.push_back(2);

        rb1.append_front(&mut rb2);

        assert_eq!(rb1.as_slice(), &[1, 2, 3, 4, 5]);
        assert!(rb2.is_empty());
        assert_eq!(rb1.len(), 5);
        assert_eq!(rb2.len(), 0);
    }

    #[test]
    fn test_append_front_zst() {
        // Test with ZST types
        let mut rb1 = SliceRingBuffer::new();
        rb1.push_back(());
        rb1.push_back(());

        let mut rb2 = SliceRingBuffer::new();
        rb2.push_back(());
        rb2.push_back(());
        rb2.push_back(());

        rb1.append_front(&mut rb2);

        assert_eq!(rb1.len(), 5);
        assert_eq!(rb2.len(), 0);
        assert!(rb2.is_empty());
    }

    #[test]
    fn test_append_front_empty_source() {
        // Test appending from an empty buffer
        let mut rb1 = SliceRingBuffer::new();
        rb1.push_back(1);
        rb1.push_back(2);

        let mut rb2 = SliceRingBuffer::new();

        rb1.append_front(&mut rb2);

        assert_eq!(rb1.as_slice(), &[1, 2]);
        assert_eq!(rb1.len(), 2);
        assert!(rb2.is_empty());
    }

    #[test]
    fn test_append_front_to_empty() {
        // Test appending to an empty buffer
        let mut rb1 = SliceRingBuffer::new();

        let mut rb2 = SliceRingBuffer::new();
        rb2.push_back(1);
        rb2.push_back(2);
        rb2.push_back(3);

        rb1.append_front(&mut rb2);

        assert_eq!(rb1.as_slice(), &[1, 2, 3]);
        assert_eq!(rb1.len(), 3);
        assert!(rb2.is_empty());
    }

    #[test]
    fn test_append_front_both_empty() {
        // Test with both buffers empty
        let mut rb1: SliceRingBuffer<i32> = SliceRingBuffer::new();
        let mut rb2: SliceRingBuffer<i32> = SliceRingBuffer::new();

        rb1.append_front(&mut rb2);

        assert!(rb1.is_empty());
        assert!(rb2.is_empty());
        assert_eq!(rb1.len(), 0);
        assert_eq!(rb2.len(), 0);
    }

    #[test]
    fn test_append_front_with_reallocation() {
        // Test append_front when it requires reallocation
        let mut rb1 = SliceRingBuffer::with_capacity(2);
        rb1.push_back(3);
        rb1.push_back(4);

        let mut rb2 = SliceRingBuffer::new();
        rb2.push_back(1);
        rb2.push_back(2);

        rb1.append_front(&mut rb2);

        assert_eq!(rb1.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(rb1.len(), 4);
        assert!(rb2.is_empty());
    }
    #[test]
    fn test_append_front_wraparound() {
        // Test append_front when the buffer has wrapped around
        let ag = allocation_granularity();
        let mut rb1 = SliceRingBuffer::with_capacity(ag + 2);

        // Fill the buffer with bool values
        for _ in 0..ag {
            rb1.push_back(false);
        }

        // Remove some elements from front to create space
        for _ in 0..3 {
            rb1.pop_front();
        }

        // Add new bool elements at the end
        rb1.push_back(true);
        rb1.push_back(true);

        // Now append some bool elements to the front
        let mut rb2 = SliceRingBuffer::new();
        rb2.push_back(false);
        rb2.push_back(false);

        rb1.append_front(&mut rb2);

        // Check the result - first two elements should be false (from rb2)
        assert!(!rb1.as_slice()[0]);
        assert!(!rb1.as_slice()[1]);
        // Next elements should be false (remaining from original rb1)
        assert!(!rb1.as_slice()[2]);
        // The true values we added later
        assert!(rb1.as_slice()[rb1.len() - 2]);
        assert!(rb1.as_slice()[rb1.len() - 1]);

        assert_eq!(rb1.len(), ag - 3 + 2 + 2); // remaining + added + prepended
        assert!(rb2.is_empty());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::VecDeque;

    // Defines the set of operations to be randomly generated for the model test.
    #[derive(Debug, Clone)]
    enum Op<T> {
        PushBack(T),
        PopBack,
        PushFront(T),
        PopFront,
        Insert(usize, T),
        Remove(usize),
        Clear,
    }

    // A proptest strategy to generate a sequence of random operations.
    // The `T: Clone` bound is necessary because `Op<T>` derives `Clone`.
    fn arb_ops<T: Arbitrary + Clone + 'static>() -> impl Strategy<Value = Vec<Op<T>>> {
        prop::collection::vec(arb_op(), 0..100)
    }

    // A proptest strategy to generate a single random operation.
    fn arb_op<T: Arbitrary + Clone + 'static>() -> impl Strategy<Value = Op<T>> {
        prop_oneof![
            any::<T>().prop_map(Op::PushBack),
            Just(Op::PopBack),
            any::<T>().prop_map(Op::PushFront),
            Just(Op::PopFront),
            (any::<usize>(), any::<T>()).prop_map(|(i, v)| Op::Insert(i, v)),
            any::<usize>().prop_map(Op::Remove),
            Just(Op::Clear),
        ]
    }

    proptest! {
        #[test]
        fn model_based_test(ops in arb_ops::<i32>()) {
            let mut srb = SliceRingBuffer::new();
            let mut model = VecDeque::new();

            for op in ops {
                // Guard: Skip operations that would cause SliceRingBuffer to exceed its
                // hard-coded MAX_CAPACITY, as this is a known design difference from VecDeque.
                match &op {
                    Op::PushBack(_) | Op::PushFront(_) | Op::Insert(_, _) => {
                        if srb.len() + 1 > SliceRingBuffer::<i32>::MAX_CAPACITY {
                            continue;
                        }
                    }
                    _ => {}
                }

                // Apply the same operation to both our implementation and the model.
                match op {
                    Op::PushBack(v) => {
                        srb.push_back(v);
                        model.push_back(v);
                    },
                    Op::PopBack => {
                        let res_srb = srb.pop_back();
                        let res_model = model.pop_back();
                        prop_assert_eq!(res_srb, res_model);
                    },
                    Op::PushFront(v) => {
                        srb.push_front(v);
                        model.push_front(v);
                    },
                    Op::PopFront => {
                        let res_srb = srb.pop_front();
                        let res_model = model.pop_front();
                        prop_assert_eq!(res_srb, res_model);
                    },
                    Op::Insert(idx, v) => {
                        let len = model.len();
                        // Insert index must be in the range 0..=len.
                        let a_idx = if len == 0 { 0 } else { idx % (len + 1) };
                        srb.insert(a_idx, v);
                        model.insert(a_idx, v);
                    },
                    Op::Remove(idx) => {
                        if !model.is_empty() {
                            let len = model.len();
                            // Remove index must be in the range 0..len.
                            let a_idx = idx % len;
                            let res_srb = srb.remove(a_idx);
                            let res_model = model.remove(a_idx).unwrap();
                            prop_assert_eq!(res_srb, res_model);
                        }
                    },
                    Op::Clear => {
                        srb.clear();
                        model.clear();
                    }
                }

                // After each operation, assert that the externally visible state is identical.
                // We specifically DO NOT compare capacity, as that's an implementation detail.
                prop_assert_eq!(srb.len(), model.len());
                prop_assert_eq!(srb.is_empty(), model.is_empty());
                // `VecDeque::make_contiguous` is the perfect counterpart to our `as_slice`.
                prop_assert_eq!(srb.as_slice(), model.make_contiguous());
                prop_assert_eq!(srb.front(), model.front());
                prop_assert_eq!(srb.back(), model.back());
            }
        }
    }
}
