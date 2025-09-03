//! Implements a low-level, mirrored memory buffer.
//!
//! A `MirroredBuffer` allocates a virtual memory region that is twice the size
//! of its physical memory capacity. The second half of the virtual memory is a
//! mirror of the first half. This layout is particularly useful for implementing
//! high-performance circular buffers or ring buffers, as it allows for sequential
//! reading and writing across the buffer's boundary without explicit wrapping logic.
//!
//! This is a foundational, `unsafe` building block. It manages raw memory and
//! does not track initialization, length, or the position of elements. It is intended
//! to be used within higher-level, safe abstractions.

use super::*;
use crate::mirrored::utils::mirrored_allocation_unit;
use core::{
    mem::{MaybeUninit, SizedTypeProperties, size_of},
    ops::{Deref, DerefMut},
    ptr::NonNull,
    slice,
};
use num::Integer;

pub type Size = core::num::niche_types::UsizeNoHighBit;

/// A contiguous, mirrored memory buffer for elements of type `T`.
///
/// See the [module-level documentation](self) for more details.
///
/// # Invariants
///
/// - If `ptr` is not dangling, `size` must be non-zero.
/// - The total virtual memory size (`size`) must not exceed `isize::MAX` bytes.
/// - The allocated memory is laid out such that the virtual address range `[ptr, ptr + size/2)` is mirrored to the
///   range `[ptr + size/2, ptr + size)`.
pub(crate) struct MirroredBuffer<T> {
    ptr: NonNull<T>,
    pub size: Size, /* The total byte length of the underlying virtual memory region. Must not exceed
                     * `isize::MAX`. */
}

impl<T> MirroredBuffer<T> {
    /// Creates a new, empty `MirroredBuffer` without allocating memory.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self { Self::with_capacity(0) }

    /// Returns the number of elements of type `T` that can be stored in the physical region.
    ///
    /// This is the effective capacity of the buffer for storing unique items.
    #[inline(always)]
    pub fn physical_capacity(&self) -> usize {
        let p_size = self.physical_size();
        let t_size = if T::IS_ZST { 1 } else { size_of::<T>() };
        debug_assert!(p_size.is_multiple_of(t_size));
        p_size / t_size
    }

    /// Returns the total byte length of the entire virtual memory region.
    #[inline(always)]
    pub fn virtual_size(&self) -> usize { self.size.as_inner() }

    /// Returns the byte length of the physical memory region (which is half of the virtual region).
    #[inline(always)]
    pub fn physical_size(&self) -> usize {
        let v_size = self.virtual_size();
        debug_assert!(v_size.is_even(), "Virtual size must be even");
        v_size / 2
    }

    /// Allocates a new `MirroredBuffer` with enough space for at least `cap` elements.
    ///
    /// The actual allocated size may be larger than requested due to system
    /// alignment and memory page size requirements.
    #[inline(always)]
    pub fn with_capacity(cap: usize) -> Self {
        if T::IS_ZST {
            assert!(cap.is_multiple_of(2));
            return Self { ptr: NonNull::dangling(), size: unsafe { Size::new_unchecked(cap) } };
        }
        let v_size = mirrored_allocation_unit::<T>(cap);
        unsafe { Self::alloc(v_size) }
    }

    /// Creates a `MirroredBuffer` from a pre-calculated virtual memory size.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `virtual_size` meets the following criteria:
    /// - If non-zero, it must be a multiple of the system's `allocation_granularity`.
    /// - It must be less than or equal to `isize::MAX`.
    /// - The alignment of `T` must be less than or equal to the `allocation_granularity`.
    ///
    /// Violating these conditions can lead to allocation failures or undefined behavior.
    #[inline(always)]
    unsafe fn alloc(v_size: usize) -> Self {
        debug_assert!(!T::IS_ZST);
        if v_size == 0 {
            return Self { ptr: NonNull::dangling(), size: unsafe { Size::new_unchecked(0) } };
        }
        debug_assert!(
            v_size.is_multiple_of(allocation_granularity()) && v_size > 0 && v_size <= MAX_VIRTUAL_BUF_SIZE,
            "virtual_size must be a positive multiple of allocation_granularity() and less than usize::MAX"
        );
        // if virtual_size is multiple of allocation_granularity(), so it could be divided by 2
        let p_size = v_size / 2;
        debug_assert!(p_size != 0, "physical_size must be in range (0, MAX_USIZE_WITHOUT_HIGHEST_BIT/ 2)");
        assert!(
            align_of::<T>() <= allocation_granularity(),
            "The alignment requirements of `T` must be smaller than the allocation granularity."
        );
        unsafe {
            let ptr = allocate_mirrored(v_size).expect("Allocation failed");
            Self { ptr: NonNull::new_unchecked(ptr as *mut T), size: Size::new_unchecked(v_size) }
        }
    }

    /// Calculates the length of the virtual slice in terms of number of `T`s.
    #[inline(always)]
    fn virtual_capacity(&self) -> usize { self.physical_capacity() * 2 }

    /// Returns the buffer's entire virtual memory region as a slice of `MaybeUninit<T>`.
    #[inline(always)]
    pub fn as_uninit_virtaul_slice(&self) -> &[MaybeUninit<T>] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast(), self.virtual_capacity()) }
    }

    /// Returns the buffer's entire virtual memory region as a mutable slice of `MaybeUninit<T>`.
    #[inline(always)]
    pub fn as_uninit_virtual_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.virtual_capacity()) }
    }

    /// Returns a view of a sub-slice of the virtual memory region.
    ///
    /// The indices `start` and `len` are relative to the entire virtual region.
    ///
    /// # Panics
    ///
    /// Panics if the range `[start, start + len)` is out of bounds of the virtual slice,
    /// or if the resulting slice's byte length would exceed `isize::MAX`.
    #[inline(always)]
    pub fn virtual_uninit_slice_at(&self, start: usize, len: usize) -> &[MaybeUninit<T>] {
        debug_assert!(start.checked_add(len) <= Some(self.virtual_capacity()), "slice bounds out of virtual capacity");
        debug_assert!(
            len.checked_mul(size_of::<T>()).is_some_and(|bytes| bytes <= MAX_VIRTUAL_BUF_SIZE),
            "slice byte length exceeds isize::MAX"
        );
        unsafe { self.as_uninit_virtaul_slice().get_unchecked(start..start + len) }
    }

    /// Returns a view of a sub-slice of the virtual memory region.
    ///
    /// The indices `start` and `len` are relative to the entire virtual region.
    ///
    /// # Panics
    ///
    /// Panics if the range `[start, start + len]` is out of bounds of the virtual slice,
    /// or if the resulting slice's byte length would exceed `isize::MAX`.
    #[inline(always)]
    pub fn virtual_uninit_slice_mut_at(&mut self, start: usize, len: usize) -> &mut [MaybeUninit<T>] {
        debug_assert!(start.checked_add(len) <= Some(self.virtual_capacity()), "slice bounds out of virtual capacity");
        debug_assert!(
            len.checked_mul(size_of::<T>()).is_some_and(|bytes| bytes <= MAX_VIRTUAL_BUF_SIZE),
            "slice byte length exceeds isize::MAX"
        );
        unsafe { self.as_uninit_virtual_mut_slice().get_unchecked_mut(start..start + len) }
    }

    /// Returns a raw, mutable pointer to the beginning of the buffer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *mut T { self.ptr.as_ptr() }
}

impl<T> Default for MirroredBuffer<T> {
    #[inline(always)]
    fn default() -> Self { Self::new() }
}

impl<T> Deref for MirroredBuffer<T> {
    type Target = [MaybeUninit<T>];

    #[inline(always)]
    fn deref(&self) -> &Self::Target { self.as_uninit_virtaul_slice() }
}

impl<T> DerefMut for MirroredBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { self.as_uninit_virtual_mut_slice() }
}

impl<T> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        if T::IS_ZST || self.virtual_size() == 0 {
            return;
        }
        unsafe {
            deallocate_mirrored(self.ptr.as_ptr() as *mut u8, self.virtual_size()).expect("Failed to deallocate memory")
        }
    }
}

unsafe impl<T> Send for MirroredBuffer<T> where T: Send {}
unsafe impl<T> Sync for MirroredBuffer<T> where T: Sync {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_default_are_empty() {
        let buf_new = MirroredBuffer::<u8>::new();
        assert_eq!(buf_new.physical_capacity(), 0);
        assert_eq!(buf_new.virtual_size(), 0);
        assert_eq!(buf_new.physical_size(), 0);

        let buf_default = MirroredBuffer::<u32>::default();
        assert_eq!(buf_default.physical_capacity(), 0);
        assert_eq!(buf_default.virtual_size(), 0);
        assert_eq!(buf_default.physical_size(), 0);
    }

    #[test]
    fn with_capacity_zero_is_empty() {
        let buf = MirroredBuffer::<u8>::with_capacity(0);
        assert_eq!(buf.physical_capacity(), 0);
        assert_eq!(buf.virtual_size(), 0);
        assert_eq!(buf.physical_size(), 0);
    }

    #[test]
    fn with_capacity_allocates_memory() {
        let cap = 10;
        let buf = MirroredBuffer::<i32>::with_capacity(cap);

        // The actual capacity might be larger due to allocation granularity.
        assert!(buf.physical_capacity() >= cap);
        assert!(buf.virtual_size() > 0);
        assert_eq!(buf.virtual_size(), buf.physical_size() * 2);
        assert_eq!(
            buf.virtual_size() % allocation_granularity(),
            0,
            "Virtual size should be a multiple of allocation granularity"
        );
    }

    #[test]
    fn drop_deallocates() {
        // This test ensures that the Drop implementation runs without panicking.
        // It doesn't verify memory is actually freed, but it's a good sanity check.
        {
            let _buf = MirroredBuffer::<u64>::with_capacity(100);
        } // _buf is dropped here

        // Also test dropping an empty buffer
        {
            let _buf = MirroredBuffer::<u8>::new();
        }

        // Test dropping a ZST buffer
        {
            let _buf = MirroredBuffer::<()>::with_capacity(100);
        }
    }

    #[test]
    fn mirrored_writes_are_correct() {
        let mut buf = MirroredBuffer::<u32>::with_capacity(4);
        let capacity = buf.physical_capacity();
        assert!(capacity >= 4);

        let val1: u32 = 12345;
        let val2: u32 = 67890;

        unsafe {
            // Write to the first half
            *buf.get_unchecked_mut(0).as_mut_ptr() = val1;
            *buf.get_unchecked_mut(2).as_mut_ptr() = val2;

            // Read from the first half
            assert_eq!(*buf.get_unchecked(0).assume_init_ref(), val1);
            assert_eq!(*buf.get_unchecked(2).assume_init_ref(), val2);

            // Read from the mirrored second half
            assert_eq!(*buf.get_unchecked(capacity).assume_init_ref(), val1);
            assert_eq!(*buf.get_unchecked(2 + capacity).assume_init_ref(), val2);
        }
    }

    #[test]
    fn mirrored_writes_in_second_half_are_correct() {
        let mut buf = MirroredBuffer::<char>::with_capacity(8);
        let capacity = buf.physical_capacity();
        assert!(capacity >= 8);

        let val1 = 'A';
        let val2 = 'Z';

        unsafe {
            // Write to the second (mirrored) half
            buf.as_uninit_virtual_mut_slice()[capacity + 1].as_mut_ptr().write(val1);
            buf.as_uninit_virtual_mut_slice()[capacity + 5].as_mut_ptr().write(val2);

            // Read from the second half
            assert_eq!(*buf.get_unchecked(capacity + 1).assume_init_ref(), val1);
            assert_eq!(*buf.get_unchecked(capacity + 5).assume_init_ref(), val2);

            // Read from the first (physical) half
            assert_eq!(*buf.get_unchecked(1).assume_init_ref(), val1);
            assert_eq!(*buf.get_unchecked(5).assume_init_ref(), val2);
        }
    }

    #[test]
    fn slice_access_in_bounds() {
        let mut buf = MirroredBuffer::<u8>::with_capacity(16);
        let capacity = buf.physical_capacity();
        let v_len = buf.virtual_capacity();
        assert_eq!(v_len, capacity * 2);

        // Get a slice of the first half
        let slice1 = buf.virtual_uninit_slice_at(0, capacity);
        assert_eq!(slice1.len(), capacity);

        // Get a slice of the second half
        let slice2 = buf.virtual_uninit_slice_at(capacity, capacity);
        assert_eq!(slice2.len(), capacity);

        // Get a mutable slice across the boundary
        let slice3 = buf.virtual_uninit_slice_mut_at(capacity - 4, 8);
        assert_eq!(slice3.len(), 8);
    }

    #[test]
    #[should_panic]
    fn slice_access_out_of_bounds() {
        let buf = MirroredBuffer::<u8>::with_capacity(16);
        let v_len = buf.virtual_capacity();

        // This should panic because start + len > virtual_slice_len
        let _slice = buf.virtual_uninit_slice_at(v_len - 4, 5);
    }

    #[test]
    #[should_panic(expected = "slice bounds out of virtual capacity")]
    fn slice_access_starts_truly_out_of_bounds() {
        let buf = MirroredBuffer::<u8>::with_capacity(16);
        let v_len = buf.virtual_capacity();

        // 这应该会 panic，因为起始点已经超出了范围 (v_len + 1)
        let _slice = buf.virtual_uninit_slice_at(v_len + 1, 0);
    }

    #[test]
    fn alignment_test() {
        // Define a type with a specific, larger-than-usual alignment
        #[repr(align(32))]
        struct AlignedType(u64);

        // This test will panic if the assertion `align_of::<T>() <= ag` fails.
        // It serves to confirm the check is in place. On most systems where page size
        // (and thus allocation granularity) is 4096, this will pass.
        let buf = MirroredBuffer::<AlignedType>::with_capacity(4);
        assert!(buf.physical_capacity() >= 4);

        // Check if the returned pointer is indeed aligned.
        let ptr_addr = buf.as_ptr() as usize;
        assert_eq!(ptr_addr % align_of::<AlignedType>(), 0, "Pointer is not correctly aligned");
    }
}
