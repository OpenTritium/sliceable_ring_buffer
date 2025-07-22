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
    mem::{size_of, MaybeUninit, SizedTypeProperties},
    ptr::NonNull,
    slice,
};

type Size = core::num::niche_types::UsizeNoHighBit;

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
pub struct MirroredBuffer<T> {
    ptr: NonNull<T>,
    size: Size, // The total byte length of the underlying virtual memory region. Must not exceed `isize::MAX`.
}

impl<T> MirroredBuffer<T> {
    /// Creates a new, empty `MirroredBuffer` without allocating memory.
    #[must_use]
    pub(crate) fn new() -> Self { Self { ptr: NonNull::dangling(), size: unsafe { Size::new_unchecked(0) } } }

    /// Returns the number of elements of type `T` that can be stored in the physical region.
    ///
    /// This is the effective capacity of the buffer for storing unique items.
    pub(crate) fn capacity(&self) -> usize {
        let cap = self.physical_size();
        let t_size = size_of::<T>();
        debug_assert!(cap % t_size == 0);
        cap / t_size
    }

    /// Returns the total byte length of the entire virtual memory region.
    #[inline]
    pub(crate) fn virtual_size(&self) -> usize { self.size.as_inner() }

    /// Returns the byte length of the physical memory region (which is half of the virtual region).
    #[inline]
    pub(crate) fn physical_size(&self) -> usize {
        let v_size = self.virtual_size();
        debug_assert!(v_size.is_power_of_two());
        v_size / 2
    }

    /// Returns `true` if the buffer has an active memory allocation.
    #[inline]
    pub(crate) fn is_not_allocated(&self) -> bool { self.size.as_inner() != 0 }

    /// Allocates a new `MirroredBuffer` with enough space for at least `cap` elements.
    ///
    /// The actual allocated size may be larger than requested due to system
    /// alignment and memory page size requirements.
    pub(crate) fn with_capacity(cap: usize) -> Self {
        let virtual_size = mirrored_allocation_unit::<T>(cap);
        unsafe { Self::uninitialized(virtual_size) }
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
    unsafe fn uninitialized(virtual_size: usize) -> Self {
        if virtual_size == 0 {
            return Self::new();
        }
        let ag = allocation_granularity();
        debug_assert!(
            virtual_size % ag == 0 && virtual_size > 0 && virtual_size <= MAX_USIZE_WITHOUT_HIGHEST_BIT,
            "virtual_size must be a positive multiple of allocation_granularity() and less than usize::MAX"
        );
        // if virtual_size is multiple of allocation_granularity(), so it could be divided by 2
        let physical_size = virtual_size / 2;
        debug_assert!(physical_size != 0, "physical_size must be in range (0, MAX_USIZE_WITHOUT_HIGHEST_BIT/ 2)");
        assert!(
            align_of::<T>() <= ag,
            "The alignment requirements of `T` must be smaller than the allocation granularity."
        );
        let ptr = allocate_mirrored(virtual_size).expect("Allocation failed");
        Self { ptr: NonNull::new_unchecked(ptr as *mut T), size: Size::new_unchecked(virtual_size) }
    }

    /// Calculates the length of the virtual slice in terms of number of `T`s.
    #[inline(always)]
    fn virtual_slice_len(&self) -> usize {
        if T::IS_ZST {
            0
        } else {
            self.capacity() * 2
        }
    }

    /// Returns the buffer's entire virtual memory region as a slice of `MaybeUninit<T>`.
    #[inline(always)]
    pub(crate) fn as_uninit_virtaul_slice(&self) -> &[MaybeUninit<T>] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast(), self.virtual_slice_len()) }
    }

    /// Returns the buffer's entire virtual memory region as a mutable slice of `MaybeUninit<T>`.
    #[inline(always)]
    pub(crate) fn as_uninit_virtual_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.virtual_slice_len()) }
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
    pub fn uninit_virtual_slice_at(&self, start: usize, len: usize) -> &[MaybeUninit<T>] {
        assert!(start.checked_add(len) < Some(self.virtual_slice_len()), "slice bounds out of virtual capacity");
        assert!(
            len.checked_mul(size_of::<T>()).map_or(false, |bytes| bytes <= MAX_USIZE_WITHOUT_HIGHEST_BIT),
            "slice byte length exceeds isize::MAX"
        );
        unsafe { slice::from_raw_parts(self.ptr.add(start).as_ptr().cast(), len) }
    }

    #[inline(always)]
    pub fn uninit_virtual_slice_mut_at(&mut self, start: usize, len: usize) -> &mut [MaybeUninit<T>] {
        assert!(start.checked_add(len) < Some(self.virtual_slice_len()), "slice bounds out of virtual capacity");
        assert!(
            len.checked_mul(size_of::<T>()).map_or(false, |bytes| bytes <= MAX_USIZE_WITHOUT_HIGHEST_BIT),
            "slice byte length exceeds isize::MAX"
        );
        unsafe { slice::from_raw_parts_mut(self.ptr.add(start).as_ptr().cast(), len) }
    }

    #[inline(always)]
    pub unsafe fn virtual_slice_at_unchecked(&self, start: usize, len: usize) -> &[T] {
        debug_assert!(start.checked_add(len) < Some(self.virtual_slice_len()), "slice bounds out of capacity");
        debug_assert!(
            len.checked_mul(size_of::<T>()).map_or(false, |bytes| bytes <= MAX_USIZE_WITHOUT_HIGHEST_BIT),
            "slice byte length exceeds isize::MAX"
        );
        unsafe { slice::from_raw_parts(self.ptr.add(start).as_ptr(), len) }
    }

    #[inline(always)]
    pub unsafe fn virtual_slice_mut_at_unchecked(&mut self, start: usize, len: usize) -> &mut [T] {
        debug_assert!(start.checked_add(len) < Some(self.virtual_slice_len()), "slice bounds out of capacity");
        debug_assert!(
            len.checked_mul(size_of::<T>()).map_or(false, |bytes| bytes <= MAX_USIZE_WITHOUT_HIGHEST_BIT),
            "slice byte length exceeds isize::MAX"
        );
        unsafe { slice::from_raw_parts_mut(self.ptr.add(start).as_ptr(), len) }
    }

    /// Returns a raw, constant pointer to the beginning of the buffer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T { self.ptr.as_ptr() }

    /// Returns a raw, mutable pointer to the beginning of the buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr.as_ptr() }

    /// Returns a reference to an element at `idx` in the virtual region, without checking for initialization.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: usize) -> &MaybeUninit<T> {
        self.as_uninit_virtaul_slice().get_unchecked(idx)
    }

    /// Returns a mutable reference to an element at `idx` in the virtual region, without checking for initialization.
    #[inline(always)]
    pub unsafe fn get_mut_unchecked(&mut self, idx: usize) -> &mut MaybeUninit<T> {
        self.as_uninit_virtual_slice_mut().get_unchecked_mut(idx)
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<&MaybeUninit<T>> { self.as_uninit_virtaul_slice().get(idx) }

    #[inline(always)]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut MaybeUninit<T>> {
        self.as_uninit_virtual_slice_mut().get_mut(idx)
    }
}

impl<T> Default for MirroredBuffer<T> {
    fn default() -> Self { Self::new() }
}

impl<T> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        if T::IS_ZST || !self.is_not_allocated() {
            return;
        }
        unsafe {
            deallocate_mirrored(self.ptr.as_ptr() as *mut u8, self.virtual_size()).expect("Failed to deallocate memory")
        }
    }
}

impl<T> Clone for MirroredBuffer<T> {
    fn clone(&self) -> Self {
        unsafe {
            let mut new_buffer = Self::uninitialized(self.virtual_size());
            new_buffer.as_mut_ptr().copy_from_nonoverlapping(self.as_ptr(), self.capacity());
            new_buffer
        }
    }
}

unsafe impl<T> Send for MirroredBuffer<T> where T: Send {}
unsafe impl<T> Sync for MirroredBuffer<T> where T: Sync {}
