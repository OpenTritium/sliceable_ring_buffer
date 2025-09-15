//! Implements the allocator hooks on top of window's virtual alloc.

use crate::mirrored::{MAX_PHYSICAL_BUF_SIZE, MAX_VIRTUAL_BUF_SIZE};
use anyhow::{Context, Result as AnyResult, bail};
use std::{
    ffi::c_void,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, INVALID_HANDLE_VALUE},
        System::{
            Memory::{
                CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
                MEM_RESERVE_PLACEHOLDER, MEM_UNMAP_NONE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS,
                PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile2, VIRTUAL_FREE_TYPE, VirtualAlloc2, VirtualFree,
            },
            SystemInformation::{GetSystemInfo, SYSTEM_INFO},
            Threading::GetCurrentProcess,
        },
    },
    core::PCWSTR,
};

/// Retrieves the system's memory allocation granularity, caching the value for performance.
///
/// The allocation granularity is the smallest unit for which virtual memory can be reserved.
/// This function caches the value after the first call to minimize subsequent syscalls.
pub(crate) fn allocation_granularity() -> usize {
    const UNINIT_ALLOCATION_GRANULARITY: usize = 0;
    static ALLOCATION_GRANULARITY: AtomicUsize = AtomicUsize::new(0);
    let cached_val = ALLOCATION_GRANULARITY.load(Ordering::Acquire);
    // fast path
    if cached_val != UNINIT_ALLOCATION_GRANULARITY {
        return cached_val;
    }
    // slow path
    let updated_val = unsafe {
        let mut system_info = MaybeUninit::<SYSTEM_INFO>::uninit();
        GetSystemInfo(system_info.as_mut_ptr());
        let system_info = system_info.assume_init();
        system_info.dwAllocationGranularity as usize
    };
    match ALLOCATION_GRANULARITY.compare_exchange(
        UNINIT_ALLOCATION_GRANULARITY, // when the value is uninit, we expect to write to it
        updated_val,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => updated_val,
        Err(val_from_other_thread) => val_from_other_thread, // another thread has updated the value
    }
}

/// Allocates a mirrored memory buffer, ideal for high-performance circular buffers.
///
/// This is achieved by mapping a single underlying physical memory region to two contiguous
/// virtual memory regions. Writes to the first half `[0, size/2)` are mirrored to the
/// second half `[size/2, size)`, and vice versa.
///
/// This implementation uses modern Windows APIs and is only available on
/// **Windows 10 (version 1803) or newer**.
///
/// # Arguments
///
/// * `virtual_size`: The total virtual size of the buffer. This must be a non-zero, even multiple of the system's
///   `allocation_granularity()`. The usable ring buffer capacity will be `virtual_size / 2`.
///
/// # Returns
///
/// On success, returns a raw pointer to the start of the `virtual_size`-byte mirrored region.
///
/// # Safety
///
/// The caller is responsible for the following invariants:
/// - The memory must be deallocated **exactly once** using `deallocate_mirrored` with the same `virtual_size`. Do not
///   use `Box` or other allocators.
/// - All memory access must be within the `[0, virtual_size)` bounds.
///
/// # Errors
///
/// Returns an `Err` if any underlying OS API call fails.
pub(crate) unsafe fn allocate_mirrored(virtual_size: usize) -> AnyResult<*mut u8> {
    debug_assert!(
        virtual_size.is_multiple_of(allocation_granularity() * 2)
            && virtual_size > 0
            && virtual_size <= MAX_VIRTUAL_BUF_SIZE,
        "virtual_size must be a multiple of double allocation_granularity() and > 0"
    );
    // if virtual_size is multiple of allocation_granularity(), so it could be divided by 2
    let physical_size = virtual_size / 2;
    debug_assert!(
        physical_size != 0 && physical_size <= MAX_PHYSICAL_BUF_SIZE,
        "physical_size must be in range (0, MAX_PHYSICAL_BUF_SIZE)"
    );
    let max_size_low = physical_size as u32;
    let max_size_high = (physical_size >> 32) as u32;
    unsafe {
        let file_mapping = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            None,
            PAGE_READWRITE | SEC_COMMIT,
            max_size_high,
            max_size_low,
            PCWSTR::null(),
        )
        .context("CreateFileMappingW failed")?;
        let current_process = GetCurrentProcess();
        let placeholder = VirtualAlloc2(
            Some(current_process),
            None, // let the system choose an address
            virtual_size,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS.0,
            None,
        );
        if placeholder.is_null() {
            CloseHandle(file_mapping)?;
            bail!("VirtualAlloc2 failed");
        }
        // free two times to divide the memory region into two halves
        let low_half_addr = placeholder;
        let high_half_addr = low_half_addr.add(physical_size);
        let partition = |addr| {
            if VirtualFree(addr, physical_size, VIRTUAL_FREE_TYPE(MEM_RELEASE.0 | MEM_PRESERVE_PLACEHOLDER.0)).is_err()
            {
                // don't care about the result of resource cleanup, ensure other resources are cleaned up
                let _ = VirtualFree(placeholder, 0, MEM_RELEASE);
                let _ = CloseHandle(file_mapping);
                bail!("Parition failed");
            }
            Ok(())
        };
        // partition(high_half_addr)?;
        partition(low_half_addr)?; // you just free any half, another half will be split automatically
        let clean_view = |view| {
            let _ = UnmapViewOfFile2(current_process, view, MEM_UNMAP_NONE);
            let _ = VirtualFree(placeholder, 0, MEM_RELEASE);
            let _ = CloseHandle(file_mapping);
        };
        // map a physical memory to two virtual memory regions
        let map_and_check_view = |addr| {
            let view = MapViewOfFile3(
                file_mapping,
                Some(current_process),
                Some(addr),
                0, // don't care about page alignment
                physical_size,
                MEM_REPLACE_PLACEHOLDER, // replace the placeholder
                PAGE_READWRITE.0,
                None,
            );
            if view.Value.is_null() {
                clean_view(view);
                bail!("MapViewOfFile3 failed")
            }
            Ok(view)
        };
        let low_view = map_and_check_view(low_half_addr)?;
        // clean low half view resources addtionally if high half view fails
        map_and_check_view(high_half_addr).inspect_err(|_| clean_view(low_view))?;
        // The file mapping handle can now be closed. The OS will keep the underlying section alive until all views are
        // unmapped.
        CloseHandle(file_mapping)?;
        Ok(low_half_addr as *mut u8)
    }
}

/// Deallocates a mirrored memory region created by `allocate_mirrored`.
///
/// This function unmaps both memory views and then frees the entire virtual address
/// space placeholder.
///
/// # Safety
///
/// The caller MUST ensure `ptr` is the valid pointer and `size` is the exact,
/// `virtual_size` is same as the one passed to `allocate_mirrored`.
/// This function must be called exactly once per allocation.
///
/// # Errors
///
/// Returns an `Err` if any underlying OS deallocation step fails. The function
/// attempts all cleanup steps regardless of intermediate failures.
pub(crate) unsafe fn deallocate_mirrored(ptr: *mut u8, virtual_size: usize) -> AnyResult<()> {
    let ptr = ptr as *mut c_void;
    debug_assert!(!ptr.is_null() && ptr.is_aligned(), "ptr must be a valid pointer and aligned");
    debug_assert!(
        virtual_size.is_multiple_of(allocation_granularity() * 2)
            && virtual_size > 0
            && virtual_size <= MAX_VIRTUAL_BUF_SIZE,
        "virtual_size must be a multiple of double allocation_granularity() and > 0"
    );
    // if virtual_size is multiple of allocation_granularity(), so it could be divided by 2
    let physical_size = virtual_size / 2;
    debug_assert!(
        physical_size != 0 && physical_size <= MAX_PHYSICAL_BUF_SIZE,
        "physical_size must be in range (0, MAX_PHYSICAL_BUF_SIZE)"
    );
    unsafe {
        let current_process = GetCurrentProcess();
        let low_ptr = ptr;
        let high_ptr = ptr.add(physical_size);
        let into_view = |p| MEMORY_MAPPED_VIEW_ADDRESS { Value: p };
        let unmap_low_result = UnmapViewOfFile2(current_process, into_view(low_ptr), MEM_UNMAP_NONE);
        let unmap_high_result = UnmapViewOfFile2(current_process, into_view(high_ptr), MEM_UNMAP_NONE);
        // no need to free the placeholder, it already has been freed by the unmap operation
        if unmap_low_result.is_err() || unmap_high_result.is_err() {
            bail!(
                "Failed to fully deallocate mirrored memory. Status: [Unmap Low: {:?}, Unmap High: {:?}]",
                unmap_low_result,
                unmap_high_result,
            )
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::slice;

    #[test]
    fn test_happy_path_allocation() {
        let virtual_size = allocation_granularity() * 16;
        let ptr = unsafe { allocate_mirrored(virtual_size).expect("Failed to allocate mirrored memory") };
        assert!(!ptr.is_null(), "Allocated pointer should not be null");
        unsafe {
            deallocate_mirrored(ptr, virtual_size).expect("Failed to deallocate mirrored memory");
        }
    }

    #[test]
    fn test_mirrored_write_read() {
        let virtual_size = allocation_granularity() * 4;
        let physical_size = virtual_size / 2;
        let ptr = unsafe { allocate_mirrored(virtual_size).expect("Allocation failed") };
        unsafe {
            let full_slice = slice::from_raw_parts_mut(ptr, virtual_size);
            let test_data = (0..physical_size).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
            full_slice[..physical_size].copy_from_slice(&test_data);
            let second_half = &full_slice[physical_size..];
            assert_eq!(
                second_half,
                test_data.as_slice(),
                "Data written to the first half was not mirrored to the second half"
            );
            full_slice.fill(0);
            assert!(full_slice.iter().all(|&b| b == 0));
            full_slice[physical_size..].copy_from_slice(&test_data);
            let first_half = &full_slice[..physical_size];
            assert_eq!(
                first_half,
                test_data.as_slice(),
                "Data written to the second half was not mirrored to the first half"
            );
            deallocate_mirrored(ptr, virtual_size).expect("Deallocation failed");
        }
    }

    #[test]
    fn test_write_across_boundary() {
        let virtual_size = allocation_granularity() * 8;
        let physical_size = virtual_size / 2;
        let ptr = unsafe { allocate_mirrored(virtual_size).expect("Allocation failed") };
        unsafe {
            let full_slice = slice::from_raw_parts_mut(ptr, virtual_size);
            let test_data = b"hello_world";
            let data_len = test_data.len();
            let start_pos = physical_size - 5;
            let target_slice = &mut full_slice[start_pos..start_pos + data_len];
            target_slice.copy_from_slice(test_data);
            let written_slice = slice::from_raw_parts(ptr.add(start_pos), data_len);
            assert_eq!(written_slice, test_data);
            assert_eq!(&full_slice[0..data_len - 5], &test_data[5..]);
            deallocate_mirrored(ptr, virtual_size).expect("Deallocation failed");
        }
    }

    #[test]
    fn test_minimum_valid_size() {
        let min_virtual_size = allocation_granularity() * 2;
        let ptr = unsafe { allocate_mirrored(min_virtual_size).expect("Allocation with minimum valid size failed") };
        assert!(!ptr.is_null());
        unsafe {
            let p = ptr.as_mut().unwrap();
            *p = 123;
            assert_eq!(*ptr.add(min_virtual_size / 2), 123);
            deallocate_mirrored(ptr, min_virtual_size).expect("Deallocation failed");
        }
    }
}
