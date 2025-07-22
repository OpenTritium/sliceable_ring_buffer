//! Implements the allocator hooks on top of window's virtual alloc.
use anyhow::{bail, Context, Result as AnyResult};
use core::{
    ffi::c_void,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};
use windows::{
    core::PCWSTR,
    Win32::{
        Foundation::{CloseHandle, INVALID_HANDLE_VALUE},
        System::{
            Memory::{
                CreateFileMappingW, MapViewOfFile3, UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
                MEMORY_MAPPED_VIEW_ADDRESS, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
                MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEM_UNMAP_NONE, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
                VIRTUAL_FREE_TYPE,
            },
            SystemInformation::{GetSystemInfo, SYSTEM_INFO},
            Threading::GetCurrentProcess,
        },
    },
};

/// Retrieves the system's memory allocation granularity, caching the value for performance.
///
/// The allocation granularity is the smallest unit for which virtual memory can be reserved.
/// This function caches the value after the first call to minimize subsequent syscalls.
pub(crate) fn allocation_granularity() -> usize {
    const UNINIT_ALLOCATION_GRANULARITY: usize = 0;
    static ALLOCATION_GRANULARITY: AtomicUsize = AtomicUsize::new(0);
    let cached_val = ALLOCATION_GRANULARITY.load(Ordering::Relaxed);
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
        Ordering::SeqCst,
        Ordering::Relaxed,
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
pub(crate) fn allocate_mirrored(virtual_size: usize) -> AnyResult<*mut u8> {
    debug_assert!(
        virtual_size % allocation_granularity() == 0 && virtual_size > 0,
        "virtual_size must be a multiple of allocation_granularity() and > 0"
    );
    // if virtual_size is multiple of allocation_granularity(), so it could be divided by 2
    let physical_size = virtual_size / 2;
    debug_assert!(
        physical_size != 0 && physical_size <= isize::MAX as usize,
        "physical_size must be in range (0, isize::MAX)"
    );
    let maximum_size_low = physical_size as u32;
    let maximum_size_high = (physical_size >> 32) as u32;
    unsafe {
        let file_mapping = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            None,
            PAGE_READWRITE | SEC_COMMIT,
            maximum_size_high,
            maximum_size_low,
            PCWSTR::null(),
        )
        .with_context(|| "CreateFileMappingW failed")?;
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
        let partition = |addr| {
            if VirtualFree(addr, physical_size, VIRTUAL_FREE_TYPE(MEM_RELEASE.0 | MEM_PRESERVE_PLACEHOLDER.0)).is_err()
            {
                // don't care about the result of resource cleanup, ensure other resources are cleaned up
                let _ = VirtualFree(placeholder, 0, MEM_RELEASE);
                let _ = CloseHandle(file_mapping);
                bail!("VirtualFree failed");
            }
            Ok(())
        };
        partition(low_half_addr)?;
        let high_half_addr = low_half_addr.offset(physical_size as isize);
        partition(high_half_addr)?;
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
    debug_assert!(!ptr.is_null() && ptr.is_aligned(), "ptr must be a valid pointer and aligned");
    debug_assert!(
        virtual_size % allocation_granularity() == 0 && virtual_size > 0,
        "virtual_size must be a multiple of allocation_granularity() and > 0"
    );
    // if virtual_size is multiple of allocation_granularity(), so it could be divided by 2
    let physical_size = virtual_size / 2;
    debug_assert!(
        physical_size != 0 && physical_size <= isize::MAX as usize,
        "physical_size must be in range (0, isize::MAX)"
    );
    let into_view = |p| MEMORY_MAPPED_VIEW_ADDRESS { Value: p as *mut c_void };
    let low_addr = into_view(ptr);
    let unmap_low_result = UnmapViewOfFile(low_addr);
    let high_addr = into_view(ptr.offset(physical_size as isize));
    let unmap_high_result = UnmapViewOfFile(high_addr);
    // when you use `MEM_RELEASE` to release a memory region, the `dwsize` parameter must be zero
    let free_result = VirtualFree(ptr as *mut c_void, 0, MEM_RELEASE);
    if unmap_low_result.is_err() || unmap_high_result.is_err() || free_result.is_err() {
        bail!(
            "Failed to fully deallocate mirrored memory. Status: [Unmap Low: {:?}, Unmap High: {:?}, VirtualFree: \
             {:?}]",
            unmap_low_result,
            unmap_high_result,
            free_result
        )
    }
    Ok(())
}
