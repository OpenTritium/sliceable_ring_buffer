//! Non-racy macOS-specific mirrored memory allocation using mach2.

use crate::mirrored::{MAX_PHYSICAL_BUF_SIZE, MAX_VIRTUAL_BUF_SIZE};
use anyhow::{Result as AnyResult, bail};
use mach2::{
    kern_return::KERN_SUCCESS,
    traps::mach_task_self,
    vm::{mach_vm_allocate, mach_vm_deallocate, mach_vm_remap},
    vm_inherit::VM_INHERIT_NONE,
    vm_page_size::vm_page_size,
    vm_prot::{VM_PROT_READ, VM_PROT_WRITE},
    vm_statistics::{VM_FLAGS_ANYWHERE, VM_FLAGS_FIXED, VM_FLAGS_OVERWRITE},
    vm_types::{mach_vm_address_t, mach_vm_size_t},
};
use std::mem::MaybeUninit;

/// Retrieves the system's memory allocation granularity (page size).
///
/// ## System APIs Used
/// - `vm_page_size`
#[inline]
pub(crate) fn allocation_granularity() -> usize { unsafe { vm_page_size as usize } }

/// Allocates a mirrored memory buffer, ideal for high-performance circular buffers.
///
/// This is achieved by allocating a physical memory region and then mapping it to two contiguous
/// virtual memory regions. Writes to the first half `[0, size/2)` are mirrored to the
/// second half `[size/2, size)`, and vice versa.
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
///
/// ## System APIs Used
/// - `mach_task_self`
/// - [`mach_vm_allocate`](https://developer.apple.com/documentation/kernel/1402376-mach_vm_allocate)
/// - [`mach_vm_remap`](https://developer.apple.com/documentation/kernel/1402218-mach_vm_remap)
pub(crate) unsafe fn allocate_mirrored(virtual_size: usize) -> AnyResult<*mut u8> {
    debug_assert!(
        virtual_size > 0
            && virtual_size.is_multiple_of(allocation_granularity() * 2)
            && virtual_size <= MAX_VIRTUAL_BUF_SIZE,
        "virtual_size must be a non-zero, even multiple of double allocation_granularity()"
    );
    let virtual_size = virtual_size as mach_vm_size_t;
    let physical_size = virtual_size / 2;
    debug_assert!(physical_size != 0 && physical_size <= MAX_PHYSICAL_BUF_SIZE as u64);
    let this_task = unsafe { mach_task_self() };
    let mut placeholder_addr = MaybeUninit::<mach_vm_address_t>::uninit();
    let result = unsafe { mach_vm_allocate(this_task, placeholder_addr.as_mut_ptr(), virtual_size, VM_FLAGS_ANYWHERE) };
    if result != KERN_SUCCESS {
        bail!("mach_vm_allocate for the placeholder region failed with error: {}", result);
    }
    let low_half_addr = unsafe { placeholder_addr.assume_init() };
    let mut high_half_addr = low_half_addr + physical_size;
    let mut rw_prot = VM_PROT_READ | VM_PROT_WRITE;
    let result = unsafe {
        mach_vm_remap(
            this_task,
            &mut high_half_addr,
            physical_size,
            0,
            VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
            this_task,
            low_half_addr,
            0,
            &mut rw_prot,
            &mut rw_prot,
            VM_INHERIT_NONE,
        )
    };
    if result != KERN_SUCCESS {
        unsafe { mach_vm_deallocate(this_task, low_half_addr, virtual_size) };
        bail!("mach_vm_remap failed with error: {}", result);
    }
    debug_assert_eq!(high_half_addr, low_half_addr + physical_size);
    Ok(low_half_addr as *mut u8)
}

/// Deallocates a mirrored memory region created by `allocate_mirrored`.
///
/// This function deallocates the entire virtual address space.
///
/// # Arguments
///
/// * `ptr`: Pointer to the memory region to deallocate
/// * `virtual_size`: The size of the virtual memory region
///
/// # Safety
///
/// The caller MUST ensure `ptr` is the valid pointer and `size` is the exact,
/// `virtual_size` is same as the one passed to `allocate_mirrored`.
/// This function must be called exactly once per allocation.
///
/// # Errors
///
/// Returns an `Err` if any underlying OS deallocation step fails.
///
/// ## System APIs Used
/// - `mach_task_self`
/// - [`mach_vm_deallocate`](https://developer.apple.com/documentation/kernel/1402285-mach_vm_deallocate)
pub(crate) unsafe fn deallocate_mirrored(ptr: *mut u8, virtual_size: usize) -> AnyResult<()> {
    debug_assert!(!ptr.is_null() && ptr.is_aligned(), "ptr must be a valid pointer and aligned");
    debug_assert!(
        virtual_size > 0 && virtual_size.is_multiple_of(allocation_granularity() * 2),
        "virtual_size must be a non-zero, even multiple of allocation_granularity()"
    );
    let this_task = unsafe { mach_task_self() };
    let result = unsafe { mach_vm_deallocate(this_task, ptr as mach_vm_address_t, virtual_size as mach_vm_size_t) };
    if result != KERN_SUCCESS {
        bail!("mach_vm_deallocate failed with error: {}", result);
    }
    Ok(())
}
