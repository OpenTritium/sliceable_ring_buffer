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

#[inline(always)]
pub(crate) fn allocation_granularity() -> usize { unsafe { vm_page_size as usize } }

/// Allocates a mirrored memory region.
/// The returned pointer points to a virtual memory region of `virtual_size`.
/// The second half of this region is a mirror of the first half.
pub(crate) unsafe fn allocate_mirrored(virtual_size: usize) -> AnyResult<*mut u8> {
    debug_assert!(
        virtual_size > 0
            && virtual_size.is_multiple_of(allocation_granularity() * 2)
            && virtual_size <= MAX_VIRTUAL_BUF_SIZE,
        "virtual_size must be a non-zero, even multiple of double allocation_granularity()"
    );
    let virtual_size = virtual_size as mach_vm_size_t;
    let physical_size = virtual_size / 2;
    debug_assert!(physical_size != 0 && physical_size <= MAX_PHYSICAL_BUF_SIZE);
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

/// Deallocates a mirrored memory region allocated with `allocate_mirrored`.
pub(crate) unsafe fn deallocate_mirrored(ptr: *mut u8, virtual_size: usize) -> AnyResult<()> {
    debug_assert!(!ptr.is_null(), "ptr must be a valid pointer");
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
