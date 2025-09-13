//! Non-racy linux-specific mirrored memory allocation.
//!
//! 功能	      Windows API	                      Linux / Android API	          macOS API	        BSD API
// 创建物理存储	CreateFileMappingW	               memfd_create 或 shm_open	        mach_vm_allocate	shm_open
// 预留虚拟地址	VirtualAlloc2	                    mmap + munmap	                mach_vm_allocate	mmap + munmap
// 映射/镜像内存	MapViewOfFile3	                mmap (使用 MAP_FIXED)	        mach_vm_remap	    mmap (使用 MAP_FIXED)
// 释放/取消映射	UnmapViewOfFile2, VirtualFree	munmap	                        mach_vm_deallocate	munmap, shm_unlink

use crate::mirrored::MAX_VIRTUAL_BUF_SIZE;
use anyhow::{Context, Result as AnyResult};
use nix::{
    errno::Errno,
    sys::mman::{MapFlags, ProtFlags, mmap, mmap_anonymous, munmap},
    unistd::{SysconfVar, close, ftruncate, sysconf},
};
use std::{
    ffi::c_void,
    num::NonZeroUsize,
    os::fd::{AsFd, OwnedFd},
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

#[inline(always)]
fn nonnull_into_nonzerousize(ptr: NonNull<c_void>) -> NonZeroUsize {
    unsafe { NonZeroUsize::new_unchecked(ptr.as_ptr() as usize) }
}

pub(crate) fn allocation_granularity() -> usize {
    const UNINIT_ALLOCATION_GRANULARITY: usize = 0;
    static ALLOCATION_GRANULARITY: AtomicUsize = AtomicUsize::new(0);
    let cached_val = ALLOCATION_GRANULARITY.load(Ordering::Acquire);
    if cached_val != UNINIT_ALLOCATION_GRANULARITY {
        return cached_val;
    }
    let updated_val = sysconf(SysconfVar::PAGE_SIZE).ok().flatten().expect("failed to obtain page size") as usize;
    match ALLOCATION_GRANULARITY.compare_exchange(
        UNINIT_ALLOCATION_GRANULARITY,
        updated_val,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => updated_val,
        Err(val_from_other_thread) => val_from_other_thread,
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn create_mem_fd() -> AnyResult<OwnedFd> {
    use nix::sys::memfd::{MFdFlags, memfd_create};
    let fd = memfd_create("mirrored_buffer", MFdFlags::MFD_CLOEXEC)?;
    Ok(fd)
}

#[cfg(all(target_family = "unix", not(any(target_os = "linux", target_os = "android"))))]
fn create_mem_fd() -> AnyResult<RawFd> {
    //     // 为了避免命名冲突，我们使用一个随机生成的名字
    // use nix::{
    //     fcntl::OFlag,
    //     sys::{
    //         mman::{shm_open, shm_unlink},
    //         stat::Mode,
    //     },
    // };

    // let mut name = [0u8; 12];
    // getrandom::getrandom(&mut name).context("Failed to generate random name for shm_open")?;
    // let name = format!("/{}", hex::encode(name));

    // let fd = shm_open(name.as_str(), OFlag::O_RDWR | OFlag::O_CREAT | OFlag::O_EXCL, Mode::from_bits_truncate(0o600))
    //     .with_context(|| format!("shm_open with name '{}' failed", name))?;

    // // shm_unlink 可以在 fd 打开时立即调用，内核会等到所有引用关闭后才真正删除它。
    // // 这可以确保即使程序崩溃，共享内存对象也会被清理。
    // shm_unlink(name.as_str()).context("shm_unlink failed")?;
    todo!()
}

pub(crate) unsafe fn allocate_mirrored(virtual_size: usize) -> AnyResult<*mut u8> {
    debug_assert!(
        virtual_size > 0 && virtual_size.is_multiple_of(allocation_granularity()),
        "virtual_size must be a non-zero, even multiple of allocation_granularity()"
    );
    let physical_size = virtual_size / 2;
    debug_assert!(
        physical_size != 0 && physical_size <= MAX_VIRTUAL_BUF_SIZE,
        "physical_size must be in range (0, isize::MAX)"
    );
    let fd = create_mem_fd()?;
    ftruncate(fd.as_fd(), physical_size as _)?;
    let placeholder = unsafe {
        mmap_anonymous(
            None,
            virtual_size.try_into().unwrap(),
            ProtFlags::PROT_NONE,
            MapFlags::MAP_PRIVATE | MapFlags::MAP_NORESERVE,
        )
        .context("mmap failed")?
    };
    let map_view = |addr, fd| unsafe {
        mmap(
            Some(nonnull_into_nonzerousize(addr)), // 映射到指定地址
            physical_size.try_into().unwrap(),
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED | MapFlags::MAP_FIXED, // MAP_FIXED 确保使用我们提供的地址
            fd,
            0,
        )
    };
    let low_half_addr = placeholder;
    let high_half_addr = unsafe { placeholder.byte_add(physical_size) };
    let low_view = map_view(low_half_addr, fd.as_fd())
        .with_context(|| format!("Failed to map low half at {:?}", low_half_addr))
        .inspect_err(|_| {
            let _ = unsafe { munmap(placeholder, virtual_size) };
        })?;
    let high_view = map_view(high_half_addr, fd.as_fd())
        .inspect_err(|_| {
            let _ = unsafe { munmap(low_view, physical_size) }; // ignore unmap result
        })
        .context("Failed to map high half")?;
    close(fd).context("Failed to close file descriptor")?;
    debug_assert_eq!(low_view, placeholder);
    debug_assert_eq!(high_view, unsafe { placeholder.byte_add(physical_size) });
    Ok(low_view.as_ptr() as *mut u8)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) unsafe fn deallocate_mirrored(ptr: *mut u8, virtual_size: usize) -> AnyResult<()> {
    debug_assert!(!ptr.is_null(), "ptr must be a valid pointer");
    debug_assert!(
        virtual_size > 0 && virtual_size.is_multiple_of(allocation_granularity()),
        "virtual_size must be a non-zero multiple of allocation_granularity()"
    );
    unsafe { munmap(NonNull::new_unchecked(ptr as *mut c_void), virtual_size) }
        .with_context(|| format!("Failed to deallocate mirrored memory with munmap. Errno: {}", Errno::last()))?;
    Ok(())
}
