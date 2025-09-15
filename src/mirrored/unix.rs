//! Non-racy unix-specific mirrored memory allocation.

use crate::mirrored::{MAX_PHYSICAL_BUF_SIZE, MAX_VIRTUAL_BUF_SIZE};
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

/// Retrieves the system's memory allocation granularity (page size), caching the value for performance.
///
/// This function caches the value after the first call to minimize subsequent syscalls.
///
/// ## System APIs Used
/// - Linux/Android: [`sysconf(_SC_PAGE_SIZE)`](https://man7.org/linux/man-pages/man3/sysconf.3.html)
/// - Other Unix: [`sysconf(_SC_PAGE_SIZE)`](https://man7.org/linux/man-pages/man3/sysconf.3.html)
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

/// Creates a file descriptor for anonymous shared memory on Linux/Android.
///
/// ## System APIs Used
/// - Linux/Android: [`memfd_create`](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
#[cfg(any(target_os = "linux", target_os = "android"))]
fn create_mem_fd() -> AnyResult<OwnedFd> {
    use nix::sys::memfd::{MFdFlags, memfd_create};
    let fd = memfd_create("mirrored_buffer", MFdFlags::MFD_CLOEXEC)?;
    Ok(fd)
}

#[cfg(all(target_family = "unix", not(any(target_os = "linux", target_os = "android"))))]
fn get_random_str(n: usize) -> Box<str> { std::iter::repeat_with(|| fastrand::alphanumeric()).take(n).collect() }

/// Creates a file descriptor for anonymous shared memory on Unix systems (except Linux/Android).
///
/// ## System APIs Used
/// - Other Unix: [`shm_open`](https://man7.org/linux/man-pages/man3/shm_open.3.html), [`shm_unlink`](https://man7.org/linux/man-pages/man3/shm_unlink.3.html)
#[cfg(all(target_family = "unix", not(any(target_os = "linux", target_os = "android"))))]
fn create_mem_fd() -> AnyResult<OwnedFd> {
    use nix::{
        fcntl::OFlag,
        sys::{
            mman::{shm_open, shm_unlink},
            stat::Mode,
        },
    };
    let name = format!("/{}", get_random_str(8));
    let fd = shm_open(name.as_str(), OFlag::O_RDWR | OFlag::O_CREAT | OFlag::O_EXCL, Mode::from_bits_truncate(0o600))?;
    shm_unlink(name.as_str()).context("shm_unlink failed")?;
    Ok(fd)
}

/// Allocates a mirrored memory buffer, ideal for high-performance circular buffers.
///
/// This is achieved by mapping a single underlying physical memory region to two contiguous
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
/// - Linux/Android: [`memfd_create`](https://man7.org/linux/man-pages/man2/memfd_create.2.html), [`mmap`](https://man7.org/linux/man-pages/man2/mmap.2.html),
///   [`mmap_anonymous`](https://man7.org/linux/man-pages/man2/mmap.2.html), [`munmap`](https://man7.org/linux/man-pages/man2/munmap.2.html),
///   [`ftruncate`](https://man7.org/linux/man-pages/man2/ftruncate.2.html), [`close`](https://man7.org/linux/man-pages/man2/close.2.html)
/// - Other Unix: [`shm_open`](https://man7.org/linux/man-pages/man3/shm_open.3.html), [`shm_unlink`](https://man7.org/linux/man-pages/man3/shm_unlink.3.html),
///   [`mmap`](https://man7.org/linux/man-pages/man2/mmap.2.html), [`mmap_anonymous`](https://man7.org/linux/man-pages/man2/mmap.2.html),
///   [`munmap`](https://man7.org/linux/man-pages/man2/munmap.2.html), [`ftruncate`](https://man7.org/linux/man-pages/man2/ftruncate.2.html),
///   [`close`](https://man7.org/linux/man-pages/man2/close.2.html)
pub(crate) unsafe fn allocate_mirrored(virtual_size: usize) -> AnyResult<*mut u8> {
    debug_assert!(
        virtual_size > 0
            && virtual_size.is_multiple_of(allocation_granularity() * 2)
            && virtual_size <= MAX_VIRTUAL_BUF_SIZE,
        "virtual_size must be a non-zero, even multiple of double allocation_granularity()"
    );
    let physical_size = virtual_size / 2;
    debug_assert!(
        physical_size != 0 && physical_size <= MAX_PHYSICAL_BUF_SIZE,
        "physical_size must be in range (0, iMAX_PHYSICAL_BUF_SIZE)"
    );
    let fd = create_mem_fd()?;
    ftruncate(fd.as_fd(), physical_size as _)?;
    let placeholder = unsafe {
        mmap_anonymous(None, virtual_size.try_into().unwrap(), ProtFlags::PROT_NONE, {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            {
                MapFlags::MAP_PRIVATE | MapFlags::MAP_NORESERVE
            }
            #[cfg(all(target_family = "unix", not(any(target_os = "linux", target_os = "android"))))]
            {
                MapFlags::MAP_PRIVATE
            }
        })
        .context("mmap failed")?
    };
    #[cfg(all(target_family = "unix", not(any(target_os = "linux", target_os = "android"))))]
    {
        unsafe { munmap(placeholder, virtual_size) }
            .with_context(|| format!("Failed to unmap placeholder at {:?}", placeholder))?;
    }
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
            #[cfg(any(target_os = "linux", target_os = "android"))]
            {
                let _ = unsafe { munmap(placeholder, virtual_size) };
            }
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

/// Deallocates a mirrored memory region created by `allocate_mirrored`.
///
/// This function unmaps the entire virtual address space.
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
/// - All Unix: [`munmap`](https://man7.org/linux/man-pages/man2/munmap.2.html)
pub(crate) unsafe fn deallocate_mirrored(ptr: *mut u8, virtual_size: usize) -> AnyResult<()> {
    debug_assert!(!ptr.is_null() && ptr.is_aligned(), "ptr must be a valid pointer and aligned");
    debug_assert!(
        virtual_size > 0
            && virtual_size.is_multiple_of(allocation_granularity() * 2)
            && virtual_size <= MAX_VIRTUAL_BUF_SIZE,
        "virtual_size must be a non-zero multiple of double allocation_granularity()"
    );
    unsafe { munmap(NonNull::new_unchecked(ptr as *mut c_void), virtual_size) }
        .with_context(|| format!("Failed to deallocate mirrored memory with munmap. Errno: {}", Errno::last()))?;
    Ok(())
}
