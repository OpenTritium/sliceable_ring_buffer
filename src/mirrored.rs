//! Mirrored memory buffer.
mod buffer;
mod utils;

pub(crate) use buffer::MirroredBuffer;
pub(crate) use utils::{MAX_PHYSICAL_BUF_SIZE, MAX_VIRTUAL_BUF_SIZE, mirrored_allocation_unit};

#[cfg(any(target_os = "linux", target_os = "android", target_os = "openbsd"))]
mod linux;
#[cfg(all(target_os = "linux", target_os = "android", target_os = "openbsd"))]
pub(crate) use linux::*;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod macos;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) use macos::*;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "windows")]
pub(crate) use windows::*;
