//! Mirrored memory buffer.
mod buffer;
mod utils;

pub(crate) use buffer::MirroredBuffer;
pub(crate) use utils::{mirrored_allocation_unit, MAX_USIZE_WITHOUT_HIGHEST_BIT};

// platform cfg

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
