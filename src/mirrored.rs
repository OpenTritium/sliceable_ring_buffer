//! Mirrored memory buffer.
mod buffer;
mod utils;

pub(crate) use buffer::MirroredBuffer;
pub(crate) use utils::{MAX_PHYSICAL_BUF_SIZE, MAX_VIRTUAL_BUF_SIZE, mirrored_allocation_unit};

#[cfg(unix)]
mod unix;

#[cfg(unix)]
pub(crate) use unix::*;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod macos;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) use macos::*;

#[cfg(windows)]
mod windows;

#[cfg(windows)]
pub(crate) use windows::*;
