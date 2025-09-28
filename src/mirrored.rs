//! Mirrored memory buffer.
mod buffer;
mod utils;

pub use buffer::MirroredBuffer;
pub use utils::{MAX_PHYSICAL_BUF_SIZE, MAX_VIRTUAL_BUF_SIZE, mirrored_allocation_unit};

#[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
mod unix;

#[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
pub use unix::*;

#[cfg(target_vendor = "apple")]
mod darwin;

#[cfg(target_vendor = "apple")]
pub(crate) use darwin::*;

#[cfg(windows)]
mod windows;

#[cfg(windows)]
pub(crate) use windows::*;
