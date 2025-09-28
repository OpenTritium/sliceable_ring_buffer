use crate::mirrored::allocation_granularity;
use num::Integer;
use std::mem::size_of;

#[cfg(not(feature = "unstable"))]
#[allow(unused_imports)]
use crate::stable::UsizeCompact;

/// Calculates the byte size of a mirrored allocation, typically used for implementing a ring buffer.
/// To ensure a valid virtual memory mapping, the size of a single mirrored region must be a
/// common multiple of `sizeof::<T>()` and the system's allocation granularity.
pub const fn is_zst<T>() -> bool { size_of::<T>() == 0 }

#[inline]
pub fn mirrored_allocation_unit<T>(count: usize) -> usize {
    if count == 0 || is_zst::<T>() {
        return 0;
    }
    let ag = allocation_granularity();
    let t_size = size_of::<T>();
    let base = ag.lcm(&t_size);
    let total = t_size.strict_mul(count);
    let blocks = total.div_ceil(base);
    blocks.strict_mul(base).strict_mul(2)
}

pub const MAX_VIRTUAL_BUF_SIZE: usize = MAX_PHYSICAL_BUF_SIZE * 2;
pub const MAX_PHYSICAL_BUF_SIZE: usize = (isize::MAX as usize >> 1) - 1;
