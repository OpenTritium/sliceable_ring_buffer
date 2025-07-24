use crate::mirrored::allocation_granularity;
use core::mem::{size_of, SizedTypeProperties};
use num::Integer;

/// Calculates the byte size of a mirrored allocation.
pub(crate) fn mirrored_allocation_unit<T>(count: usize) -> usize {
    if T::IS_ZST {
        return 0;
    }
    let ag = allocation_granularity();
    let t_size = size_of::<T>();
    let base = ag.lcm(&t_size);
    let total = size_of::<T>().strict_mul(count);
    let blocks = total.div_ceil(base);
    blocks.strict_mul(base).strict_mul(2)
}

// fn is_slice_contains_ptr<T>(s: &[T], p: *const T) -> bool { s.as_ptr_range().contains(&p) }

pub const MAX_USIZE_WITHOUT_HIGHEST_BIT: usize = isize::MAX as usize;
