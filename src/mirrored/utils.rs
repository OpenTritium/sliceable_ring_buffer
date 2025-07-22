use crate::mirrored::allocation_granularity;
use core::mem::{align_of, size_of, SizedTypeProperties};
use num::Integer;

/// Calculates the byte size of a mirrored allocation.
pub(crate) fn mirrored_allocation_unit<T>(count: usize) -> usize {
    if T::IS_ZST {
        return 0;
    }
    let ag = allocation_granularity();
    let align = align_of::<T>();
    let base = ag.lcm(&align);
    let total = size_of::<T>().checked_mul(count).expect("Allocation size overflow");
    let blocks = total.div_ceil(base);
    blocks.checked_mul(base).and_then(|s| s.checked_mul(2)).expect("Allocation size overflow")
}

fn is_slice_contains_ptr<T>(s: &[T], p: *const T) -> bool { s.as_ptr_range().contains(&p) }

pub const MAX_USIZE_WITHOUT_HIGHEST_BIT: usize = isize::MAX as usize;
