use std::{mem::MaybeUninit, ptr};
pub type Size = usize;
pub trait SizeCompact {
    unsafe fn new_unchecked(value: usize) -> Self;
    fn as_inner(&self) -> usize;
}
impl SizeCompact for Size {
    unsafe fn new_unchecked(value: usize) -> Self { value }

    fn as_inner(&self) -> usize { *self }
}

pub trait MaybeUninitCompact<T> {
    unsafe fn assume_init_ref(&self) -> &[T];
    unsafe fn assume_init_mut(&mut self) -> &mut [T];
}

impl<T> MaybeUninitCompact<T> for [MaybeUninit<T>] {
    unsafe fn assume_init_ref(&self) -> &[T] { unsafe { &*(ptr::from_ref::<Self>(self) as *const [T]) } }

    unsafe fn assume_init_mut(&mut self) -> &mut [T] { unsafe { &mut *(std::ptr::from_mut::<Self>(self) as *mut [T]) } }
}

#[allow(unused)]
pub trait UsizeCompact {
    fn strict_mul(self, rhs: Self) -> Self;
    fn strict_sub_signed(self, rhs: isize) -> Self;
    fn strict_sub(self, rhs: Self) -> Self;
    fn overflowing_sub_signed(self, rhs: isize) -> (usize, bool);
}

impl UsizeCompact for usize {
    fn strict_mul(self, rhs: Self) -> Self {
        let (a, b) = self.overflowing_mul(rhs);
        if b { panic!("mul overflow") } else { a }
    }

    fn overflowing_sub_signed(self, rhs: isize) -> (Self, bool) {
        let (res, overflow) = self.overflowing_sub(rhs.cast_unsigned());

        (res, overflow ^ (rhs < 0))
    }

    fn strict_sub_signed(self, rhs: isize) -> Self {
        let (a, b) = self.overflowing_sub_signed(rhs);
        if b { panic!("sub signed overflow") } else { a }
    }

    fn strict_sub(self, rhs: Self) -> Self {
        let (a, b) = self.overflowing_sub(rhs);
        if b { panic!("sub overflow") } else { a }
    }
}
