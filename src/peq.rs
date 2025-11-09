use core::arch::x86_64::*;
use core::mem::size_of;
use core::ops::{BitAnd, Index, Sub};

pub trait Word {
    /// This word entirely zeroed.
    const ZERO: Self;

    /// This word's LSB set to 1 and the rest zeroed.
    const ONE: Self;

    /// Compute a bitmask such that bit `i` is set to 1 and the rest of the word is zeroed.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that `i < 8 * std::mem::size_of::<Self>()`.
    unsafe fn bit_at_unchecked(i: usize) -> Self;

    /// Compute the bitwise OR operation between `self` and `rhs`. Equivalent to
    /// [`BitOr`](`std::ops::BitOr`) but has custom implementations in this crate for SIMD types.
    fn bit_or(self, rhs: Self) -> Self;
}

impl Word for usize {
    const ZERO: usize = 0;
    const ONE: usize = 1;

    #[inline(always)]
    unsafe fn bit_at_unchecked(i: usize) -> usize {
        1_usize.unchecked_shl(i as u32)
    }

    #[inline(always)]
    fn bit_or(self, rhs: usize) -> usize {
        self | rhs
    }
}

#[cfg(feature = "avx512")]
impl Word for __m512i {
    // Safety: `__m512i` is POD so this value is valid.
    const ZERO: __m512i = unsafe { core::mem::zeroed() };

    // Safety: `__m512i` has an exact underlying representation of `[i64; 8]`.
    const ONE: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([1, 0, 0, 0, 0, 0, 0, 0]) };

    #[inline(always)]
    unsafe fn bit_at_unchecked(i: usize) -> __m512i {
        // Mask for lane containing bit to set. 1_u8 << (i / usize::BITS).
        let k = 1_u8.unchecked_shl(i.unchecked_shr(usize::BITS.trailing_zeros()) as u32);

        // Set bit inside selected lane. 1_i64 << (i % usize::BITS).
        let a = 1_i64.unchecked_shl(i.bitand(usize::BITS.sub(1) as usize) as u32);

        _mm512_mask_set1_epi64(Self::ZERO, k, a)
    }

    #[inline(always)]
    fn bit_or(self, rhs: __m512i) -> __m512i {
        // Safety: we guarantee that avx512f is present if avx512 crate feature compiles.
        unsafe { _mm512_or_si512(self, rhs) }
    }
}

#[repr(align(64))]
pub struct SingleWordPeq<T> {
    peq: [T; 256],
    len: usize,
}

impl<T: Word + Copy> SingleWordPeq<T> {
    pub fn new() -> SingleWordPeq<T> {
        Self::default()
    }

    pub fn from_bytes(a: &[u8]) -> SingleWordPeq<T> {
        let mut peq = Self::new();

        assert!(
            a.len() <= 8 * size_of::<T>(),
            "Input byte array must be smaller than {} bytes",
            8 * size_of::<T>()
        );

        peq.len = a.len();

        // Encode the position of each character in the relevant mask.
        //
        // TODO: Could re-work this to take an `alphabet` and then search for those
        // symbols in the input. This should be faster because we can search for those
        // symbols across SIMD lanes.
        for (i, &x) in a.iter().enumerate() {
            // Infallible: `*x as usize` \in [0, 255] and `peq.peq.len() == 256`.
            // Safety: `i < a.len() = 8 * size_of::<T>()` as required by function.
            peq[x as usize] = unsafe { peq[x as usize].bit_or(T::bit_at_unchecked(i)) };
        }

        peq
    }

    pub fn from_string<S: AsRef<str>>(s: S) -> SingleWordPeq<T> {
        assert!(
            // str::len() returns number of bytes, not necessarily number of elements.
            // This is helpful because we actually do want the number of bytes here.
            s.as_ref().len() <= 8 * size_of::<T>(),
            "Input string must be smaller than {} bytes",
            8 * size_of::<T>()
        );

        assert!(
            s.as_ref().is_ascii(),
            "Input string must only contain ASCII characters"
        );

        Self::from_bytes(s.as_ref().as_bytes())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Index<usize> for SingleWordPeq<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.peq[idx]
    }
}

/// Crate private implementation of IndexMut because we don't want to leak
/// access to inner slice to users to guarantee correctness of the .len value.
pub(crate) mod __private_index_mut {
    use super::*;
    use core::ops::IndexMut;

    impl<T> IndexMut<usize> for SingleWordPeq<T> {
        fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
            &mut self.peq[idx]
        }
    }
}

impl<T: Word> Default for SingleWordPeq<T> {
    fn default() -> Self {
        SingleWordPeq {
            peq: [T::ZERO; 256],
            len: 0,
        }
    }
}
