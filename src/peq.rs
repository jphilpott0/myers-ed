use core::arch::x86_64::*;
use core::mem::size_of;
use core::ops::Index;

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

impl Word for u64 {
    const ZERO: u64 = 0;
    const ONE: u64 = 1;

    #[inline(always)]
    unsafe fn bit_at_unchecked(i: usize) -> u64 {
        1_u64 << i
    }

    #[inline(always)]
    fn bit_or(self, rhs: u64) -> u64 {
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
        // Mask for lane containing bit to set.
        let k = 1_u8 << (i >> 6);

        // Set bit inside selected lane.
        let a = 1_i64 << (i & 63);

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
    pub fn from_bytes<B: AsRef<[u8]>>(s: B) -> SingleWordPeq<T> {
        let mut peq = Self::default();
        peq.len = s.as_ref().len();

        assert!(
            peq.len <= 8 * size_of::<T>(),
            "Input byte array must be smaller than {} bytes",
            8 * size_of::<T>()
        );

        // Encode the position of each character in the relevant mask.
        for (i, &x) in s.as_ref().iter().enumerate() {
            // Infallible: `*x as usize` \in [0, 255] and `peq.peq.len() == 256`.
            // Safety: `i < a.len() = 8 * size_of::<T>()` as required by function.
            peq[x as usize] = unsafe { peq[x as usize].bit_or(T::bit_at_unchecked(i)) };
        }

        peq
    }

    pub fn from_bytes_and_alphabet<B: AsRef<[u8]>>(s: B, a: BitAlphabet) -> SingleWordPeq<T> {
        let mut peq = Self::default();

        // TODO: So probably need to just call a Word trait here. Because we want different
        // implementations depending on ISA. Scalar will just iterate over symbols, iterate
        // over string, set bit in peq. AVX-512 will iterate over symbol, spread chars into
        // lanes, check if char == symbol and return mask, then OR that mask directly into
        // the peq (frame adjusted ofc). AVX-512 can handle 64x8-bit lanes, so pretty stupid
        // fast peq construction here. The bigger string should always be put inside the peq.
        // SIMD construction is O(alphabet_size + alphabet_size * ceil(n / w)), which reduces
        // to O(alphabet_size * ceil(n / w)). For single-word, n < w, so this reduces further
        // to O(alphabet_size). A 512-bit DNA string can be handled in 4 iterations. Each
        // iteration contains a load, a maskz_cmp, and an OR. Previous implementation was O(n)
        // and did way more work per iter. This is probably now only a few ns.
        //
        // Could always add a `const`
        // version of BitAlphabet that knows its length at compile-time, so we can unroll
        // this loop exactly (if size is small). Can add the `const` using a `const` generic
        // special method in the builder.

        todo!()
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
pub(super) mod __private_index_mut {
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

const trait AssertTrue<const B: bool, const E: usize> {
    const ERR_MSGS: [&str; 2] = [
        // Error message 0.
        "No input string was provided. Add with `.with_string()`.",
        // Error message 1. Technically very difficult to cause this error. If no alphabet is
        // provided, then its automatically generated during Peq construction. Added just in case.
        "No alphabet has been generated. Add with `.with_alphabet()` or infer it during `.build_*()`.",
    ];

    const CHECK: () = {
        assert!(B, "{}", Self::ERR_MSGS[E]);
    };
}

impl<const B: bool, const T: usize> AssertTrue<B, T> for () {}

#[derive(Default)]
pub struct PeqBuilder<const STRING_CHECK: bool = false, const ALPHABET_CHECK: bool = false> {
    string: Option<Vec<u8>>,
    alphabet: Option<BitAlphabet>,
}

impl<const STRING_CHECK: bool, const ALPHABET_CHECK: bool>
    PeqBuilder<STRING_CHECK, ALPHABET_CHECK>
{
    pub fn new(
        string: Option<Vec<u8>>,
        alphabet: Option<BitAlphabet>,
    ) -> PeqBuilder<STRING_CHECK, ALPHABET_CHECK> {
        PeqBuilder::<STRING_CHECK, ALPHABET_CHECK> { string, alphabet }
    }

    #[inline(always)]
    pub fn with_string<S: Into<Vec<u8>>>(mut self, string: S) -> PeqBuilder<true, ALPHABET_CHECK> {
        self.string = Some(string.into());

        PeqBuilder::<true, ALPHABET_CHECK>::new(self.string, self.alphabet)
    }

    #[inline(always)]
    pub fn with_alphabet<S: AsRef<[u8]>>(mut self, alphabet: S) -> PeqBuilder<STRING_CHECK, true> {
        let mut a = [0_u64; 4];

        for c in alphabet.as_ref() {
            a[(c >> 6) as usize] = 1 << (c & 63);
        }

        self.alphabet = Some(BitAlphabet::new(a));

        PeqBuilder::<STRING_CHECK, true>::new(self.string, self.alphabet)
    }

    #[inline(always)]
    pub(crate) fn get_alphabet(&self) -> BitAlphabet
    where
        (): AssertTrue<STRING_CHECK, 0>,
    {
        let mut a = [0_u64; 4];

        // Safety: `AssertTrue` check guarantees string has been provided if this compiles.
        for c in unsafe { self.string.as_ref().unwrap_unchecked() } {
            a[(c >> 6) as usize] |= 1 << (c & 63);
        }

        BitAlphabet::new(a)
    }
}

pub struct BitAlphabet {
    bits: [u64; 4],
}

impl BitAlphabet {
    pub(crate) fn new(bits: [u64; 4]) -> Self {
        BitAlphabet { bits }
    }

    pub(crate) fn into_iter(&self) -> BitAlphabetIter {
        BitAlphabetIter::new(0, self.bits)
    }
}

pub struct BitAlphabetIter {
    block_idx: usize,
    bits: [u64; 4],
}

impl BitAlphabetIter {
    pub(crate) fn new(block_idx: usize, bits: [u64; 4]) -> Self {
        BitAlphabetIter { block_idx, bits }
    }
}

impl Iterator for BitAlphabetIter {
    type Item = u8;

    /// `tzcnt` loop to find all set bits and return index. Has `O(alphabet_size)` time-complexity,
    /// very fast considering typically `alphabet_size << 256`.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.block_idx < 4 {
            // Get 64-bit block of 256-bit bitvector.
            let block = self.bits[self.block_idx];

            // Check if we have set bits in the block.
            if block != 0 {
                // Find index of next LSB from block start.
                let lsb = block.trailing_zeros() as u8;

                // Remove this bit from block. Since this is LSB, `block - 1` cannot be
                // set at said LSB and all lower bits in `block` than LSB are 0 by definition.
                self.bits[self.block_idx] &= block - 1;

                let idx = (self.block_idx as u8) * 64 + lsb;
                return Some(idx);
            }

            // Goto next block.
            self.block_idx += 1;
        }

        None
    }
}
