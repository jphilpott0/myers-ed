#![cfg(feature = "avx512")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(cfg(feature = "avx512")))]

use core::arch::x86_64::*;
use core::ops::BitAnd;

use anyhow::{Result, anyhow};

use crate::assert_target_features;
use crate::peq::{SingleWordPeq, Word};

// Safety: the safe functions in this module are always safe because we guarantee
// that avx512f and avx512vpopcntdq are available if the avx512 crate feature compiles.
assert_target_features! { "avx512", "avx512f", "avx512vpopcntdq" }

/// Perform Myers algorithm to find the edit distance between `a` and `b`. Uses SIMD AVX-512 with 512-bit words.
/// Input bytes `a` must be `<= 512` bytes. Input bytes `b` can be any length.
///
/// # Examples
///
/// ```
/// # use myers_ed::avx512::myers_ed_single_avx512;
/// # fn main() {
/// let d = myers_ed_single_avx512(b"ACCCT", b"ACCTT");
///
/// assert_eq!(d, 1);
/// # }
/// ```
pub fn myers_ed_single_avx512(a: &[u8], b: &[u8]) -> usize {
    assert!(a.len() <= 512, "Input must be <= 512 bytes");

    // Infallible: we've verified a.len() <= 512.
    let peq = SingleWordPeq::from_bytes(a);

    myers_ed_single_avx512_with_peq(peq, b)
}

pub fn try_myers_ed_single_avx512(a: &[u8], b: &[u8]) -> Result<usize> {
    if a.len() > 512 {
        return Err(anyhow!("Input must be be <= 512 bytes"));
    }

    // Infallible: we've verified a.len() <= 512.
    let peq = SingleWordPeq::from_bytes(a);

    Ok(myers_ed_single_avx512_with_peq(peq, b))
}

pub fn myers_ed_single_avx512_with_peq(peq: SingleWordPeq<__m512i>, b: &[u8]) -> usize {
    // Safety
    //
    // The `avx512f` and `avx512vpopcntdq `target_features` must be available.
    #[inline(always)]
    unsafe fn __inner_myers_ed_single_avx512_with_peq(
        peq: SingleWordPeq<__m512i>,
        b: &[u8],
    ) -> usize {
        // Vertical positive delta bit-vector.
        let mut vp = _mm512_set1_epi64(-1_i64);

        // Vertical negative delta bit-vector.
        let mut vn = _mm512_setzero_si512();

        // Update loop.
        for &x in b {
            // Get the equality mask for the current character.
            //
            // Infallible: `x as usize` \in [0, 255] and `peq.peq.len() == 256`.
            let eq = peq[x as usize];

            // Calculate diagonal zero delta bit-vector. This is d0 = (((eq & vp) + vp) ^ vp) | eq.
            let d0 = _mm512_ternarylogic_epi64(
                _mm512_add_si512_custom::<1>(_mm512_and_si512(eq, vp), vp),
                vp,
                eq,
                0xBE,
            );

            // Calculate horizontal positive delta bit-vector. This is hp = vn | !(vp | d0).
            let mut hp = _mm512_ternarylogic_epi64(vn, vp, d0, 0xF1);

            // Calculate horizontal negative delta bit-vector.
            let hn = _mm512_and_si512(vp, d0);

            // Calculate intermediate mask for next column's vertical delta bits.
            let xh = _mm512_or_si512(eq, vn);

            // Move one column right in DP matrix. Uses custom `Word` trait for `ONE` const.
            // This is hp = (hp << 1_u32) | 1_m512i.
            hp = _mm512_or_si512(_mm512_slli_si512_custom::<1>(hp), __m512i::ONE);

            // Update positive vertical delta bit-vector. This is (hn << 1_u32) | !(xh | hp).
            vp = _mm512_ternarylogic_epi64(_mm512_slli_si512_custom::<1>(hn), xh, hp, 0xF1);

            // Update negative vertical delta bit-vector.
            vn = _mm512_and_si512(hp, xh);
        }

        // Compute mask to get only real bits. This is dst[l:0] = 1 and dst[511:l+1] = 0,
        // for `l = peq.len()`.
        //
        // Safety: `peq.len()` must be `<=512`, which is guaranteed by `SingleWordPeq`.
        let m = unsafe { _mm512_mask_upto_si512_custom(peq.len()) };

        // Compute final edit distance.
        let vp_popcnt = _mm512_popcnt_si512_custom(_mm512_and_si512(vp, m)) as usize;
        let vn_popcnt = _mm512_popcnt_si512_custom(_mm512_and_si512(vn, m)) as usize;

        b.len() + vp_popcnt - vn_popcnt
    }

    // Safety: we guarantee that avx512f and avx512vpopcntdq are present when the avx512 crate feature compiles.
    unsafe { __inner_myers_ed_single_avx512_with_peq(peq, b) }
}

/// Custom AVX-512 helper "intrinsics" for supporting the Myers algorithm implementation.
pub mod plumbing {
    use super::*;

    /// Add two `_m512i` as if they were both one-lane 512-bit integers. Accepts a
    /// compile-time hint about the number of expected carries between lanes there
    /// might be (`LIKELY_CARRY_ROUNDS`). The operation works with 64-bit lanes, meaning
    /// if any lane of the sum of `a` and `b` overflows, we must carry a bit into the next
    /// lane. We do this in a number of carry rounds. If you expect that carrying between
    /// lanes is unlikely, for instance when one of `a` or `b` is very small, then provide
    /// a hint of `LIKELY_CARRY_ROUNDS = 0`. Otherwise, or if you are unsure, provide
    /// `LIKELY_CARRY_ROUNDS = 1`. It is extremely unlikely that you would ever expect more
    /// than `1` carry round. Providing this hint removes the need for some checks and
    /// slightly improves performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::arch::x86_64::*;
    /// # use myers_ed::avx512::plumbing::_mm512_add_si512_custom;
    /// # fn main() {
    /// let a: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([1, 0, 0, 0, 0, 0, 0, 1]) };
    /// let b: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([1, 0, 0, 0, 0, 0, 0, -1]) };
    ///
    /// let s1: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([2, 0, 0, 0, 0, 0, 1, 0]) };
    /// let s2: __m512i = _mm512_add_si512_custom::<1>(a, b);
    /// # // __m512i doesn't implement PartialEq, so quietly transmute back to arrays.
    /// # let s1: [i64; 8] = unsafe { core::mem::transmute::<__m512i, _>(s1)};
    /// # let s2: [i64; 8] = unsafe { core::mem::transmute::<__m512i, _>(s2)};
    ///
    /// assert_eq!(s1, s2);
    /// # }
    /// ```
    #[inline(always)]
    pub fn _mm512_add_si512_custom<const LIKELY_CARRY_ROUNDS: u32>(
        a: __m512i,
        b: __m512i,
    ) -> __m512i {
        // Safety
        //
        // The `avx512f` `target_feature` must be available.
        #[inline(always)]
        unsafe fn __inner_mm512_add_si512_custom<const LIKELY_CARRY_ROUNDS: u32>(
            mut a: __m512i,
            b: __m512i,
        ) -> __m512i {
            // Add a and b together as 64-bit lanes without carry between.
            let mut s = _mm512_add_epi64(a, b);

            // Initalise carry mask. Since 0..=LIKELY_CARRY_ROUNDS must always run once, we
            // will always populate the carry mask with some data.
            #[allow(unused_assignments)]
            let mut cm = 0;

            // Perform `LIKELY_CARRY_ROUNDS` of unchecked non-branching carry rounds. When
            // `LIKELY_CARRY_ROUNDS` is small, llvm can unroll this loop since its a `const`
            // generic and known at compile-time. If `LIKELY_CARRY_ROUNDS = 0`, then llvm
            // will exclude the loop entirely.
            for _ in 0..LIKELY_CARRY_ROUNDS {
                // Mask of carry bits. If s < a, then we overflowed and need a carry bit.
                cm = _mm512_cmp_epu64_mask(s, a, _MM_CMPINT_LT);

                // Broadcast carry bits across lanes. Right shift mask to propagate bits up along lanes.
                //
                // Safety: `rhs = 1 < 8 * std::mem::size_of::<u8>()`.
                let cb = unsafe { _mm512_maskz_set1_epi64(cm.unchecked_shr(1), 1_i64) };

                // Save current pre-carry lanes.
                a = s;

                // Add carry bits into s.
                s = _mm512_add_epi64(s, cb);
            }

            // Loop indefinitely doing checked carry rounds until we have no more carries to do.
            // If `LIKELY_CARRY_ROUNDS` was a good hint, then we hope to return immediately.
            loop {
                // Check if we had any more overflows and need to do more carry logic.
                cm = _mm512_cmp_epu64_mask(s, a, _MM_CMPINT_LT);

                // It is extremely unlikely, but not impossible, that we need another carry round.
                // Assume that s = a + b overflows ==> s = a + b - 2^64. Now, s + 1 = a + b - 2^64 + 1.
                // Overflow occurs when x + y >= 2^64 , so s + 1 overflows when a + b - 2^64 + 1 >= 2^64
                // ==> a + b + 1 >= 2^65. Since max_{a,b}(a + b) = 2 * (2^64 - 1) = 2^65 - 2, it follows that
                // max_{a,b}(a + b) + 1 = 2^65 - 1 < 2^65 (the value required to overflow), therefore if a + b
                // overflows once, it cannot overflow again when adding the carry bit. So the only way to
                // cause a second carry round is to *not* have an overflow in a + b, but that a + b =
                // 0xFFFFFFFFFFFFFFFF exactly AND for a + b in the previous lane to have overflowed AND
                // for this to not be occuring in the highest-lane (because an overflow there is discarded).
                // The likelihood for a + b to be exactly 0xFFFFFFFFFFFFFFFF is nearly impossible, except
                // in the starting iteration of the algorithm where VP=0xFFFFFFFFFFFFFFFF, in which case
                // EQ must be exactly 0x0000000000000000 in that lane AND the previous lane must overflow,
                // in which case EQ in that lane must be non-zero. After the first few iterations, the bits
                // of VP become essentially random and the likelihood that a + b is exactly 0xFFFFFFFFFFFFFFFF
                // is almost zero (1 in 2^64 chance). Nevertheless, for the edge-case of the first iteration,
                // and the insanely low chance thereafter, we need to handle further carry rounds for
                // correctness. However, since its so rare, we hint the branch predictor that it's unlikely
                // to happen. It probably will already detect this dynamically, but this is a nice compile-time
                // micro-optimisation that saves the branch predictor some time.
                if core::hint::likely(cm == 0) {
                    return s;
                }

                // If the `LIKELY_CARRY_ROUNDS` hint was bad, and we still have carries to
                // propagate, then continue propagating.

                // Broadcast carry bits across lanes. Right shift mask to propagate bits up along lanes.
                //
                // Safety: `rhs = 1 < 8 * std::mem::size_of::<u8>()`.
                let cb = unsafe { _mm512_maskz_set1_epi64(cm.unchecked_shr(1), 1_i64) };

                // Save current pre-carry lanes.
                a = s;

                // Add carry bits into s.
                s = _mm512_add_epi64(s, cb);
            }
        }

        // Safety: We guarantee that the avx512f target_feature is available when the avx512 crate feature compiles.
        unsafe { __inner_mm512_add_si512_custom::<LIKELY_CARRY_ROUNDS>(a, b) }
    }

    // Trick to evaluate a `const` context that doesn't cause a well-formedness check cycle.
    // This way the `const` context can be evaluated separately from a well-formedness proof.
    trait ConstExpr<const X: u32> {}
    impl<const X: u32> ConstExpr<X> for () {}

    /// Left shift all bits within a 512-bit `__m512i` type by `IMM8` bits, while shifting in zeros.
    /// `IMM8` is a `const` generic immediate and must be less than or equal to `64`. `IMM8` must be
    /// passed as a `u32` (matching Rust's intrinsic ABI), but also must have a valid representation
    /// as a `u8`. The `u8` representation is what is actually used through some downstream compiler magic.
    /// If your `u32` is less than or equal to `64`, as also required, then it also has a valid `u8` representation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::arch::x86_64::*;
    /// # use myers_ed::avx512::plumbing::_mm512_slli_si512_custom;
    /// # fn main() {
    /// let a: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([0, 0, 0, 0, 0, 0, 0, 1]) };
    /// let b: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([0, 0, 0, 0, 0, 0, 0, 8]) };
    ///
    /// let s: __m512i = _mm512_slli_si512_custom::<3>(a);
    /// # // __m512i doesn't implement PartialEq, so quietly transmute back to arrays.
    /// # let b: [i64; 8] = unsafe { core::mem::transmute::<__m512i, _>(b)};
    /// # let s: [i64; 8] = unsafe { core::mem::transmute::<__m512i, _>(s)};
    ///
    /// assert_eq!(b, s);
    /// # }
    /// ```
    #[inline(always)]
    #[allow(private_bounds)]
    pub fn _mm512_slli_si512_custom<const IMM8: u32>(a: __m512i) -> __m512i
    where
        // Constrain IMM8 to be <= 64. We need this `ConstExpr` trick because the standard
        // `[(); { 64 - IMM8 } as usize]` method causes cyclic well-formedness checks.
        (): ConstExpr<{ 64 - IMM8 }>,
    {
        // Loads of funky type logic here. A really specific and tricky setup here
        // is required to satisfy the type checker and trait solver, while allowing
        // us to calculate `64 - IMM8` at compile-time so we can use immediate sll and slr.
        // You can't use inlined `const` expressions that reference a `const` generic
        // from an outer item (in this case the outer `fn` item). And `fn` items cannot
        // have default `const` generic values. This slightly convoluted way below actually
        // has a monomorphic trait solution (no unconstrained generics), with no cycles,
        // no well-formedness check overflows, is actually able to use the original `const`
        // generic from the outer `fn` item, and doesn't cause an ICE (had a system that
        // passed type checking but failed at MIR generation??).
        struct Complementor;

        trait ComplementShift<const N: u32, const Q: u32> {
            // Perform the actual left shift calculation now that we are in a scope with both `N`
            // and `Q` as available `const` generics. This lets us use immediate variants of sll
            // and srl, which are faster than the non-immediate ones (3 less cycles of latency).
            //
            // Safety
            //
            // `N` must be less than or equal to 64. The `avx512f` `target_feature` must be available.
            #[inline(always)]
            unsafe fn __inner_mm512_slli_si512_custom(a: __m512i) -> __m512i {
                // Left shift lanes in a by N without carrying between lanes.
                let s = _mm512_slli_epi64(a, N);

                // Overflow bits. We right shift by Q = 64 - N to get all the bits that overflowed.
                let o = _mm512_srli_epi64(a, Q);

                // Shift all the overflowed bits along by 64-bits (to be in line with next lane).
                let m = _mm512_alignr_epi64(__m512i::ZERO, o, 1);

                // Fill shifted in zero bits with overflowed bits from previous lane. Since we're
                // adding into zero bits, OR and ADD are the same, and we use the logical op ports
                // so much that the ports ADD uses will be slightly less pressured.
                _mm512_add_epi64(s, m)
            }
        }

        impl<const N: u32, const Q: u32> ComplementShift<N, Q> for Complementor {}

        // Safety: outer function signature guarantees that `N < 64`. We also guarantee that
        // the avx512f target_feature is available when the avx512 crate feature compiles.
        unsafe {
            <Complementor as ComplementShift<IMM8, { 64 - IMM8 }>>::__inner_mm512_slli_si512_custom(
                a,
            )
        }
    }

    /// Return a bitmask of set bits up to bit `i`. I.e. `dst[511:(i+1)] = 0` and `dst[i:0] = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::arch::x86_64::*;
    /// # use myers_ed::avx512::plumbing::_mm512_mask_upto_si512_custom;
    /// # fn main() {
    /// // Note that memory-layout is little-endian.
    /// let m1: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([255, 0, 0, 0, 0, 0, 0, 0]) };
    ///
    /// // Safety: 8 < 512.
    /// let m2: __m512i = unsafe { _mm512_mask_upto_si512_custom(8) };
    /// # // __m512i doesn't implement PartialEq, so quietly transmute back to arrays.
    /// # let m1: [i64; 8] = unsafe { core::mem::transmute::<__m512i, _>(m1)};
    /// # let m2: [i64; 8] = unsafe { core::mem::transmute::<__m512i, _>(m2)};
    ///
    /// assert_eq!(m1, m2);
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// `i` must not be greater than 512. Otherwise, this is UB.
    #[inline(always)]
    pub unsafe fn _mm512_mask_upto_si512_custom(i: usize) -> __m512i {
        // Lane containing intended MSB. 1_u8 << (i / 64).
        let lane = 1_u8.unchecked_shl(i.unchecked_shr(6) as u32);

        // Mask of lanes before lane that contains intended MSB. max(1_u8 << (i / 64) - 1_u8, 0).
        let low_lanes = lane.saturating_sub(1);

        // Start with dst[511:0] = 0 and set dst[((i / 64) * 64) - 1:0] = 1.
        let m = _mm512_mask_set1_epi64(__m512i::ZERO, low_lanes, -1_i64);

        // Get intended MSB + 1, subtract 1 to fill intended MSB:LSB with set bits. 1_i64 << (i % 64) - 1.
        let bits = 1_i64.unchecked_shl(i.bitand(63) as u32).unchecked_sub(1);

        // Set dst[i:((i / 64) * 64)] = 1.
        _mm512_mask_set1_epi64(m, lane, bits)
    }

    /// Find number of set bits in 512-bit `__m512i`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::arch::x86_64::*;
    /// # use myers_ed::avx512::plumbing::_mm512_popcnt_si512_custom;
    /// # fn main() {
    /// let a: __m512i = unsafe { core::mem::transmute::<[i64; 8], _>([1, 0, 0, 0, 0, 0, 0, 15]) };
    ///
    /// assert_eq!(5, _mm512_popcnt_si512_custom(a));
    /// # }
    /// ```
    #[inline(always)]
    pub fn _mm512_popcnt_si512_custom(a: __m512i) -> i64 {
        // Safety
        //
        // The `avx512f` and `avx512vpopcntdq` `target_features` must be available.
        #[inline(always)]
        unsafe fn __inner_mm512_popcnt_si512_custom(a: __m512i) -> i64 {
            _mm512_reduce_add_epi64(_mm512_popcnt_epi64(a))
        }

        // Safety: we guarantee that avx512f and avx512vpopcntdq are present when the avx512 crate feature compiles.
        unsafe { __inner_mm512_popcnt_si512_custom(a) }
    }
}

pub(crate) use plumbing::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let d = myers_ed_single_avx512(b"AATTC", b"AATTCA");

        assert_eq!(d, 1);
    }
}
