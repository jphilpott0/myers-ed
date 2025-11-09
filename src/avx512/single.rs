#![cfg(feature = "avx512")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(cfg(feature = "avx512")))]

use core::arch::x86_64::*;

use anyhow::{Result, anyhow};

use crate::avx512::plumbing::*;
use crate::peq::{SingleWordPeq, Word};

/// Perform Myers algorithm to find the edit distance between `a` and `b`. Uses SIMD AVX-512 with 512-bit words.
/// Input bytes `a` must be `<= 512` bytes. Input bytes `b` can be any length.
///
/// # Examples
///
/// ```
/// # use myers_ed::avx512::single::myers_ed_single_avx512;
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

    myers_ed_single_avx512_with_peq(&peq, b)
}

pub fn try_myers_ed_single_avx512(a: &[u8], b: &[u8]) -> Result<usize> {
    if a.len() > 512 {
        return Err(anyhow!("Input must be be <= 512 bytes"));
    }

    // Infallible: we've verified a.len() <= 512.
    let peq = SingleWordPeq::from_bytes(a);

    Ok(myers_ed_single_avx512_with_peq(&peq, b))
}

pub fn myers_ed_single_avx512_with_peq(peq: &SingleWordPeq<__m512i>, b: &[u8]) -> usize {
    // Safety
    //
    // The `avx512f` and `avx512vpopcntdq `target_features` must be available.
    #[inline(always)]
    unsafe fn __inner_myers_ed_single_avx512_with_peq(
        peq: &SingleWordPeq<__m512i>,
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
