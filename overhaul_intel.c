// Design for faster mainloop. Not compilable, just shorthand. WIP.

// Assumed preliminaries (from previous iterations or starting conditions):

__m512i peq[256];

char b[];
uint64_t b_len;

__m512i vp;
__m512i vn;

alignas(64) static const uint64_t top_arr[8] = {
    0x0000000100000001, 0x0000000100000001,
    0x0000000100000001, 0x0000000100000001,
    0x0000000100000001, 0x0000000100000001,
    0x0000000100000001, 0x0000000100000001
};

alignas(64) static const uint64_t one_msb_arr[8] = {
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x8000000000000000
};

__m512i zero            = _mm512_setzero_si512();
__m512i topmask         = _mm512_load_epi64(top_arr);
__m512i one_msb         = _mm512_load_epi64(one_msb_arr);

// Standard Mainloop (Intel - Alder Lake-P):
//
// Current Stats:
// - Lat: 11c.
// - Total: 25 uops.

// mid algorithm iteration. we may assume that a previous iteration has been
// run and that these variables are initialised.
uint64_t n; 
unsigned char failed;

// Assume existing from previous iteration.
__m512i eq;

for (i = n; i < b_len; i++) {
    __m512i vpeq    = _mm512_and_si512(eq, vp); // 1c. eq & vp. p05 => p0.
    __m512i _n_vpeq = _mm512_andnot_si512(vp, eq); // 1c. eq & ~vp. p05 => p5.

    vp                  = _mm512_sub_epi32(vp, carry_partial); // 1c. vp + carry bits. p05 => p5.
    __m512i xh_rshifted = _mm512_shrdi_epi32(xh, topmask, 1); // 1c. xh >> 1 | 01. p0 => p0.

    __m512i sum     = _mm512_add_epi32(vpeq, vp); // 1c. vpeq + vp. p05 => p5.
    __m512i vpeq_or = _mm512_or_si512(eq, vp); // 1c. vp | eq. p05 => p0.
    if (failed != 0) { // 1c. check if carry predictor failed. p6 => p6. fused test + jcc.
        // todo: write mispredict branch.
    }

    __mmask16 carry_check = _mm512_test_epi32_mask(n_vpeq, zero); // 3c. test if eq & ~vp = 0. p5 => p5.
    __m512i hp            = _mm512_ternarylogic_epi32(vpeq_or, xh, sum, 0x0D); // 1c. ~vpeq_or & (xh | ~sum). p05 => p0. todo: recheck this.
    char x                = b[i+1]; // 1c. get next char. p49 + p78 => p49 + p78.

    __m512i hp_carry = _mm512_alignr_epi32(one_msb, hp, 1); // 3c. right-shift hp by 1 lane. p5 => p5.
    __m512i hn       = _mm512_ternarylogic_epi32(vp, sum, eq, 0xB0); // 1c. vp & (~sum | eq). p05 => p0.
    __m512i _eq      = _mm512_load_si512(&peq[x]); // 5c. get next peq. p23A => p23A.

    __m512i hn_carry      = _mm512_alignr_epi32(zero, hn, 1); // 3c. right-shift hn by 1 lane. p5 => p5.
    __m512i eq_rshifted   = _mm512_srli_epi32(eq, 1); // 1c. eq >> 1. p0 => p0.

    __m512i vp_partial    = _mm512_ternarylogic_epi32(hn, xh_rshifted, hp, 0xF1); // 1c. hn | ~(xh_rshifted | hp). p05 => p5.
    __m512i vpeq_partial  = _mm512_and_si512(vp, eq_rshifted); // 1c. vp & eq_rshifted. p05 => p0.

    __m512i carry_partial = _mm512_add_epi32(vp_partial, vpeq_partial); // 1c. near-optimally predict carry bits. p05 => p5.
    __m512i hp_lshifted   = _mm512_shldi_epi32(hp, hp_carry, 1); // 1c. merge overflowed bit in. p0 => p0.

    carry_partial         = _mm512_srai_epi32(carry_partial, 31); // 1c. dst[i+31:i] = src[i+31]. p0 => p0.
    __m512i xh            = _mm512_ternarylogic_epi32(eq, hp_lshifted, xh, 0xF8); // 1c. eq | (hp & xh). p05 => p5.

    carry_partial         = _mm512_alignr_epi32(zero, carry_partial, 1); // 3c. right-shift carry bits by 1 lane. p5 => p5.
    __m512i hn_lshifted   = _mm512_shldi_epi32(hn, hn_carry, 1); // 1c. merge overflowed bit in. p0 => p0.

    unsigned char failed  = _cvtmask16_u32(carry_check); // 3c. kmovw __mmask16 to char. p0 => p0.
    vp                    = _mm512_ternarylogic_epi32(hn_lshifted, xh, hp_lshifted, 0xF1); // 1c. hn_lshifted | ~(xh | hp_lshifted). p05 => p5.
    eq                    = _eq; // 0c. variable rename only.
    n_vpeq                = _n_vpeq; // 0c. variable rename only.
}
