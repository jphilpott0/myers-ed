// Design for faster mainloop. Not compilable, just shorthand. WIP.

// Assumed preliminaries (from previous iterations or starting conditions):

__m512i peq[256];

char b[];
uint64_t b_len;

__m512i vp;
__m512i vn;

alignas(64) static const uint64_t top_arr[8] = {
    0x8000000080000000, 0x8000000080000000,
    0x8000000080000000, 0x8000000080000000,
    0x8000000080000000, 0x8000000080000000,
    0x8000000080000000, 0x8000000080000000
};

alignas(64) static const uint64_t one_shifted_arr[8] = {
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x8000000000000000
};

__m512i zero            = _mm512_setzero_si512();
__m512i topmask         = _mm512_load_epi64(top_arr);
__m512i one_shifted     = _mm512_load_epi64(one_shifted_arr);

// Standard Mainloop:
//
// Current Stats:
// - Lat: 9/14 + (Intel/AMD).
// - CPI: 18c.

// mid algorithm iteration. say 5+.
// we may assume that a previous iteration has been run.
uint64_t n; 

// Assume existing from previous iteration.
__m512i eq;

for (i = n; i < b_len; i++) {
    __m512i vpeq_and  = _mm512_and_si512(eq, vp); // hidden (1c; prev iter). eq & vp.
    __m512i xh        = _mm512_or_si512(eq, vn); // hidden (1c). eq | vn.
    __m512i vpeq_or   = _mm512_or_si512(eq, vp); // hidden (1c). eq | vp.
    __m512i vpeq_nand = _mm512_andnot_si512(eq, vp); // hidden (1c). ~eq & vp.

    vp           = _mm512_sub_epi32(vp, carry_partial); // 1c. vp + carry bits.
    __m512i eq_rshifted = _mm512_srli_epi32(eq, 1); // hidden (1c). eq >> 1.
    __m512i xh_rshifted = _mm512_srli_epi32(xh, 1); // hidden (1c). xh >> 1.
    __m512i _eq    = _mm512_load_si512(&peq[x]); // hidden (5c). get next peq.

    __m512i sum     = _mm512_add_epi32(vpeq_and, vp); // 1c.
    __m512i vpeqreq = _mm512_ternarylogic_epi32(vp, eq, eq_rshifted, 0x20); // hidden (1c). vp & ~eq & eq_rshifted.
    xh_rshifted     = _mm512_or_si512(xh_rshifted, topmask); // hidden (1c). xh_rshifted | 0^511 1.
    __m512i eq      = _eq; // rename for clarity only (0c).

    // todo: write exit condition, check, and misprediction branch.

    __m512i hp    = _mm512_ternarylogic_epi32(vn, vpeq_or, sum, 0xF1); // 1c. vn | ~(vpeq_or | sum).
    __m512i hn    = _mm512_ternarylogic_epi32(vpeq_nand, sum, vp, 0x90); // hidden (1c). vpeq_nand & ~(sum ^ vp).
    __m512i hnreq = _mm512_ternarylogic_epi32(vpeqreq, sum, vp, 0x90); // hidden (1c). vpeqreq & ~(sum ^ vp).
    __m512i n_xheq = _mm512_andnot_si512(xh_rshifted, eq_rshifted); // hidden (1c). ~xh_rshifted & eq_rshifted.

    __m512i hp_carry      = _mm512_alignr_epi32(one_shifted, hp, 1); // 3/5c. right-shift hp by 1 lane.
    __m512i hn_carry      = _mm512_alignr_epi32(zero, hn, 1); // hidden (3/5c). right-shift hn by 1 lane. 
    __m512i vp_partial    = _mm512_ternarylogic_epi32(hn, xh_rshifted, hp, 0xF1); // hidden (1c). hn | ~(xh_rshifted | hp).
    __m512i vpeq_partial  = _mm512_ternarylogic_epi32(hnreq, n_xheq, hp, 0xF4); // hidden (1c). hnreq | (n_xheq & ~hp).
    __m512i carry_partial = _mm512_add_epi32(vp_partial, vpeq_partial); // hidden (1c). near-optimally predict carry bits.
    carry_partial         = _mm512_srai_epi32(carry_partial, 31); // hidden (1c). dst[i+31:i] = src[i+31].

    carry_partial       = _mm512_alignr_epi32(zero, carry_partial, 1); // 3/5c. right-shift carry bits by 1 lane. 
    __m512i hp_lshifted = _mm512_shldi_epi32(hp, hp_carry, 1); // hidden (1/2c). merge overflowed bit in.
    __m512i hn_lshifted = _mm512_shldi_epi32(hn, hn_carry, 1); // hidden (1/2c). merge overflowed bit in.
    char x = b[i+1]; // hidden (1c). get next char.
    vp = _mm512_ternarylogic_epi32(hn_lshifted, xh, hp_lshifted, 0xF1); // hidden (1c). hn_lshifted | ~(xh | hp_lshifted).
    vn = _mm512_and_si512(hp, xh); // hidden (1c). hp & xh.
}
