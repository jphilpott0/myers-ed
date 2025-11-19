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
// - Lat: 9/12 + (Intel/AMD).
// - CPI: 18c.

// mid algorithm iteration. say 5+.
// we may assume that a previous iteration has been run.
uint64_t n; 

// Assume existing from previous iteration.
__m512i eq;

for (i = n; i < b_len; i++) {
    __m512i vpeq = _mm512_and_si512(eq, vp); // 1c.
    __m512i xh   = _mm512_or_si512(eq, vn); // hidden by above (1c).
    vp           = _mm512_sub_epi32(vp, carry_partial); // hidden by above (1c).

    __m512i sum         = _mm512_add_epi32(vpeq, vp); // 1c (!!).
    __m512i eq_rshifted = _mm512_srli_epi32(eq, 1); // hidden by above (1c).
    __m512i xh_rshifted = _mm512_srli_epi32(xh, 1); // hidden by above (1c).
    xh_rshifted         = _mm512_or_si512(xh_rshifted, topmask); // hidden by above (1c).

    // todo: write exit condition, check, and misprediction branch.

    __m512i d0     = _mm512_ternarylogic_epi32(sum, vp, eq, 0xBE); // 1c.
    __m512i n_xheq = _mm512_andnot_si512(xh_rshifted, eq_rshifted); // hidden by above (1c).
    char x = b[i+1]; // hidden by above (1c).

    __m512i hp   = _mm512_ternarylogic_epi32(vn, vp, d0, 0xF1); // 1c.
    __m512i hn   = _mm512_and_si512(vp, d0); // hidden by above (1c).
    __m512i hneq = _mm512_ternarylogic_epi32(vp, d0, eq_rshifted, 0x80); // hidden by above (1c).
    __m512i eq   = _mm512_load_si512(&peq[x]); // hidden by above and below (5c).

    __m512i hp_carry      = _mm512_alignr_epi32(one_shifted, hp, 1); // 3/5c.
    __m512i hn_carry      = _mm512_alignr_epi32(zero, hn, 1); // hidden by above (3/5c).
    __m512i vp_partial    = _mm512_ternarylogic_epi32(hn, xh_rshifted, hp, 0xF1); // hidden by above (1c).
    __m512i vpeq_partial  = _mm512_ternarylogic_epi32(hneq, n_xheq, hp, 0xF4); // hidden by above (1c).
    __m512i carry_partial = _mm512_add_epi32(vp_partial, vpeq_partial); // hidden by above (1c).
    carry_partial       = _mm512_srai_epi32(carry_partial, 31); // hidden by above (1c).

    __m512i hp_lshifted = _mm512_shldi_epi32(hp, hp_carry, 1); // 1/2c.
    __m512i hn_lshifted = _mm512_shldi_epi32(hn, hn_carry, 1); // hidden by above (1/2c).

    vp = _mm512_ternarylogic_epi32(hn_lshifted, xh, hp_lshifted, 0xF1); // 1c.
    vn = _mm512_and_si512(hp, xh); // hidden by above (1c).
}
