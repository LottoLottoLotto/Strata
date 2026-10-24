// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu (no shipped
// model has IQ1_M expert rows: the 'Coder IQ1_M' pack's gate/up are IQ2_S / IQ3_XXS / IQ3_S).
#include "strata/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// E-2: the IQ3 grids by one AVX-512 gather (the same words, the same results) - measured 3-5% SLOWER than the scalar
// lookups on a Ryzen 5 7600 (Zen 4 gathers are microcoded), so opt-in: STRATA_IQ_GATHER=1
const bool gather = std::getenv("STRATA_IQ_GATHER") != nullptr;
// E-2: software prefetch distance in bytes (STRATA_IQ_PREFETCH; 0 = off): -2-3% gate/up time on IQ3_S decode
// (the rows are read at ~30 GB/s by 6 cores, near what DDR5 with 4 KB pages gives: the pool is memory-bound)
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // E-2: one gather of the 16 grid words (the scalar lookups assembled with set_epi32 cost ~30 uops)
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // E-2: the 9-bit indices (the high bit from qh under a mask), one gather
            __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q));
            idx = _mm512_mask_add_epi32(idx, (__mmask16) (h0 | (h1 << 8)), idx, _mm512_set1_epi32(256));
            g = _mm512_i32gather_epi32(idx, iq3s_grid, 4);
        } else {
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        }
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

// Wide groups (more than MAXT tokens of one expert, the TT port's prompt reads): a block's 4 chunks are decoded ONCE
// for every token (row_dot decodes them once per call of at most 8 tokens), then the tokens run 8 at a time against
// the decoded values, their float accumulators in memory.  Each token's operations and their order are row_dot's,
// so every result is bitwise row_dot<TY, NT>'s, for any NT.
template <int N>
inline void apply_block(const __m512i* g, const __mmask64* m, const __m512i* sc, const block_q8_K* const* y, int i,
                        float dx, __m512* accf) {
    const __m512i zero = _mm512_setzero_si512();
    __m512i acci[N];
    for (int t = 0; t < N; ++t) acci[t] = _mm512_setzero_si512();
    for (int j = 0; j < 4; ++j)
        for (int t = 0; t < N; ++t) {
            const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
            const __m512i ys = _mm512_mask_sub_epi8(yv, m[j], zero, yv);
            acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g[j], ys), sc[j]));
        }
    for (int t = 0; t < N; ++t)
        accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
}

template <int TY>
void row_dot_wide(const uint8_t* row, int nblocks, const block_q8_K* const* y, int nt, float* res) {
    __m512 accf[kIq512WideMax];
    for (int t = 0; t < nt; ++t) accf[t] = _mm512_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i g[4], sc[4];
        __mmask64 m[4];
        for (int j = 0; j < 4; ++j) Fmt<TY>::decode(blk, j, g[j], m[j], sc[j]);
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        int t0 = 0;
        for (; t0 + 8 <= nt; t0 += 8) apply_block<8>(g, m, sc, y + t0, i, dx, accf + t0);
        switch (nt - t0) {
            case 1: apply_block<1>(g, m, sc, y + t0, i, dx, accf + t0); break;
            case 2: apply_block<2>(g, m, sc, y + t0, i, dx, accf + t0); break;
            case 3: apply_block<3>(g, m, sc, y + t0, i, dx, accf + t0); break;
            case 4: apply_block<4>(g, m, sc, y + t0, i, dx, accf + t0); break;
            case 5: apply_block<5>(g, m, sc, y + t0, i, dx, accf + t0); break;
            case 6: apply_block<6>(g, m, sc, y + t0, i, dx, accf + t0); break;
            case 7: apply_block<7>(g, m, sc, y + t0, i, dx, accf + t0); break;
            default: break;
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY>
void gu_rows_wide(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                  float* const* ff, int r0, int r1) {
    const block_q8_K* y[kIq512WideMax];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[kIq512WideMax], u[kIq512WideMax];
    for (int r = r0; r < r1; ++r) {
        row_dot_wide<TY>(blob + (size_t) r * gu_row, nb, y, nt, g);
        row_dot_wide<TY>(blob + up_off + (size_t) r * gu_row, nb, y, nt, u);
        for (int t = 0; t < nt; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 8: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows_wide<TY>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;   // 9..kIq512WideMax
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// ---- IQ4_NL rows (type 20: 32-value blocks of 18 bytes, f16 d + 16 nibble bytes) against Q8_0 activations, two
// tokens per 512-bit register.  Each 256-bit half carries one token through iq_avx2's iq4nl_rows arithmetic lane for
// lane: the |w| x sign-applied q8 integer dot of every 4 bytes (vpdpbusd; maddubs cannot saturate there, so the ints
// are its), times dx * dy in the half's 8 float lanes by fmadd, and its hsum8 at the end - so every result is bitwise
// the AVX2 kernel's.  The sign: q8 negated where w < 0 (sign_epi8 also zeroes where w == 0, but |w| = 0 there).
inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template <int NP, bool ODD>
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out, int r0,
                int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m512i zero = _mm512_setzero_si512();
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m512 acc[NP > 0 ? NP : 1];
        for (int p = 0; p < NP; ++p) acc[p] = _mm512_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            if (prefetch_ahead > 0) _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);
            const __m512i aqz = _mm512_broadcast_i64x4(aq);
            const __mmask64 neg = _mm512_movepi8_mask(_mm512_broadcast_i64x4(q4));
            const float dx = h2f(u16(blk));
            for (int p = 0; p < NP; ++p) {
                const block_q8_0& a = y[2 * p][ib];
                const block_q8_0& b = y[2 * p + 1][ib];
                const __m512i q8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) a.qs)),
                                                      _mm256_loadu_si256((const __m256i*) b.qs), 1);
                const __m512i pz = _mm512_dpbusd_epi32(zero, aqz, _mm512_mask_sub_epi8(q8, neg, zero, q8));
                const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(dx * h2f(a.d)), _mm256_set1_ps(dx * h2f(b.d)), 1);
                acc[p] = _mm512_fmadd_ps(sc, _mm512_cvtepi32_ps(pz), acc[p]);
            }
            if constexpr (ODD) {
                const block_q8_0& a = y[2 * NP][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) a.qs);
                const __m256i p1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aq, _mm256_sign_epi8(q8, q4));
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(a.d)), _mm256_cvtepi32_ps(p1), acc1);
            }
        }
        for (int p = 0; p < NP; ++p) {
            out[2 * p][r] = hsum8(_mm512_castps512_ps256(acc[p]));
            out[2 * p + 1][r] = hsum8(_mm512_extractf32x8_ps(acc[p], 1));
        }
        if constexpr (ODD) out[2 * NP][r] = hsum8(acc1);
    }
}

}  // namespace

bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: iq4nl_rows<0, true>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows<1, false>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows<1, true>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows<2, false>(w, row_bytes, n, y, out, r0, r1); break;
        case 5: iq4nl_rows<2, true>(w, row_bytes, n, y, out, r0, r1); break;
        case 6: iq4nl_rows<3, false>(w, row_bytes, n, y, out, r0, r1); break;
        case 7: iq4nl_rows<3, true>(w, row_bytes, n, y, out, r0, r1); break;
        default: iq4nl_rows<4, false>(w, row_bytes, n, y, out, r0, r1); break;   // 8 (callers pass at most MAXT)
    }
}

}  // namespace strata::kernels::cpu
