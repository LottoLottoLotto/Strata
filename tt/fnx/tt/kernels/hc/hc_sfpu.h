// HC fused kernels: custom fp32 SFPU passes over the DST tiles of the current section (tile k = dst_reg[32 * k]).
// Every pass works on whole tiles (4 faces x 8 rows of 32 lanes).
#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// fp32 -> nearest-even bfloat16, returned as an fp32 value with the low 16 bits clear (exact in a bf16 tile).
sfpi_inline sfpi::vFloat hc_rne_bf16(sfpi::vFloat v) {
    sfpi::vUInt u = sfpi::as<sfpi::vUInt>(v);
    sfpi::vUInt lsb = (u >> 16u) & 1u;
    u = u + lsb;
    u = u + (int32_t)0x7fff;
    u = u & 0xffff0000u;
    return sfpi::as<sfpi::vFloat>(u);
}

// d0 = R, d1 = gamma*sqrt(N)  ->  d0 = y = R*gs, d1 = bf16(y), d2 = bf16(y - d1), d3 = bf16(R^2), d4 = bf16(R^2 - d3)
// (needs a full-sync DST section: 5 fp32 tiles)
inline void hc_prep_fn() {
#pragma GCC unroll 2
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat r = sfpi::dst_reg[0];
        sfpi::vFloat g = sfpi::dst_reg[32];
        sfpi::vFloat y = r * g;
        sfpi::vFloat hi = hc_rne_bf16(y);
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg[32] = hi;
        sfpi::dst_reg[64] = hc_rne_bf16(y - hi);
        sfpi::vFloat x2 = r * r;
        sfpi::vFloat x2hi = hc_rne_bf16(x2);
        sfpi::dst_reg[96] = x2hi;
        sfpi::dst_reg[128] = hc_rne_bf16(x2 - x2hi);
        sfpi::dst_reg++;
    }
}

// d0 = S_a, d1 = S_b (per-row sums of squares, broadcast over columns)  ->  d0 = rsqrt(S_a + S_b + eps_bits)
inline void hc_rs_fn(uint32_t eps_bits) {
    const sfpi::vFloat eps = Converter::as_float(eps_bits);
#pragma GCC unroll 2
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[32];
        sfpi::vFloat s = a + b;
        sfpi::dst_reg[0] = _calculate_sqrt_body_<false, true, false>(s + eps);
        sfpi::dst_reg++;
    }
}

// d0..d7 = P_0..P_7 (slice partials)  ->  d_c = P_{2c} + P_{2c+1}, c = 0..3 (the stream partials)
inline void hc_pairsum_fn() {
#pragma GCC unroll 2
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat p0 = sfpi::dst_reg[0], p1 = sfpi::dst_reg[32], p2 = sfpi::dst_reg[64], p3 = sfpi::dst_reg[96];
        sfpi::vFloat p4 = sfpi::dst_reg[128], p5 = sfpi::dst_reg[160], p6 = sfpi::dst_reg[192], p7 = sfpi::dst_reg[224];
        sfpi::dst_reg[0] = p0 + p1;
        sfpi::dst_reg[32] = p2 + p3;
        sfpi::dst_reg[64] = p4 + p5;
        sfpi::dst_reg[96] = p6 + p7;
        sfpi::dst_reg++;
    }
}

// d0..d3 = q_c (stream partials), d4..d7 = rs_c  ->  p = sum_c q_c * rs_c;
// silu: d0 = bf16(silu(p)), d1 = bf16(silu(p) - d0); else d0 = p
inline void hc_rsdot_fn(uint32_t silu) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat q0 = sfpi::dst_reg[0], r0 = sfpi::dst_reg[128];
        sfpi::vFloat p = q0 * r0;
        sfpi::vFloat q1 = sfpi::dst_reg[32], r1 = sfpi::dst_reg[160];
        p = p + q1 * r1;
        sfpi::vFloat q2 = sfpi::dst_reg[64], r2 = sfpi::dst_reg[192];
        p = p + q2 * r2;
        sfpi::vFloat q3 = sfpi::dst_reg[96], r3 = sfpi::dst_reg[224];
        p = p + q3 * r3;
        if (silu) {
            sfpi::vFloat g = p * _sfpu_sigmoid_<true>(p);
            sfpi::vFloat hi = hc_rne_bf16(g);
            sfpi::dst_reg[0] = hi;
            sfpi::dst_reg[32] = hc_rne_bf16(g - hi);
        } else {
            sfpi::dst_reg[0] = p;
        }
        sfpi::dst_reg++;
    }
}

// d0..d3 = y_c, d4..d7 = rs_c  ->  d_c = xn_c = y_c * rs_c
inline void hc_xn_fn() {
#pragma GCC unroll 2
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat y0 = sfpi::dst_reg[0], y1 = sfpi::dst_reg[32], y2 = sfpi::dst_reg[64], y3 = sfpi::dst_reg[96];
        sfpi::vFloat r0 = sfpi::dst_reg[128], r1 = sfpi::dst_reg[160], r2 = sfpi::dst_reg[192], r3 = sfpi::dst_reg[224];
        sfpi::dst_reg[0] = y0 * r0;
        sfpi::dst_reg[32] = y1 * r1;
        sfpi::dst_reg[64] = y2 * r2;
        sfpi::dst_reg[96] = y3 * r3;
        sfpi::dst_reg++;
    }
}

// d0..d3 = gate_c, d4..d7 = xn_c  ->  d0 = 1/4 sum_c xn_c * sigmoid(gate_c) (nearest-even bf16 if round_bf16)
inline void hc_mix_fn(uint32_t round_bf16) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat g0 = sfpi::dst_reg[0], x0 = sfpi::dst_reg[128];
        sfpi::vFloat t = x0 * _sfpu_sigmoid_<true>(g0);
        sfpi::vFloat g1 = sfpi::dst_reg[32], x1 = sfpi::dst_reg[160];
        t = t + x1 * _sfpu_sigmoid_<true>(g1);
        sfpi::vFloat g2 = sfpi::dst_reg[64], x2 = sfpi::dst_reg[192];
        t = t + x2 * _sfpu_sigmoid_<true>(g2);
        sfpi::vFloat g3 = sfpi::dst_reg[96], x3 = sfpi::dst_reg[224];
        t = (t + x3 * _sfpu_sigmoid_<true>(g3)) * 0.25f;
        if (round_bf16) {
            t = hc_rne_bf16(t);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg++;
    }
}

// d0 = inj  ->  w = 2*sigmoid(inj/4); d0 = bf16(w), d1 = bf16(w - d0)
inline void hc_wsplit_fn() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat w = 2.0f * _sfpu_sigmoid_<true>(x * 0.25f);
        sfpi::vFloat hi = hc_rne_bf16(w);
        sfpi::dst_reg[0] = hi;
        sfpi::dst_reg[32] = hc_rne_bf16(w - hi);
        sfpi::dst_reg++;
    }
}

// d0 = w (broadcast), d1 = R, d2 = bo  ->  d1 = R + w * bo
inline void hc_write_fn() {
#pragma GCC unroll 4
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat w = sfpi::dst_reg[0];
        sfpi::vFloat r = sfpi::dst_reg[32];
        sfpi::vFloat bo = sfpi::dst_reg[64];
        sfpi::dst_reg[32] = r + w * bo;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace ckernel {

// SFPU state for the passes without programmable constants.
ALWI void hc_sfpu_init_plain() { MATH(SFPU_UNARY_INIT(unused)); }
// sigmoid (reciprocal constant) / rsqrt (sqrt constants)
ALWI void hc_sfpu_init_sigmoid() { sigmoid_tile_init<false>(); }
ALWI void hc_sfpu_init_rsqrt() { rsqrt_tile_init<false>(); }

ALWI void hc_prep() { MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_prep_fn, 0, VectorMode::RC))); }
ALWI void hc_rs(uint32_t eps_bits) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_rs_fn, 0, VectorMode::RC, eps_bits)));
}
ALWI void hc_pairsum() { MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_pairsum_fn, 0, VectorMode::RC))); }
ALWI void hc_rsdot(uint32_t silu) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_rsdot_fn, 0, VectorMode::RC, silu)));
}
ALWI void hc_xn() { MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_xn_fn, 0, VectorMode::RC))); }
ALWI void hc_mix(uint32_t round_bf16) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_mix_fn, 0, VectorMode::RC, round_bf16)));
}
ALWI void hc_wsplit() { MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_wsplit_fn, 0, VectorMode::RC))); }
ALWI void hc_write_upd() { MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::hc_write_fn, 0, VectorMode::RC))); }

}  // namespace ckernel
