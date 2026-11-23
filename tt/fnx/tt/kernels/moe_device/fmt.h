// Run-time weight-format switching for the all-device MoE compute kernels.
//
// Weight CBs: cb_w(f) = 1 + f per format (common.h).  The Bfp2_b CB is declared bfloat4_b (ttnn has no DataType for
// it) with 320 B pages: its compile-time unpacker config is wrong, so every switch of the weight format programs the
// SrcA unpacker directly (format + tile size; register format = the L1 format, as tt-metal's data-format inference
// does for block-float and BF16 sources).  The math side only re-derives the int8 flag on Blackhole.
//
// Fidelity per format: block-float 4/2 weights have <= 4 significant bits, exact in SrcA's 5 MSBs, so their matmuls run
// at FID_LO (named compile-time arg "fid_lo"; LoFi keeps the BF16 activation's 7 MSBs); bfp8 / bf16 weights (the
// shared expert) need SrcA's low bits and the activation's last bit: HiFi4 (all four phases, exact products).
#pragma once
#include "common.h"

constexpr uint32_t cb_w(uint32_t f) { return 1 + f; }
constexpr ckernel::MathFidelity FID_LO = static_cast<ckernel::MathFidelity>(get_named_compile_time_arg_val("fid_lo"));
constexpr ckernel::MathFidelity FID_HI = ckernel::MathFidelity::HiFi4;

ALWI bool hi_fidelity(uint32_t f) { return f == fnx_moe::F_BFP8 || f == fnx_moe::F_BF16; }

// SrcA <- format f, and the matmul MOPs (fidelity of f) for in0 x cb_w(f).
ALWI void mm_select(uint32_t in0, uint32_t f) {
    const uint32_t df = fnx_moe::data_format(f);
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
        df, df, fnx_moe::tile_bytes(f) >> 4, 16, 4)));
    MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE>(df)));
    if (hi_fidelity(f)) {
        MATH((llk_math_matmul_init<FID_HI, MM_THROTTLE>(in0, cb_w(f), 0)));
    } else {
        MATH((llk_math_matmul_init<FID_LO, MM_THROTTLE>(in0, cb_w(f), 0)));
    }
    UNPACK((llk_unpack_AB_matmul_init(in0, cb_w(f), 0)));
}

// DEST[dst] += in0[i0] @ cb_w(f)[i1] at the fidelity of f (mm_select(in0, f) first).
ALWI void mm_tile(uint32_t in0, uint32_t f, uint32_t i0, uint32_t i1, uint32_t dst) {
    UNPACK((llk_unpack_AB_matmul(in0, cb_w(f), i0, i1)));
    if (hi_fidelity(f)) {
        MATH((llk_math_matmul<FID_HI, MM_THROTTLE>(dst)));
    } else {
        MATH((llk_math_matmul<FID_LO, MM_THROTTLE>(dst)));
    }
}
