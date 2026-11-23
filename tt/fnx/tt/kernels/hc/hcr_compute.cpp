// HcRead, compute (HiFi4, fp32 DST, half-sync: 4 fp32 tiles per section).
// Matmuls take bf16 operands only (activations split exactly into hi + lo bf16 tiles); everything elementwise runs
// in fp32 on the SFPU (hc_sfpu.h); f32 CBs are unpacked straight to DST.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "hc_common.h"
#include "hc_sfpu.h"

using namespace hc;

constexpr uint32_t NN = get_compile_time_arg_val(0);
constexpr uint32_t PRECISE = get_compile_time_arg_val(1);
constexpr uint32_t OUT_PAGE = get_compile_time_arg_val(3);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);

FORCE_INLINE void copy_init(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
}

FORCE_INLINE void mm_init(uint32_t in0, uint32_t in1) {
    reconfig_data_format_srca(in1);
    reconfig_data_format_srcb(in0);
    matmul_init(in0, in1);
}

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
}

#ifdef HC_PROF
#include "tools/profiler/kernel_profiler.hpp"
#define HC_ZONE(name) DeviceZoneScopedN(name)
#else
#define HC_ZONE(name)
#endif

void kernel_main() {
    const uint32_t lx = get_absolute_logical_x();
#if ROLE == 1  // MAIN
    const uint32_t ntp = lx <= 6 ? 4 : 3;
    compute_kernel_hw_startup(CB_R, CB_GS, CB_Y);
    // PREP: y = R*gs, hi/lo(y), hi/lo(R^2)   (full-sync DST: 5 fp32 tiles)
    {
    HC_ZONE("PREP");
    cb_wait_front(CB_R, ntp);
    cb_wait_front(CB_GS, ntp);
    for (uint32_t i = 0; i < ntp; ++i) {
        tile_regs_acquire();
        copy_init(CB_R);
        copy_tile(CB_R, i, 0);
        copy_tile(CB_GS, i, 1);
        hc_sfpu_init_plain();
        hc_prep();
        tile_regs_commit();
        cb_reserve_back(CB_Y, 1);
        cb_reserve_back(CB_HL, 2);
        cb_reserve_back(CB_X2, 2);
        tile_regs_wait();
        pack_to(0, CB_Y);
        pack_to(1, CB_HL);
        pack_tile(2, CB_HL);
        pack_to(3, CB_X2);
        pack_tile(4, CB_X2);
        tile_regs_release();
        cb_push_back(CB_Y, 1);
        cb_push_back(CB_HL, 2);
        cb_push_back(CB_X2, 2);
    }
    cb_pop_front(CB_R, ntp);
    cb_pop_front(CB_GS, ntp);
    }
    // DOWN: partial of N-tile lx over the 40 K-tiles of slice ly (each weight chunk as soon as it has landed; the
    // accumulation order is k = 0..39 either way)
    if (lx < NN) {
        HC_ZONE("DOWN");
        cb_wait_front(CB_ACT, 2 * K_SLICE);
        mm_init(CB_ACT, CB_WD);
        tile_regs_acquire();
        for (uint32_t c = 0; c < WD_CHUNKS; ++c) {
            cb_wait_front(CB_WD, (c + 1) * WD_CHUNK);
            for (uint32_t k = c * WD_CHUNK; k < (c + 1) * WD_CHUNK; ++k) {
                matmul_tiles(CB_ACT, CB_WD, 2 * k, k, 0);
                if constexpr (PRECISE) {
                    matmul_tiles(CB_ACT, CB_WD, 2 * k + 1, k, 0);
                }
            }
        }
        tile_regs_commit();
        cb_reserve_back(CB_PART, 1);
        tile_regs_wait();
        pack_to(0, CB_PART);
        tile_regs_release();
        cb_push_back(CB_PART, 1);
        cb_pop_front(CB_WD, K_SLICE);
        cb_pop_front(CB_ACT, 2 * K_SLICE);
    }
    // MIX: gate_c = silu(p) @ Wu[:, column (c, j)], mixed_j = 1/4 sum_c xn_c * sigmoid(gate_c), xn_c = y_c * rs_c
    if (lx < 10) {
        HC_ZONE("MIX");
        cb_wait_front(CB_YMIX, 4);  // xn first: rs lands before silu(p) does
        cb_wait_front(CB_RS, 4);
        tile_regs_acquire();
        copy_init(CB_YMIX);
        for (uint32_t c = 0; c < 4; ++c) {
            copy_tile(CB_YMIX, c, c);
            copy_tile(CB_RS, c, 4 + c);
        }
        hc_sfpu_init_plain();
        hc_xn();
        tile_regs_commit();
        cb_reserve_back(CB_XN, 4);
        tile_regs_wait();
        pack_reconfig_data_format(CB_XN);
        for (uint32_t c = 0; c < 4; ++c) {
            pack_tile(c, CB_XN);
        }
        tile_regs_release();
        cb_push_back(CB_XN, 4);
        cb_pop_front(CB_YMIX, 4);
        cb_pop_front(CB_RS, 4);

        cb_wait_front(CB_WU, 40);
        cb_wait_front(CB_LO, 20);
        cb_wait_front(CB_XN, 4);
        mm_init(CB_LO, CB_WU);
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            for (uint32_t n = 0; n < 10; ++n) {
                matmul_tiles(CB_LO, CB_WU, 2 * n, c * 10 + n, c);
                if constexpr (PRECISE) {
                    matmul_tiles(CB_LO, CB_WU, 2 * n + 1, c * 10 + n, c);
                }
            }
        }
        copy_init(CB_XN);
        for (uint32_t c = 0; c < 4; ++c) {
            copy_tile(CB_XN, c, 4 + c);
        }
        hc_sfpu_init_sigmoid();
        hc_mix(OUT_PAGE == BF16_TILE);
        tile_regs_commit();
        cb_reserve_back(CB_OUT, 1);
        tile_regs_wait();
        pack_to(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_WU, 40);
        cb_pop_front(CB_LO, 20);
        cb_pop_front(CB_XN, 4);
    }
#elif ROLE == 2  // SSQ: per-row sum of R^2 (hi + lo) over the slice, broadcast over the 32 columns (x2 @ ones)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X2IN, CB_ONES, CB_SSQ);
    matmul_init(CB_X2IN, CB_ONES);
    cb_wait_front(CB_ONES, 1);
    cb_wait_front(CB_X2IN, 2 * K_SLICE);
    tile_regs_acquire();
    for (uint32_t k = 0; k < 2 * K_SLICE; ++k) {
        matmul_tiles(CB_X2IN, CB_ONES, k, 0, 0);
    }
    tile_regs_commit();
    cb_reserve_back(CB_SSQ, 1);
    tile_regs_wait();
    pack_tile(0, CB_SSQ);
    tile_regs_release();
    cb_push_back(CB_SSQ, 1);
    cb_pop_front(CB_X2IN, 2 * K_SLICE);
    cb_pop_front(CB_ONES, 1);
#elif ROLE == 3  // RS: rs_c = rsqrt(S_{2c} + S_{2c+1} + N*eps)
    compute_kernel_hw_startup(CB_SSQIN, CB_SSQIN, CB_RS);
    cb_wait_front(CB_SSQIN, N_SLICE);
    for (uint32_t c = 0; c < 4; ++c) {
        tile_regs_acquire();
        copy_init(CB_SSQIN);
        copy_tile(CB_SSQIN, 2 * c, 0);
        copy_tile(CB_SSQIN, 2 * c + 1, 1);
        hc_sfpu_init_rsqrt();
        hc_rs(EPS_BITS);
        tile_regs_commit();
        cb_reserve_back(CB_RS, 1);
        tile_regs_wait();
        pack_to(0, CB_RS);
        tile_regs_release();
        cb_push_back(CB_RS, 1);
    }
    cb_pop_front(CB_SSQIN, N_SLICE);
#elif ROLE == 4  // RED(n): p = sum_c rs_c * (P_{2c} + P_{2c+1}); n < 10: hi/lo of silu(p); n == 10: inject
    compute_kernel_hw_startup(CB_PIN, CB_PIN, CB_ACC);
    cb_wait_front(CB_PIN, N_SLICE);  // stream partials first: rs may still be on its way
    tile_regs_acquire();
    copy_init(CB_PIN);
    for (uint32_t s = 0; s < N_SLICE; ++s) {
        copy_tile(CB_PIN, s, s);
    }
    hc_sfpu_init_plain();
    hc_pairsum();
    tile_regs_commit();
    cb_reserve_back(CB_ACC, 4);
    tile_regs_wait();
    pack_reconfig_data_format(CB_ACC);
    for (uint32_t c = 0; c < 4; ++c) {
        pack_tile(c, CB_ACC);
    }
    tile_regs_release();
    cb_push_back(CB_ACC, 4);
    cb_pop_front(CB_PIN, N_SLICE);

    cb_wait_front(CB_ACC, 4);
    cb_wait_front(CB_RS, 4);
    tile_regs_acquire();
    copy_init(CB_ACC);
    for (uint32_t c = 0; c < 4; ++c) {
        copy_tile(CB_ACC, c, c);
        copy_tile(CB_RS, c, 4 + c);
    }
    if (lx < 10) {
        hc_sfpu_init_sigmoid();
        hc_rsdot(1);
    } else {
        hc_sfpu_init_plain();
        hc_rsdot(0);
    }
    tile_regs_commit();
    if (lx < 10) {
        cb_reserve_back(CB_RED, 2);
        tile_regs_wait();
        pack_to(0, CB_RED);
        pack_tile(1, CB_RED);
        tile_regs_release();
        cb_push_back(CB_RED, 2);
    } else {
        cb_reserve_back(CB_INJ, 1);
        tile_regs_wait();
        pack_to(0, CB_INJ);
        tile_regs_release();
        cb_push_back(CB_INJ, 1);
    }
    cb_pop_front(CB_ACC, 4);
    cb_pop_front(CB_RS, 4);
#endif
}
