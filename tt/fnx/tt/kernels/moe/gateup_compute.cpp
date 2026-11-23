// Device experts, stage A compute: h[s][j] = SiLU(x_g @ G_e[j]^T) * (x_g @ U_e[j]^T) * route[s] (FP32 DEST, one
// BF16 rounding at the pack).  DEST 0 = gate, 1 = up, 2 = route tile; units as in gateup_reader.cpp.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_r = 3, cb_meta = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_named_compile_time_arg_val("kt");
    constexpr uint32_t JT = get_named_compile_time_arg_val("jt");
    constexpr uint32_t BK = get_named_compile_time_arg_val("bk");
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    const uint32_t core = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t units = read_tile_value(cb_meta, 0, 0) * JT;
    uint32_t cur_g = 0xFFFFFFFFu;
    uint32_t srca = cb_w;   // operand currently configured in SrcA (in1 of the matmul, the copied route tile)
    for (uint32_t u = core; u < units; u += NC) {
        const uint32_t s = u / JT;
        const uint32_t g = read_tile_value(cb_meta, 0, 64 + s) >> 16;
        if (g != cur_g) {
            if (cur_g != 0xFFFFFFFFu) {
                cb_pop_front(cb_x, KT);
            }
            cb_wait_front(cb_x, KT);
            cur_g = g;
        }
        reconfig_data_format_srca(srca, cb_w);
        srca = cb_w;
        matmul_init(cb_x, cb_w, 1);
        tile_regs_acquire();
        for (uint32_t op = 0; op < 2; ++op) {
            for (uint32_t kb = 0; kb < KT; kb += BK) {
                cb_wait_front(cb_w, BK);
                for (uint32_t k = 0; k < BK; ++k) {
                    matmul_tiles(cb_x, cb_w, kb + k, k, op);
                }
                cb_pop_front(cb_w, BK);
            }
        }
        cb_wait_front(cb_r, 1);
        reconfig_data_format_srca(srca, cb_r);
        srca = cb_r;
        copy_tile_to_dst_init_short(cb_r);
        copy_tile(cb_r, 0, 2);
        cb_pop_front(cb_r, 1);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    if (cur_g != 0xFFFFFFFFu) {
        cb_pop_front(cb_x, KT);
    }
}
