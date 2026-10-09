// All-device MoE, fused router compute (column core c < N_COL): logits tile c = x @ W[:, 32 c : 32 c + 32] over the KT
// k-tiles (f32 x -> SrcB, BF16 weights -> SrcA, the program's fidelity (HiFi4), one FP32 DEST accumulation), f32 out.
// k-tiles 0..39 arrive in cb_w (reader), 40..79 in cb_w2 (writer), in blocks of RB.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "common.h"
#include "prof.h"

using namespace fnx_moe;

constexpr uint32_t cb_x = 0, cb_w = 1, cb_w2 = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t RB = get_named_compile_time_arg_val("rb");
    const uint32_t core = get_arg_val<uint32_t>(0);
    if (core >= N_COL) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_init(cb_x, cb_w, 0);
    {
        MD_ZONE("RC-WAITX");
        cb_wait_front(cb_x, KT);
    }
    MD_ZONE("RC-MM");
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < KT; ++kt) {
        const uint32_t cb = kt < KT / 2 ? cb_w : cb_w2;
        const uint32_t i = kt % RB;
        if (i == 0) {
            cb_wait_front(cb, RB);
        }
        matmul_tiles(cb_x, cb, kt, i, 0);
        if (i == RB - 1) {
            cb_pop_front(cb, RB);
        }
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_x, KT);
}
