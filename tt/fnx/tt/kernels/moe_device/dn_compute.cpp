// All-device MoE, stage B compute: y[n] = sum over the selected entries s (ascending id, the shared expert last) of
// h[s] @ D_e[:, n], all in one FP32 DEST accumulation (the routed and shared weights are already in h), f32 out.
// Units as in dn_reader.cpp; the weight format (SrcA unpacker + fidelity) follows each entry (fmt.h).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "common.h"
#include "fmt.h"

using namespace fnx_moe;

constexpr uint32_t cb_h = 0, cb_sel = 6, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    const uint32_t core = get_arg_val<uint32_t>(0);
    if (core >= NT) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_w(F_BFP4), cb_out);
    cb_wait_front(cb_sel, 1);
    const uint32_t S = read_tile_value(cb_sel, 0, 0);
    uint32_t cur = 0xffffffffu;
    for (uint32_t n = core; n < NT; n += NC) {
        tile_regs_acquire();
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t f = word_fmt(read_tile_value(cb_sel, 0, read_tile_value(cb_sel, 0, SEL_ENT + s) + 2));
            if (f != cur) {
                mm_select(cb_h, f);
                cur = f;
            }
            const uint32_t cb = cb_w(f);
            cb_wait_front(cb_h, JT);
            cb_wait_front(cb, JT);
            for (uint32_t k = 0; k < JT; ++k) {
                mm_tile(cb_h, f, k, k, 0);
            }
            cb_pop_front(cb, JT);
            cb_pop_front(cb_h, JT);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
