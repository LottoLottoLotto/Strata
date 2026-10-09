// All-device MoE, stage A compute: h[s][j] = SiLU(x @ G_e[:, j]) * (x @ U_e[:, j]) * route[s] (FP32 DEST, one BF16
// rounding at the pack).  DEST 0 = gate, 1 = up, 2 = route tile (unpacked to DEST in FP32); units: sched.h (the
// readers' order).  The weight operand (in1 -> SrcA) changes format per matrix (fmt.h: unpacker + fidelity).
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
#include "common.h"
#include "fmt.h"
#include "sched.h"

using namespace fnx_moe;

constexpr uint32_t cb_x = 0, cb_r = 5, cb_sel = 6, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t BT = get_named_compile_time_arg_val("bt");
    constexpr uint32_t HCORES = get_named_compile_time_arg_val("hcores");
    const uint32_t core = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w(F_BFP4), cb_out);
    cb_wait_front(cb_sel, 1);
    const uint32_t S = read_tile_value(cb_sel, 0, 0);
    auto units = make_units([](uint32_t i) { return read_tile_value(cb_sel, 0, i); }, S, core, NC, HCORES);
    if (!units.any()) {
        return;
    }
    cb_wait_front(cb_x, KT);
    uint32_t cur = 0xffffffffu;   // weight format the unpacker / matmul MOPs are set up for (none after the route copy)
    uint32_t s, j;
    while (units.next(s, j)) {
        const uint32_t off = read_tile_value(cb_sel, 0, SEL_ENT + s);
        const uint32_t fg = word_fmt(read_tile_value(cb_sel, 0, off));
        const uint32_t fu = word_fmt(read_tile_value(cb_sel, 0, off + 1));
        tile_regs_acquire();
        for (uint32_t op = 0; op < 2; ++op) {
            const uint32_t f = op ? fu : fg;
            if (f != cur) {
                mm_select(cb_x, f);
                cur = f;
            }
            const uint32_t cb = cb_w(f);
            for (uint32_t kb = 0; kb < KT; kb += BT) {
                cb_wait_front(cb, BT);
                for (uint32_t k = 0; k < BT; ++k) {
                    mm_tile(cb_x, f, kb + k, k, op);
                }
                cb_pop_front(cb, BT);
            }
        }
        cb_wait_front(cb_r, 1);
        reconfig_data_format_srca(cb_r);
        cur = 0xffffffffu;
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
    cb_pop_front(cb_x, KT);
}
