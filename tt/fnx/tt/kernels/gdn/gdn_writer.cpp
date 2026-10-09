// SPDX-License-Identifier: Apache-2.0
// Fused GDN writer (BRISC), one v-head h per core: loads the FP32 recurrent state S_h (16 tiles) from s_in for token
// 0, writes y row t of the 4 output tiles of head h per token (OUT), and stores the final state to s_out once the
// compute signals it; with T = 0 the loaded state goes to s_out unchanged (rollback to row 0).  SNAP (verify window):
// copies the state after row 0 to snap_s and then releases that buffer to the compute (which reuses it at row 2).
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t c_s0 = 6, c_s1 = 8, c_y = 23, c_sdone = 24, c_si = 25, c_sim = 26, c_s0ready = 27, c_snapok = 28;
constexpr uint32_t TB = 4096;
constexpr uint32_t OUT = 1, SNAP = 2;
inline uint32_t row_off(uint32_t row, uint32_t half) { return ((row >> 4) * 2 + half) * 1024 + (row & 15) * 64; }
}  // namespace

void kernel_main() {
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_in_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_out_addr = get_arg_val<uint32_t>(3);
    const uint32_t snap_addr = get_arg_val<uint32_t>(4);
    const uint32_t T = get_arg_val<uint32_t>(5);
    const uint32_t mode = get_arg_val<uint32_t>(6);
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto yacc = TensorAccessor(y_args, y_addr, TB);
    const auto s_in = TensorAccessor(s_args, s_in_addr, TB);
    const auto s_out = TensorAccessor(s_args, s_out_addr, TB);
    const auto s_snap = TensorAccessor(s_args, snap_addr, TB);
    const bool out = mode & OUT, snap = (mode & SNAP) && T > 1;

    cb_reserve_back(c_si, 16);
    const uint32_t si = get_write_ptr(c_si);
    for (uint32_t i = 0; i < 16; ++i) {
        noc_async_read(s_in.get_noc_addr(16 * h + i), si + i * TB, TB);
    }
    noc_async_read_barrier();
    if (T == 0) {
        for (uint32_t i = 0; i < 16; ++i) {
            noc_async_write(si + i * TB, s_out.get_noc_addr(16 * h + i), TB);
        }
        noc_async_write_barrier();
        return;
    }
    cb_push_back(c_si, 16);
    cb_reserve_back(c_sim, 16);
    cb_push_back(c_sim, 16);

    for (uint32_t t = 0; t < T; ++t) {
        if (snap && t == 0) {   // the state after row 0 (c_s0) -> snap_s
            cb_wait_front(c_s0ready, 1);
            const uint32_t st = get_read_ptr(c_s0);
            for (uint32_t i = 0; i < 16; ++i) {
                noc_async_write(st + i * TB, s_snap.get_noc_addr(16 * h + i), TB);
            }
        }
        if (t + 1 == T) {
            cb_wait_front(c_sdone, 1);
            const uint32_t fin = get_read_ptr(((T - 1) & 1) ? c_s1 : c_s0);
            for (uint32_t i = 0; i < 16; ++i) {
                noc_async_write(fin + i * TB, s_out.get_noc_addr(16 * h + i), TB);
            }
        }
        if (out) {
            cb_wait_front(c_y, 4);
            const uint32_t src = get_read_ptr(c_y);
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(src + r * TB + half * 1024, yacc.get_noc_addr(4 * h + r, row_off(t, half)), 64);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(c_y, 4);
        }
        if (snap && t == 0) {
            noc_async_write_barrier();
            cb_pop_front(c_s0ready, 1);
            cb_reserve_back(c_snapok, 1);
            cb_push_back(c_snapok, 1);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(c_sdone, 1);
}
