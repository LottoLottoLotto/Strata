// All-device MoE, stage A writer (BRISC): x rows, route tiles, hidden output.
//
// Waits for the selection (gu_reader), reads the T valid rows of the BF16 input x (KT tiles, rows < T of the first
// 32-row tile; with T <= 16 only faces 0/1, else whole tiles) into cb_x once, then per work unit (s, j) (sched.h) builds the f32
// route tile (row r = entry s's weight for row r in all 32 columns, rows >= T zero) and writes the activated hidden
// tile h[s * JT + j] (only its valid rows when T <= 16).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"
#include "prof.h"
#include "sched.h"
#include "stream.h"

using namespace fnx_moe;

constexpr uint32_t cb_x = 0, cb_r = 5, cb_sel = 6, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t HCORES = get_named_compile_time_arg_val("hcores");
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto h_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t x_tile = get_tile_size(cb_x), h_tile = get_tile_size(cb_out);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), x_tile);
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(1), h_tile);

    // route tiles: a ring of 2 f32 tiles, zeroed once (rows >= T stay zero)
    cb_reserve_back(cb_r, 2);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_r));
        for (uint32_t i = 0; i < 2 * 1024; ++i) {
            z[i] = 0;
        }
    }

    cb_wait_front(cb_sel, 1);
    invalidate_l1_cache();
    const uint32_t* sel = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_sel));
    const uint32_t S = sel[0], T = sel[1];
    auto units = make_units([sel](uint32_t i) { return sel[i]; }, S, core, NC, HCORES);
    if (!units.any()) {
        return;
    }
    const bool part = T <= 16;
    {
    MD_ZONE("GW-X");
    noc_async_read_set_trid(TRID_W);
    cb_reserve_back(cb_x, KT);
    const uint32_t xp = get_write_ptr(cb_x);
    for (uint32_t i = 0; i < KT; ++i) {
        const uint32_t k = (i + core) % KT;          // start at a core-dependent tile: no queueing on one bank
        const uint64_t src = get_noc_addr(k, x);
        if (part) {
            noc_async_read(src, xp + k * x_tile, T * 32);
            noc_async_read(src + 512, xp + k * x_tile + 512, T * 32);
        } else {
            noc_async_read(src, xp + k * x_tile, x_tile);
        }
    }
    noc_async_read_barrier_with_trid(TRID_W);
    noc_async_read_set_trid(0);
    cb_push_back(cb_x, KT);
    }

    uint32_t s, j;
    while (units.next(s, j)) {
        const uint32_t u = s * JT + j;
        cb_reserve_back(cb_r, 1);
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_r));
        const uint32_t e = sel[SEL_ID + s];
        for (uint32_t r = 0; r < T; ++r) {
            volatile tt_l1_ptr uint32_t* f0 = t + (r >> 4) * 512 + (r & 15) * 16;   // faces 0/2: columns 0-15
            volatile tt_l1_ptr uint32_t* f1 = f0 + 256;                                // faces 1/3: columns 16-31
            const uint32_t v = row_weight(sel + SEL_ROWS + r * ENT_WORDS, e);
            for (uint32_t c = 0; c < 16; ++c) {
                f0[c] = v;
                f1[c] = v;
            }
        }
        cb_push_back(cb_r, 1);

        cb_wait_front(cb_out, 1);
        const uint32_t rp = get_read_ptr(cb_out);
        const uint64_t dst = get_noc_addr(u, h);
        if (part) {
            noc_async_write(rp, dst, T * 32);
            noc_async_write(rp + 512, dst + 512, T * 32);
        } else {
            noc_async_write(rp, dst, h_tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
