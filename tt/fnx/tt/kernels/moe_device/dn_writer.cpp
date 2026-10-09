// All-device MoE, stage B writer (BRISC): hidden rows in, output tile out.
//
// Per work unit n (dn_reader.cpp) and selected entry s: the entry's JT hidden tiles h[s * JT + k] (only the T valid
// rows of faces 0/1 when T <= 16, else whole tiles) into cb_h; then the f32 output tile y[n] (first 32-row tile).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"
#include "stream.h"

using namespace fnx_moe;

constexpr uint32_t cb_h = 0, cb_sel = 6, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr auto h_args = TensorAccessorArgs<0>();
    constexpr auto y_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t h_tile = get_tile_size(cb_h), y_tile = get_tile_size(cb_out);
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0), h_tile);
    const auto y = TensorAccessor(y_args, get_common_arg_val<uint32_t>(1), y_tile);
    if (core >= NT) {
        return;
    }
    cb_wait_front(cb_sel, 1);
    invalidate_l1_cache();
    const uint32_t* sel = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_sel));
    const uint32_t S = sel[0], T = sel[1];
    const bool part = T <= 16;
    noc_async_read_set_trid(TRID_W);
    for (uint32_t n = core; n < NT; n += NC) {
        for (uint32_t s = 0; s < S; ++s) {
            cb_reserve_back(cb_h, JT);
            const uint32_t hp = get_write_ptr(cb_h);
            for (uint32_t i = 0; i < JT; ++i) {
                const uint32_t k = (i + core) % JT;
                const uint64_t src = get_noc_addr(s * JT + k, h);
                if (part) {
                    noc_async_read(src, hp + k * h_tile, T * 32);
                    noc_async_read(src + 512, hp + k * h_tile + 512, T * 32);
                } else {
                    noc_async_read(src, hp + k * h_tile, h_tile);
                }
            }
            noc_async_read_barrier_with_trid(TRID_W);
            cb_push_back(cb_h, JT);
        }
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), get_noc_addr(n, y), y_tile);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
}
