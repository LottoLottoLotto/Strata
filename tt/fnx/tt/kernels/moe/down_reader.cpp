// Device experts, stage B reader (NCRISC): down projection streaming for fnx.tt.moe.
//
// Work unit u = (group g, output tile n), u = g * NT + n; core c takes u = c, c + NC, ...  For every entry s of group g
// (meta[4 + g] <= s < meta[5 + g]) it streams the entry's JT hidden tiles (h tile s * JT + k)
// and the expert's down tile row n (JT tiles of the [n_out, n_in] weight, transposed by the matmul).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_h = 0, cb_w = 1, cb_meta = 4;

void kernel_main() {
    constexpr uint32_t NT = get_named_compile_time_arg_val("nt");
    constexpr uint32_t JT = get_named_compile_time_arg_val("jt");
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t META_BYTES = get_named_compile_time_arg_val("meta_bytes");
    constexpr uint32_t META_PAGE = get_named_compile_time_arg_val("meta_page");
    constexpr bool IN_L1 = get_named_compile_time_arg_val("in_l1") != 0;
    constexpr auto h_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t h_tile = get_tile_size(cb_h);
    const uint32_t w_tile = get_tile_size(cb_w);
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0), h_tile);
    const auto meta = TensorAccessor(m_args, get_common_arg_val<uint32_t>(1), META_PAGE);
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2), w_tile);

    cb_reserve_back(cb_meta, 1);
    const uint32_t mp = get_write_ptr(cb_meta);
    noc_async_read(get_noc_addr(0, meta), mp, META_BYTES);
    noc_async_read_barrier();
    cb_push_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp);
    const uint32_t units = m[1] * NT;
    const uint32_t g0 = m[2], T = m[3];

    for (uint32_t u = core; u < units; u += NC) {
        const uint32_t g = u / NT, n = u - g * NT;
        const uint32_t s_end = m[5 + g];
        const uint32_t left = T - 32 * (g0 + g);
        const uint32_t nr = left < 32 ? left : 32;   // valid rows of the group; the others are padding
        for (uint32_t s = m[4 + g]; s < s_end; ++s) {
            const uint32_t e = m[64 + s] & 0xFFFFu;
            // every unit reads the entry's JT hidden tiles (only the valid rows of faces 0/1 when <= 16 of them, from
            // L1); start at a core-dependent tile (no queueing on one bank)
            cb_reserve_back(cb_h, JT);
            const uint32_t hp = get_write_ptr(cb_h);
            for (uint32_t i = 0; i < JT; ++i) {
                const uint32_t k = (i + core) % JT;
                const uint64_t src = get_noc_addr(s * JT + k, h);
                if (IN_L1 && nr <= 16) {
                    noc_async_read(src, hp + k * h_tile, nr * 32);
                    noc_async_read(src + 512, hp + k * h_tile + 512, nr * 32);
                } else {
                    noc_async_read(src, hp + k * h_tile, h_tile);
                }
            }
            cb_reserve_back(cb_w, JT);
            uint32_t wp = get_write_ptr(cb_w);
            const uint32_t base = (e * NT + n) * JT;
            for (uint32_t k = 0; k < JT; ++k) {
                noc_async_read(get_noc_addr(base + k, w), wp, w_tile);
                wp += w_tile;
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, JT);
            cb_push_back(cb_h, JT);
        }
    }
}
