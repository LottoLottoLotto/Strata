// Device experts, stage A reader (NCRISC): gate/up weight streaming for fnx.tt.moe.
//
// Work unit u = (entry s, intermediate tile j), u = s * JT + j; core c takes u = c, c + NC, ...
// Metadata (uint32 row, one page, fnx.tt.moe.DeviceExperts.run): [0] S entries, [3] token rows T, [64 + s] =
// group << 16 | resident expert, f32 route weights of entry s at byte WOFF + 128 s (32 rows of its 32-token group).
// Per unit: x tiles of the entry's group (cb_x, reloaded only when the group changes), the f32 route tile (row r =
// weight of token row r in every column), then the gate tile row j and the up tile row j of the expert (KT tiles each,
// blocks of BK; weights are stored [n_out, n_in] per expert, the matmul transposes them).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_r = 3, cb_meta = 4;

void kernel_main() {
    constexpr uint32_t KT = get_named_compile_time_arg_val("kt");
    constexpr uint32_t JT = get_named_compile_time_arg_val("jt");
    constexpr uint32_t BK = get_named_compile_time_arg_val("bk");
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t META_BYTES = get_named_compile_time_arg_val("meta_bytes");
    constexpr uint32_t META_PAGE = get_named_compile_time_arg_val("meta_page");
    constexpr uint32_t WOFF = get_named_compile_time_arg_val("woff");
    constexpr bool IN_L1 = get_named_compile_time_arg_val("in_l1") != 0;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t x_tile = get_tile_size(cb_x);
    const uint32_t w_tile = get_tile_size(cb_w);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), x_tile);
    const auto meta = TensorAccessor(m_args, get_common_arg_val<uint32_t>(1), META_PAGE);
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2), w_tile);

    const uint64_t meta_noc = get_noc_addr(0, meta);
    cb_reserve_back(cb_meta, 1);
    const uint32_t mp = get_write_ptr(cb_meta);
    noc_async_read(meta_noc, mp, META_BYTES);
    noc_async_read_barrier();
    cb_push_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp);
    const uint32_t units = m[0] * JT;
    const uint32_t T = m[3];   // token rows of the call: group g has min(32, T - 32 g) valid rows

    uint32_t cur_g = 0xFFFFFFFFu;
    for (uint32_t u = core; u < units; u += NC) {
        const uint32_t s = u / JT, j = u - s * JT;
        const uint32_t ent = m[64 + s];
        const uint32_t g = ent >> 16, e = ent & 0xFFFFu;
        if (g != cur_g) {
            // Every core reads all KT x tiles (110 cores x 160 KB was ~70 us of NOC traffic at T=1): with at most 16
            // valid rows (L1 source: 16 B NOC alignment) only those rows of faces 0 and 1 are read; the other rows
            // are padding.  Start at a core-dependent tile so the cores do not queue on one bank.
            const uint32_t nr = T - 32 * g < 32 ? T - 32 * g : 32;
            cb_reserve_back(cb_x, KT);
            const uint32_t wp = get_write_ptr(cb_x);
            for (uint32_t i = 0; i < KT; ++i) {
                const uint32_t k = (i + core) % KT;
                const uint64_t src = get_noc_addr(g * KT + k, x);
                if (IN_L1 && nr <= 16) {
                    noc_async_read(src, wp + k * x_tile, nr * 32);
                    noc_async_read(src + 512, wp + k * x_tile + 512, nr * 32);
                } else {
                    noc_async_read(src, wp + k * x_tile, x_tile);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, KT);
            cur_g = g;
        }
        // route tile: the 32 f32 weights land in the tile's last 128 B, then each valid row is filled with its
        // weight (rows past the call's tokens are padding: whatever they hold only reaches padding rows of h)
        cb_reserve_back(cb_r, 1);
        const uint32_t rp = get_write_ptr(cb_r);
        noc_async_read(meta_noc + WOFF + s * 128, rp + 4096 - 128, 128);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rp);
        const uint32_t nr = T - 32 * g < 32 ? T - 32 * g : 32;
        uint32_t wv[32];
        for (uint32_t r = 0; r < nr; ++r) {
            wv[r] = t[1024 - 32 + r];
        }
        for (uint32_t r = 0; r < nr; ++r) {
            volatile tt_l1_ptr uint32_t* f0 = t + (r >> 4) * 512 + (r & 15) * 16;   // faces 0/2: columns 0-15
            volatile tt_l1_ptr uint32_t* f1 = f0 + 256;                                // faces 1/3: columns 16-31
            const uint32_t v = wv[r];
            for (uint32_t c = 0; c < 16; ++c) {
                f0[c] = v;
                f1[c] = v;
            }
        }
        cb_push_back(cb_r, 1);
        for (uint32_t op = 0; op < 2; ++op) {
            const uint32_t base = ((e * 2 + op) * JT + j) * KT;
            for (uint32_t kb = 0; kb < KT; kb += BK) {
                cb_reserve_back(cb_w, BK);
                uint32_t wp = get_write_ptr(cb_w);
                for (uint32_t k = 0; k < BK; ++k) {
                    noc_async_read(get_noc_addr(base + kb + k, w), wp, w_tile);
                    wp += w_tile;
                }
                noc_async_read_barrier();
                cb_push_back(cb_w, BK);
            }
        }
    }
}
