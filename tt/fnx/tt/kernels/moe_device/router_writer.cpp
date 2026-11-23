// All-device MoE, fused router (BRISC, NOC1).  Common args: [0] logits, [1] route buffer, [2] x (f32 TILE), [3] router
// weight, [4] T, [5] xb (BF16 TILE: stage A's input), [6] the layer's expert table, [7] the entry pages (common.h),
// then the NOC (x, y) of route core r = program core r < T.
//   column core c < N_COL (9; 17 for 512 experts): k-tiles 40..79 of weight column c (the reader takes 0..39) ->
//            cb_w2, in flight while the T valid rows of x (KT f32 tiles, faces 0/1 rows < T, or whole tiles for
//            T > 16) land in cb_x; while compute multiplies: x's BF16 copy (round to nearest even, = ttnn.typecast) of
//            k-tiles c, c + N_COL, ... -> xb; then the logits tile from compute -> logits tile c, and each row r < T's
//            32 values -> route core r's gather page at 128 c, one semaphore increment per row core
//   route core r < T: fetches the expert table (in flight during the column work), waits for the N_COL column
//            increments, then route_row (topk.h) on its gathered N_EXP + 1 logits -> route buffer row r, and the row
//            with its TOPK experts' and the shared expert's table entries -> entry page r (common.h: stage A / B read
//            these T pages, not the whole table: no core but the T route cores touches the table's DRAM page); resets
//            the semaphore for the next launch.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"
#include "prof.h"
#include "stream.h"
#include "topk.h"

using namespace fnx_moe;

constexpr uint32_t cb_x = 0, cb_w2 = 2, cb_xb = 3, cb_tab = 5, cb_g = 7, cb_out = 16;

void kernel_main() {
    constexpr uint32_t RB = get_named_compile_time_arg_val("rb");
    constexpr uint32_t NP = KT / RB / 2;          // weight pages per half column
    constexpr auto l_args = TensorAccessorArgs<0>();
    constexpr auto r_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr uint32_t TRID_TAB = NOC_MAX_TRANSACTION_ID;     // never a column stream's (KT / RB / 2 + 1 ids)
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t T = get_common_arg_val<uint32_t>(4);
    const uint32_t sem = get_semaphore(0);
    const uint32_t g = get_write_ptr(cb_g);     // the gather page: same L1 address on every core of the program
    const uint32_t tab = get_write_ptr(cb_tab);
    if (core < T) {
        const auto table = TensorAccessor(t_args, get_common_arg_val<uint32_t>(6), TABLE_WORDS * 4);
        noc_async_read_set_trid(TRID_TAB);
        noc_async_read(get_noc_addr(0, table), tab, TABLE_WORDS * 4);
        noc_async_read_set_trid(0);
    }

    if (core < N_COL) {
        const uint32_t x_tile = get_tile_size(cb_x), page = RB * get_tile_size(cb_w2), o_tile = get_tile_size(cb_out);
        const auto lg = TensorAccessor(l_args, get_common_arg_val<uint32_t>(0), o_tile);
        const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(2), x_tile);
        const auto wr = TensorAccessor(w_args, get_common_arg_val<uint32_t>(3), page);
        const bool part = T <= 16;
        uint32_t xp;
        {
        MD_ZONE("RW-X+W2");
        BlockStream<NP> st;                           // k-tiles 40..79: the column's second half of pages, in flight
        for (uint32_t p = 0; p < NP; ++p) {           // while x lands (its own transaction id)
            st.issue(cb_w2, get_noc_addr(core * 2 * NP + NP + p, wr), page, RB, NP);
        }
        constexpr uint32_t TRID_X = NP + 1;
        noc_async_read_set_trid(TRID_X);
        cb_reserve_back(cb_x, KT);
        xp = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < KT; ++i) {
            const uint32_t k = (i + 9 * core) % KT;
            const uint64_t src = get_noc_addr(k, x);
            if (part) {
                noc_async_read(src, xp + k * x_tile, T * 64);
                noc_async_read(src + 1024, xp + k * x_tile + 1024, T * 64);
            } else {
                noc_async_read(src, xp + k * x_tile, x_tile);
            }
        }
        noc_async_read_barrier_with_trid(TRID_X);
        cb_push_back(cb_x, KT);
        st.drain();
        }
        {
        MD_ZONE("RW-XB");
        const auto xb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(5), get_tile_size(cb_xb));
        const uint32_t sp = get_write_ptr(cb_xb), rows = part ? T : 32;
        invalidate_l1_cache();
        for (uint32_t k = core, i = 0; k < KT; k += N_COL, ++i) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(xp + k * x_tile);
            uint32_t* dst = reinterpret_cast<uint32_t*>(sp + i * 2048);         // two BF16 values per word
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t h = 0; h < 2; ++h) {         // face (r / 16) * 2 + h: columns 16 h .. 16 h + 15
                    const uint32_t off = ((r >> 4) * 2 + h) * 256 + (r & 15) * 16;
                    const uint32_t* s = src + off;
                    uint32_t* d = dst + off / 2;
#pragma GCC unroll 8
                    for (uint32_t c = 0; c < 8; ++c) {
                        const uint32_t u0 = s[2 * c], u1 = s[2 * c + 1];
                        d[c] = ((u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16) |
                               ((u1 + 0x7fffu + ((u1 >> 16) & 1u)) & 0xffff0000u);
                    }
                }
            }
            const uint64_t d = get_noc_addr(k, xb);
            if (part) {
                noc_async_write(sp + i * 2048, d, T * 32);
                noc_async_write(sp + i * 2048 + 512, d + 512, T * 32);
            } else {
                noc_async_write(sp + i * 2048, d, 2048);
            }
        }
        }
        {
        MD_ZONE("RW-WAITMM");
        cb_wait_front(cb_out, 1);
        }
        const uint32_t op = get_read_ptr(cb_out);
        noc_async_write(op, get_noc_addr(core, lg), o_tile);
        for (uint32_t r = 0; r < T; ++r) {
            const uint32_t src = op + (r >> 4) * 2048 + (r & 15) * 64;     // faces (r / 16) * 2 + {0, 1}, row r % 16
            const uint32_t vx = get_common_arg_val<uint32_t>(8 + 2 * r), vy = get_common_arg_val<uint32_t>(9 + 2 * r);
            noc_async_write(src, get_noc_addr(vx, vy, g + core * 128), 64);
            noc_async_write(src + 1024, get_noc_addr(vx, vy, g + core * 128 + 64), 64);
        }
        noc_async_write_barrier();
        for (uint32_t r = 0; r < T; ++r) {
            const uint32_t vx = get_common_arg_val<uint32_t>(8 + 2 * r), vy = get_common_arg_val<uint32_t>(9 + 2 * r);
            noc_semaphore_inc(get_noc_addr(vx, vy, sem), 1);
        }
        cb_pop_front(cb_out, 1);
    }

    if (core < T) {
        const auto rt = TensorAccessor(r_args, get_common_arg_val<uint32_t>(1), ROUTE_WORDS * 4);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        {
            MD_ZONE("RW-GATHER");
            noc_semaphore_wait_min(s, N_COL);
        }
        MD_ZONE("RW-TOPK");
        invalidate_l1_cache();
        uint32_t* out = reinterpret_cast<uint32_t*>(g + N_COL * 128);
        route_row(reinterpret_cast<const uint32_t*>(g), out);
        noc_async_write(g + N_COL * 128, get_noc_addr(core, rt), ROUTE_WORDS * 4);
        noc_async_read_barrier_with_trid(TRID_TAB);
        const uint32_t* tw = reinterpret_cast<const uint32_t*>(tab);
        for (uint32_t k = 0; k <= TOPK; ++k) {
            const uint32_t* src = tw + TABLE_STRIDE * (k < TOPK ? out[R_IDS + k] : SHARED);
            for (uint32_t i = 0; i < TABLE_STRIDE; ++i) {
                out[ENT_TAB + TABLE_STRIDE * k + i] = src[i];
            }
        }
        const auto ep = TensorAccessor(e_args, get_common_arg_val<uint32_t>(7), ENT_WORDS * 4);
        noc_async_write(g + N_COL * 128, get_noc_addr(core, ep), ENT_WORDS * 4);
        noc_semaphore_set(s, 0);
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
