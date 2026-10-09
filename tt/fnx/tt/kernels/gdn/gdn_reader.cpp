// SPDX-License-Identifier: Apache-2.0
// Fused GDN reader (NCRISC), one v-head h per core (see gdn_compute.cpp for the layouts).
// Setup: the head's parameter page and conv-history page (hist_in) in one DRAM read each, staged in the z buffer
// (unused before token 0) and scattered by local reads: conv weights (4 taps x P/V rows tiles), ssm_norm row tiles,
// the 3 conv-history elements into the element ring (element e in ring slot e % 4, P tile then V tile); a ones tile.  Per token t: x_t (raw
// q/k/v of key head h % 16 and v-head h) into ring slot (t + 3) % 4 and the raw a_hi/a_lo/b_hi/b_lo scalars into
// words 0..3 of the token tile, read from the mega projection (forward) or from the saved window rows (REPLAY); OUT:
// z_t into row tiles.  SNAP (verify window): the conv history after row 0 goes to snap_hist and rows 1.. (raw x and
// the a/b words, as read) to the rows buffer.  End: the last three window elements to hist_out.
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t c_ring = 0, c_w = 1, c_tok = 2, c_z = 3, c_norm = 4, c_ones = 5;
constexpr uint32_t TB = 4096;                 // fp32 tile bytes
constexpr uint32_t ROWS = 12;                 // compact rows per element: q0..3, k0..3, v0..3 (32 floats each)
constexpr uint32_t PAR_BYTES = 4 * ROWS * 128 + 4 * 128;
constexpr uint32_t HIST_BYTES = 3 * ROWS * 128;
constexpr uint32_t ROW_BYTES = ROWS * 128 + 64;   // saved window row: 12 compact rows + the 4 a/b words (padded)
constexpr uint32_t HV = 48;
constexpr uint32_t Z0 = 320, AB0 = 512;       // mega tile columns of z and of the a/b hi/lo block
constexpr uint32_t OUT = 1, SNAP = 2, REPLAY = 4;

// byte offset of (row, column half) in an fp32 tile
inline uint32_t row_off(uint32_t row, uint32_t half) { return ((row >> 4) * 2 + half) * 1024 + (row & 15) * 64; }
// compact row i -> byte offset of its tile (0 = P, 1 = V) and row
inline uint32_t crow(uint32_t i, uint32_t half) {
    return i < 8 ? row_off(4 * i, half) : TB + row_off(4 * (i - 8), half);
}
}  // namespace

void kernel_main() {
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t mega_addr = get_arg_val<uint32_t>(1);
    const uint32_t par_addr = get_arg_val<uint32_t>(2);
    const uint32_t hist_in_addr = get_arg_val<uint32_t>(3);
    const uint32_t hist_out_addr = get_arg_val<uint32_t>(4);
    const uint32_t rows_addr = get_arg_val<uint32_t>(5);
    const uint32_t snaph_addr = get_arg_val<uint32_t>(6);
    const uint32_t T = get_arg_val<uint32_t>(7);
    const uint32_t mode = get_arg_val<uint32_t>(8);
    constexpr auto mega_args = TensorAccessorArgs<0>();
    constexpr auto par_args = TensorAccessorArgs<mega_args.next_compile_time_args_offset()>();
    constexpr auto hist_args = TensorAccessorArgs<par_args.next_compile_time_args_offset()>();
    constexpr auto rows_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
    const auto mega = TensorAccessor(mega_args, mega_addr, TB);
    const auto par = TensorAccessor(par_args, par_addr, PAR_BYTES);
    const auto hist_in = TensorAccessor(hist_args, hist_in_addr, HIST_BYTES);
    const auto hist_out = TensorAccessor(hist_args, hist_out_addr, HIST_BYTES);
    const auto snaph = TensorAccessor(hist_args, snaph_addr, HIST_BYTES);
    const auto rows = TensorAccessor(rows_args, rows_addr, ROW_BYTES);
    const uint32_t kh = h & 15;
    const bool out = mode & OUT, keep = (mode & SNAP) && T > 1, replay = mode & REPLAY;

    cb_reserve_back(c_w, 8);
    cb_reserve_back(c_norm, 4);
    cb_reserve_back(c_ones, 1);
    cb_reserve_back(c_ring, 8);
    const uint32_t w = get_write_ptr(c_w);
    const uint32_t nrm = get_write_ptr(c_norm);
    const uint32_t ring = get_write_ptr(c_ring);
    const uint32_t stage = get_write_ptr(c_z);
    noc_async_read(par.get_noc_addr(h), stage, PAR_BYTES);
    noc_async_read(hist_in.get_noc_addr(h), stage + PAR_BYTES, HIST_BYTES);
    volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(c_ones));
    for (uint32_t i = 0; i < 1024; ++i) {
        ones[i] = 0x3f800000u;
    }
    noc_async_read_barrier();
    const uint64_t pbase = get_noc_addr(stage);
    const uint64_t hbase = get_noc_addr(stage + PAR_BYTES);
    for (uint32_t tap = 0; tap < 4; ++tap) {
        for (uint32_t i = 0; i < ROWS; ++i) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(pbase + (tap * ROWS + i) * 128 + half * 64, w + 2 * tap * TB + crow(i, half), 64);
            }
        }
    }
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t half = 0; half < 2; ++half) {
            noc_async_read(pbase + 4 * ROWS * 128 + r * 128 + half * 64, nrm + r * TB + row_off(0, half), 64);
        }
    }
    for (uint32_t e = 0; e < 3; ++e) {
        for (uint32_t i = 0; i < ROWS; ++i) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(hbase + (e * ROWS + i) * 128 + half * 64, ring + 2 * e * TB + crow(i, half), 64);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(c_w, 8);
    cb_push_back(c_norm, 4);
    cb_push_back(c_ones, 1);
    cb_push_back(c_ring, 8);

    // the last three window elements after n tokens (ring slots (n + e) % 4) -> history page `dst`
    auto put_hist = [&](uint64_t dst, uint32_t n) {
        for (uint32_t e = 0; e < 3; ++e) {
            const uint32_t el = ring + 2 * ((n + e) & 3) * TB;
            for (uint32_t i = 0; i < ROWS; ++i) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(el + crow(i, half), dst + (e * ROWS + i) * 128 + half * 64, 64);
                }
            }
        }
    };

    for (uint32_t t = 0; t < T; ++t) {
        cb_reserve_back(c_tok, 1);  // the compute is done with ring element t - 1 (slot (t + 3) % 4)
        if (keep) {
            noc_async_write_barrier();  // the saves of row t - 1 (its ring slot, the token tile) have left
        }
        const uint32_t el = ring + 2 * ((t + 3) & 3) * TB;
        const uint32_t tok = get_write_ptr(c_tok);
        if (replay) {
            const uint64_t src = rows.get_noc_addr(t * HV + h);
            for (uint32_t i = 0; i < ROWS; ++i) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_read(src + i * 128 + half * 64, el + crow(i, half), 64);
                }
            }
            noc_async_read(src + ROWS * 128, tok, 64);
            noc_async_read_barrier();
        } else {
            uint32_t zt = 0;
            if (out) {
                cb_reserve_back(c_z, 4);
                zt = get_write_ptr(c_z);
            }
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t src = row_off(t, half);
                for (uint32_t r = 0; r < 4; ++r) {
                    noc_async_read(mega.get_noc_addr(kh * 4 + r, src), el + crow(r, half), 64);
                    noc_async_read(mega.get_noc_addr(64 + kh * 4 + r, src), el + crow(4 + r, half), 64);
                    noc_async_read(mega.get_noc_addr(128 + h * 4 + r, src), el + crow(8 + r, half), 64);
                    if (out) {
                        noc_async_read(mega.get_noc_addr(Z0 + h * 4 + r, src), zt + r * TB + row_off(0, half), 64);
                    }
                }
            }
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t col = 64 * j + h;
                noc_async_read(mega.get_noc_addr(AB0 + col / 32, row_off(t, (col % 32) / 16)), tok + 2048 + 64 * j, 64);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok);
            for (uint32_t j = 0; j < 4; ++j) {
                words[j] = words[(2048 + 64 * j) / 4 + (h % 16)];
            }
            if (keep) {
                if (t == 0) {
                    put_hist(snaph.get_noc_addr(h), 1);   // the conv history after row 0
                } else {
                    const uint64_t dst = rows.get_noc_addr((t - 1) * HV + h);
                    for (uint32_t i = 0; i < ROWS; ++i) {
                        for (uint32_t half = 0; half < 2; ++half) {
                            noc_async_write(el + crow(i, half), dst + i * 128 + half * 64, 64);
                        }
                    }
                    noc_async_write(tok, dst + ROWS * 128, 64);
                }
            }
            if (out) {
                cb_push_back(c_z, 4);
            }
        }
        cb_push_back(c_tok, 1);
    }

    put_hist(hist_out.get_noc_addr(h), T);
    noc_async_write_barrier();
}
