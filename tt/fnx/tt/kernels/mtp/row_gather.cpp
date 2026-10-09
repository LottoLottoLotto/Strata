// MTP drafter (fnx.tt.mtp.EmbedRows): the token embedding row (IQ4_XS, 10 blocks x 136 B) of a draft-vocabulary
// index from pinned host memory, dequantized exactly to float32, at the index a device argmax wrote: a draft chain
// step embeds the previous step's draft without a host round trip.  One core, one data-movement RISC (integer code).
// Row i (ROW bytes: the 1360 B IQ4_XS row, zero-padded to a 64 B multiple) sits at the 64-bit bus address in words
// (2 (i % 8), 2 (i % 8) + 1) of page i / 8 of the address tensor (uint32 ROW_MAJOR, 64 B pages: lo, hi per row).
// Common args: [0] index tensor (uint32 ROW_MAJOR, word 0 = the row), [1] address tensor, [2] out (float32
// ROW_MAJOR [1, 1, 1, 2560], one page).  An index >= N_ROWS (never written by an argmax over N_ROWS columns) reads
// row N_ROWS - 1.
// Dequantization (ggml dequantize_row_iq4_xs, as fnx.ref.embed.iq4xs_rows): value = kv * ((ls - 32) * d), d the
// block's fp16 scale, ls its 6-bit sub-block scale, kv the IQ4_NL code value; |kv (ls - 32)| <= 4064 (12 bits) times
// d's 11-bit mantissa fits 24 bits, so every value is exact in float32 and is built here from integers.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../hostmem/hostmem.h"

constexpr uint32_t cb_buf = 0;
constexpr uint32_t N = 2560, QK = 256, BLOCK = 136;
constexpr int32_t KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// float32 bits of m * d exactly (m: a nonzero-or-zero integer |m| <= 4064, d: fp16 bits); signed zeros as numpy
// (the sign of kv * ((ls - 32) * d) with (ls - 32) = 0 read as +0.0: sign(kv) ^ sign(d))
inline uint32_t exact(int32_t kv, int32_t s, uint32_t d16) {
    const uint32_t sign = ((kv < 0) ^ (s < 0) ^ (d16 >> 15)) << 31;
    const uint32_t e = (d16 >> 10) & 0x1f, man = d16 & 0x3ff;
    const uint32_t D = e ? (man | 0x400) : man;                 // d = D * 2^(E), E = e - 25 (subnormal: -24)
    const int32_t E = e ? int32_t(e) - 25 : -24;
    const uint32_t P = uint32_t(kv < 0 ? -kv : kv) * uint32_t(s < 0 ? -s : s) * D;
    if (P == 0) {
        return sign;
    }
    const int32_t n = 31 - __builtin_clz(P);                    // P < 2^23
    return sign | uint32_t(n + E + 127) << 23 | ((P << (23 - n)) & 0x7fffff);
}

void kernel_main() {
    constexpr uint32_t ROW = get_named_compile_time_arg_val("row");               // bytes (64 B multiple)
    constexpr uint32_t N_ROWS = get_named_compile_time_arg_val("n_rows");
    constexpr auto i_args = TensorAccessorArgs<0>();
    constexpr auto a_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto idx = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0), 64);
    const auto addrs = TensorAccessor(a_args, get_common_arg_val<uint32_t>(1), 64);
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), N * 4);

    cb_reserve_back(cb_buf, 1);
    const uint32_t base = (get_write_ptr(cb_buf) + 63) & ~63u;      // 64 B aligned: DRAM and PCIe reads
    const uint32_t iw = base, aw = base + 64, raw = base + 128, row = raw + ROW;
    noc_async_read(idx.get_noc_addr(0), iw, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t i = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iw)[0];
    if (i >= N_ROWS) {
        i = N_ROWS - 1;
    }
    noc_async_read(addrs.get_noc_addr(i / 8), aw, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aw) + 2 * (i % 8);
    hostmem_read(hostmem_addr(a[0], a[1]), raw, ROW);
    noc_async_read_barrier();
    invalidate_l1_cache();

    const volatile tt_l1_ptr uint8_t* b = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(raw);
    volatile tt_l1_ptr uint32_t* y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
    for (uint32_t blk = 0; blk < N / QK; ++blk, b += BLOCK, y += QK) {
        const uint32_t d16 = b[0] | (uint32_t(b[1]) << 8);
        const uint32_t sh = b[2] | (uint32_t(b[3]) << 8);
        for (uint32_t ib = 0; ib < QK / 32; ++ib) {
            const int32_t ls = int32_t(((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4));
            const int32_t s = ls - 32;
            const volatile tt_l1_ptr uint8_t* qs = b + 8 + 16 * ib;
            for (uint32_t j = 0; j < 16; ++j) {
                const uint32_t q = qs[j];
                y[32 * ib + j] = exact(KV[q & 0xf], s, d16);
                y[32 * ib + 16 + j] = exact(KV[q >> 4], s, d16);
            }
        }
    }
    noc_async_write(row, out.get_noc_addr(0), N * 4);
    noc_async_write_barrier();
}
