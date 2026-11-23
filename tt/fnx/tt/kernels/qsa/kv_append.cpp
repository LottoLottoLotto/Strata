// Fused QSA KV append (fnx.tt.qsa.kv_append): one core reads the T rows of the roped keys kr (float32 TILE
// [1, T, 2, 256]) and of the values (float32 TILE qkv [1, 1, T, 32 * QKV_WT], column tiles V0T .. V0T + 16), rounds
// them to bfloat16 exactly as ttnn.typecast(float32 -> bfloat16) does (round to nearest, ties to even), and writes row
// (head j, token t) to row j * cells + p0 + t of the row-major bfloat16 caches K and V [1, 1, 2 * cells, 256].
// p0 = word 0 of the uint32 position page when its address is nonzero (a captured trace writes where the refreshed
// page says), else the runtime argument (eager calls).  The same bytes as typecast + reshape/permute + to_layout +
// kv_write / slice_write of each tensor.  Tokens are processed in groups of G (bounded L1 staging).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_buf = 0;
constexpr uint32_t TB = 4096;                  // float32 tile bytes
constexpr uint32_t D = 256, ROW = D * 2;       // head dim, bfloat16 row bytes
constexpr uint32_t HK = 2;                     // kv heads
constexpr uint32_t G = 8;                      // tokens per group

// byte offset of (row, column half) in a float32 tile
inline uint32_t row_off(uint32_t row, uint32_t half) { return ((row >> 4) * 2 + half) * 1024 + (row & 15) * 64; }

inline uint16_t bf16_rne(uint32_t u) {
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) {
        return static_cast<uint16_t>((u >> 16) | 0x40u);               // NaN stays NaN
    }
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t POS_PAGE = get_named_compile_time_arg_val("pos_page");
    constexpr uint32_t QKV_WT = get_named_compile_time_arg_val("qkv_wt");
    constexpr uint32_t V0T = get_named_compile_time_arg_val("v0t");
    constexpr auto k_args = TensorAccessorArgs<0>();
    constexpr auto q_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto kc_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    const auto kr = TensorAccessor(k_args, get_common_arg_val<uint32_t>(0), TB);
    const auto qkv = TensorAccessor(q_args, get_common_arg_val<uint32_t>(1), TB);
    const auto kc = TensorAccessor(kc_args, get_common_arg_val<uint32_t>(2), ROW);
    const auto vc = TensorAccessor(vc_args, get_common_arg_val<uint32_t>(3), ROW);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t T = get_common_arg_val<uint32_t>(5);
    const uint32_t cells = get_common_arg_val<uint32_t>(6);
    uint32_t p0 = get_common_arg_val<uint32_t>(7);

    cb_reserve_back(cb_buf, 1);
    const uint32_t l1 = get_write_ptr(cb_buf);
    if (pos_addr != 0) {
        const auto pos = TensorAccessor(p_args, pos_addr, POS_PAGE);
        noc_async_read(pos.get_noc_addr(0), l1, POS_PAGE);
        noc_async_read_barrier();
        p0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1)[0];
    }
    // staging per group: float32 rows [m = k/v][j][g][256], then the same rows as bfloat16
    const uint32_t f32_base = l1 + POS_PAGE;
    const uint32_t b16_base = f32_base + 2 * HK * G * D * 4;
    for (uint32_t t0 = 0; t0 < T; t0 += G) {
        const uint32_t g = T - t0 < G ? T - t0 : G;
        for (uint32_t i = 0; i < g; ++i) {
            const uint32_t t = t0 + i;
            for (uint32_t j = 0; j < HK; ++j) {
                const uint32_t kdst = f32_base + ((0 * HK + j) * G + i) * D * 4;
                const uint32_t vdst = f32_base + ((1 * HK + j) * G + i) * D * 4;
                for (uint32_t c = 0; c < D / 32; ++c) {
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t off = (c * 32 + half * 16) * 4;
                        noc_async_read(kr.get_noc_addr(t * (D / 32) + c, row_off(j, half)), kdst + off, 64);
                        noc_async_read(
                            qkv.get_noc_addr((t >> 5) * QKV_WT + V0T + j * (D / 32) + c, row_off(t & 31, half)),
                            vdst + off, 64);
                    }
                }
            }
        }
        noc_async_read_barrier();
        noc_async_write_barrier();             // the previous group's writes read the bfloat16 staging
        for (uint32_t m = 0; m < 2; ++m) {
            for (uint32_t j = 0; j < HK; ++j) {
                const uint32_t r0 = (m * HK + j) * G;
                const volatile tt_l1_ptr uint32_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(f32_base + r0 * D * 4);
                volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b16_base + r0 * ROW);
                for (uint32_t e = 0; e < g * D; ++e) {
                    dst[e] = bf16_rne(src[e]);
                }
            }
        }
        for (uint32_t j = 0; j < HK; ++j) {
            for (uint32_t i = 0; i < g; ++i) {
                noc_async_write(b16_base + ((0 * HK + j) * G + i) * ROW, kc.get_noc_addr(j * cells + p0 + t0 + i), ROW);
                noc_async_write(b16_base + ((1 * HK + j) * G + i) * ROW, vc.get_noc_addr(j * cells + p0 + t0 + i), ROW);
            }
        }
    }
    noc_async_write_barrier();
}
