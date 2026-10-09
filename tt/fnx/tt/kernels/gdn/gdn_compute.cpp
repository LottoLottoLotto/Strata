// SPDX-License-Identifier: Apache-2.0
// Flash-Next fused GDN decode core (fnx.tt.gdn.GdnFused): one core per v-head h (key head h % 16), T tokens in
// sequence. FP32 throughout: elementwise math on the SFPU (exact FP32), contractions on the FPU (inputs rounded to
// ~10 mantissa bits, FP32 accumulation), FP32 recurrent state carried in L1 between the tokens of a call.
//
// Layouts (fp32 32x32 tiles; SFPU "slot" s of a tile = face s / 8, face rows 4 * ((s % 8) / 2) .. +3, even (s even)
// or odd columns, SFPU lane 8 i + c = slot row i):
//   rows tile: 128-wide chunks as single tile rows, chunk c at row 4 c (P: q chunks 0..3, k chunks 4..7;
//              V: v chunks 0..3), rows 4 c + 1 .. 4 c + 3 unspecified.
//   row tile:  a 128-vector as row 0 of 4 tiles (chunk R in tile R), every other row exactly 0.
// Per token: conv (4 taps) + SiLU on the P and V rows tiles; gate g = exp(A softplus(a + dt)) and beta = sigmoid(b)
// as scalars; q, k L2-normalized (q also by 1/sqrt(128)) and moved into row tiles; kv = k S; delta = (v - g kv) beta;
// S <- g S + k^T delta; o = q S; y = o rsqrt(mean o^2 + eps) * norm * sigmoid(z).  The scalars (the a/b words of the
// token tile, g and beta) reach the MATH thread, their only user, through the UNPACK -> MATH mailbox (math_word).
// Modes (runtime arg): OUT (forward: o and y; without it, rollback replay: only the state advances), SNAP (verify
// window: signal the writer once the state after row 0 is packed (c_s0) and wait for its snapshot copy before
// reusing that buffer at row 2).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#ifdef TRISC_MATH
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
using sfpi::dst_reg;
using sfpi::vFloat;
constexpr int TS = 32;  // sfpi rows per FP32 dest tile

sfpi_inline void recip_consts() { sfpi::vConstFloatPrgm0 = 2.0f; }
sfpi_inline void sqrt_consts() {
    sfpi::vConstIntPrgm0 = 0x5f1110a0;
    sfpi::vConstFloatPrgm1 = 2.2825186f;
    sfpi::vConstFloatPrgm2 = 2.2533049f;
}
sfpi_inline void log1p_consts() {
    sfpi::vConstFloatPrgm0 = 0.693147182f * 1.19209290e-7f;
    sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
    sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
}
sfpi_inline vFloat rsqrt_acc(vFloat x) { return _calculate_sqrt_body_<false, true, false>(x); }

// d0..d3 = tap inputs, d4..d7 = tap weights -> d0 = silu(sum_t in_t * w_t); SQ: d1 = d0^2
template <bool SQ>
inline void gdn_conv() {
    recip_consts();
#pragma GCC unroll 2
    for (int k = 0; k < 32; ++k) {
        vFloat s = dst_reg[k] * dst_reg[4 * TS + k];
        s = s + dst_reg[TS + k] * dst_reg[5 * TS + k];
        s = s + dst_reg[2 * TS + k] * dst_reg[6 * TS + k];
        s = s + dst_reg[3 * TS + k] * dst_reg[7 * TS + k];
        const vFloat h = s * _sfpu_sigmoid_<true>(s);
        dst_reg[k] = h;
        if constexpr (SQ) {
            dst_reg[TS + k] = h * h;
        }
    }
}

// d0 slot 0 = g = exp(A * softplus20(a_hi + a_lo + dt)), slot 1 = beta = sigmoid(b_hi + b_lo) (all lanes)
inline void gdn_gates(uint32_t a_hi, uint32_t a_lo, uint32_t b_hi, uint32_t b_lo, uint32_t dt, uint32_t A) {
    const vFloat a = vFloat(Converter::as_float(a_hi)) + vFloat(Converter::as_float(a_lo));
    const vFloat x = a + vFloat(Converter::as_float(dt));
    const vFloat e = _sfpu_exp_fp32_accurate_<false>(x);
    log1p_consts();
    vFloat sp = calculate_log1p_fp32<true>(e);
    v_if(x > 20.0f) { sp = x; }
    v_endif;
    const vFloat g = _sfpu_exp_fp32_accurate_<false>(sp * vFloat(Converter::as_float(A)));
    const vFloat b = vFloat(Converter::as_float(b_hi)) + vFloat(Converter::as_float(b_lo));
    recip_consts();
    const vFloat beta = _sfpu_sigmoid_<true>(b);
    dst_reg[0] = g;
    dst_reg[1] = beta;
}

// Zero lanes 8..31 (slot rows 1..3).
sfpi_inline vFloat row0(vFloat v) {
    v_if(sfpi::vConstTileId >= 16) { v = 0.0f; }
    v_endif;
    return v;
}

// Write row tile `tile` (all slots) from rows-tile slots base + 2R (+1, +8, +9) scaled by `scale`.
sfpi_inline void put_row_tile(int tile, int src, vFloat scale) {
    const vFloat zero = 0.0f;
#pragma GCC unroll 8
    for (int k = 0; k < 32; ++k) {
        dst_reg[tile * TS + k] = zero;
    }
    dst_reg[tile * TS + 0] = row0(dst_reg[src] * scale);
    dst_reg[tile * TS + 1] = row0(dst_reg[src + 1] * scale);
    dst_reg[tile * TS + 8] = row0(dst_reg[src + 8] * scale);
    dst_reg[tile * TS + 9] = row0(dst_reg[src + 9] * scale);
}

// d0 = row sums of P^2 (FPU), d1 = P rows tile -> d2..d5 = q / sqrt(128 (sum q^2 + eps)) row tiles; the k chunks
// of d1 are scaled in place by 1 / sqrt(sum k^2 + eps).
inline void gdn_normqk() {
    sqrt_consts();
    const vFloat q0 = dst_reg[0], q1 = dst_reg[2], q2 = dst_reg[4], q3 = dst_reg[6];
    const vFloat k0 = dst_reg[16], k1 = dst_reg[18], k2 = dst_reg[20], k3 = dst_reg[22];
    const vFloat nq = (q0 + q1) + (q2 + q3);
    const vFloat nk = (k0 + k1) + (k2 + k3);
    const vFloat iq = rsqrt_acc((nq + 1e-6f) * 128.0f);
    const vFloat ik = rsqrt_acc(nk + 1e-6f);
#pragma GCC unroll 1
    for (int r = 0; r < 4; ++r) {
        put_row_tile(2 + r, TS + 2 * r, iq);
    }
#pragma GCC unroll 1
    for (int r = 0; r < 4; ++r) {
        const int s = TS + 16 + 2 * r;
        dst_reg[s] = dst_reg[s] * ik;
        dst_reg[s + 1] = dst_reg[s + 1] * ik;
        dst_reg[s + 8] = dst_reg[s + 8] * ik;
        dst_reg[s + 9] = dst_reg[s + 9] * ik;
    }
}

// d0 = rows tile -> d1..d4 = row tiles of the chunks at slot base BASE (0: rows 0..12, 16: rows 16..28)
template <int BASE>
inline void gdn_relayout() {
    const vFloat one = 1.0f;
#pragma GCC unroll 1
    for (int r = 0; r < 4; ++r) {
        put_row_tile(1 + r, BASE + 2 * r, one);
    }
}

// d0..d3 = kv, d4..d7 = v -> d4..d7 = (v - g kv) beta (row 0 slots; the other slots are 0 in both)
inline void gdn_delta(uint32_t gb, uint32_t bb) {
    const vFloat g = Converter::as_float(gb);
    const vFloat beta = Converter::as_float(bb);
#pragma GCC unroll 1
    for (int c = 0; c < 4; ++c) {
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const int s = (j & 1) + 8 * (j >> 1);
            const vFloat kv = dst_reg[c * TS + s];
            const vFloat v = dst_reg[(4 + c) * TS + s];
            dst_reg[(4 + c) * TS + s] = (v - g * kv) * beta;
        }
    }
}

// d0..d(n-1) *= g
template <int N>
inline void gdn_scale(uint32_t gb) {
    const vFloat g = Converter::as_float(gb);
#pragma GCC unroll 8
    for (int k = 0; k < N * TS; ++k) {
        dst_reg[k] = dst_reg[k] * g;
    }
}

// d0..d3 = o row tiles -> d4: slot 0 = sum over tiles and row-0 slots of o^2, every other slot 0
inline void gdn_osq() {
    vFloat acc = 0.0f;
#pragma GCC unroll 1
    for (int c = 0; c < 4; ++c) {
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const int s = (j & 1) + 8 * (j >> 1);
            const vFloat o = dst_reg[c * TS + s];
            acc = acc + o * o;
        }
    }
    const vFloat zero = 0.0f;
#pragma GCC unroll 8
    for (int k = 1; k < 32; ++k) {
        dst_reg[4 * TS + k] = zero;
    }
    dst_reg[4 * TS] = acc;
}

// d0 = row sums of the o^2 tile (row 0 = sum o^2), d1 = o, d2 = z, d3 = norm -> d(4 + c) row 0 = y
inline void gdn_y(int c) {
    sqrt_consts();
    const vFloat inv = rsqrt_acc(dst_reg[0] * 0.0078125f + 1e-6f);
    recip_consts();
#pragma GCC unroll 4
    for (int j = 0; j < 4; ++j) {
        const int s = (j & 1) + 8 * (j >> 1);
        const vFloat z = dst_reg[2 * TS + s];
        const vFloat y = dst_reg[TS + s] * inv;
        dst_reg[(4 + c) * TS + s] = (y * dst_reg[3 * TS + s]) * _sfpu_sigmoid_<true>(z);
    }
}
}  // namespace ckernel::sfpu
#define SFPU(...) MATH((_llk_math_eltwise_unary_sfpu_params_(__VA_ARGS__)))
#else
#define SFPU(...)
#endif

namespace ckernel {
// Word i of the front tile of `cb` for the MATH thread, the only one that uses these scalars: UNPACK reads it from
// L1 and hands it over through the UNPACK -> MATH mailbox, MATH consumes it at once.  Not read_tile_value: that also
// hands the word to PACK, which leaves its mailbox loads unconsumed, so on the Blackhole baby RISC-V two of them can
// be in flight at once; the second, stalled in front of the mailbox until UNPACK pushes, blocks PACK's queued
// Tensix instruction pushes behind it (tt-isa-documentation BabyRISCV/MemoryOrdering.md: loads overtake queued
// stores; a pending FIFO read stalls later requests to its region).  For the c_gb words those pushes are the pack of
// c_gb itself, which UNPACK waits for before it writes them: a deadlock that depended on the threads' timing.
ALWI uint32_t math_word(uint32_t cb_id, uint32_t element_offset) {
    uint32_t value = 0;
    UNPACK({
        value = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(cb_id), 0))[element_offset];
        mailbox_write(ckernel::ThreadId::MathThreadId, value);
    })
    MATH({
        volatile uint32_t got = mailbox_read(ckernel::ThreadId::UnpackThreadId);  // consumed: the RISC waits here
        value = got;
    })
    return value;
}
}  // namespace ckernel

namespace {
constexpr uint32_t c_ring = 0, c_w = 1, c_tok = 2, c_z = 3, c_norm = 4, c_ones = 5, c_s0 = 6, c_s0m = 7;
constexpr uint32_t c_s1 = 8, c_s1m = 9, c_hp = 10, c_hv = 11, c_sq = 12, c_q = 13, c_k = 14, c_v = 15;
constexpr uint32_t c_kn = 16, c_gb = 17, c_kv = 18, c_delta = 19, c_kcol = 20, c_o = 21, c_osq = 22, c_y = 23;
constexpr uint32_t c_sdone = 24, c_si = 25, c_sim = 26, c_s0ready = 27, c_snapok = 28;
constexpr uint32_t OUT = 1, SNAP = 2;

inline void sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>())); }

inline void pack_out(uint32_t first, uint32_t n, uint32_t cb) {
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(first + i, cb);
    }
}

inline void copy_n(uint32_t cb, uint32_t first, uint32_t n, uint32_t dst) {
    copy_tile_init(cb);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb, first + i, dst + i);
    }
}
}  // namespace

void kernel_main() {
    const uint32_t T = get_arg_val<uint32_t>(0);
    const uint32_t dt_bits = get_arg_val<uint32_t>(1);
    const uint32_t a_bits = get_arg_val<uint32_t>(2);
    const uint32_t mode = get_arg_val<uint32_t>(3);
    const bool out = mode & OUT, snap = (mode & SNAP) && T > 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_k, c_sim, c_kv);
    cb_wait_front(c_w, 8);
    cb_wait_front(c_norm, 4);
    cb_wait_front(c_ones, 1);
    cb_wait_front(c_ring, 8);

    for (uint32_t t = 0; t < T; ++t) {
        // state buffers: token 0 reads the loaded state (c_si), then A (c_s0) / B (c_s1) alternate; each CB index
        // has a single producer (c_si: writer kernel; A, B: pack), c_si shares B's memory (disjoint lifetimes)
        const uint32_t cur = t == 0 ? c_si : ((t & 1) ? c_s0 : c_s1);
        const uint32_t nxt = (t & 1) ? c_s1 : c_s0;
        cb_wait_front(c_tok, 1);
        const uint32_t a_hi = math_word(c_tok, 0);
        const uint32_t a_lo = math_word(c_tok, 1);
        const uint32_t b_hi = math_word(c_tok, 2);
        const uint32_t b_lo = math_word(c_tok, 3);

        // conv + SiLU: P (q, k) with its squares, then V
        for (uint32_t part = 0; part < 2; ++part) {
            tile_regs_acquire();
            copy_tile_init(c_ring);
            for (uint32_t tap = 0; tap < 4; ++tap) {
                copy_tile(c_ring, 2 * ((t + tap) & 3) + part, tap);
            }
            copy_tile_init(c_w);
            for (uint32_t tap = 0; tap < 4; ++tap) {
                copy_tile(c_w, 2 * tap + part, 4 + tap);
            }
            sfpu_init();
            if (part == 0) {
                SFPU(ckernel::sfpu::gdn_conv<true>, 0, VectorMode::None);
            } else {
                SFPU(ckernel::sfpu::gdn_conv<false>, 0, VectorMode::None);
            }
            tile_regs_commit();
            if (part == 0) {
                cb_reserve_back(c_hp, 1);
                cb_reserve_back(c_sq, 1);
                tile_regs_wait();
                pack_tile(0, c_hp);
                pack_tile(1, c_sq);
                tile_regs_release();
                cb_push_back(c_hp, 1);
                cb_push_back(c_sq, 1);
            } else {
                cb_reserve_back(c_hv, 1);
                tile_regs_wait();
                pack_tile(0, c_hv);
                tile_regs_release();
                cb_push_back(c_hv, 1);
            }
        }
        cb_pop_front(c_tok, 1);  // the reader may now replace ring element t with element t + 4

        // gates
        tile_regs_acquire();
        sfpu_init();
        SFPU(ckernel::sfpu::gdn_gates, 0, VectorMode::None, a_hi, a_lo, b_hi, b_lo, dt_bits, a_bits);
        tile_regs_commit();
        cb_reserve_back(c_gb, 1);
        tile_regs_wait();
        pack_tile(0, c_gb);
        tile_regs_release();
        cb_push_back(c_gb, 1);
        cb_wait_front(c_gb, 1);
        const uint32_t g_bits = math_word(c_gb, 0);
        const uint32_t beta_bits = math_word(c_gb, 1);
        cb_pop_front(c_gb, 1);

        // q, k L2 norms; q row tiles
        cb_wait_front(c_sq, 1);
        cb_wait_front(c_hp, 1);
        tile_regs_acquire();
        matmul_init(c_sq, c_ones);
        matmul_tiles(c_sq, c_ones, 0, 0, 0);
        copy_n(c_hp, 0, 1, 1);
        sfpu_init();
        SFPU(ckernel::sfpu::gdn_normqk, 0, VectorMode::None);
        tile_regs_commit();
        cb_reserve_back(c_q, 4);
        cb_reserve_back(c_kn, 1);
        tile_regs_wait();
        pack_out(2, 4, c_q);
        pack_tile(1, c_kn);
        tile_regs_release();
        cb_push_back(c_q, 4);
        cb_push_back(c_kn, 1);
        cb_pop_front(c_sq, 1);
        cb_pop_front(c_hp, 1);

        // k and v row tiles
        for (uint32_t part = 0; part < 2; ++part) {
            const uint32_t src = part == 0 ? c_kn : c_hv;
            const uint32_t dst = part == 0 ? c_k : c_v;
            cb_wait_front(src, 1);
            tile_regs_acquire();
            copy_n(src, 0, 1, 0);
            sfpu_init();
            if (part == 0) {
                SFPU(ckernel::sfpu::gdn_relayout<16>, 0, VectorMode::None);
            } else {
                SFPU(ckernel::sfpu::gdn_relayout<0>, 0, VectorMode::None);
            }
            tile_regs_commit();
            cb_reserve_back(dst, 4);
            tile_regs_wait();
            pack_out(1, 4, dst);
            tile_regs_release();
            cb_push_back(dst, 4);
            cb_pop_front(src, 1);
        }

        // kv = k S (d0..d3), k^T (d4..d7)
        const uint32_t cur_m = cur + 1, nxt_m = nxt + 1;
        cb_wait_front(c_k, 4);
        cb_wait_front(cur, 16);
        cb_wait_front(cur_m, 16);
        tile_regs_acquire();
        matmul_init(c_k, cur_m);
        for (uint32_t c = 0; c < 4; ++c) {
            for (uint32_t r = 0; r < 4; ++r) {
                matmul_tiles(c_k, cur_m, r, 4 * r + c, c);
            }
        }
        transpose_init(c_k);
        for (uint32_t r = 0; r < 4; ++r) {
            transpose_tile(c_k, r, 4 + r);
        }
        tile_regs_commit();
        cb_reserve_back(c_kv, 4);
        cb_reserve_back(c_kcol, 4);
        tile_regs_wait();
        pack_out(0, 4, c_kv);
        pack_out(4, 4, c_kcol);
        tile_regs_release();
        cb_push_back(c_kv, 4);
        cb_push_back(c_kcol, 4);

        // delta = (v - g kv) beta
        cb_wait_front(c_kv, 4);
        cb_wait_front(c_v, 4);
        tile_regs_acquire();
        copy_n(c_kv, 0, 4, 0);
        copy_n(c_v, 0, 4, 4);
        sfpu_init();
        SFPU(ckernel::sfpu::gdn_delta, 0, VectorMode::None, g_bits, beta_bits);
        tile_regs_commit();
        cb_reserve_back(c_delta, 4);
        tile_regs_wait();
        pack_out(4, 4, c_delta);
        tile_regs_release();
        cb_push_back(c_delta, 4);
        cb_pop_front(c_kv, 4);
        cb_pop_front(c_v, 4);

        // S <- g S + k^T delta (FP32 S in DEST: exact scaling, FPU outer product accumulated in FP32)
        if (snap && t == 2) {
            cb_wait_front(c_snapok, 1);  // the writer's copy of the state after row 0 (in c_s0) is done
            cb_pop_front(c_snapok, 1);
        }
        cb_wait_front(c_delta, 4);
        cb_wait_front(c_kcol, 4);
        cb_reserve_back(nxt, 16);
        for (uint32_t half = 0; half < 2; ++half) {
            tile_regs_acquire();
            copy_n(cur, 8 * half, 8, 0);
            sfpu_init();
            SFPU(ckernel::sfpu::gdn_scale<8>, 0, VectorMode::None, g_bits);
            matmul_init(c_kcol, c_delta);
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t tile = 8 * half + i;
                matmul_tiles(c_kcol, c_delta, tile >> 2, tile & 3, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_out(0, 8, nxt);
            tile_regs_release();
        }
        cb_push_back(nxt, 16);
        cb_reserve_back(nxt_m, 16);
        cb_push_back(nxt_m, 16);
        cb_pop_front(cur, 16);
        cb_pop_front(cur_m, 16);
        cb_pop_front(c_delta, 4);
        cb_pop_front(c_kcol, 4);
        if (snap && t == 0) {
            cb_reserve_back(c_s0ready, 1);
            cb_push_back(c_s0ready, 1);
        }
        if (t + 1 == T) {
            cb_reserve_back(c_sdone, 1);
            cb_push_back(c_sdone, 1);
        }
        if (!out) {
            cb_wait_front(c_q, 4);
            cb_pop_front(c_q, 4);
            cb_pop_front(c_k, 4);
            continue;
        }

        // o = q S
        cb_wait_front(nxt_m, 16);
        cb_wait_front(c_q, 4);
        tile_regs_acquire();
        matmul_init(c_q, nxt_m);
        for (uint32_t c = 0; c < 4; ++c) {
            for (uint32_t r = 0; r < 4; ++r) {
                matmul_tiles(c_q, nxt_m, r, 4 * r + c, c);
            }
        }
        tile_regs_commit();
        cb_reserve_back(c_o, 4);
        tile_regs_wait();
        pack_out(0, 4, c_o);
        tile_regs_release();
        cb_push_back(c_o, 4);
        cb_pop_front(c_q, 4);
        cb_pop_front(c_k, 4);

        // y = o rsqrt(mean o^2 + eps) norm sigmoid(z)
        cb_wait_front(c_o, 4);
        cb_wait_front(c_z, 4);
        tile_regs_acquire();
        copy_n(c_o, 0, 4, 0);
        sfpu_init();
        SFPU(ckernel::sfpu::gdn_osq, 0, VectorMode::None);
        tile_regs_commit();
        cb_reserve_back(c_osq, 1);
        tile_regs_wait();
        pack_tile(4, c_osq);
        tile_regs_release();
        cb_push_back(c_osq, 1);
        cb_wait_front(c_osq, 1);
        tile_regs_acquire();
        matmul_init(c_osq, c_ones);
        matmul_tiles(c_osq, c_ones, 0, 0, 0);
        for (uint32_t c = 0; c < 4; ++c) {
            copy_n(c_o, c, 1, 1);
            copy_n(c_z, c, 1, 2);
            copy_n(c_norm, c, 1, 3);
            sfpu_init();
            SFPU(ckernel::sfpu::gdn_y, 0, VectorMode::None, (int)c);
        }
        tile_regs_commit();
        cb_reserve_back(c_y, 4);
        tile_regs_wait();
        pack_out(4, 4, c_y);
        tile_regs_release();
        cb_push_back(c_y, 4);
        cb_pop_front(c_osq, 1);
        cb_pop_front(c_o, 4);
        cb_pop_front(c_z, 4);
    }
}
