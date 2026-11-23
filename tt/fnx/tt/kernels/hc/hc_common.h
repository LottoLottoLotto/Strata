// HC fused kernels: circular buffers, semaphores and argument layout shared by the read / write programs.
//
// HcRead program (one launch per 32-row tile `rt` of R), logical worker grid 11 x 10:
//   MAIN (x, s), s < 8 : PREP K-tiles 40s + x + 11i of slice s (stream s/2), DOWN partial of N-tile x (x < NN),
//                        MIX of output column j = 10s + x (x < 10); (GATHER_X, s) gathers and multicasts the row's
//                        PREP output (the slice's hi/lo activations)
//   SSQ  (s, 8), s < 8 : per-row sum of squares of slice s
//   RS   (8, 8)        : rs_c = rsqrt(sum_c + N*eps) for the 4 streams; then the LO hub (silu tiles -> MIX cores)
//   RED  (n, 9), n < NN: p_n = sum_c rs_c * (P_{2c,n} + P_{2c+1,n}); n < 10: silu split for MIX; n == 10: inject
#pragma once

#include <cstdint>

namespace hc {

constexpr uint32_t F32_TILE = 4096;
constexpr uint32_t BF16_TILE = 2048;

// circular buffers (the same indices / addresses on every core of the program)
constexpr uint32_t CB_R = 0;       // f32  4   R tiles (PREP) / hc_write: R tiles
constexpr uint32_t CB_GS = 1;      // f32  4   gamma*sqrt(N) tiles (PREP) / hc_write: bo tile
constexpr uint32_t CB_Y = 2;       // f32  4   y = R*gs (PREP -> MIX)
constexpr uint32_t CB_HL = 3;      // bf16 8   hi/lo of y (PREP -> row multicast) / hc_write: hi/lo of w
constexpr uint32_t CB_X2 = 4;      // bf16 4   bf16(R^2) (PREP -> SSQ) / hc_write: E_c selector tiles
constexpr uint32_t CB_WD = 5;      // bf16 40  down|inject weights of (s, n)
constexpr uint32_t CB_ACT = 6;     // bf16 80  hi/lo of the 40 slice K-tiles (remote written)
constexpr uint32_t CB_PART = 7;    // f32  1   DOWN partial
constexpr uint32_t CB_WU = 8;      // bf16 40  up weights of column j (c, n)
constexpr uint32_t CB_LO = 9;      // bf16 20  hi/lo of silu(p_n), n < 10 (remote written)
constexpr uint32_t CB_RS = 10;     // f32  4   rs_c (remote written)
constexpr uint32_t CB_YMIX = 11;   // f32  4   y tiles of column j (remote written)
constexpr uint32_t CB_ACC = 12;    // f32  4   RED stream partials
constexpr uint32_t CB_OUT = 13;    // out  1   mixed tile / hc_write: output tile
constexpr uint32_t CB_X2IN = 14;   // bf16 40  (aliases CB_ACT) bf16(R^2) of slice s (remote written)
constexpr uint32_t CB_ONES = 15;   // bf16 1   all-ones tile
constexpr uint32_t CB_SSQ = 16;    // f32  1   SSQ result
constexpr uint32_t CB_SSQIN = 17;  // f32  8   (aliases CB_ACT) slice sums (remote written)
constexpr uint32_t CB_PIN = 18;    // f32  8   (aliases CB_ACT) DOWN partials of N-tile n (remote written)
constexpr uint32_t CB_RED = 19;    // bf16 2   hi/lo of silu(p_n)
constexpr uint32_t CB_INJ = 20;    // f32  1   inject tile (RED n == 10) / hc_write: inj input tile
constexpr uint32_t CB_XN = 21;     // f32  4   xn_c = y_c * rs_c (MIX)

constexpr uint32_t SEM_ACT = 0;   // MAIN: += K-tiles landed in CB_ACT (target 40)
constexpr uint32_t SEM_X2 = 1;    // SSQ:  += K-tiles landed in CB_X2IN (target 40)
constexpr uint32_t SEM_Y = 2;     // MIX:  += y tiles landed in CB_YMIX (target 4)
constexpr uint32_t SEM_PART = 3;  // RED:  += partials landed in CB_PIN (target 8)
constexpr uint32_t SEM_SSQ = 4;   // RS:   += slice sums landed in CB_SSQIN (target 8)
constexpr uint32_t SEM_RS = 5;    // MIX, RED: rs landed (target 1)
constexpr uint32_t SEM_LO = 6;    // MIX:  += silu tiles landed in CB_LO (target 10); LO hub: += RED parts (10)
constexpr uint32_t SEM_GATHER = 7;  // ACT gather core of a row: += K-tiles landed in its CB_ACT (target 40)
constexpr uint32_t SEM_WU = 8;      // MIX: BRISC's share of the up weights landed (target 1)

// ACT gather: the 11 MAIN cores of row s write their hi/lo tiles into the CB_ACT memory of core (GATHER_X, s), which
// multicasts the whole slice (80 tiles) to the other 10 (11 concurrent overlapping row multicasts stalled up to 37 us)
constexpr uint32_t GATHER_X = 10;

constexpr uint32_t ROLE_MAIN = 1, ROLE_SSQ = 2, ROLE_RS = 3, ROLE_RED = 4;

constexpr uint32_t K_SLICE = 40;  // K-tiles per slice
constexpr uint32_t N_SLICE = 8;   // slices (2 per stream)
constexpr uint32_t N_MIX = 80;    // MIX cores / output column tiles
constexpr uint32_t ROW_KT = 320;  // K-tiles per 32-row tile of R
constexpr uint32_t GS_ROW = 128;  // gamma*sqrt(N) page: the 32 f32 of one K-tile (row-major [320, 32])

// LO hub: the RS core gathers the 10 RED cores' hi/lo silu tiles and multicasts all 20 to the MIX cores at once
// (10 concurrent 2-tile multicasts into the same 80 cores measured up to 22 us; one 20-tile multicast ~1 us)
constexpr uint32_t HUB_X = 8, HUB_Y = 8;

// MAIN reader transaction ids (NOC0): PREP's R / gamma reads, the down weights in WD_CHUNKS chunks, the up weights
constexpr uint32_t TRID_PREP = 1;
constexpr uint32_t TRID_WD0 = 2;
constexpr uint32_t WD_CHUNKS = 4;
constexpr uint32_t WD_CHUNK = K_SLICE / WD_CHUNKS;
constexpr uint32_t TRID_WU = TRID_WD0 + WD_CHUNKS;
// up weights of a MIX column: the last WU_B of its 40 tiles come over NOC1 (BRISC, after PREP's traffic), the rest
// over NOC0 once the down weights have landed (traced T=1: 0.0542 ms; 0.0586 with all 40 on NOC0 behind the down
// weights, 0.0557 with 16 on NOC1 but NOC0's issued together with the down weights, 0.0623 with 32 on NOC1)
constexpr uint32_t WU_B = 16;

// compile-time args: 0 NN, 1 PRECISE, 2 R_IS_DRAM, 3 OUT_PAGE, 4 EPS bits, 5..15 VX[0..10], 16..25 VY[0..9]
constexpr uint32_t CT_VX = 5;
constexpr uint32_t CT_VY = 16;

}  // namespace hc
