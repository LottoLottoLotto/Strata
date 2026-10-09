// Shared definitions of the all-device MoE kernels (fnx.tt.moe_device).
//
// Route buffer (written by the router, int32 ROW_MAJOR [1, 1, 32, 32], one 128 B page per token row r < T):
//   [0, 10) routed expert ids in descending logit order (ties -> lower id), [10, 20) their softmax weights (f32 bits),
//   [20] sigmoid(shared-expert gate logit) (f32 bits), the rest 0.
// Entry pages (written by the router next to the route buffer, int32 ROW_MAJOR [1, 1, 32, ENT_WORDS], one page per row
//   r < T: what stage A / B read instead of the whole expert table): [0, 32) = route row r, then at ENT_TAB +
//   TABLE_STRIDE k the expert table entry of the row's k-th id (k < TOPK) and, k = TOPK, of the shared expert.
// Expert table (per layer, uint32 [9 * (N_EXP + 1)]): entry e (N_EXP = the shared expert) -> gate, up, down words
//   tier << 28 | fmt << 24 | slot, fmt = F_BFP4 / F_BFP2 / F_BFP8 / F_BF16, then the gate, up, down host words
//   (lo, hi pairs).  Tier 0: the matrix is page `slot` (* its page count) of that format's DRAM store; tier 1
//   (TIER_HOST): it lives in pinned host memory (a physically contiguous fnx.tt.hostmem.HostPool item: its pages
//   back to back) at the host bus address (lo, hi), read over PCIe (hostmem_read); tier 2 (TIER_PAGED): the same in
//   a paged item (the pool's 4k region), lo = its row of the store's page table (stream.h fetch_pages: the bus
//   address of each 4 KiB of the matrix), read with hostmem_read_paged.
// Stores (per layer and format, DRAM interleaved): "A" (K = 2560: gate and up) page slot * JT + j = the KT tiles of
//   output column j of W^T [2560, 640]; "B" (K = 640: down) page slot * NT + n = the JT tiles of output column n of
//   W^T [640, 2560].  Tiles [K, N] orientation: a block-float face row spans 16 OUTPUT channels (GPTQ-BFP layout).
// Selection (an L1 page every core builds): the T entry pages at SEL_ROWS (read straight into the page), then
//   sel[0] = S entries, sel[1] = T rows; entry s (ascending expert id, the shared expert last): sel[SEL_ENT + s] =
//   word offset in the page of its expert-table entry (gate, up, down word, then the gate, up, down host words lo,
//   hi: one of the entry pages' copies), sel[SEL_ID + s] = its expert id.  Route weights: row_weight.
// Scratch page (cb_tmp): page table windows of paged host reads (stream.h fetch_pages).
// Expert count: FNX_N_EXP (a program define; default 256 = the main layers, 512 = the MTP drafter's MoE).
#pragma once
#include <stdint.h>

#ifndef FNX_N_EXP
#define FNX_N_EXP 256
#endif

namespace fnx_moe {

constexpr uint32_t N_EXP = FNX_N_EXP, TOPK = 10, NE_MAX = N_EXP + 1, SHARED = N_EXP;
constexpr uint32_t MASK_WORDS = (NE_MAX + 31) / 32;     // selection bit mask words
constexpr uint32_t N_COL = (N_EXP + 32) / 32;           // router logits column tiles (routed experts + shared gate)
constexpr uint32_t KT = 80, JT = 20, NT = 80;           // hidden tiles (K of A, N of B), intermediate tiles
constexpr uint32_t ROUTE_WORDS = 32;                    // one route row = 128 B
constexpr uint32_t R_IDS = 0, R_W = TOPK, R_SH = 2 * TOPK;
constexpr uint32_t TABLE_STRIDE = 9, SEL_STRIDE = 1 + TABLE_STRIDE;     // words per table / selection entry
constexpr uint32_t TABLE_WORDS = TABLE_STRIDE * NE_MAX;
constexpr uint32_t ENT_TAB = ROUTE_WORDS, ENT_WORDS = 160;              // entry page: route row, TOPK + 1 entries
static_assert(ENT_TAB + TABLE_STRIDE * (TOPK + 1) <= ENT_WORDS && ENT_WORDS * 4 % 64 == 0);
constexpr uint32_t SEL_ENT = 16, SEL_ID = SEL_ENT + NE_MAX, SEL_ROWS = (SEL_ID + NE_MAX + 15) / 16 * 16;
constexpr uint32_t SEL_WORDS = SEL_ROWS + 32 * ENT_WORDS, SEL_BYTES = SEL_WORDS * 4;
constexpr uint32_t TMP_BYTES = 16384;
constexpr uint32_t F_BFP4 = 0, F_BFP2 = 1, F_BFP8 = 2, F_BF16 = 3, N_FMT = 4;
constexpr uint32_t TIER_HOST = 1, TIER_PAGED = 2;

inline uint32_t word_fmt(uint32_t w) { return (w >> 24) & 0xf; }
inline uint32_t word_tier(uint32_t w) { return w >> 28; }
inline uint32_t word_slot(uint32_t w) { return w & 0xffffffu; }

// tile bytes and tt::DataFormat code per format
constexpr uint32_t tile_bytes(uint32_t f) { return f == F_BFP4 ? 576 : f == F_BFP2 ? 320 : f == F_BFP8 ? 1088 : 2048; }
constexpr uint32_t data_format(uint32_t f) { return f == F_BFP4 ? 7 : f == F_BFP2 ? 15 : f == F_BFP8 ? 6 : 5; }

// Build the selection from the T entry pages at sel + SEL_ROWS (rows < T).  with_shared: entry N_EXP always present.
// Returns S.
inline uint32_t build_selection(uint32_t* sel, uint32_t T, bool with_shared) {
    const uint32_t* rows = sel + SEL_ROWS;
    uint32_t mask[MASK_WORDS] = {};
    uint16_t pos[NE_MAX];                               // e -> (row, k) of its first table entry copy
    for (uint32_t r = 0; r < T; ++r) {
        const uint32_t* row = rows + r * ENT_WORDS;
        for (uint32_t k = 0; k < TOPK; ++k) {
            const uint32_t e = row[R_IDS + k];
            if (!(mask[e >> 5] & (1u << (e & 31)))) {
                mask[e >> 5] |= 1u << (e & 31);
                pos[e] = r << 4 | k;
            }
        }
    }
    if (with_shared) {
        mask[SHARED >> 5] |= 1u << (SHARED & 31);
        pos[SHARED] = TOPK;                             // row 0, entry slot TOPK
    }
    uint32_t S = 0;
    for (uint32_t w = 0; w < MASK_WORDS; ++w) {
        uint32_t m = mask[w];
        while (m) {
            const uint32_t e = w * 32 + __builtin_ctz(m);
            m &= m - 1;
            sel[SEL_ENT + S] = SEL_ROWS + (pos[e] >> 4) * ENT_WORDS + ENT_TAB + TABLE_STRIDE * (pos[e] & 15);
            sel[SEL_ID + S] = e;
            ++S;
        }
    }
    sel[0] = S;
    sel[1] = T;
    return S;
}

// Route weight (f32 bits) of an entry page's row for expert e: its softmax weight, 0 if the row did not pick e, the
// shared gate for the shared expert.
inline uint32_t row_weight(const uint32_t* row, uint32_t e) {
    if (e == SHARED) {
        return row[R_SH];
    }
    for (uint32_t k = 0; k < TOPK; ++k) {
        if (row[R_IDS + k] == e) {
            return row[R_W + k];
        }
    }
    return 0;
}

}  // namespace fnx_moe
