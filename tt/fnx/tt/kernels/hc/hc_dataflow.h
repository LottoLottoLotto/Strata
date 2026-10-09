// HC fused kernels: helpers shared by the data-movement kernels.
#pragma once

#include <cstdint>

namespace hc {

// Gamma rows -> gamma tiles: `n` tiles at base + i * F32_TILE, each with its row (32 f32) read into words 0..31
// (face 0 rows 0 and 1), get that row in every row.  Tile memory = 4 row-major 16x16 f32 faces of 1 KB (columns
// 0..15 are faces 0 and 2, 16..31 faces 1 and 3; a face row is 64 B).  Local NoC copies that double the filled rows
// (6 dependent steps for all tiles at once, ~1 us; RISC-V stores took 4 us per tile).
FORCE_INLINE void gamma_tiles(uint32_t base, uint32_t n) {
    constexpr uint32_t ROW = 64, FACE = 1024;
    auto copy = [](uint32_t src, uint32_t dst, uint32_t bytes) { noc_async_write(src, get_noc_addr(dst), bytes); };
    for (uint32_t i = 0; i < n; ++i) {  // columns 16..31: face 0 row 1 -> face 1 row 0
        copy(base + i * F32_TILE + ROW, base + i * F32_TILE + FACE, ROW);
    }
    noc_async_write_barrier();
    for (uint32_t rows = 1; rows < 16; rows *= 2) {  // rows [0, rows) -> [rows, 2 rows) in faces 0 and 1
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = base + i * F32_TILE;
            copy(t, t + rows * ROW, rows * ROW);
            copy(t + FACE, t + FACE + rows * ROW, rows * ROW);
        }
        noc_async_write_barrier();
    }
    for (uint32_t i = 0; i < n; ++i) {  // faces 0, 1 -> faces 2, 3
        copy(base + i * F32_TILE, base + i * F32_TILE + 2 * FACE, 2 * FACE);
    }
    noc_async_write_barrier();
}

}  // namespace hc
