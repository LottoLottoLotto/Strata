// Row argmax of float32 TILE logits without the untilize (fnx.tt.tile_argmax): rows 0..ROWS-1 (ROWS <= 32) of the
// first tile row of [1, 1, 32, V] logits -> uint32 indices, one per row, written to page r of a ROW_MAJOR [1, 1, ROWS, 1]
// tensor (ttnn.argmax's keepdim output).
//
// Workers (core, lane): this kernel runs on BRISC (lane 0) and NCRISC (lane 1) of every core of the grid rectangle;
// worker w = core * 2 + lane scans tiles [w * TILES / WORKERS, (w + 1) * TILES / WORKERS) (increasing columns in worker
// order).  Of each tile it reads only the face rows holding rows 0..ROWS-1 (faces 0/1 rows 0..min(ROWS,16)-1, faces
// 2/3 rows 0..ROWS-17), i.e. 128 B per row and tile.  KEEP_RM: the same face rows also go to a ROW_MAJOR float32 copy
// of rows 0..ROWS-1 (the bits ttnn.to_layout produces) laid out as [1, 1, ROWS * RM_K, V / RM_K]: row r = pages
// r * RM_K .. r * RM_K + RM_K - 1, spread over the DRAM banks (a [.., ROWS, V] page per row would put each row's 1 MB
// into one bank).  ROWS > 1: the scan, which has each face row's 16 values in registers, also stores them as one
// contiguous run per row (CB 4), so a worker writes a row of a chunk in one transfer per page it touches.
//
// Order (ttnn.argmax on float32: reader_argmax_interleaved_multicore.cpp + float32_greater): the first index of the
// row maximum; NaN never wins; -0 == +0; when no value exceeds -inf the index is 0.  A worker keeps the bits bu of its
// first strict maximum (start -inf, index 0).  While bu is a non-negative float (int32 >= 0), a value can only beat it
// if its bits, as int32, are larger (negative floats and -0 are negative int32s): one compare per value; the rare taken
// branch rejects positive NaN.  Otherwise (bu negative or -0) values compare by key = u ^ (sign ? ~0 : 0x80000000)
// (monotonic in the float order; -0 one below +0, positive NaN above +inf, negative NaN below key(-inf)); the taken
// branch rejects positive NaN and +0 after -0.  Each worker posts (key(bu), index) of row r (-0 posted as +0) to the
// row's gatherer, lane r % 2 of core r / 2, which keeps the first strict maximum in worker order (= increasing
// columns) and writes index page r.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t TILES = get_named_compile_time_arg_val("tiles");
constexpr uint32_t WORKERS = get_named_compile_time_arg_val("workers");
constexpr uint32_t LANE = get_named_compile_time_arg_val("lane");
constexpr uint32_t GRID_X = get_named_compile_time_arg_val("grid_x");  // cores per grid row
constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk");  // tiles per read chunk

constexpr uint32_t UP = ROWS < 16 ? ROWS : 16;      // rows in faces 0 / 1
constexpr uint32_t LO = ROWS > 16 ? ROWS - 16 : 0;  // rows in faces 2 / 3
constexpr uint32_t FROW = 64;                        // one face row: 16 float32
constexpr uint32_t TB = 2 * ROWS * FROW;             // L1 bytes per tile: [f0 rows | f1 rows | f2 rows | f3 rows]
constexpr uint32_t SLOT = 16;                        // gather slot: key, index
constexpr uint32_t NEG_INF = 0xFF800000u;
constexpr uint32_t POS_INF = 0x7F800000u;
constexpr uint32_t KEY_POSINF = 0xFF800000u;  // key(+inf): larger keys are positive NaN
constexpr uint32_t KEY_NZERO = 0x7FFFFFFFu;   // key(-0)
constexpr uint32_t KEY_PZERO = 0x80000000u;   // key(+0)
constexpr uint32_t KEY_NEGINF = 0x007FFFFFu;  // key(-inf)
static_assert(ROWS >= 1 && ROWS <= 32 && TILES >= WORKERS);

constexpr uint32_t row_off(uint32_t r, uint32_t h) {
    return r < 16 ? h * UP * FROW + r * FROW : 2 * 16 * FROW + h * LO * FROW + (r - 16) * FROW;
}

FORCE_INLINE uint32_t key_of(uint32_t u) {
    return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

FORCE_INLINE uint32_t align64(uint32_t a) { return (a + 63) & ~63u; }

template <typename Acc>
FORCE_INLINE void read_chunk(const Acc& lg, uint32_t t0, uint32_t n, uint32_t l1) {
    for (uint32_t t = t0; t < t0 + n; ++t, l1 += TB) {
        const uint64_t src = lg.get_noc_addr(t);
        if constexpr (UP == 16) {
            noc_async_read<2048>(src, l1, 2048);
        } else {
            noc_async_read<2048>(src, l1, UP * FROW);
            noc_async_read<2048>(src + 1024, l1 + UP * FROW, UP * FROW);
        }
        if constexpr (LO == 16) {
            noc_async_read<2048>(src + 2048, l1 + 2048, 2048);
        } else if constexpr (LO > 0) {
            noc_async_read<2048>(src + 2048, l1 + 2048, LO * FROW);
            noc_async_read<2048>(src + 3072, l1 + 2048 + LO * FROW, LO * FROW);
        }
    }
}

#ifdef KEEP_RM
constexpr uint32_t RM_K = get_named_compile_time_arg_val("rm_pages");  // pages per row of the row-major copy
constexpr uint32_t RM_TPP = TILES / RM_K;                             // tile columns (128 B) per page
static_assert(RM_K * RM_TPP == TILES);

// Row r of tiles [t0, t1) from contiguous L1 (128 B per tile) to the row-major copy, one write per page touched.
template <typename Acc>
FORCE_INLINE void write_row(const Acc& rm, uint32_t r, uint32_t t0, uint32_t t1, uint32_t src) {
    while (t0 < t1) {
        const uint32_t page = t0 / RM_TPP;
        const uint32_t end = (page + 1) * RM_TPP < t1 ? (page + 1) * RM_TPP : t1;
        noc_async_write(src, rm.get_noc_addr(r * RM_K + page, (t0 - page * RM_TPP) * 128), (end - t0) * 128);
        src += (end - t0) * 128;
        t0 = end;
    }
}
#endif

#if defined(KEEP_RM)
constexpr bool JOIN = ROWS > 1;  // the scan also stores each row's values contiguously for the row-major copy
#else
constexpr bool JOIN = false;
#endif

// 16 values of one face row (columns col..col+15), best so far (bu bits, bi index) updated in place; JOIN: the values
// are also stored to out[0..15].
FORCE_INLINE void scan16(const uint32_t* p, uint32_t col, uint32_t& bu, uint32_t& bi, uint32_t* out) {
    uint32_t v[16];
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 16; ++c) {
        v[c] = p[c];
    }
    if constexpr (JOIN) {
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            out[c] = v[c];
        }
    }
    if (static_cast<int32_t>(bu) >= 0) {
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            if (__builtin_expect(static_cast<int32_t>(v[c]) > static_cast<int32_t>(bu), 0)) {
                asm volatile("");  // keeps the NaN test off the common path (no if-conversion into it)
                if (v[c] <= POS_INF) {
                    bu = v[c];
                    bi = col + c;
                }
            }
        }
    } else {
        uint32_t bk = key_of(bu);
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t k = key_of(v[c]);
            if (__builtin_expect(k > bk, 0)) {
                asm volatile("");
                if (k <= KEY_POSINF && !(k == KEY_PZERO && bk == KEY_NZERO)) {
                    bk = k;
                    bu = v[c];
                    bi = col + c;
                }
            }
        }
    }
}

void kernel_main() {
    const uint32_t core = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();  // row-major
    const uint32_t worker = core * 2 + LANE;
    const uint32_t first = worker * TILES / WORKERS;
    const uint32_t last = (worker + 1) * TILES / WORKERS;

    constexpr auto lg_args = TensorAccessorArgs<0>();
    constexpr auto id_args = TensorAccessorArgs<lg_args.next_compile_time_args_offset()>();
    const auto lg = TensorAccessor(lg_args, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(id_args, get_common_arg_val<uint32_t>(1));
#ifdef KEEP_RM
    constexpr auto rm_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    const auto rm = TensorAccessor(rm_args, get_common_arg_val<uint32_t>(2));
#endif
    // JOIN: CB 4 holds each lane's chunk row by row, [lane][row][tile] 128 B (the source of the row-major writes)
    const uint32_t rowbuf = JOIN ? align64(get_write_ptr(4)) + LANE * ROWS * CHUNK * 128 : 0;

    const uint32_t buf0 = align64(get_write_ptr(LANE));  // CB 0 (BRISC) / CB 1 (NCRISC): 1 or 2 chunk buffers
    const uint32_t slots = align64(get_write_ptr(2));    // gather slots [lane][worker], same address on every core
    const uint32_t stage = align64(get_write_ptr(3)) + LANE * 64 * (ROWS + 1);  // [index out | posts (16 B / row)]

    uint32_t best_u[ROWS];
    uint32_t best_i[ROWS];
    for (uint32_t r = 0; r < ROWS; ++r) {
        best_u[r] = NEG_INF;
        best_i[r] = 0;
    }

    uint32_t cur = buf0;
    uint32_t alt = buf0 + CHUNK * TB;
    read_chunk(lg, first, (last - first) < CHUNK ? (last - first) : CHUNK, cur);
    for (uint32_t c0 = first; c0 < last; c0 += CHUNK) {
        const uint32_t n = (last - c0) < CHUNK ? (last - c0) : CHUNK;
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        const uint32_t c1 = c0 + n;
#ifdef KEEP_RM
        noc_async_writes_flushed();  // the previous chunk's row writes left their L1 source (`alt` / rowbuf)
#endif
        if (c1 < last) {
            read_chunk(lg, c1, (last - c1) < CHUNK ? (last - c1) : CHUNK, alt);
        }
#ifdef KEEP_RM
        if constexpr (!JOIN) {  // ROWS == 1: [h0 | h1] per tile, tiles contiguous: the chunk is already row-major
            write_row(rm, 0, c0, c1, cur);
        }
#endif
        for (uint32_t r = 0; r < ROWS; ++r) {
            uint32_t bu = best_u[r], bi = best_i[r];
            const uint32_t o0 = row_off(r, 0), o1 = row_off(r, 1);
            uint32_t l1 = cur;
            uint32_t* out = reinterpret_cast<uint32_t*>(rowbuf + r * CHUNK * 128);
            for (uint32_t t = c0; t < c1; ++t, l1 += TB, out += 32) {
                scan16(reinterpret_cast<const uint32_t*>(l1 + o0), t * 32, bu, bi, out);
                scan16(reinterpret_cast<const uint32_t*>(l1 + o1), t * 32 + 16, bu, bi, out + 16);
            }
            best_u[r] = bu;
            best_i[r] = bi;
#ifdef KEEP_RM
            if constexpr (JOIN) {
                asm volatile("fence" ::: "memory");  // the stored row is in L1 before the NoC reads it
                write_row(rm, r, c0, c1, rowbuf + r * CHUNK * 128);
            }
#endif
        }
        const uint32_t tmp = cur;
        cur = alt;
        alt = tmp;
    }

    // post row r to its gatherer (lane r % 2 of core r / 2): slot [lane][worker] on that core, then its semaphore
    volatile tt_l1_ptr uint32_t* post = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64);
    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t k = key_of(best_u[r]);
        post[4 * r] = k == KEY_NZERO ? KEY_PZERO : k;
        post[4 * r + 1] = best_i[r];
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t gx = get_common_arg_val<uint32_t>(3 + 2 * (r / 2));
        const uint32_t gy = get_common_arg_val<uint32_t>(4 + 2 * (r / 2));
        noc_async_write<SLOT>(stage + 64 + r * SLOT,
                              get_noc_addr(gx, gy, slots + ((r % 2) * WORKERS + worker) * SLOT), SLOT);
    }
    noc_async_write_barrier();
    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t gx = get_common_arg_val<uint32_t>(3 + 2 * (r / 2));
        const uint32_t gy = get_common_arg_val<uint32_t>(4 + 2 * (r / 2));
        noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(r % 2)), 1);
    }
    noc_async_atomic_barrier();

    const uint32_t r = core * 2 + LANE;  // the row this worker gathers, if any
    if (r < ROWS) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(LANE)), WORKERS);
        asm volatile("" ::: "memory");
        const uint32_t* s = reinterpret_cast<const uint32_t*>(slots + LANE * WORKERS * SLOT);
        uint32_t key = KEY_NEGINF, idx = 0;
#pragma GCC unroll 4
        for (uint32_t w = 0; w < WORKERS; ++w) {
            const uint32_t k = s[w * (SLOT / 4)];
            if (k > key) {
                key = k;
                idx = s[w * (SLOT / 4) + 1];
            }
        }
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0] = idx;
        noc_async_write<32>(stage, ids.get_noc_addr(r), 32);
    }
    noc_async_write_barrier();
}
