// Host-memory read benchmark / copy (fnx/tests/test_hostmem.py).  Each core reads its n requests of REQ bytes
// (requests core*n .. core*n + n - 1, `reps` times) into a ring of DEPTH L1 slots with DEPTH reads in flight
// (transaction ids 1..DEPTH), optionally writing every landed request to page r of an interleaved DRAM tensor.
// Request r comes from the host iff host_every > 0 and r % host_every == 0, else from page r of an interleaved DRAM
// tensor (page = REQ bytes).  Host request r = bytes (r % PAGES) * REQ of HostPool item r / PAGES: item i is at
// chunk i / PER_CHUNK, offset (i % PER_CHUNK) * STRIDE (the pool's chunk table: u32 lo, hi per chunk), or, PAGED,
// a paged item: the table holds (lo, hi) per 4 KiB page of item 0, 1, ... (PPI pages per item, HostPool.pages) and
// the read goes through hostmem_read_paged.
// Stats page (16 u32) per core: t0 lo/hi, t1 lo/hi (wall clock before the first / after the last request landed),
// cycles of a lone 64 B read and of a lone REQ read of the core's first request (issued before t0), PCIE_NOC_X/Y.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostmem.h"

constexpr uint32_t cb_buf = 0;

FORCE_INLINE uint64_t wall_clock() {
    const uint32_t lo = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);  // latches the high word
    const uint32_t hi = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (uint64_t(hi) << 32) | lo;
}

void kernel_main() {
    constexpr uint32_t HOST_EVERY = get_named_compile_time_arg_val("host_every");
    constexpr uint32_t REQ = get_named_compile_time_arg_val("req");
    constexpr uint32_t DEPTH = get_named_compile_time_arg_val("depth");
    constexpr uint32_t WRITE_OUT = get_named_compile_time_arg_val("write_out");
    constexpr uint32_t PAGES = get_named_compile_time_arg_val("pages");
    constexpr uint32_t PER_CHUNK = get_named_compile_time_arg_val("per_chunk");
    constexpr uint32_t STRIDE = get_named_compile_time_arg_val("stride");
    constexpr uint32_t TABLE_BYTES = get_named_compile_time_arg_val("table_bytes");
    constexpr uint64_t WINDOW = get_named_compile_time_arg_val("relaxed") ? HOSTMEM_RELAXED : 0;
    constexpr uint32_t PAGED = get_named_compile_time_arg_val("paged");
    constexpr uint32_t PPI = get_named_compile_time_arg_val("ppi");
    static_assert(DEPTH >= 1 && DEPTH <= NOC_MAX_TRANSACTION_ID, "one transaction id per slot");
    constexpr auto t_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(t_args, get_common_arg_val<uint32_t>(0), TABLE_BYTES);
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), REQ);
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), REQ);
    const auto stats = TensorAccessor(st_args, get_common_arg_val<uint32_t>(3), 64);
    const uint32_t n = get_common_arg_val<uint32_t>(4);
    const uint32_t reps = get_common_arg_val<uint32_t>(5);
    const uint32_t core = get_arg_val<uint32_t>(0);

    cb_reserve_back(cb_buf, 1);
    const uint32_t tab = get_write_ptr(cb_buf);
    const uint32_t st = tab + TABLE_BYTES;
    const uint32_t ring = st + 64;
    volatile tt_l1_ptr uint32_t* tw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
    if constexpr (HOST_EVERY > 0) {
        uint32_t t0b = 0, t1b = TABLE_BYTES;
        if constexpr (PAGED) {      // only this core's items' page tables (a whole table per core would queue every
                                    // core on the table's one DRAM bank before its first request)
            const uint32_t r0 = core * n;
            t0b = ((r0 / PAGES) * PPI * 8) & ~63u;
            t1b = (((r0 + n - 1) / PAGES + 1) * PPI * 8 + 63) & ~63u;
            t1b = t1b < TABLE_BYTES ? t1b : TABLE_BYTES;
        }
        noc_async_read(table.get_noc_addr(0, t0b), tab + t0b, t1b - t0b);
        noc_async_read_barrier();
    }

    auto issue = [&](uint32_t r, uint32_t dst, uint32_t len) {
        if constexpr (HOST_EVERY > 0) {
            if (r % HOST_EVERY == 0) {
                const uint32_t item = r / PAGES;
                if constexpr (PAGED) {
                    const uint32_t off = (r % PAGES) * REQ;
                    hostmem_read_paged(tw + 2 * (item * PPI + off / HOSTMEM_PAGE), off, dst, len);
                    return;
                }
                const uint32_t c = item / PER_CHUNK;
                const uint64_t base = hostmem_addr(tw[2 * c], tw[2 * c + 1]) | WINDOW;
                hostmem_read(base + (item % PER_CHUNK) * STRIDE + (r % PAGES) * REQ, dst, len);
                return;
            }
        }
        noc_async_read(src.get_noc_addr(r), dst, len);
    };

    const uint32_t first = core * n;
    uint64_t t = wall_clock();
    issue(first, ring, 64);
    noc_async_read_barrier();
    const uint32_t lat64 = uint32_t(wall_clock() - t);
    t = wall_clock();
    issue(first, ring, REQ);
    noc_async_read_barrier();
    const uint32_t lat_req = uint32_t(wall_clock() - t);

    auto retire = [&](uint32_t g) {
        const uint32_t slot = g % DEPTH;
        noc_async_read_barrier_with_trid(1 + slot);
        if constexpr (WRITE_OUT) {
            noc_async_write(ring + slot * REQ, out.get_noc_addr(first + g % n), REQ);
            noc_async_writes_flushed();
        }
    };

    const uint64_t t0 = wall_clock();
    const uint32_t total = n * reps;
    for (uint32_t g = 0; g < total; ++g) {
        const uint32_t slot = g % DEPTH;
        if (g >= DEPTH) {
            retire(g - DEPTH);
        }
        noc_async_read_set_trid(1 + slot);
        issue(first + g % n, ring + slot * REQ, REQ);
    }
    for (uint32_t g = total > DEPTH ? total - DEPTH : 0; g < total; ++g) {
        retire(g);
    }
    noc_async_read_set_trid(0);
    if constexpr (WRITE_OUT) {
        noc_async_write_barrier();
    }
    const uint64_t t1 = wall_clock();

    sw[0] = uint32_t(t0);
    sw[1] = uint32_t(t0 >> 32);
    sw[2] = uint32_t(t1);
    sw[3] = uint32_t(t1 >> 32);
    sw[4] = lat64;
    sw[5] = lat_req;
    sw[6] = PCIE_NOC_X;
    sw[7] = PCIE_NOC_Y;
    noc_async_write(st, stats.get_noc_addr(core), 64);
    noc_async_write_barrier();
}
