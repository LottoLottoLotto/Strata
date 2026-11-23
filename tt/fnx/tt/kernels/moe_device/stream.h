// Pipelined block reads into circular buffers for the all-device MoE readers (data-movement kernels only).
//
// BlockStream keeps up to DEPTH block reads in flight (NOC transaction ids 1..DEPTH): issue() reserves room in the
// block's CB behind the blocks of that CB already in flight, starts the read with its own transaction id, and only
// the oldest block is waited for and pushed when the queue is full (retire()).  Blocks are pushed in issue order,
// which is the order the compute kernel consumes them.  Every CB holds a whole number of blocks of one size (its
// write pointer stays block aligned), at least 2.  A block comes from a NoC address (DRAM store page), host=true,
// from a 64-bit host bus address (a contiguous pinned host item over PCIe, ../hostmem/hostmem.h: 64 B aligned source
// and destination; the CB block sizes are multiples of 64 B), or (issue_paged) from a paged host item through its
// page table entries in L1 (fetch_pages); every read carries the slot's transaction id.
#pragma once
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../hostmem/hostmem.h"

// Transaction id of the page table fetches (never a BlockStream slot's: DEPTH < TRID_PT)
constexpr uint32_t TRID_PT = NOC_MAX_TRANSACTION_ID;
// Transaction id of the writers' (BRISC) reads: with dual-NoC streams the reader's block reads share the writer's NoC,
// and a plain noc_async_read_barrier (dynamic NoC mode: every RISC's reads on that NoC) would wait for them too
constexpr uint32_t TRID_W = NOC_MAX_TRANSACTION_ID - 1;

// Page table entries per fetch slot: a fetch window of the entries a read of `len` bytes touches, 64 B aligned at both
// ends (fetch_pages)
constexpr uint32_t pt_window(uint32_t len) {
    return ((len + HOSTMEM_PAGE - 1) / HOSTMEM_PAGE + 1) * 8 + 128;
}

// Paged host matrices (table word tier 2: an item of the HostPool's 4k region): the store's page table `ptab` (DRAM
// interleaved, one row per paged matrix of the layer: (lo, hi) bus address of each 4 KiB of the matrix).  Reads the
// entries that bytes [off, off + len) of row `row` touch into L1 buf (64 B aligned, >= pt_window(len) bytes) with the
// current transaction id and returns the L1 address of the entry of page off / 4096 (hostmem_read_paged's pt).
template <typename Acc>
FORCE_INLINE uint32_t fetch_pages(const Acc& ptab, uint32_t row, uint32_t off, uint32_t len, uint32_t buf) {
    const uint32_t e0 = off / HOSTMEM_PAGE * 8, e1 = ((off + len - 1) / HOSTMEM_PAGE + 1) * 8;
    const uint32_t w0 = e0 & ~63u, w1 = (e1 + 63) & ~63u;
    noc_async_read(ptab.get_noc_addr(row, w0), buf, w1 - w0);
    return buf + (e0 - w0);
}

// The L1 page table entry of byte `off` in a window fetch_pages(.., off0, .., buf) fetched, for off0 in the same
// 4 KiB page as off (= fetch_pages' return value)
FORCE_INLINE volatile tt_l1_ptr uint32_t* pt_entry(uint32_t buf, uint32_t off) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + ((off / HOSTMEM_PAGE * 8) & 63));
}

template <uint32_t DEPTH, bool DUAL = false>
struct BlockStream {
    static_assert(DEPTH >= 1 && DEPTH < TRID_W, "transaction ids 1..DEPTH, TRID_W / TRID_PT for the others");
    static constexpr uint8_t ALT = 1 - noc_index;   // DUAL: the other NoC (the kernel runs in DM_DYNAMIC_NOC mode)
    uint8_t cb[DEPTH];
    uint16_t pages[DEPTH];
    uint32_t head = 0, count = 0, issued = 0;

    uint32_t pending(uint32_t c) const {
        uint32_t n = 0;
        for (uint32_t i = 0; i < count; ++i) {
            n += cb[(head + i) % DEPTH] == c;
        }
        return n;
    }

    void retire() {
        const uint32_t i = head;
        noc_async_read_barrier_with_trid(1 + i);
        if constexpr (DUAL) {
            noc_async_read_barrier_with_trid(1 + i, ALT);
        }
        cb_push_back(cb[i], pages[i]);
        head = (head + 1) % DEPTH;
        --count;
    }

    // The next block of CB c (bytes = npages pages of it; cap_blocks: the CB's block count): room reserved behind
    // the blocks in flight, its transaction id set; returns its L1 write address (the caller starts the read).
    uint32_t reserve(uint32_t c, uint32_t bytes, uint32_t npages, uint32_t cap_blocks) {
        if (count == DEPTH) {
            retire();
        }
        uint32_t pend = pending(c);
        while (pend + 1 > cap_blocks) {      // the CB's ring is full of blocks in flight: wait for the oldest
            retire();
            pend = pending(c);
        }
        cb_reserve_back(c, (pend + 1) * npages);
        auto& ci = get_local_cb_interface(c);
        uint32_t wp = ci.fifo_wr_ptr + pend * bytes;
        if (wp >= ci.fifo_limit) {
            wp -= ci.fifo_size;
        }
        const uint32_t slot = (head + count) % DEPTH;
        noc_async_read_set_trid(1 + slot);
        if constexpr (DUAL) {
            noc_async_read_set_trid(1 + slot, ALT);
        }
        cb[slot] = c;
        pages[slot] = npages;
        ++count;
        ++issued;
        return wp;
    }

    // Read `bytes` (= npages pages of CB c) from src (host: a host bus address) into the next block of CB c.
    void issue(uint32_t c, uint64_t src, uint32_t bytes, uint32_t npages, uint32_t cap_blocks, bool host = false) {
        const uint32_t wp = reserve(c, bytes, npages, cap_blocks);
        if (host) {
            hostmem_read(src, wp, bytes);
        } else {
            noc_async_read(src, wp, bytes);
        }
    }

    // Read `bytes` of a DRAM page into the next block of CB c: src / src_alt = the same bytes' address on this
    // kernel's NoC / the other one.  DUAL: the first half (DRAM_ALIGNMENT aligned) on noc_index, the rest on ALT, so
    // each block streams over both NoCs (and both DRAM endpoints of its bank) into the same bytes of the CB.
    void issue_dram(uint32_t c, uint64_t src, uint64_t src_alt, uint32_t bytes, uint32_t npages, uint32_t cap_blocks) {
        const uint32_t wp = reserve(c, bytes, npages, cap_blocks);
        if constexpr (DUAL) {
            const uint32_t first = (bytes / 2) & ~(DRAM_ALIGNMENT - 1);
            noc_async_read(src, wp, first);
            noc_async_read(src_alt + first, wp + first, bytes - first, ALT);
        } else {
            noc_async_read(src, wp, bytes);
        }
    }

    // Read bytes [off, off + bytes) of a paged host item (pt: the L1 page table entry of page off / 4096) into the
    // next block of CB c.
    void issue_paged(uint32_t c, const volatile tt_l1_ptr uint32_t* pt, uint32_t off, uint32_t bytes, uint32_t npages,
                     uint32_t cap_blocks) {
        const uint32_t wp = reserve(c, bytes, npages, cap_blocks);
        hostmem_read_paged(pt, off, wp, bytes);
    }

    void drain() {
        while (count) {
            retire();
        }
        noc_async_read_set_trid(0);
        if constexpr (DUAL) {
            noc_async_read_set_trid(0, ALT);
        }
    }
};
