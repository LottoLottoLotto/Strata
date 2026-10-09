// Device reads of pinned host memory (fnx.tt.hostmem.HostPool) over PCIe, for data-movement kernels.
//
// On Blackhole a NoC request to the PCIe tile carries a full 64-bit address: [0, 2^58) goes straight to the host
// (through the IOMMU when it translates; this host runs iommu=pt, so the address is the physical address that
// HostPool's KMD pin returned).  The standard noc_async_read cannot express such an address (its 64-bit argument
// packs a 36-bit local address under the x/y coordinate), so hostmem_read programs the read command buffer with the
// split form (coordinate register + 64-bit address), one NOC packet (<= NOC_MAX_BURST_SIZE, 16 KiB) per command.
//
// Contract:
//  - dedicated-NoC kernels (the default noc_mode); reads are counted like noc_async_read, so noc_async_read_barrier
//    covers them, and they carry whatever transaction id noc_async_read_set_trid last set (it is not touched), so
//    noc_async_read_barrier_with_trid works too;
//  - addr and dst 64 B aligned (NOC_PCIE_READ_ALIGNMENT_BYTES), the read inside physically contiguous memory: a
//    contiguous HostPool item (HostPool.contig), or for a paged item (the pool's 4k region) hostmem_read_paged with
//    the item's page table (HostPool.pages: one bus address per 4 KiB of the item);
//  - it overwrites the read command buffer's address / coordinate / length registers: a caller that uses the
//    *_set_state / *_with_state read APIs on read_cmd_buf must set its state again afterwards (noc_async_read
//    itself programs every register each call).
#pragma once
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// HostPool.relaxed: the same host addresses through the PCIe relaxed-ordering window of the NoC-to-host TLBs
constexpr uint64_t HOSTMEM_RELAXED = 1ull << 58;

// NoC x/y encoding of the PCIe tile that is connected to the host (PCIE_NOC_X/Y: the JIT's translated PCIe core, the
// same value tt-metal's own H2D sockets use on either NoC)
FORCE_INLINE uint32_t hostmem_pcie_xy() { return NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y); }

// Asynchronous read of len bytes at host bus address addr into L1 dst on NoC `noc`.
FORCE_INLINE void hostmem_read(uint64_t addr, uint32_t dst, uint32_t len, uint8_t noc = noc_index) {
    const uint32_t xy = hostmem_pcie_xy();
    while (len > NOC_MAX_BURST_SIZE) {
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            noc, xy, addr, dst, NOC_MAX_BURST_SIZE);
        addr += NOC_MAX_BURST_SIZE;
        dst += NOC_MAX_BURST_SIZE;
        len -= NOC_MAX_BURST_SIZE;
    }
    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(noc, xy, addr, dst, len);
}

// (lo, hi) runtime-arg pair -> 64-bit host bus address (HostPool.lohi)
FORCE_INLINE uint64_t hostmem_addr(uint32_t lo, uint32_t hi) { return (uint64_t(hi) << 32) | lo; }

// HostPool 4k region: a paged item's reads split at multiples of HOSTMEM_PAGE of the item
constexpr uint32_t HOSTMEM_PAGE = 4096;

// Asynchronous read of len bytes at byte `off` of a paged item into L1 dst (off, dst, len 64 B multiples).  pt: L1
// words (lo, hi) of the bus addresses of the item's 4 KiB pages off / 4096, off / 4096 + 1, ... (as many as the read
// touches).  Physically consecutive pages are read with one request, so a read inside one physical run is one
// hostmem_read.
FORCE_INLINE void hostmem_read_paged(const volatile tt_l1_ptr uint32_t* pt, uint32_t off, uint32_t dst, uint32_t len,
                                     uint8_t noc = noc_index) {
    const uint32_t in_page = off & (HOSTMEM_PAGE - 1);
    uint64_t a = hostmem_addr(pt[0], pt[1]) + in_page;
    uint32_t n = HOSTMEM_PAGE - in_page;          // bytes physically contiguous from a
    while (n < len) {
        pt += 2;
        const uint64_t b = hostmem_addr(pt[0], pt[1]);
        if (b != a + n) {
            hostmem_read(a, dst, n, noc);
            dst += n;
            len -= n;
            a = b;
            n = 0;
        }
        n += HOSTMEM_PAGE;
    }
    hostmem_read(a, dst, len, noc);
}
