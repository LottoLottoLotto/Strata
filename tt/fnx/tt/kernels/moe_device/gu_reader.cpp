// All-device MoE, stage A reader (NCRISC): selection + gate/up weight streaming.
//
// Builds the selection page (common.h) from the T entry pages the router wrote (route rows + their experts' table
// entries), pushes it to cb_sel (the writer and compute read it), then streams the weights of this core's work units
// (sched.h: host-tier units on their own cores, resident units on the rest): gate column j then up column j of entry
// s, each one contiguous store page of KT tiles (a DRAM store page, split over both NoCs (stream.h issue_dram), or
// bytes j * KT tiles of the matrix's host pool item for host-tier matrices: a contiguous item over PCIe, or a paged
// one through the column's page table entries, fetched into the scratch page first), in blocks of BT tiles into the
// CB of the matrix's format (cap_f blocks), DEPTH block reads in flight (stream.h).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"
#include "prof.h"
#include "sched.h"
#include "stream.h"

using namespace fnx_moe;

constexpr uint32_t cb_sel = 6, cb_tmp = 7;
constexpr uint32_t cb_w(uint32_t f) { return 1 + f; }

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t BT = get_named_compile_time_arg_val("bt");
    constexpr uint32_t CAP[N_FMT] = {get_named_compile_time_arg_val("cap0"), get_named_compile_time_arg_val("cap1"),
                                     get_named_compile_time_arg_val("cap2"), get_named_compile_time_arg_val("cap3")};
    constexpr uint32_t DEPTH = get_named_compile_time_arg_val("depth");   // block reads in flight
    constexpr uint32_t HCORES = get_named_compile_time_arg_val("hcores"); // cores for the host-tier units (sched.h)
    constexpr auto e_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t T = get_common_arg_val<uint32_t>(0);
    const auto ents = TensorAccessor(e_args, get_common_arg_val<uint32_t>(1), ENT_WORDS * 4);
    // store A of format f: common arg 2 + f (formats without matrices may point anywhere: never read)
    const auto a0 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(2), KT * tile_bytes(0));
    const auto a1 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(3), KT * tile_bytes(1));
    const auto a2 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(4), KT * tile_bytes(2));
    const auto a3 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(5), KT * tile_bytes(3));
    // page table of the paged host matrices (common args 6, 7: address, row bytes; never read without them)
    const auto ptab = TensorAccessor(p_args, get_common_arg_val<uint32_t>(6), get_common_arg_val<uint32_t>(7));

    uint32_t S;
    uint32_t* sel;
    uint32_t pt_buf;                        // page table scratch: the tmp page
    {
    MD_ZONE("GR-SEL");
    cb_reserve_back(cb_tmp, 1);
    pt_buf = (get_write_ptr(cb_tmp) + 63) & ~63u;
    cb_reserve_back(cb_sel, 1);
    sel = reinterpret_cast<uint32_t*>(get_write_ptr(cb_sel));
    for (uint32_t r = 0; r < T; ++r) {
        noc_async_read(get_noc_addr(r, ents), get_write_ptr(cb_sel) + (SEL_ROWS + r * ENT_WORDS) * 4, ENT_WORDS * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    S = build_selection(sel, T, true);
    cb_push_back(cb_sel, 1);
    }
    MD_ZONE("GR-STREAM");

    // Paged host columns: the page table windows of a batch of PT_UNITS units are fetched together (one DRAM round
    // trip per batch, while the previous batch's blocks are still in flight), then the batch's blocks are issued.
    constexpr uint32_t PT_SLOT = (pt_window(KT * tile_bytes(F_BF16)) + 63) / 64 * 64;
    constexpr uint32_t PT_UNITS = (TMP_BYTES - 64) / PT_SLOT / 2;
    auto units = make_units([sel](uint32_t i) { return sel[i]; }, S, core, NC, HCORES);
    BlockStream<DEPTH, true> st;   // DRAM blocks split over both NoCs (DM_DYNAMIC_NOC)
    uint32_t s, j;
    while (true) {
        uint32_t nb = 0;
        bool fetched = false;
        auto ahead = units;
        while (nb < PT_UNITS && ahead.next(s, j)) {
            const uint32_t* tw = sel + sel[SEL_ENT + s];
            for (uint32_t op = 0; op < 2; ++op) {
                const uint32_t f = word_fmt(tw[op]), col = KT * tile_bytes(f);
                if (word_tier(tw[op]) == TIER_PAGED) {
                    if (!fetched) {
                        noc_async_read_set_trid(TRID_PT);
                        fetched = true;
                    }
                    fetch_pages(ptab, tw[3 + 2 * op], j * col, col, pt_buf + (2 * nb + op) * PT_SLOT);
                }
            }
            ++nb;
        }
        if (nb == 0) {
            break;
        }
        if (fetched) {
            noc_async_read_barrier_with_trid(TRID_PT);
            invalidate_l1_cache();
        }
        for (uint32_t b = 0; b < nb; ++b) {
            units.next(s, j);
            const uint32_t* tw = sel + sel[SEL_ENT + s];
            for (uint32_t op = 0; op < 2; ++op) {
                const uint32_t word = tw[op];
                const uint32_t f = word_fmt(word), tier = word_tier(word);
                const uint32_t cb = cb_w(f), blk = BT * tile_bytes(f), col = KT * tile_bytes(f);
                if (tier == TIER_PAGED) {   // page j at j * KT tiles, through the window fetched above
                    const uint32_t slot = pt_buf + (2 * b + op) * PT_SLOT;
                    const volatile tt_l1_ptr uint32_t* pt = pt_entry(slot, j * col);
                    uint32_t off = j * col;
                    const uint32_t p0 = off / HOSTMEM_PAGE;
                    for (uint32_t kb = 0; kb < KT; kb += BT) {
                        st.issue_paged(cb, pt + 2 * (off / HOSTMEM_PAGE - p0), off, blk, BT, CAP[f]);
                        off += blk;
                    }
                    continue;
                }
                if (tier != 0) {            // contiguous host pool item: page j at j * KT tiles
                    uint64_t src = hostmem_addr(tw[3 + 2 * op], tw[4 + 2 * op]) + uint64_t(j) * col;
                    for (uint32_t kb = 0; kb < KT; kb += BT) {
                        st.issue(cb, src, blk, BT, CAP[f], true);
                        src += blk;
                    }
                    continue;
                }
                const uint32_t page = word_slot(word) * JT + j;
                uint64_t src, alt;
                switch (f) {
                    case F_BFP4: src = get_noc_addr(page, a0); alt = a0.get_noc_addr(page, 0, st.ALT); break;
                    case F_BFP2: src = get_noc_addr(page, a1); alt = a1.get_noc_addr(page, 0, st.ALT); break;
                    case F_BFP8: src = get_noc_addr(page, a2); alt = a2.get_noc_addr(page, 0, st.ALT); break;
                    default: src = get_noc_addr(page, a3); alt = a3.get_noc_addr(page, 0, st.ALT); break;
                }
                for (uint32_t kb = 0; kb < KT; kb += BT) {
                    st.issue_dram(cb, src, alt, blk, BT, CAP[f]);
                    src += blk;
                    alt += blk;
                }
            }
        }
    }
    st.drain();
}
