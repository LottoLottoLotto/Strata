// All-device MoE, stage B reader (NCRISC): selection + down weight streaming.
//
// Work unit n = output tile column (n = core, core + NC, ... < NT): for every selected entry s (ascending id, the
// shared expert last) the JT tiles of the entry's down column n, one contiguous store page (DRAM, or bytes n * JT
// tiles of the matrix's host pool item for host-tier matrices: a contiguous item over PCIe, or a paged one through
// the block's page table entries, fetched into the scratch page first), into the CB of its format, consumed in entry
// order: a ring of cap_f blocks per format CB with DEPTH block reads in flight (stream.h), DRAM blocks split over
// both NoCs.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"
#include "prof.h"
#include "stream.h"

using namespace fnx_moe;

constexpr uint32_t cb_sel = 6, cb_tmp = 7;
constexpr uint32_t cb_w(uint32_t f) { return 1 + f; }

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t CAP[N_FMT] = {get_named_compile_time_arg_val("cap0"), get_named_compile_time_arg_val("cap1"),
                                     get_named_compile_time_arg_val("cap2"), get_named_compile_time_arg_val("cap3")};
    constexpr uint32_t DEPTH = get_named_compile_time_arg_val("depth");   // block reads in flight
    constexpr auto e_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    // page table scratch slots (the tmp page): one block's window each
    constexpr uint32_t PT_SLOT = (pt_window(JT * tile_bytes(F_BF16)) + 63) / 64 * 64;
    constexpr uint32_t PT_SLOTS = (TMP_BYTES - 64) / PT_SLOT;

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t T = get_common_arg_val<uint32_t>(0);
    const auto ents = TensorAccessor(e_args, get_common_arg_val<uint32_t>(1), ENT_WORDS * 4);
    const auto b0 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(2), JT * tile_bytes(0));
    const auto b1 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(3), JT * tile_bytes(1));
    const auto b2 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(4), JT * tile_bytes(2));
    const auto b3 = TensorAccessor(s_args, get_common_arg_val<uint32_t>(5), JT * tile_bytes(3));
    // page table of the paged host matrices (common args 6, 7: address, row bytes; never read without them)
    const auto ptab = TensorAccessor(p_args, get_common_arg_val<uint32_t>(6), get_common_arg_val<uint32_t>(7));

    uint32_t S;
    uint32_t* sel;
    uint32_t pt_buf;
    {
    MD_ZONE("DR-SEL");
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
    MD_ZONE("DR-STREAM");

    // the source of entry s's down column n: format, tier, address (tier 0: NoC address of the DRAM store page;
    // TIER_HOST: host bus address; TIER_PAGED: the byte offset in the item, row = page table row)
    auto source = [&](uint32_t s, uint32_t n, uint32_t& f, uint32_t& tier, uint32_t& row) -> uint64_t {
        const uint32_t* tw = sel + sel[SEL_ENT + s];
        const uint32_t word = tw[2];
        f = word_fmt(word);
        tier = word_tier(word);
        if (tier == TIER_PAGED) {
            row = tw[7];
            return uint64_t(n) * (JT * tile_bytes(f));
        }
        if (tier != 0) {                    // host pool item: page n at n * JT tiles
            return hostmem_addr(tw[7], tw[8]) + uint64_t(n) * (JT * tile_bytes(f));
        }
        const uint32_t page = word_slot(word) * NT + n;
        switch (f) {
            case F_BFP4: return get_noc_addr(page, b0);
            case F_BFP2: return get_noc_addr(page, b1);
            case F_BFP8: return get_noc_addr(page, b2);
            default: return get_noc_addr(page, b3);
        }
    };
    // ring: per column, the page table windows of up to PT_SLOTS paged entries are fetched together (one DRAM round
    // trip, while earlier blocks are in flight), then those entries' blocks are issued in entry order
    BlockStream<DEPTH, true> st;   // DRAM blocks split over both NoCs (DM_DYNAMIC_NOC)
    for (uint32_t n = core; n < NT; n += NC) {
        for (uint32_t s0 = 0; s0 < S;) {
            uint32_t s1 = s0, k = 0;
            for (; s1 < S; ++s1) {
                uint32_t f, tier, row;
                const uint32_t off = uint32_t(source(s1, n, f, tier, row));
                if (tier == TIER_PAGED) {
                    if (k == PT_SLOTS) {
                        break;
                    }
                    if (k == 0) {
                        noc_async_read_set_trid(TRID_PT);
                    }
                    fetch_pages(ptab, row, off, JT * tile_bytes(f), pt_buf + k++ * PT_SLOT);
                }
            }
            if (k) {
                noc_async_read_barrier_with_trid(TRID_PT);
                invalidate_l1_cache();
            }
            k = 0;
            for (uint32_t s = s0; s < s1; ++s) {
                uint32_t f, tier, row;
                const uint64_t src = source(s, n, f, tier, row);
                const uint32_t bytes = JT * tile_bytes(f);
                if (tier == TIER_PAGED) {
                    st.issue_paged(cb_w(f), pt_entry(pt_buf + k++ * PT_SLOT, uint32_t(src)), uint32_t(src), bytes, JT,
                                   CAP[f]);
                } else if (tier != 0) {
                    st.issue(cb_w(f), src, bytes, JT, CAP[f], true);
                } else {
                    const uint32_t page = word_slot(sel[sel[SEL_ENT + s] + 2]) * NT + n;
                    uint64_t alt;
                    switch (f) {
                        case F_BFP4: alt = b0.get_noc_addr(page, 0, st.ALT); break;
                        case F_BFP2: alt = b1.get_noc_addr(page, 0, st.ALT); break;
                        case F_BFP8: alt = b2.get_noc_addr(page, 0, st.ALT); break;
                        default: alt = b3.get_noc_addr(page, 0, st.ALT); break;
                    }
                    st.issue_dram(cb_w(f), src, alt, bytes, JT, CAP[f]);
                }
            }
            s0 = s1;
        }
    }
    st.drain();
}
