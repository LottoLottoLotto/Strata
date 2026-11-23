// Stage A work schedule of the all-device MoE: gu_reader / gu_writer / gu_compute enumerate the same units of a core
// in the same order through Units (any kernel type: the selection words come through an accessor).
//
// Unit (s, j) = h tile s * JT + j: intermediate column j of selection entry s.  Without host-tier entries: units
// u = core, core + NC, ... of all S * JT units in entry order (s = u / JT).  With nh > 0 host-tier units (the entries
// whose matrices live in the host pool, x JT): cores 0 .. CH - 1, CH = min(nh, HCORES, NC - 1), take the host units
// i = core, core + CH, ... < nh and cores CH .. NC - 1 the resident units i = core - CH, core - CH + (NC - CH), ...;
// class unit i -> the (i / JT)-th entry of its tier in ascending s, j = i % JT.  The PCIe transfer (link-bound: a few
// cores' reads in flight saturate it) then runs from the program's first cycle on its own cores, overlapping the DRAM
// streaming of the others instead of queueing behind or in front of it on the same cores.  Every unit's arithmetic is
// the same on any core, so the schedule never changes a result.
#pragma once
#include <stdint.h>

#include "common.h"

namespace fnx_moe {

template <typename Word>   // Word(i) -> word i of the selection page
struct Units {
    Word word;
    uint32_t S, step, i, n, s, k;
    uint8_t cls;            // 0: every entry, 1: host-tier entries, 2: resident entries

    bool member(uint32_t t) {
        if (cls == 0) {
            return true;
        }
        const bool host = word_tier(word(word(SEL_ENT + t))) != 0;
        return host == (cls == 1);
    }

    Units(Word w, uint32_t S_, uint32_t core, uint32_t NC, uint32_t hcores) : word(w), S(S_) {
        uint32_t nhe = 0;
        for (uint32_t t = 0; t < S; ++t) {
            nhe += word_tier(word(word(SEL_ENT + t))) != 0;
        }
        const uint32_t nh = nhe * JT;
        if (nh == 0 || hcores == 0) {
            cls = 0, i = core, step = NC, n = S * JT;
        } else {
            uint32_t ch = nh < hcores ? nh : hcores;
            ch = ch < NC - 1 ? ch : NC - 1;
            if (core < ch) {
                cls = 1, i = core, step = ch, n = nh;
            } else {
                cls = 2, i = core - ch, step = NC - ch, n = (S - nhe) * JT;
            }
        }
        k = 0;
        s = 0;
        while (s < S && !member(s)) {
            ++s;
        }
    }

    bool any() const { return i < n; }

    // The next unit of this core: (entry, column); false when done.
    bool next(uint32_t& se, uint32_t& j) {
        if (i >= n) {
            return false;
        }
        const uint32_t ke = i / JT;
        while (k < ke) {
            ++s;
            while (!member(s)) {
                ++s;
            }
            ++k;
        }
        se = s;
        j = i - ke * JT;
        i += step;
        return true;
    }
};

template <typename Word>
Units<Word> make_units(Word w, uint32_t S, uint32_t core, uint32_t NC, uint32_t hcores) {
    return Units<Word>(w, S, core, NC, hcores);
}

}  // namespace fnx_moe
