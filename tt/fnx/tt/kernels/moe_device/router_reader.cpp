// All-device MoE, fused router (NCRISC, NOC0): on column core c < N_COL, k-tiles 0..39 of column c of the BF16 router
// weight [W_r^T | w_sg | 0], stored as column-contiguous pages of RB tiles (page c * (KT / RB) + p = k-tiles
// RB p .. RB p + RB - 1), all reads in flight at once (stream.h); the writer streams k-tiles 40..79.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"
#include "prof.h"
#include "stream.h"

using namespace fnx_moe;

constexpr uint32_t cb_w = 1;

void kernel_main() {
    constexpr uint32_t RB = get_named_compile_time_arg_val("rb");
    constexpr uint32_t NP = KT / RB / 2;          // pages per half column
    constexpr auto w_args = TensorAccessorArgs<0>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    if (core >= N_COL) {
        return;
    }
    const uint32_t page = RB * get_tile_size(cb_w);
    const auto wr = TensorAccessor(w_args, get_common_arg_val<uint32_t>(3), page);
    MD_ZONE("RR-W1");
    BlockStream<NP> st;
    for (uint32_t p = 0; p < NP; ++p) {
        st.issue(cb_w, get_noc_addr(core * 2 * NP + p, wr), page, RB, NP);
    }
    st.drain();
}
