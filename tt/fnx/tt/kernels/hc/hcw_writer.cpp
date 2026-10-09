// hc_write, BRISC: the 4 updated R tiles (c, j) of row tile rt -> out (may be R itself: each tile has one owner).
#include "api/dataflow/dataflow_api.h"
#include "hc_common.h"

using namespace hc;

constexpr bool R_IS_DRAM = get_compile_time_arg_val(0) == 1;

void kernel_main() {
    const uint32_t j = get_absolute_logical_y() * 10 + get_absolute_logical_x();
    const uint32_t rt = get_common_arg_val<uint32_t>(4);
    const auto OUT = TensorAccessor(tensor_accessor::make_interleaved_dspec<R_IS_DRAM>(),
                                    get_common_arg_val<uint32_t>(3), F32_TILE);
    for (uint32_t c = 0; c < 4; ++c) {
        cb_wait_front(CB_OUT, c + 1);
        noc_async_write_page(rt * ROW_KT + c * N_MIX + j, OUT, get_read_ptr(CB_OUT) + c * F32_TILE);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, 4);
}
