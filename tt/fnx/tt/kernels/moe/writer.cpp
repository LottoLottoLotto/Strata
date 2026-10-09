// Device experts writer (BRISC), both stages: unit u's output tile goes to tile base + u of the output tensor
// (stage A: h tile s * JT + j, base 0; stage B: y tile (meta[2] + g) * NT + n, meta[2] = the round's first group).
// units = meta[count_word] * per.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_meta = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    constexpr uint32_t COUNT_WORD = get_named_compile_time_arg_val("count_word");
    constexpr uint32_t PER = get_named_compile_time_arg_val("per");
    constexpr auto o_args = TensorAccessorArgs<0>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t o_tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), o_tile);

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
    const uint32_t units = m[COUNT_WORD] * PER;
    const uint32_t base = COUNT_WORD == 1 ? m[2] * PER : 0;
    for (uint32_t u = core; u < units; u += NC) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), get_noc_addr(base + u, out), o_tile);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
