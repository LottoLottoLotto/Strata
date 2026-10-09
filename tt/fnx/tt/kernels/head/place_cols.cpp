// Exact placement of a column block into the LM head weight (fnx.tt.head): the TILE pages of part [K, W] (W/32 tile
// columns, DRAM interleaved) are copied to tile columns col0 .. col0 + W/32 of dst [K, N] (N/32 tile columns), i.e.
// page r * PW + c of the part -> page r * DW + col0 + c of dst.  Pure data movement (block-float pages unchanged), so
// the joined weight is bitwise the converted blocks.  Core i of n copies tile rows i, i + n, ...
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_buf = 0;

void kernel_main() {
    constexpr uint32_t PAGE = get_named_compile_time_arg_val("page_bytes");
    constexpr uint32_t PW = get_named_compile_time_arg_val("part_cols");
    constexpr uint32_t DW = get_named_compile_time_arg_val("dst_cols");
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0), PAGE);
    const auto dst = TensorAccessor(d_args, get_common_arg_val<uint32_t>(1), PAGE);
    const uint32_t rows = get_common_arg_val<uint32_t>(2);
    const uint32_t col0 = get_common_arg_val<uint32_t>(3);
    const uint32_t n_cores = get_common_arg_val<uint32_t>(4);
    const uint32_t core = get_arg_val<uint32_t>(0);

    cb_reserve_back(cb_buf, 1);
    const uint32_t l1 = get_write_ptr(cb_buf);
    for (uint32_t r = core; r < rows; r += n_cores) {
        for (uint32_t c = 0; c < PW; ++c) {
            noc_async_read(src.get_noc_addr(r * PW + c), l1 + c * PAGE, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < PW; ++c) {
            noc_async_write(l1 + c * PAGE, dst.get_noc_addr(r * DW + col0 + c), PAGE);
        }
        noc_async_write_barrier();
    }
}
