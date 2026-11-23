// KV append at a DEVICE position (trace-safe ttnn.experimental.slice_write): one core copies the H*T row-major rows of
// src [1, H, T, ROW] (row j*T + t) to dst rows j*cells + p0 + t of a [1, 1, H*cells, ROW] row-major cache, p0 = word 0
// of the uint32 position tensor (refreshed with copy_host_to_device_tensor before a trace replay).  Exact copy.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_buf = 0;

void kernel_main() {
    constexpr uint32_t ROW = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t POS_PAGE = get_named_compile_time_arg_val("pos_page");
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0), ROW);
    const auto dst = TensorAccessor(d_args, get_common_arg_val<uint32_t>(1), ROW);
    const auto pos = TensorAccessor(p_args, get_common_arg_val<uint32_t>(2), POS_PAGE);
    const uint32_t H = get_common_arg_val<uint32_t>(3);
    const uint32_t T = get_common_arg_val<uint32_t>(4);
    const uint32_t cells = get_common_arg_val<uint32_t>(5);

    cb_reserve_back(cb_buf, 1);
    const uint32_t l1 = get_write_ptr(cb_buf);
    noc_async_read(get_noc_addr(0, pos), l1, POS_PAGE);
    noc_async_read_barrier();
    const uint32_t p0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1)[0];
    const uint32_t rows = l1 + POS_PAGE;
    for (uint32_t r = 0; r < H * T; ++r) {
        noc_async_read(get_noc_addr(r, src), rows + r * ROW, ROW);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < H; ++j) {
        for (uint32_t t = 0; t < T; ++t) {
            noc_async_write(rows + (j * T + t) * ROW, get_noc_addr(j * cells + p0 + t, dst), ROW);
        }
    }
    noc_async_write_barrier();
}
