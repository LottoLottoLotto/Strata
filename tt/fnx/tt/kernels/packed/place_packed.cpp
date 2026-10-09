// Exact placement of a column block into a bank-packed weight (fnx.tt.packed_matmul.place_packed): TILE page
// (k, c) of part [K, PW] (DRAM interleaved) -> page packed(k, col0 + c) of dst [K, N], the worker-major / bank-rotating
// order packed_in1_reader.cpp streams (fnx.tt.packed_matmul.PackedLayout.pages):
//   n = col0 + c, worker w = n / COLS, j = n % COLS, valid = COLS (LAST_COLS for the last worker),
//   group g = k / GROUP, h = k % GROUP, bank = (w + g) % 8,
//   bank page = w * (K / 8) * COLS + (g / 8) * GROUP * valid + h * valid + j, page = bank + 8 * bank page.
// Pure page copies (block-float pages unchanged).  Core i of n copies tile rows i, i + n, ...
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_buf = 0;

void kernel_main() {
    constexpr uint32_t PAGE = get_named_compile_time_arg_val("page_bytes");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk");      // pages per L1 round (CB size)
    constexpr uint32_t PW = get_named_compile_time_arg_val("part_cols");
    constexpr uint32_t KT = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t COLS = get_named_compile_time_arg_val("cols");
    constexpr uint32_t GROUP = get_named_compile_time_arg_val("group");
    constexpr uint32_t LAST_W = get_named_compile_time_arg_val("last_worker");
    constexpr uint32_t LAST_COLS = get_named_compile_time_arg_val("last_cols");
    constexpr uint32_t BANKS = 8;
    constexpr uint32_t PPWB = (KT / BANKS) * COLS;
    static_assert(KT % (BANKS * GROUP) == 0);
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0), PAGE);
    const auto dst = TensorAccessor(d_args, get_common_arg_val<uint32_t>(1), PAGE);
    const uint32_t col0 = get_common_arg_val<uint32_t>(2);
    const uint32_t n_cores = get_common_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(0);

    cb_reserve_back(cb_buf, 1);
    const uint32_t l1 = get_write_ptr(cb_buf);
    for (uint32_t k = core; k < KT; k += n_cores) {
        const uint32_t g = k / GROUP;
        const uint32_t h = k % GROUP;
        for (uint32_t c0 = 0; c0 < PW; c0 += CHUNK) {
            const uint32_t nc = PW - c0 < CHUNK ? PW - c0 : CHUNK;
            for (uint32_t c = 0; c < nc; ++c) {
                noc_async_read(src.get_noc_addr(k * PW + c0 + c), l1 + c * PAGE, PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t c = 0; c < nc; ++c) {
                const uint32_t n = col0 + c0 + c;
                const uint32_t w = n / COLS;
                const uint32_t j = n - w * COLS;
                const uint32_t valid = w == LAST_W ? LAST_COLS : COLS;
                const uint32_t bank = (w + g) % BANKS;
                const uint32_t bank_page = w * PPWB + (g / BANKS) * GROUP * valid + h * valid + j;
                noc_async_write(l1 + c * PAGE, dst.get_noc_addr(bank + BANKS * bank_page), PAGE);
            }
            noc_async_write_barrier();
        }
    }
}
