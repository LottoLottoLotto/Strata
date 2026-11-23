// hc_write, NCRISC: core (x, y), x < 10, y < 8 owns output column tile j = 10y + x of row tile rt:
// R tiles (c, j) for the 4 streams, bo tile j, the inj tile, and the 4 stream selector tiles E_c (row c = 1).
#include "api/dataflow/dataflow_api.h"
#include "hc_common.h"

using namespace hc;

constexpr bool R_IS_DRAM = get_compile_time_arg_val(0) == 1;
constexpr uint32_t BO_PAGE = get_compile_time_arg_val(1);

void kernel_main() {
    const uint32_t j = get_absolute_logical_y() * 10 + get_absolute_logical_x();
    const uint32_t rt = get_common_arg_val<uint32_t>(4);
    const auto R = TensorAccessor(tensor_accessor::make_interleaved_dspec<R_IS_DRAM>(),
                                  get_common_arg_val<uint32_t>(0), F32_TILE);
    const auto BO = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                   get_common_arg_val<uint32_t>(1), BO_PAGE);
    const auto INJ = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                    get_common_arg_val<uint32_t>(2), F32_TILE);
    cb_reserve_back(CB_INJ, 1);
    noc_async_read_page(rt, INJ, get_write_ptr(CB_INJ));
    cb_reserve_back(CB_R, 4);
    const uint32_t pr = get_write_ptr(CB_R);
    for (uint32_t c = 0; c < 4; ++c) {
        noc_async_read_page(rt * ROW_KT + c * N_MIX + j, R, pr + c * F32_TILE);
    }
    cb_reserve_back(CB_GS, 1);
    noc_async_read_page(rt * N_MIX + j, BO, get_write_ptr(CB_GS));
    // E_c: bf16 tile with row c all ones. Tile memory = 4 row-major 16x16 faces; row c < 16 lies in faces 0 and 1.
    cb_reserve_back(CB_X2, 4);
    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_X2));
    for (uint32_t i = 0; i < 4 * BF16_TILE / 4; ++i) {
        e[i] = 0;
    }
    for (uint32_t c = 0; c < 4; ++c) {
        volatile tt_l1_ptr uint32_t* t = e + c * (BF16_TILE / 4);
        for (uint32_t i = 0; i < 8; ++i) {
            t[(c * 16) / 2 + i] = 0x3F803F80u;          // face 0, row c, cols 0..15
            t[(256 + c * 16) / 2 + i] = 0x3F803F80u;    // face 1, row c, cols 16..31
        }
    }
    cb_push_back(CB_X2, 4);
    noc_async_read_barrier();
    cb_push_back(CB_INJ, 1);
    cb_push_back(CB_R, 4);
    cb_push_back(CB_GS, 1);
}
