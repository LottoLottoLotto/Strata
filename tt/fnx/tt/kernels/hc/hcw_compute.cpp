// hc_write, compute: w = 2*sigmoid(inj/4) (cols 0..3 = streams), broadcast of column c over a tile as the exact
// bf16 matmul (hi(w) + lo(w)) @ E_c, then out_c = R_c + w_c * bo in fp32.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "hc_common.h"
#include "hc_sfpu.h"

using namespace hc;

constexpr uint32_t PRECISE = get_compile_time_arg_val(2);

void kernel_main() {
    compute_kernel_hw_startup(CB_INJ, CB_INJ, CB_HL);
    cb_wait_front(CB_INJ, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_INJ);
    copy_tile_to_dst_init_short(CB_INJ);
    copy_tile(CB_INJ, 0, 0);
    hc_sfpu_init_sigmoid();
    hc_wsplit();
    tile_regs_commit();
    cb_reserve_back(CB_HL, 2);
    tile_regs_wait();
    pack_reconfig_data_format(CB_HL);
    pack_tile(0, CB_HL);
    pack_tile(1, CB_HL);
    tile_regs_release();
    cb_push_back(CB_HL, 2);
    cb_pop_front(CB_INJ, 1);

    cb_wait_front(CB_HL, 2);
    cb_wait_front(CB_X2, 4);
    cb_wait_front(CB_R, 4);
    cb_wait_front(CB_GS, 1);
    pack_reconfig_data_format(CB_OUT);
    for (uint32_t c = 0; c < 4; ++c) {
        reconfig_data_format_srca(CB_X2);
        reconfig_data_format_srcb(CB_HL);
        matmul_init(CB_HL, CB_X2);
        tile_regs_acquire();
        matmul_tiles(CB_HL, CB_X2, 0, c, 0);
        if constexpr (PRECISE) {
            matmul_tiles(CB_HL, CB_X2, 1, c, 0);
        }
        reconfig_data_format_srca(CB_R);
        copy_tile_to_dst_init_short(CB_R);
        copy_tile(CB_R, c, 1);
        reconfig_data_format_srca(CB_GS);
        copy_tile_to_dst_init_short(CB_GS);
        copy_tile(CB_GS, 0, 2);
        hc_sfpu_init_plain();
        hc_write_upd();
        tile_regs_commit();
        cb_reserve_back(CB_OUT, 1);
        tile_regs_wait();
        pack_tile(1, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
    }
    cb_pop_front(CB_HL, 2);
    cb_pop_front(CB_X2, 4);
    cb_pop_front(CB_R, 4);
    cb_pop_front(CB_GS, 1);
}
