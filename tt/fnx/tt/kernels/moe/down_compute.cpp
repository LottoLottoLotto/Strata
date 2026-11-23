// Device experts, stage B compute: y[g][n] = sum over the group's entries s of h[s] @ D_e[n]^T, FP32 DEST, FP32 out.
// Units as in down_reader.cpp; every group has at least one entry (DeviceExperts.entries).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"

constexpr uint32_t cb_h = 0, cb_w = 1, cb_meta = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NT = get_named_compile_time_arg_val("nt");
    constexpr uint32_t JT = get_named_compile_time_arg_val("jt");
    constexpr uint32_t NC = get_named_compile_time_arg_val("ncores");
    const uint32_t core = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t units = read_tile_value(cb_meta, 0, 1) * NT;
    matmul_init(cb_h, cb_w, 1);
    for (uint32_t u = core; u < units; u += NC) {
        const uint32_t g = u / NT;
        const uint32_t s0 = read_tile_value(cb_meta, 0, 4 + g);
        const uint32_t s1 = read_tile_value(cb_meta, 0, 5 + g);
        tile_regs_acquire();
        for (uint32_t s = s0; s < s1; ++s) {
            cb_wait_front(cb_h, JT);
            cb_wait_front(cb_w, JT);
            for (uint32_t k = 0; k < JT; ++k) {
                matmul_tiles(cb_h, cb_w, k, k, 0);
            }
            cb_pop_front(cb_w, JT);
            cb_pop_front(cb_h, JT);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
