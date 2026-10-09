// HcRead, NCRISC: the MAIN cores' weight reads (down and up projections, on NOC0: the fast DRAM read path) and the
// waits that turn remote writes into local CB pushes.
#include "api/dataflow/dataflow_api.h"
#include "hc_common.h"

#ifdef HC_PROF
#include "tools/profiler/kernel_profiler.hpp"
#define HC_ZONE(name) DeviceZoneScopedN(name)
#else
#define HC_ZONE(name)
#endif

using namespace hc;

constexpr uint32_t NN = get_compile_time_arg_val(0);

FORCE_INLINE void wait_ge(uint32_t sem, uint32_t val) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
    noc_semaphore_wait_min(p, val);
}

// Remote producers write straight into this core's CB memory; the local push hands the tiles to compute.
FORCE_INLINE void land(uint32_t cb, uint32_t tiles, uint32_t sem, uint32_t target) {
    cb_reserve_back(cb, tiles);
    wait_ge(sem, target);
    cb_push_back(cb, tiles);
}

void kernel_main() {
    const uint32_t lx = get_absolute_logical_x();
    const uint32_t ly = get_absolute_logical_y();
#if ROLE == 1  // MAIN
    const uint32_t s = ly;
    const uint32_t ntp = lx <= 6 ? 4 : 3;
    // NOC0 streams the down weights from the start (BRISC reads PREP's R tiles and gamma rows on NOC1 meanwhile) in
    // WD_CHUNKS chunks (transaction ids TRID_WD0.., DOWN multiplies a chunk as soon as it has landed), then this core's
    // share of the up weights (TRID_WU; MIX needs them last).
    if (lx < NN) {
        const auto WD = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                       get_common_arg_val<uint32_t>(2), BF16_TILE);
        cb_reserve_back(CB_WD, K_SLICE);
        const uint32_t p = get_write_ptr(CB_WD);
        const uint32_t first = (s * NN + lx) * K_SLICE;
        for (uint32_t c = 0; c < WD_CHUNKS; ++c) {
            noc_async_read_set_trid(TRID_WD0 + c);
            for (uint32_t k = c * WD_CHUNK; k < (c + 1) * WD_CHUNK; ++k) {
                noc_async_read_page(first + k, WD, p + k * BF16_TILE);
            }
        }
    }
    noc_async_read_set_trid(0);
    if (lx < NN) {
        HC_ZONE("WD");
        for (uint32_t c = 0; c < WD_CHUNKS; ++c) {
            noc_async_read_barrier_with_trid(TRID_WD0 + c);
            cb_push_back(CB_WD, WD_CHUNK);
        }
    }
    if (lx < 10) {  // MIX column j = 10s + x
        const auto WU = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                       get_common_arg_val<uint32_t>(3), BF16_TILE);
        cb_reserve_back(CB_WU, 40);
        const uint32_t p = get_write_ptr(CB_WU);
        const uint32_t j = ly * 10 + lx;
        noc_async_read_set_trid(TRID_WU);
        for (uint32_t q = 0; q < 40 - WU_B; ++q) {   // the last WU_B tiles: BRISC on NOC1 (hcr_writer.cpp)
            noc_async_read_page(j * 40 + q, WU, p + q * BF16_TILE);
        }
        noc_async_read_set_trid(0);
    }
    if (lx < NN) {
        HC_ZONE("ACT");
        land(CB_ACT, 2 * K_SLICE, SEM_ACT, K_SLICE);
    }
    if (lx < 10) {
        {
            HC_ZONE("WU");
            noc_async_read_barrier_with_trid(TRID_WU);
            if constexpr (WU_B > 0) {
                wait_ge(SEM_WU, 1);
            }
            cb_push_back(CB_WU, 40);
        }
        HC_ZONE("LAND");
        land(CB_YMIX, 4, SEM_Y, 4);
        land(CB_RS, 4, SEM_RS, 1);
        land(CB_LO, 20, SEM_LO, 10);
    }
    noc_async_read_barrier();
#elif ROLE == 2  // SSQ
    cb_reserve_back(CB_ONES, 1);
    volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONES));
    for (uint32_t i = 0; i < BF16_TILE / 4; ++i) {
        ones[i] = 0x3F803F80u;  // bf16 1.0 pairs
    }
    cb_push_back(CB_ONES, 1);
    land(CB_X2IN, 2 * K_SLICE, SEM_X2, K_SLICE);
#elif ROLE == 3  // RS
    land(CB_SSQIN, N_SLICE, SEM_SSQ, N_SLICE);
#elif ROLE == 4  // RED
    land(CB_PIN, N_SLICE, SEM_PART, N_SLICE);
    land(CB_RS, 4, SEM_RS, 1);
#endif
}
