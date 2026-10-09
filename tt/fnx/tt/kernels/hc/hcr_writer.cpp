// HcRead, BRISC: PREP's R / gamma reads, every remote write / multicast / semaphore signal and the DRAM output writes
// (MAIN cores on NOC1 while NCRISC streams the weights on NOC0; the other roles on NOC0, see hc.py _read_program).
#include "api/dataflow/dataflow_api.h"
#include "hc_common.h"
#include "hc_dataflow.h"

using namespace hc;

constexpr uint32_t NN = get_compile_time_arg_val(0);
constexpr uint32_t OUT_PAGE = get_compile_time_arg_val(3);
constexpr bool R_IS_DRAM = get_compile_time_arg_val(2) == 1;

FORCE_INLINE uint32_t vx(uint32_t x) { return kernel_compile_time_args[CT_VX + x]; }
FORCE_INLINE uint32_t vy(uint32_t y) { return kernel_compile_time_args[CT_VY + y]; }
FORCE_INLINE uint64_t at(uint32_t x, uint32_t y, uint32_t addr) { return get_noc_addr(vx(x), vy(y), addr); }
// multicast to the logical rectangle [x0, x1] x [y0, y1]; a NOC1 multicast starts at the far corner (start / end
// swapped, the tt-metal convention for NOC1)
FORCE_INLINE uint64_t rect(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr) {
    if constexpr (noc_index == 1) {
        return get_noc_multicast_addr(vx(x1), vy(y1), vx(x0), vy(y0), addr);
    }
    return get_noc_multicast_addr(vx(x0), vy(y0), vx(x1), vy(y1), addr);
}

#ifdef HC_PROF
#include "tools/profiler/kernel_profiler.hpp"
#define HC_ZONE(name) DeviceZoneScopedN(name)
#else
#define HC_ZONE(name)
#endif

void kernel_main() {
    const uint32_t lx = get_absolute_logical_x();
    const uint32_t ly = get_absolute_logical_y();
    const uint32_t rt = get_common_arg_val<uint32_t>(6);
#if ROLE == 1  // MAIN
    const uint32_t s = ly, c = ly >> 1, h = ly & 1;
    const uint32_t ntp = lx <= 6 ? 4 : 3;
    // CB bases are identical on every core of the program; BRISC never produces into these CBs.
    const uint32_t act_base = get_write_ptr(CB_ACT);
    const uint32_t x2in_base = get_write_ptr(CB_X2IN);
    const uint32_t ymix_base = get_write_ptr(CB_YMIX);
    const uint32_t pin_base = get_write_ptr(CB_PIN);
    // PREP's inputs on this NoC (NOC1) while NCRISC streams the weights on NOC0: R tiles and gamma rows (-> tiles)
    cb_reserve_back(CB_R, ntp);
    cb_reserve_back(CB_GS, ntp);
    const uint32_t pg = get_write_ptr(CB_GS);
    {
        const auto R = TensorAccessor(tensor_accessor::make_interleaved_dspec<R_IS_DRAM>(),
                                      get_common_arg_val<uint32_t>(0), F32_TILE);
        const auto GS = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                       get_common_arg_val<uint32_t>(1), GS_ROW);
        const uint32_t pr = get_write_ptr(CB_R);
        for (uint32_t i = 0; i < ntp; ++i) {
            const uint32_t kt = K_SLICE * s + lx + 11 * i;
            noc_async_read_page(rt * ROW_KT + kt, R, pr + i * F32_TILE);
            noc_async_read_page(kt, GS, pg + i * F32_TILE);    // the row -> words 0..31 of the gamma tile
        }
        noc_async_read_barrier();
        cb_push_back(CB_R, ntp);
    }
    {
        HC_ZONE("GFILL");
        gamma_tiles(pg, ntp);
    }
    cb_push_back(CB_GS, ntp);
    // PREP: hi/lo of y -> the row's gather core (GATHER_X, s), hi/lo of R^2 -> SSQ(s), y -> MIX core of the column.
    for (uint32_t i = 0; i < ntp; ++i) {
        HC_ZONE("SEND");
        const uint32_t k = lx + 11 * i;
        cb_wait_front(CB_HL, 2 * (i + 1));
        noc_async_write(get_read_ptr(CB_HL) + i * 2 * BF16_TILE, at(GATHER_X, s, act_base + k * 2 * BF16_TILE),
                        2 * BF16_TILE);
        cb_wait_front(CB_X2, 2 * (i + 1));
        noc_async_write(get_read_ptr(CB_X2) + i * 2 * BF16_TILE, at(s, 8, x2in_base + k * 2 * BF16_TILE),
                        2 * BF16_TILE);
        cb_wait_front(CB_Y, i + 1);
        const uint32_t j = 40 * h + k;
        noc_async_write(get_read_ptr(CB_Y) + i * F32_TILE, at(j % 10, j / 10, ymix_base + c * F32_TILE), F32_TILE);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(at(GATHER_X, s, get_semaphore(SEM_GATHER)), ntp);
    noc_semaphore_inc(at(s, 8, get_semaphore(SEM_X2)), ntp);
    for (uint32_t i = 0; i < ntp; ++i) {
        const uint32_t j = 40 * h + lx + 11 * i;
        noc_semaphore_inc(at(j % 10, j / 10, get_semaphore(SEM_Y)), 1);
    }
    cb_pop_front(CB_HL, 2 * ntp);
    cb_pop_front(CB_X2, 2 * ntp);
    cb_pop_front(CB_Y, ntp);
    if (lx == GATHER_X) {  // the row's slice, complete -> the other 10 cores of the row (and this core's NCRISC)
        HC_ZONE("GATHER");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER)), K_SLICE);
        noc_async_write_multicast(act_base, rect(0, s, GATHER_X - 1, s, act_base), 2 * K_SLICE * BF16_TILE,
                                  GATHER_X);
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(rect(0, s, GATHER_X - 1, s, get_semaphore(SEM_ACT)), K_SLICE, GATHER_X);
        noc_semaphore_inc(at(GATHER_X, s, get_semaphore(SEM_ACT)), K_SLICE);
    }
    if constexpr (WU_B > 0) {
        if (lx < 10) {  // the last WU_B up-weight tiles of MIX column j on NOC1 (NCRISC reads the rest on NOC0)
            const auto WU = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                           get_common_arg_val<uint32_t>(3), BF16_TILE);
            const uint32_t p = get_write_ptr(CB_WU);   // the CB base: NCRISC produces CB_WU, BRISC only fills it
            const uint32_t j = ly * 10 + lx;
            for (uint32_t q = 40 - WU_B; q < 40; ++q) {
                noc_async_read_page(j * 40 + q, WU, p + q * BF16_TILE);
            }
        }
    }
    if (lx < NN) {  // DOWN partial -> RED(lx) slot s
        cb_wait_front(CB_PART, 1);
        noc_async_write(get_read_ptr(CB_PART), at(lx, 9, pin_base + s * F32_TILE), F32_TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(at(lx, 9, get_semaphore(SEM_PART)), 1);
        cb_pop_front(CB_PART, 1);
    }
    if constexpr (WU_B > 0) {
        if (lx < 10) {
            noc_async_read_barrier();
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WU)) = 1;
        }
    }
    if (lx < 10) {  // MIX output tile
        const auto OUT = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(4), OUT_PAGE);
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(rt * N_MIX + ly * 10 + lx, OUT, get_read_ptr(CB_OUT));
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
#elif ROLE == 2  // SSQ(lx) -> RS slot lx
    const uint32_t ssqin_base = get_write_ptr(CB_SSQIN);
    cb_wait_front(CB_SSQ, 1);
    noc_async_write(get_read_ptr(CB_SSQ), at(8, 8, ssqin_base + lx * F32_TILE), F32_TILE);
    noc_async_write_barrier();
    noc_semaphore_inc(at(8, 8, get_semaphore(SEM_SSQ)), 1);
    cb_pop_front(CB_SSQ, 1);
#elif ROLE == 3  // RS -> MIX cores and RED cores; then the LO hub: the 10 RED cores' silu tiles -> MIX cores
    cb_wait_front(CB_RS, 4);
    const uint32_t src = get_read_ptr(CB_RS);  // == the CB base, the remote destination address too
    noc_async_write_multicast(src, rect(0, 0, 9, 7, src), 4 * F32_TILE, N_MIX);
    noc_async_write_multicast(src, rect(0, 9, NN - 1, 9, src), 4 * F32_TILE, NN);
    noc_async_write_barrier();
    noc_semaphore_inc_multicast(rect(0, 0, 9, 7, get_semaphore(SEM_RS)), 1, N_MIX);
    noc_semaphore_inc_multicast(rect(0, 9, NN - 1, 9, get_semaphore(SEM_RS)), 1, NN);
    cb_pop_front(CB_RS, 4);
    {
        HC_ZONE("HUB");
        const uint32_t lo_base = get_write_ptr(CB_LO);   // this core's CB_LO memory: the gather buffer
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_LO)), 10);
        noc_async_write_multicast(lo_base, rect(0, 0, 9, 7, lo_base), 20 * BF16_TILE, N_MIX);
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(rect(0, 0, 9, 7, get_semaphore(SEM_LO)), 10, N_MIX);
    }
#elif ROLE == 4  // RED(lx)
    if (lx < 10) {  // hi/lo of silu(p_lx) -> the LO hub, slot lx
        const uint32_t lo_base = get_write_ptr(CB_LO);
        cb_wait_front(CB_RED, 2);
        noc_async_write(get_read_ptr(CB_RED), at(HUB_X, HUB_Y, lo_base + lx * 2 * BF16_TILE), 2 * BF16_TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(at(HUB_X, HUB_Y, get_semaphore(SEM_LO)), 1);
        cb_pop_front(CB_RED, 2);
    } else {
        const auto INJ = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(5), F32_TILE);
        cb_wait_front(CB_INJ, 1);
        noc_async_write_page(rt, INJ, get_read_ptr(CB_INJ));
        noc_async_write_barrier();
        cb_pop_front(CB_INJ, 1);
    }
#endif
    noc_async_atomic_barrier();
}
