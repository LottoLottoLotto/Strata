// Device-profiler zones of the all-device MoE kernels: MD_ZONE("name") opens a zone for the rest of the scope when
// the program is built with MD_PROF (fnx.tt.moe_device: FNX_MD_PROF=1, with TT_METAL_DEVICE_PROFILER=1), else nothing.
#pragma once
#ifdef MD_PROF
#include "tools/profiler/kernel_profiler.hpp"
#define MD_ZONE(name) DeviceZoneScopedN(name)
#else
#define MD_ZONE(name)
#endif
