"""Device DRAM left for routed experts, measured on exactly what the serving engine builds.

    tools/devrun.sh dram tools/dram_budget.py [--plan NAME] [--ctx 32768] [--spec 4]

Builds TtModel(device MoE, plan, ctx, with the MTP drafter's host items reserved), the MTP drafter, runs the engine's
warm-up and captures the traces (enable_trace), then prints allocated/free DRAM, the trace region use, and the expert
budget a new tiered plan may use while keeping --headroom GB free: budget = plan's resident bytes + free - headroom.
"""
import argparse
import os
import time

import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default=None, help="default: fnx.tt.model.DEFAULT_PLAN")
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--spec", type=int, default=4)
ap.add_argument("--headroom", type=float, default=1.5, help="GB of DRAM to keep free")
a = ap.parse_args()
if a.plan:
    os.environ["FNX_MOE_PLAN"] = a.plan

from fnx.config import N_EXPERT, N_LAYERS  # noqa: E402
from fnx.cpu_experts import CpuExperts  # noqa: E402
from fnx.serve_engine import make_drafter, warm_up  # noqa: E402
from fnx.tt import model as tmodel  # noqa: E402
from fnx.tt.mtp import reserve_items  # noqa: E402
from fnx.weights import Weights  # noqa: E402


def dram(mesh) -> tuple[float, float]:
    m = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    nb = m.num_banks
    return m.total_bytes_allocated_per_bank * nb / 1e9, m.total_bytes_free_per_bank * nb / 1e9


w = Weights()
mesh = tmodel.open_mesh()
EXPERT_BYTES = 2_764_800            # one routed expert, GPTQ-BFP4 gate + up + down (fnx.quant tiered plans)

try:
    plan = os.environ.get("FNX_MOE_PLAN", tmodel.DEFAULT_PLAN)
    t0 = time.time()
    model = tmodel.TtModel(mesh, w, CpuExperts(w.model, N_LAYERS, 4), max_ctx=a.ctx, verbose=False, moe="device",
                           host_reserve=reserve_items() if a.spec > 1 else 0)
    stores = [lay[3].store for lay in model.layers]
    n_res = sum(int((~s.host[:N_EXPERT]).sum()) for s in stores)     # routed experts resident in DRAM
    resident = n_res * EXPERT_BYTES / 1e9                             # what a tiered plan's budget counts
    al, fr = dram(mesh)
    print(f"TtModel(plan {plan}, ctx {a.ctx}) in {time.time() - t0:.0f} s: allocated {al:.3f} GB, free {fr:.3f} GB; "
          f"{n_res} routed experts resident = {resident:.3f} GB", flush=True)
    drafter = make_drafter(mesh, w, a.ctx, a.spec - 1, model=model) if a.spec > 1 else None
    al, fr = dram(mesh)
    print(f"+ MTP drafter: allocated {al:.3f} GB, free {fr:.3f} GB", flush=True)
    warm_up(model, drafter, 128)
    model.enable_trace(Ts=(1, a.spec) if a.spec > 1 else (1,), verbose=False,
                       drafter=drafter.mtp if drafter is not None else None)
    al, fr = dram(mesh)
    tr = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
    print(f"+ warm-up + traces: allocated {al:.3f} GB, free {fr:.3f} GB; trace region "
          f"{tr.total_bytes_allocated_per_bank * tr.num_banks / 1e9:.3f} of "
          f"{tr.total_bytes_per_bank * tr.num_banks / 1e9:.3f} GB used", flush=True)
    budget = resident + fr - a.headroom
    print(f"BUDGET: resident expert bytes for a new tiered plan keeping {a.headroom} GB free: {budget * 1e9:.4g}",
          flush=True)
finally:
    ttnn.close_mesh_device(mesh)
