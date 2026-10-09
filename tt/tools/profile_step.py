"""Device-time profile of one traced decode step (T rows) of TtModel on the P150, per segment and per op type.

Run on the device:  tools/devrun.sh profile-step tools/profile_step.py [--golden chat] [--reps 30] [--extra 2]
                    [--kinds hc_read,hc_write,...] [--layers 48] [--T 1] [--ctx 512] [--no-host-whatif]
The MoE is TtModel's (FNX_MOE=device | cpu, FNX_MOE_PLAN): with the all-device MoE (fnx.tt.moe_device) a dense step
is one segment and its MoE op types are moe_router / moe_gate_up / moe_down.

1. Segments: every captured segment of the T-row sequence (fnx.tt.trace.Sequence; dense span 512 after the golden
   prompt) replayed alone `reps` times back to back, then a sync: device ms per segment kind and the step total.
2. Op types, marginal method: the traces are recaptured with every call of one module function run 1 + `extra` times
   inside the captures (the extra copies on the same inputs, their outputs freed in the capture: the same programs,
   buffers and order as the real step) and the step's device time grows by `extra` x that op type's device time.
   "other" = the step total minus the attributed op types (embedding input, the QSA step-input prep and read-back
   concat, ...).  Nested kinds (gdn_recur inside gdn) are listed but not summed.  "weights" = the device weight
   bytes one call streams (fixed sets; for the device MoE stages the step's routed + shared expert bytes, all tiers),
   with the implied rate.
3. Device MoE only: the step's routing (route buffers of the replayed step: experts per layer, how many in the host
   tier, their bytes) and the host-tier what-if: the step recaptured with every host-tier matrix read from a DRAM
   store page of the same format instead (wrong values, same byte count and access pattern): the step-time
   difference is what the PCIe reads cost beyond resident reads.
The replays rerun stateful programs (GDN state, KV cells at the step's position): the model state after a profile is
meaningless; the tool only times.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import ttnn

from fnx.config import N_LAYERS
from fnx.cpu_experts import CpuExperts
from fnx.ref.model import GOLDEN, read_ids
from fnx.serve_engine import warm_up
from fnx.tt import moe_device as md
from fnx.tt import trace as trace_mod
from fnx.tt.hc import HcRead
from fnx.tt.head import Head
from fnx.tt.model import TtModel, open_mesh
from fnx.tt.moe import DeviceExperts, Moe
from fnx.tt.moe_device import MoeDevice
from fnx.tt.ple import Ple
from fnx.tt.qsa import Qsa
from fnx.weights import Weights

_BYTES = {ttnn.float32: 4.0, ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1088 / 1024, ttnn.bfloat4_b: 576 / 1024,
          ttnn.int32: 4.0}


def tensors(obj):
    """The ttnn tensors in a (nested tuple / list / dict) value."""
    if isinstance(obj, ttnn.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from tensors(o)


def nbytes(t) -> float:
    return float(np.prod(list(t.padded_shape))) * _BYTES[t.dtype]


class Kind:
    """One op type: the function `owner.name` (a method, staticmethod or module function); `weights()`: the device
    bytes of one call's weight stream (None: not a fixed set)."""

    def __init__(self, owner, name: str, weights=None, extra_kwargs=None, nested: bool = False):
        self.owner, self.name, self.weights, self.extra_kwargs = owner, name, weights, extra_kwargs or {}
        self.nested = nested
        self.calls = 0
        self.counting = False

    def patch(self, extra: int) -> None:
        raw = self.owner.__dict__[self.name] if isinstance(self.owner, type) else getattr(self.owner, self.name)
        self.orig = raw
        static = isinstance(raw, staticmethod)
        fn = raw.__func__ if static else raw
        kind = self

        def wrapper(*args, **kwargs):
            if kind.counting:
                kind.calls += 1
            inputs = {id(t) for t in tensors((args, kwargs))}
            for _ in range(extra):
                out = fn(*args, **{**kwargs, **kind.extra_kwargs})
                for t in tensors(out):
                    if id(t) not in inputs:
                        ttnn.deallocate(t)
            return fn(*args, **kwargs)
        setattr(self.owner, self.name, staticmethod(wrapper) if static else wrapper)

    def unpatch(self) -> None:
        setattr(self.owner, self.name, self.orig)


def _wb(f):
    """weights() of a kind: the bytes of the tensors f() lists (None when the module has no such attributes)."""
    def g():
        try:
            return sum(nbytes(x) for x in f())
        except AttributeError:
            return None
    return g


def make_kinds(m, moe_bytes: dict) -> dict[str, Kind]:
    mix_g = next((mix for _, mix, _, _ in m.layers if not isinstance(mix, Qsa)), None)
    L0 = m.layers[0]
    q = next((mix for _, mix, _, _ in m.layers if isinstance(mix, Qsa)), None)
    k = {"hc_read": Kind(HcRead, "forward", _wb(lambda: [L0[0].wd, L0[0].wu, L0[0].gs])),
         "hc_write": Kind(trace_mod, "hc_write")}
    if mix_g is not None:
        k["gdn"] = Kind(type(mix_g), "forward", _wb(lambda: [mix_g.w_mega, mix_g.w_out]))
        if hasattr(type(mix_g), "_recur"):
            k["gdn_recur"] = Kind(type(mix_g), "_recur", _wb(lambda: [mix_g.w_mega]), nested=True)
    if q is not None:
        k["qsa_project"] = Kind(Qsa, "_project", _wb(lambda: [q.w_qkvg]))
        k["qsa_attend"] = Kind(Qsa, "_attend", _wb(lambda: [q.w_o]), extra_kwargs={"free_inputs": False})
    if isinstance(L0[3], MoeDevice):
        k["moe_router"] = Kind(MoeDevice, "route", _wb(lambda: [L0[3].wr]))
        k["moe_gate_up"] = Kind(MoeDevice, "stage_a", lambda: moe_bytes.get("A"))
        k["moe_down"] = Kind(MoeDevice, "stage_b", lambda: moe_bytes.get("B"))
    else:
        k["moe_pre"] = Kind(Moe, "pre", _wb(lambda: [L0[3].wr]))
        k["moe_shared"] = Kind(Moe, "shared", _wb(lambda: [L0[3].wg, L0[3].wu, L0[3].wd]))
        k["moe_dev_experts"] = Kind(DeviceExperts, "run_fixed")
        k["moe_post"] = Kind(Moe, "post")
    if m.n_layers > 1:
        k["ple"] = Kind(Ple, "forward", _wb(lambda: [m.ple.wkv]))
    k["head_matmul"] = Kind(Head, "forward", _wb(lambda: [m.head.w]))
    k["head_argmax"] = Kind(Head, "argmax")
    return k


def seg_times(mesh, seq, reps: int) -> list[tuple[str, float]]:
    """Device ms of every segment of `seq`, each replayed alone `reps` times back to back."""
    out = []
    for (_, _, kind), tid in zip(seq.segs, seq.tids):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        out.append((kind, 1e3 * (time.perf_counter() - t0) / reps))
    return out


def recapture(tr) -> None:
    tr.n_prog = -1                  # forces Tracer.ensure() to recapture every sequence (compile run first)
    tr.ensure("profile_step")


# ------------------------------------------------------------------------------------------------ device MoE
def moe_routing(m, T: int) -> dict:
    """The replayed step's routing per device-MoE layer (route buffers kept by a capture with keep_route): expert
    selections, host-tier counts and the stage A / B weight bytes (resident and host)."""
    out = {"A": 0.0, "B": 0.0, "A_host": 0.0, "B_host": 0.0, "slots": 0, "host_slots": 0, "experts": 0,
           "host_experts": 0}
    for _, _, _, moe in m.layers:
        r = ttnn.to_torch(moe.last_route).numpy().reshape(32, 32)[:T]
        ids = r[:, :10].astype(np.int64)
        st = moe.store
        sel = sorted(set(ids.ravel().tolist())) + [md.SHARED]
        out["slots"] += ids.size
        out["host_slots"] += int(st.host[ids].sum())
        out["experts"] += len(sel) - 1
        out["host_experts"] += int(sum(st.host[e] for e in sel))
        for e in sel:
            for mt in range(3):
                b = md.matrix_bytes(int(st.fmt[e, mt]))
                key = "B" if mt == 2 else "A"
                out[key] += b
                if st.host[e]:
                    out[key + "_host"] += b
    return out


def resident_tables(mesh, m) -> list:
    """Per device-MoE layer with a host tier: (store, table) where every host-tier word points at page 0 of the DRAM
    store of its format and matrix kind instead (the what-if of an all-resident layer: same bytes, wrong values)."""
    out = []
    for _, _, _, moe in m.layers:
        st = moe.store
        if not st.n_host:
            continue
        tab = ttnn.to_torch(st.table).numpy().reshape(-1).view(np.uint32).copy()
        ent = tab[:md.TABLE_STRIDE * (st.n_exp + 1)].reshape(st.n_exp + 1, md.TABLE_STRIDE)
        for e in np.nonzero(st.host)[0]:
            for mt in range(3):
                f = int(st.fmt[e, mt])
                if (st.B if mt == 2 else st.A)[f] is None:
                    raise RuntimeError(f"what-if: no resident {md.FMTS[f]} store for a host-tier matrix")
                ent[e, mt] = f << 24
        t = ttnn.from_torch(torch.from_numpy(tab.view(np.int32)).reshape(1, 1, 1, -1), dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.append((st, t))
    return out


def swap_tables(pairs: list) -> list:
    """Exchange each store's table with its what-if table (call again to swap back)."""
    out = []
    for st, t in pairs:
        old = st.table
        st.table = t
        out.append((st, old))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="chat")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--extra", type=int, default=2, help="extra copies per call in the marginal runs")
    ap.add_argument("--kinds", default="", help="comma list (default all)")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--T", type=int, default=1, help="rows of the profiled step (T > 1: a verify window)")
    ap.add_argument("--ctx", type=int, default=512, help="TtModel max_ctx (512: only the span-512 sequence)")
    ap.add_argument("--no-host-whatif", action="store_true")
    a = ap.parse_args()
    w = Weights()
    experts = CpuExperts(w.model, a.layers, a.threads)
    mesh = open_mesh()
    try:
        t0 = time.time()
        m = TtModel(mesh, w, experts, n_layers=a.layers, max_ctx=a.ctx, verbose=False)
        dev_moe = isinstance(m.layers[0][3], MoeDevice)
        plan = os.environ.get("FNX_MOE_PLAN", "(default)") if dev_moe else "-"
        print(f"TtModel: moe {m.moe}, plan {plan}, max_ctx {a.ctx}, built in {time.time() - t0:.0f} s; "
              f"load average {os.getloadavg()[0]:.1f}", flush=True)
        warm_up(m, None, a.chunk)
        prompt = read_ids(GOLDEN / "prompts" / f"{a.golden}.ids")
        n_ctx = min(len(prompt) - 1, a.ctx - 8 - a.T)
        for i in range(0, n_ctx, a.chunk):
            m.forward(prompt[i:min(i + a.chunk, n_ctx)], head=False)
        spans = tuple(s for s in (512, 2304) if s <= max(512, a.ctx)) if a.ctx > 512 else (512,)
        m.enable_trace(Ts=(a.T,), spans=spans)
        tok = prompt[n_ctx]
        if a.T == 1:
            for _ in range(3):              # traced steps: the captured program runs end to end
                _, ids = m.forward([tok], read_logits=False)
                tok = int(ids[0])
        else:
            m.forward([tok] * a.T, read_logits=False, verify=True)
            m.commit(a.T)
        tr = m._trace
        key = (a.T, 512)
        seq = tr.seqs[key]
        moe_bytes: dict = {}
        kinds = make_kinds(m, moe_bytes)
        sel = [s for s in a.kinds.split(",") if s] or list(kinds)

        # counting run (no extra copies): calls per step, counted during the captures of `seq` only; with the
        # device MoE the route buffers are kept, so one replay gives the step's routing
        orig_capture = seq.capture

        def counted_capture():
            for kd in kinds.values():
                kd.counting = True
            try:
                orig_capture()
            finally:
                for kd in kinds.values():
                    kd.counting = False
        seq.capture = counted_capture
        for kd in kinds.values():
            kd.patch(0)
        if dev_moe:
            for _, _, _, moe in m.layers:
                moe.keep_route = True
        recapture(tr)
        for kd in kinds.values():
            kd.unpatch()
        seq.capture = orig_capture
        rt = None
        if dev_moe:
            seg_times(mesh, seq, 1)
            rt = moe_routing(m, a.T)
            moe_bytes.update(A=rt["A"] / m.n_layers, B=rt["B"] / m.n_layers)
            for _, _, _, moe in m.layers:
                moe.keep_route = False
                for t in (moe.last_route, moe.last_logits):
                    if t is not None:
                        ttnn.deallocate(t)
                moe.last_route = moe.last_logits = None
            recapture(tr)
        base = seg_times(mesh, seq, a.reps)
        total = sum(ms for _, ms in base)
        by_kind: dict[str, list[float]] = {}
        for kind, ms in base:
            by_kind.setdefault(kind, []).append(ms)
        print(f"\nsegments of the T={a.T} step (span {key[1]}), each replayed alone {a.reps}x: total {total:.3f} ms "
              f"device; load average {os.getloadavg()[0]:.1f}")
        for kind, v in by_kind.items():
            print(f"  {kind:8s} {len(v):3d} x {np.mean(v):.3f} ms (min {min(v):.3f} max {max(v):.3f}) = {sum(v):.3f} ms")
        if rt is not None:
            st0 = [lay[3].store for lay in m.layers]
            print(f"device MoE routing of the replayed step: {rt['experts']} expert selections ({rt['experts'] / m.n_layers:.1f}"
                  f"/layer), {rt['host_experts']} in the host tier ({100 * rt['host_experts'] / max(1, rt['experts']):.1f}%; "
                  f"routed slots {rt['host_slots']}/{rt['slots']}); weights per step A {rt['A'] / 1e6:.0f} MB "
                  f"({rt['A_host'] / 1e6:.0f} host) + B {rt['B'] / 1e6:.0f} MB ({rt['B_host'] / 1e6:.0f} host); store "
                  f"{sum(s.nbytes for s in st0) / 1e9:.2f} GB DRAM + {sum(s.host_bytes for s in st0) / 1e9:.2f} GB host",
                  flush=True)

        rows = []
        for name in sel:
            kd = kinds[name]
            if not kd.calls:
                print(f"  {name}: no calls in this step", flush=True)
                continue
            kd.patch(a.extra)
            try:
                recapture(tr)
            finally:
                kd.unpatch()
            t = sum(ms for _, ms in seg_times(mesh, seq, a.reps))
            per_step = (t - total) / a.extra
            n = kd.calls
            wb = kd.weights() if kd.weights is not None else None
            rows.append((name, n, per_step / n, per_step, wb, kd.nested))
            print(f"  marginal {name}: step {total:.3f} -> {t:.3f} ms with {a.extra} extra copies of {n} calls",
                  flush=True)
        recapture(tr)
        end = sum(ms for _, ms in seg_times(mesh, seq, a.reps))
        attributed = sum(r[3] for r in rows if not r[5])
        rows.append(("other (unattributed)", 0, float("nan"), total - attributed, None, False))
        rows.sort(key=lambda r: -r[3])
        print(f"\ndevice profile of one traced T={a.T} decode step: {total:.3f} ms (re-measured at the end: "
              f"{end:.3f} ms); load average {os.getloadavg()[0]:.1f}")
        print("| op type | calls/step | ms/call | ms/step | % of step | weights MB/call | GB/s |")
        print("|---|---|---|---|---|---|---|")
        for name, n, per_call, per_step, wb, nested in rows:
            mb = f"{wb / 1e6:.1f}" if wb else ""
            gbs = f"{wb / 1e6 / per_call:.0f}" if wb and per_call > 0 else ""
            label = f"({name}, nested)" if nested else name
            pc = f"{per_call:.4f}" if n else ""
            print(f"| {label} | {n or ''} | {pc} | {per_step:.3f} | {100 * per_step / total:.1f} | {mb} | {gbs} |")

        if dev_moe and not a.no_host_whatif and any(lay[3].store.n_host for lay in m.layers):
            pairs = resident_tables(mesh, m)
            back = swap_tables(pairs)
            try:
                recapture(tr)
                t_res = sum(ms for _, ms in seg_times(mesh, seq, a.reps))
            finally:
                swap_tables(back)
            for _, t in pairs:
                ttnn.deallocate(t)
            recapture(tr)
            hb = rt["A_host"] + rt["B_host"]
            print(f"\nhost-tier what-if (every host-tier matrix read from DRAM instead): step {total:.3f} -> "
                  f"{t_res:.3f} ms: the {rt['host_experts']} host-tier experts ({hb / 1e6:.0f} MB over PCIe) cost "
                  f"{total - t_res:.3f} ms per step ({hb / 1e6 / max(total - t_res, 1e-6):.1f} GB/s marginal)",
                  flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
