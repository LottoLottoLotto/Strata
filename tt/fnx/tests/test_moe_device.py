"""test_moe_device: fnx.tt.moe_device.MoeDevice (router, top-10, every routed expert and the shared expert on the
device, no host round trip) on real layers against the numpy references.

    tools/devrun.sh test_moe_device fnx/tests/test_moe_device.py [--layers 0,4] [--tokens 32] [--store rtn:96]
        [--inputs PATH.npz] [--fidelity LoFi] [--iters 50] [--workers 16]

--store rtn:N   round-to-nearest of the GGUF dequantized experts, the layer's N hottest experts (Strata's Coder
                profile) bfp4, the rest bfp2 (rtn:256 = all bfp4, rtn:0 = all bfp2); tiles cached under
                /var/tmp/strata-tt/dmoe/rtn (converted in --workers processes before the device opens)
--store quant:NAME   ExpertQuant's GPTQ plan NAME (fnx.quant.store.Plan)
--inputs        real MoE inputs (npz with mixed_L<L> [n, 2560] f32 rows); default: realistic FFN-half rows (token
                embeddings as a 4-stream residual read through hc_ffn, as test_moe)
--full NAME     only: TtModel(max_ctx 32768, moe="device", FNX_MOE_PLAN=NAME: resident experts in DRAM, the host tier
                in TtModel's pinned host pool) + the MTP drafter; FAIL unless >= HEADROOM_GB stay free; then an eager
                greedy decode of the chat prompt
--capture PATH  write such an npz (every 4th row of each layer's input over the golden chat / code / doc prompts,
                TtModel eager with the current Moe) and exit

Checks per layer (FAIL on any):
  routing    device ids (route buffer) vs the f64 host router (fnx.ref.moe.route) as sets per row: flips allowed
             only where the exact 10th-11th logit margin is below ROUTE_MARGIN (device f32 logit error, ~3e-4
             relative per product); route weights vs the host softmax on the device's own selection (<= 2e-5 abs)
             and the shared sigmoid gate (<= 2e-5 abs)
  kernel     ||bo - bo_emu|| / ||bo_emu|| <= KERNEL_TOL: bo_emu = the device's arithmetic in f64 on its own routing,
             with the exact stored weights (block-float tiles decoded), BF16 x and BF16 h
  quality    (reported) ||bo - bo_float|| / ||bo_float||, bo_float = the all-float reference (exact routing, f32
             dequantized GGUF experts, f32 shared expert); and the emulation on exact routing vs bo_float (the
             quantization error alone)
  invariance a T=4 window's rows are bitwise the T=1 results
  timing     device ms per layer call at T=1 and T=4 (traced replay) and the expert bytes read / s
  host tier  (--host-split 1, default) the same matrices with the odd experts in pinned host memory (HostPool, read
             over PCIe inside the kernels), once per --host-modes pool mode (2m: contiguous items; 4k: the 4k region,
             paged items read through the store's page table; 4k-paged: the same pool with every item treated as
             paged): T=1 and T=4 results and routes bitwise the all-resident ones; traced ms per call against all
             resident
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fnx.config import N_EMBD, N_EXPERT, N_FF
from fnx.ref import moe as M
from fnx.ref.hc import hc_read, hc_weights
from fnx.weights import Weights

F64 = np.float64
ROUTE_MARGIN = 2e-3          # exact 10th-11th logit gap below which a flip is a rounding tie
KERNEL_TOL = 5e-3
WT_TOL = 2e-5
HEADROOM_GB = 1.5            # free device DRAM required after TtModel(32K) + MTP + every layer's expert store
RTN_DIR = Path(os.environ.get("FNX_DMOE_RTN", "/var/tmp/strata-tt/dmoe/rtn"))


def parse_layers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, F64) - b) / np.linalg.norm(b))


def bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16                        # round to nearest even
    return u.astype(np.uint32).view(np.float32)


def realistic_mixed(w, L: int, E: np.ndarray, rng) -> np.ndarray:
    T = E.shape[0]
    other = E[rng.permutation(T)]
    R = E[:, None, :] + 0.5 * rng.standard_normal((T, 4, 1)).astype(np.float32) * other[:, None, :]
    g, Wd, Wu, _ = hc_weights(w, f"blk.{L}.hc_ffn")
    with np.errstate(over="ignore"):
        mixed, _ = hc_read(R.astype(np.float32), g, Wd, Wu)
    return mixed


def embeddings(model, toks) -> np.ndarray:
    rb = 2560 // 256 * 136
    raw = model.raw("token_embd.weight")
    return np.stack([model._dequant("IQ4_XS", raw[t * rb:(t + 1) * rb], 2560) for t in toks])


# ---------------------------------------------------------------------------------------- RTN tile cache (host)
_W = None


def _rtn_job(args):
    import torch
    from fnx.tt import moe_device as md
    global _W
    L, e, m, f = args
    if _W is None:
        torch.set_num_threads(1)
        _W = Weights()
    W = _W.model.get_expert(f"blk.{L}.ffn_{md.MATS[m]}_exps.weight", e)
    return md.rtn_tiles(np.ascontiguousarray(W.T), f)


def rtn_tiles_cached(w, L: int, fmt: np.ndarray, workers: int) -> dict:
    """{(e, m): tiles} of layer L for the [257, 3] format codes: routed matrices from per-format caches of all 256 x 3
    (built in `workers` spawned processes when missing), the shared expert converted here."""
    import multiprocessing

    from fnx.tt import moe_device as md
    RTN_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in sorted(set(fmt[:N_EXPERT].ravel().tolist())):
        path = RTN_DIR / f"L{L:02d}.{md.FMTS[f]}.npy"
        if not path.exists():
            t0 = time.time()
            jobs = [(L, e, m, f) for e in range(N_EXPERT) for m in range(3)]
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
                arr = np.stack(list(ex.map(_rtn_job, jobs, chunksize=4)))
            np.save(str(path) + ".tmp.npy", arr)
            os.replace(str(path) + ".tmp.npy", path)
            print(f"  RTN cache L{L} {md.FMTS[f]}: {arr.nbytes / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
        arr = np.load(path, mmap_mode="r")
        for e, m in zip(*np.nonzero(fmt[:N_EXPERT] == f)):
            out[(int(e), int(m))] = arr[e * 3 + m]
    for m in range(3):
        out[(md.SHARED, m)] = md.shared_tiles(w, L, m, int(fmt[md.SHARED, m]))
    return out


def capture_inputs(path: str, every: int = 4, chunk: int = 32) -> None:
    """Real MoE inputs: the golden chat / code / doc prompts through TtModel (eager, current Moe with CPU experts);
    every `every`-th row of each layer's `mixed` -> npz mixed_L<L> [n, 2560] f32 (+ src: corpus per row)."""
    import ttnn

    from fnx.config import N_LAYERS
    from fnx.cpu_experts import CpuExperts
    from fnx.tt import moe as tmoe
    from fnx.tt.model import TtModel, open_mesh
    w = Weights()
    rows: dict[int, list] = {L: [] for L in range(N_LAYERS)}
    src: list[str] = []
    orig = tmoe.Moe.forward
    state = {"name": ""}

    def hook(self, mixed, T=None):
        T = int(mixed.shape[-2]) if T is None else int(T)
        x = ttnn.to_torch(mixed).float().numpy().reshape(-1, N_EMBD)[:T]
        rows[self.L].append(x[::every].copy())
        if self.L == 0:
            src.extend([state["name"]] * len(x[::every]))
        return orig(self, mixed, T)

    tmoe.Moe.forward = hook
    mesh = open_mesh()
    try:
        model = TtModel(mesh, w, CpuExperts(w.model, N_LAYERS, 16), max_ctx=4096, verbose=False)
        for name in ("chat", "code", "doc"):
            toks = [int(t) for t in Path(f"/var/tmp/strata-tt/golden/prompts/{name}.ids").read_text().split(",") if t]
            state["name"] = name
            model.reset()
            t0 = time.time()
            for i in range(0, len(toks), chunk):
                model.forward(toks[i:i + chunk], head=False)
            print(f"capture {name}: {len(toks)} tokens in {time.time() - t0:.0f} s", flush=True)
    finally:
        tmoe.Moe.forward = orig
        ttnn.close_mesh_device(mesh)
    np.savez(path, src=np.array(src), **{f"mixed_L{L}": np.concatenate(v) for L, v in rows.items()})
    print(f"wrote {path}: {len(src)} rows per layer")


def _free(*ts) -> None:
    import ttnn
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def timed(mesh, fn, iters: int) -> float:
    """Device seconds per call of fn (device work only): one eager warm-up (compiles), one trace capture, `iters`
    back-to-back replays."""
    import ttnn
    fn()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / iters
    ttnn.release_trace(mesh, tid)
    return dt


# ---------------------------------------------------------------------------------------- references
class Emu:
    """The device's arithmetic in f64 on given routing: exact stored weights, BF16 x and h."""

    def __init__(self, lm, L: int):
        from fnx.tt import moe_device as md
        self.md, self.lm, self.L, self.cache = md, lm, L, {}

    def weights(self, e: int):
        if e not in self.cache:
            md = self.md
            shp = ((N_EMBD, N_FF), (N_EMBD, N_FF), (N_FF, N_EMBD))
            self.cache[e] = [md.tiles_to_float(np.asarray(self.lm.tiles(e, m)), *shp[m], int(self.lm.fmt[e, m]))
                             .astype(F64) for m in range(3)]
        return self.cache[e]

    def row(self, x: np.ndarray, ids, wts, sg: float) -> np.ndarray:
        xb = bf16(x).astype(F64)
        y = np.zeros(N_EMBD, F64)
        for e, wt in list(zip(ids, wts)) + [(self.md.SHARED, sg)]:
            G, U, D = self.weights(int(e))
            g, u = xb @ G, xb @ U
            h = bf16((g / (1 + np.exp(-g))) * u * float(wt)).astype(F64)
            y += h @ D
        return y


def full_model(plan_name: str, prompt: str = "chat") -> bool:
    """TtModel(max_ctx 32768, moe="device", FNX_MOE_PLAN = the GPTQ plan: resident experts in DRAM, the rest in the
    host pool) + the MTP drafter: the device must keep >= HEADROOM_GB free; then one eager pass over a golden prompt
    (ms per eager T=1 step)."""
    import ttnn

    from fnx.config import N_LAYERS
    from fnx.cpu_experts import CpuExperts
    from fnx.serve_engine import make_drafter
    from fnx.tt import model as tmodel
    w = Weights()
    os.environ["FNX_MOE_PLAN"] = plan_name
    mesh = tmodel.open_mesh()
    try:
        cx = CpuExperts(w.model, N_LAYERS, 16)
        t0 = time.time()
        from fnx.tt.mtp import reserve_items
        model = tmodel.TtModel(mesh, w, cx, max_ctx=32768, verbose=False, moe="device", host_reserve=reserve_items())
        t_model = time.time() - t0
        make_drafter(mesh, w, 32768, 3, model=model)
        m = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        free = m.total_bytes_free_per_bank * m.num_banks / 1e9
        st = [lay[3].store for lay in model.layers]
        fit = free >= HEADROOM_GB
        print(f"full model: TtModel(32K, plan {plan_name}: {sum(s.nbytes for s in st) / 1e9:.2f} GB experts in DRAM + "
              f"{sum(s.host_bytes for s in st) / 1e9:.2f} GB in the host pool, "
              f"{(model.host_pool.nbytes if model.host_pool else 0) / 1e9:.2f} GB pinned"
              f"{f', {model.host_pool.alloc_stats}' if model.host_pool else ''}) in {t_model:.0f} s + MTP: "
              f"allocated {m.total_bytes_allocated_per_bank * m.num_banks / 1e9:.3f} GB, free {free:.3f} GB (largest "
              f"contiguous {m.largest_contiguous_bytes_free_per_bank / 1e9:.3f} GB/bank) {'OK' if fit else 'FAIL'} "
              f"(>= {HEADROOM_GB} GB headroom)", flush=True)
        toks = [int(t) for t in Path(f"/var/tmp/strata-tt/golden/prompts/{prompt}.ids").read_text().split(",") if t]
        model.reset()
        model.forward(toks[:-1], head=False)
        out, tok = [], toks[-1]
        times = []
        for _ in range(16):
            t0 = time.perf_counter()
            tok = int(model.forward([tok], read_logits=False)[1][0])
            times.append(time.perf_counter() - t0)
            out.append(tok)
        print(f"  eager greedy decode after the {prompt} prompt: {out}; eager step {1e3 * np.median(times):.1f} ms "
              f"(median of 16)")
        return fit
    finally:
        ttnn.close_mesh_device(mesh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,4")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--store", default="rtn:96")
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--fidelity", default="HiFi3,LoFi", help="comma list: the first is tested, the others reported")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture", default=None, help="only write real MoE inputs (TtModel on the golden prompts) here")
    ap.add_argument("--full", default=None, help="only the full-model check with this GPTQ plan (fnx.quant.store)")
    ap.add_argument("--host-split", type=int, default=1, help="1: also the host-tier check (odd experts in a HostPool)")
    ap.add_argument("--host-modes", default="2m,4k,4k-paged", help="host-tier check pool modes (fnx.tt.hostmem)")
    a = ap.parse_args()
    if a.capture:
        capture_inputs(a.capture)
        return 0
    if a.full:
        return 0 if full_model(a.full) else 1
    layers = parse_layers(a.layers)
    w = Weights()
    model = w.model

    from fnx.tt import moe_device as md
    rng = np.random.default_rng(a.seed)
    if a.inputs:
        inp = np.load(a.inputs)
        X = {L: np.ascontiguousarray(inp[f"mixed_L{L}"][:a.tokens], np.float32) for L in layers}
    else:
        E = embeddings(model, rng.integers(1000, 150000, size=a.tokens))
        X = {L: realistic_mixed(w, L, E, rng) for L in layers}

    # ---- host side first (the conversion forks workers: before the device opens)
    LM, REF = {}, {}
    kind, _, arg = a.store.partition(":")
    plan = None
    if kind == "quant":
        from fnx.quant.store import Plan
        plan = Plan(arg)
    for L in layers:
        if kind == "rtn":
            fmt = md.rtn_layer(w, L, md.profile_plan(L, int(arg))).fmt
            tiles = rtn_tiles_cached(w, L, fmt, a.workers)
            LM[L] = md.LayerMatrices(fmt, lambda e, m, t=tiles: t[(e, m)])
        else:
            LM[L] = md.quant_layer(w, L, plan)
        tr = {}
        REF[L] = (M.moe(w, L, X[L], float_experts=True, trace=tr), tr)
        n4 = int((LM[L].fmt[:N_EXPERT] == md.F_BFP4).sum())
        print(f"L{L}: store {a.store}: {n4} bfp4 / {3 * N_EXPERT - n4} bfp2 routed matrices, "
              f"{sum(md.matrix_bytes(int(f)) for f in LM[L].fmt[:N_EXPERT].ravel()) / 1e9:.3f} GB", flush=True)

    import torch
    import ttnn

    from fnx.tt.model import open_mesh
    fids = [getattr(ttnn.MathFidelity, f) for f in a.fidelity.split(",")]
    fid = fids[0]
    mesh = open_mesh()
    ok = True
    try:
        def up(x):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(1, 1, *x.shape),
                                   dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def down(t, T, cols=N_EMBD):
            return ttnn.to_torch(t).float().numpy().reshape(-1, cols)[:T]

        for L in layers:
            t0 = time.time()
            moe = md.MoeDevice(mesh, w, L, LM[L], fidelity=fid)
            print(f"\n== L{L}: store on device {moe.store.nbytes / 1e9:.3f} GB in {time.time() - t0:.1f} s", flush=True)
            moe.keep_route = True
            x = X[L]
            N = len(x)
            bo_ref, tr = REF[L]

            def run(T, mod=moe):
                bo, rt, lg = [], [], []
                for c in range(0, N, T):
                    xc = x[c:c + T]
                    xd = up(xc)
                    y = mod.forward(xd)
                    bo.append(down(y, len(xc)))
                    ttnn.deallocate(y)
                    ttnn.deallocate(xd)
                    rt.append(ttnn.to_torch(mod.last_route).numpy().reshape(32, 32)[:len(xc)].copy())
                    lg.append(down(mod.last_logits, len(xc), md.ROUTER_COLS))
                return np.concatenate(bo), np.concatenate(rt), np.concatenate(lg)

            bo1, rt1, lg1 = run(1)
            bo4, rt4, _ = run(4)
            moe.keep_route = False
            ids = rt1[:, :10].astype(np.int64)
            wt = rt1[:, 10:20].copy().view(np.float32)
            sg = rt1[:, 20].copy().view(np.float32)

            # routing
            lg_ex = tr["logits"].astype(F64)
            ids_ex = tr["ids"]
            srt = -np.sort(-lg_ex, axis=1)
            margin = srt[:, 9] - srt[:, 10]
            flips = [t for t in range(N) if set(ids[t]) != set(ids_ex[t])]
            order = sum(not np.array_equal(ids[t], ids_ex[t]) for t in range(N) if t not in flips)
            bad = [t for t in flips if margin[t] > ROUTE_MARGIN]
            # the device's softmax / sigmoid against f64 math on its own f32 logits
            sel_lg = np.take_along_axis(lg1[:, :N_EXPERT].astype(F64), ids, 1)
            p = np.exp(sel_lg - sel_lg.max(1, keepdims=True))
            werr = float(np.abs(wt - p / p.sum(1, keepdims=True)).max())
            sgerr = float(np.abs(sg - 1 / (1 + np.exp(-lg1[:, N_EXPERT].astype(F64)))).max())
            # the device's own top-10 of its f32 logits (stable, ties -> lower id)
            own = np.argsort(-lg1[:, :N_EXPERT], axis=1, kind="stable")[:, :10]
            own_ok = np.array_equal(own, ids)
            lg_err = float(np.abs(lg1[:, :N_EXPERT] - lg_ex).max() / np.abs(lg_ex).max())
            sg_ex = M.sigmoid(M.shared_gate_logit(w, L, x))
            r_ok = not bad and werr <= WT_TOL and sgerr <= WT_TOL and own_ok
            ok &= r_ok
            print(f"  routing: {len(flips)}/{N} rows with a different expert set vs the f64 router "
                  f"(exact 10th-11th margins {[f'{margin[t]:.2e}' for t in flips]}), {order} rank-order-only "
                  f"differences; device logits max |err| / max|logit| {lg_err:.1e}; top-10 == stable top-10 of the "
                  f"device logits: {own_ok}; weights vs f64 softmax of the device logits max |d| {werr:.1e}, shared "
                  f"gate {sgerr:.1e} (tol {WT_TOL}) {'OK' if r_ok else 'FAIL'}")

            # kernel arithmetic and quality
            emu = Emu(LM[L], L)
            bo_emu = np.stack([emu.row(x[t], ids[t], wt[t], float(sg[t])) for t in range(N)])
            bo_emu_ex = np.stack([emu.row(x[t], ids_ex[t], tr["wt"][t], float(sg_ex[t])) for t in range(N)])
            for k, f in enumerate(fids):
                if k:
                    moe.fidelity = f
                    bo_f = run(1)[0]
                else:
                    bo_f = bo1
                k_err = rel(bo_f, bo_emu)
                ok &= k > 0 or k_err <= KERNEL_TOL
                print(f"  kernel ({f.name} for bfp4/bfp2): ||bo - emu|| / ||emu|| = {k_err:.2e} (tol {KERNEL_TOL}"
                      f"{'' if k == 0 else ', informational'}) {'OK' if k_err <= KERNEL_TOL else 'FAIL'}; per-row max "
                      f"{max(rel(bo_f[t], bo_emu[t]) for t in range(N)):.2e}")
            moe.fidelity = fids[0]
            print(f"  quality: bo vs all-float {rel(bo1, bo_ref):.4f}; quantization alone (emu on exact routing) "
                  f"{rel(bo_emu_ex, bo_ref):.4f}; routed part only {rel(bo_emu_ex - (tr['sh']), tr['routed']):.4f}")
            inv = np.array_equal(bo1.view(np.uint32), bo4.view(np.uint32)) and np.array_equal(rt1, rt4)
            ok &= inv
            print(f"  invariance: T=4 windows == T=1 rows bitwise: {inv} "
                  f"(max |d| {np.abs(bo1 - bo4).max():.2e}) {'OK' if inv else 'FAIL'}")

            # timing: traced replays of one call and of each stage alone
            print(f"  host load average {os.getloadavg()[0]:.1f} (device times: traced replays)")
            for T in (1, 4):
                xd = up(x[:T])
                route, logits, xb, ents = moe.route(xd, T)
                h = moe.hidden(T)
                y = ttnn.empty([1, 1, T, N_EMBD], ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
                stages = {
                    "all": lambda: ttnn.deallocate(moe.forward(xd)),
                    "route": lambda: [ttnn.deallocate(t) for t in moe.route(xd, T)],
                    "A": lambda: moe.stage_a(xb, ents, h, T),
                    "B": lambda: moe.stage_b(h, ents, y, T)}
                ms = {k: 1e3 * timed(mesh, f, a.iters) for k, f in stages.items()}
                _free(xd, route, logits, xb, ents, h, y)
                sel = sorted(set(ids[:T].ravel().tolist()))
                b_a = sum(md.matrix_bytes(int(LM[L].fmt[e, m])) for e in sel + [md.SHARED] for m in (0, 1))
                b_b = sum(md.matrix_bytes(int(LM[L].fmt[e, 2])) for e in sel + [md.SHARED])
                print(f"  T={T}: {ms['all']:.3f} ms per call (traced), {len(sel)} experts + shared, "
                      f"{(b_a + b_b) / 1e6:.1f} MB weights -> {(b_a + b_b) / ms['all'] / 1e6:.0f} GB/s; stages alone: "
                      f"route {ms['route']:.3f}, A {ms['A']:.3f} "
                      f"({b_a / ms['A'] / 1e6:.0f} GB/s), B {ms['B']:.3f} ({b_b / ms['B'] / 1e6:.0f} GB/s)", flush=True)

            # host tier: the same matrices with the odd experts in pinned host memory (read over PCIe) == all resident
            if a.host_split:
                from fnx.tt.hostmem import HostPool
                hm = np.zeros(md.NE, bool)
                hm[1:N_EXPERT:2] = True
                lmh = md.LayerMatrices(LM[L].fmt, LM[L].tiles, LM[L].rows, hm)
                n_h, item_h = md.host_matrices(lmh)
                for mode in a.host_modes.split(","):
                    t0 = time.time()
                    pool = HostPool(n_h, item_h, mode.split("-")[0])
                    t_pool = time.time() - t0
                    if mode.endswith("-paged"):
                        pool.contig[:] = False          # every item through its page table
                    mh = md.MoeDevice(mesh, w, L, lmh, fidelity=fid, pool=pool)
                    print(f"  host tier [{mode}]: odd experts in a HostPool: {mh.store.nbytes / 1e9:.3f} GB DRAM "
                          f"(page table {mh.store.n_paged} rows x {mh.store.pt_row} B) + {mh.store.host_bytes / 1e9:.3f} "
                          f"GB host ({pool.nbytes / 1e9:.3f} GB pinned in {t_pool:.1f} s, {pool.alloc_stats}) in "
                          f"{time.time() - t0:.1f} s", flush=True)
                    mh.keep_route = True
                    bh1, rh1, _ = run(1, mh)
                    bh4, rh4, _ = run(4, mh)
                    mh.keep_route = False
                    hit = float(np.mean(ids % 2 == 1))
                    h_ok = (np.array_equal(bh1.view(np.uint32), bo1.view(np.uint32)) and
                            np.array_equal(bh4.view(np.uint32), bo4.view(np.uint32)) and
                            np.array_equal(rh1, rt1) and np.array_equal(rh4, rt4))
                    ok &= h_ok
                    print(f"  host tier [{mode}]: T=1 and T=4 bo == all-resident bitwise: {h_ok} ({100 * hit:.0f}% of "
                          f"the routed slots read from the host) {'OK' if h_ok else 'FAIL'}")
                    for T in (1, 4):
                        xd = up(x[:T])
                        ms_r = 1e3 * timed(mesh, lambda: ttnn.deallocate(moe.forward(xd)), a.iters)
                        ms_h = 1e3 * timed(mesh, lambda: ttnn.deallocate(mh.forward(xd)), a.iters)
                        sel = sorted(set(ids[:T].ravel().tolist()))
                        nh = sum(e % 2 for e in sel)
                        hb = sum(md.matrix_bytes(int(LM[L].fmt[e, m])) for e in sel if e % 2 for m in range(3))
                        print(f"  host tier [{mode}] T={T}: {ms_h:.3f} ms per call vs {ms_r:.3f} all resident (traced; "
                              f"{nh} of {len(sel)} experts from the host, {hb / 1e6:.1f} MB over PCIe -> "
                              f"{hb / max(ms_h - ms_r, 1e-6) / 1e6:.1f} GB/s marginal); load {os.getloadavg()[0]:.1f}",
                              flush=True)
                        _free(xd)
                    for t in (mh.last_route, mh.last_logits):
                        if t is not None:
                            ttnn.deallocate(t)
                    mh.store.release()
                    ttnn.synchronize_device(mesh)
                    pool.close()
            for t in (moe.last_route, moe.last_logits):
                if t is not None:
                    ttnn.deallocate(t)
            moe.store.release()
    finally:
        ttnn.close_mesh_device(mesh)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
