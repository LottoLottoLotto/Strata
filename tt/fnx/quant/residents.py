"""GPTQ-BFP4 values of the cpu MoE's resident routed experts (IQ3_S; RTX 3090, torch, float32).

    cd tt          (the repository's tt/; the host venv's python)
    FNX_MODEL=iq3s python -m fnx.quant.residents [--layers 0-47] [--n N] [--n-min 8192] [--batch 8]

The resident experts (fnx.quant.datafree.DEFAULT_RESIDENT: the variant's N hottest (layer, expert) pairs by measured
routing counts, the set fnx.tt.moe keeps in device DRAM) are held by ttnn as W = [out, in] itself (DeviceExperts), so
a BFP block is 16 consecutive input columns of one output row: fnx.quant.methods.gptq_rows, not the device MoE's
column-block gptq.  Calibration as fnx.quant.pipeline (its Stream: fnx.eval.torchref's float32 forward of the GGUF,
layer-major over fnx.quant.calib_data's 128 calibration + 16 held-out sequences x 2048 tokens), unweighted second
moments (the Coder's chosen settings: --weighted 0, --n-min 8192) of the resident experts only:
    Hgu[e] = sum x x^T over the rows routed to e (gate and up share it), Hd[e] = sum h h^T, h = silu(Wg x) * (Wu x)
with a prior of pad = max(n_min - n_e, 0) pseudo-rows: Hgu += pad * (the layer's per-token mean x x^T), Hd += pad *
diag(Hd[e]) / n_e.

Outputs under FNX_QUANT_DIR/residents (fnx.paths.QUANT):
  L{LL}.npy        uint8 [3 n, 921600]  row 3 i + m: matrix m (0 gate, 1 up, 2 down) of the layer's i-th resident
                   expert (ascending id) as an fnx.quant.bfp store row of W [out, in] (blocks along in)
  L{LL}.ids.npy    int32 [n]            the resident experts
  L{LL}.proxy.npz  per resident: n (calibration rows), n_held, rel_h_{gate,up,down}_{search,gptq} (in-sample
                   tr(dW H dW^T)/tr(W H W^T)), held_{search,gptq} (held-out sum w^2 ||f_q(x) - f(x)||^2 of the whole
                   expert), held_f2 (sum w^2 ||f(x)||^2); search = the min-MSE BFP4 the engine uses without this store
  manifest.json    {"layers": {L: {"n", "sha1"}}, "args": ...}: the sha1 keys fnx.tt.moe's device cache (gptq4)
A layer is done when its manifest entry exists (written last); a rerun only repeats the forward for it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..config import N_EMBD, N_EXPERT, N_FF, N_LAYERS, N_USED

DIR = paths.QUANT / "residents"
FMT = "bfp4"
MATS = ("gate", "up", "down")
SHAPE = {0: (N_FF, N_EMBD), 1: (N_FF, N_EMBD), 2: (N_EMBD, N_FF)}      # W [out, in] = the store row's [K, N]


def _manifest() -> dict:
    p = DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"layers": {}}


def layer_digest(L: int, resident: np.ndarray) -> str:
    """The sha1 of layer L's store file, checked to hold exactly `resident`."""
    ent = _manifest()["layers"].get(str(L))
    if ent is None:
        raise FileNotFoundError(f"{DIR}: no GPTQ residents of layer {L} (run fnx.quant.residents, or set the "
                                f"resident format to bfloat4_b)")
    ids = np.load(DIR / f"L{L:02d}.ids.npy")
    if not np.array_equal(ids, np.asarray(resident, np.int32)):
        raise ValueError(f"{DIR}/L{L:02d}: GPTQ residents of other experts than the placement's ({len(ids)} vs "
                         f"{len(resident)})")
    return ent["sha1"]


def load_layer(L: int, resident: np.ndarray, part: str) -> np.ndarray:
    """fnx.tt.moe.DeviceExperts' f32 operand from the store: part "gate_up" -> [n * 2 * N_FF, N_EMBD] (gate then up
    of each expert), "down" -> [n * N_EMBD, N_FF]; exactly the values tt-metal's bfloat4_b keeps."""
    from . import bfp
    layer_digest(L, resident)
    rows = np.load(DIR / f"L{L:02d}.npy", mmap_mode="r")
    n = len(resident)
    mats = (0, 1) if part == "gate_up" else (2,)
    K, N = SHAPE[mats[0]]
    out = np.empty((n * len(mats) * K, N), np.float32)
    for i0 in range(0, n, 16):
        sel = [3 * i + m for i in range(i0, min(i0 + 16, n)) for m in mats]
        v = bfp.unpack_batch(torch.from_numpy(np.ascontiguousarray(rows[sel])), FMT, K, N)
        out[len(mats) * i0 * K:len(mats) * i0 * K + v.numel() // N] = v.reshape(-1, N).numpy()
    return out


class Acc:
    """Second moments of one layer's resident experts (calibration rows) + the layer's per-token x x^T."""

    def __init__(self, ex: dict, sel: np.ndarray):
        from ..eval.torchref import DEV
        self.ex, n = ex, len(sel)
        self.sel = torch.from_numpy(sel.astype(np.int64)).to(DEV)
        self.slot = torch.full((N_EXPERT,), -1, dtype=torch.int64, device=DEV)
        self.slot[self.sel] = torch.arange(n, device=DEV)
        self.Hgu = torch.zeros(n, N_EMBD, N_EMBD, device=DEV)
        self.Hd = torch.zeros(n, N_FF, N_FF, device=DEV)
        self.n = torch.zeros(n, dtype=torch.float64, device=DEV)
        self.P = torch.zeros(N_EMBD, N_EMBD, device=DEV)
        self.T = 0

    def add(self, x, ids, wt):
        from ..eval.torchref import silu
        self.P.addmm_(x.T, x)
        self.T += x.shape[0]
        flat = self.slot[ids.reshape(-1).long()]
        hit = torch.nonzero(flat >= 0).reshape(-1)
        s = flat[hit]
        order = torch.argsort(s, stable=True)
        hit = hit[order]
        counts = torch.bincount(s, minlength=len(self.n)).tolist()
        pos = 0
        for i, c in enumerate(counts):
            if not c:
                continue
            t = hit[pos:pos + c] // N_USED
            pos += c
            e = int(self.sel[i])
            xe = x[t]
            self.Hgu[i].addmm_(xe.T, xe)
            h = silu(xe @ self.ex["gate"][e].T) * (xe @ self.ex["up"][e].T)
            self.Hd[i].addmm_(h.T, h)
            self.n[i] += c


def quantize_layer(L: int, exr: dict, acc: Acc, held: dict, a) -> tuple[np.ndarray, dict]:
    """GPTQ rows (uint8 [3 n, 921600]) + per-expert proxies (module docstring).  exr: the resident experts' float
    weights, in acc.sel order."""
    from ..eval.torchref import DEV
    from . import bfp, methods
    from .pipeline import ffn
    sel = acc.sel.cpu().numpy()
    n = len(sel)
    rows = np.empty((3 * n, bfp.row_bytes(FMT, N_FF, N_EMBD)), np.uint8)
    P = {k: np.zeros(n) for k in ["n", "n_held", "held_f2", "held_search", "held_gptq"] +
         [f"rel_h_{m}_{q}" for m in MATS for q in ("search", "gptq")]}
    P["n"][:] = acc.n.cpu().numpy()
    Pgu = acc.P / acc.T
    hflat = held["ids"].reshape(-1).long()
    for i0 in range(0, n, a.chunk):
        I = list(range(i0, min(i0 + a.chunk, n)))
        E = [int(sel[i]) for i in I]
        Wg, Wu, Wd = exr["gate"][I], exr["up"][I], exr["down"][I]
        Ho_gu, Ho_d = acc.Hgu[I], acc.Hd[I]
        Hgu, Hd = Ho_gu.clone(), Ho_d.clone()
        for j, i in enumerate(I):
            ne = float(acc.n[i])
            pad = max(a.n_min - ne, 0.0)
            if pad > 0:
                Hgu[j] += pad * Pgu
                dd = Hd[j].diagonal()
                Hd[j] += torch.diag(pad * (dd / ne if ne > 0 else torch.ones_like(dd)))
        qgu = methods.gptq_rows(torch.cat([Wg, Wu], 1), Hgu, FMT, damp=a.damp)
        Q = (qgu[:, :N_FF], qgu[:, N_FF:], methods.gptq_rows(Wd, Hd, FMT, damp=a.damp))
        del qgu, Hgu, Hd
        S = tuple(methods.blockq(W, FMT) for W in (Wg, Wu, Wd))
        for q in Q:
            assert bfp.is_snapped(q, FMT), f"layer {L}: GPTQ values tt-metal would not keep"
        for m, (W, H) in enumerate(((Wg, Ho_gu), (Wu, Ho_gu), (Wd, Ho_d))):
            for tag, R in (("search", S), ("gptq", Q)):
                P[f"rel_h_{MATS[m]}_{tag}"][I] = methods.rel_err_h(W, R[m], H).double().cpu().numpy()
        for j, (i, e) in enumerate(zip(I, E)):
            r = torch.nonzero(hflat == e).reshape(-1)
            P["n_held"][i] = r.numel()
            if not r.numel():
                continue
            x = held["x"][r // N_USED]
            w2 = held["wt"].reshape(-1)[r].float() ** 2
            f = ffn(Wg[j], Wu[j], Wd[j], x)
            P["held_f2"][i] = float((w2 * (f * f).sum(-1)).sum())
            for tag, R in (("search", S), ("gptq", Q)):
                d = ffn(R[0][j], R[1][j], R[2][j], x) - f
                P[f"held_{tag}"][i] = float((w2 * (d * d).sum(-1)).sum())
        for m in range(3):
            r = bfp.pack_batch(Q[m].contiguous(), FMT).cpu().numpy()
            for j, i in enumerate(I):
                rows[3 * i + m] = r[j]
        del Q, S, Ho_gu, Ho_d
        torch.cuda.synchronize()
    return rows, P


def main(argv=None) -> int:
    from .datafree import DEFAULT_RESIDENT, resident_layers
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", default=f"0-{N_LAYERS - 1}", help="layers to quantize (a-b or comma list)")
    ap.add_argument("--n", type=int, default=DEFAULT_RESIDENT[paths.MODEL][0], help="resident (layer, expert) pairs")
    ap.add_argument("--batch", type=int, default=8, help="sequences per forward batch")
    ap.add_argument("--chunk", type=int, default=32, help="experts per GPTQ batch")
    ap.add_argument("--n-min", type=int, default=8192)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    from ..eval.torchref import ExpertLoader, setup_torch
    from ..weights import Weights
    from . import calib_data
    from .pipeline import Stream, shm_hygiene
    if "-" in a.layers:
        lo, hi = a.layers.split("-")
        want = set(range(int(lo), int(hi) + 1))
    else:
        want = {int(x) for x in a.layers.split(",") if x.strip()}
    setup_torch()
    shm_hygiene()
    DIR.mkdir(parents=True, exist_ok=True)
    placement = resident_layers(a.n)
    man = _manifest()
    man["args"] = vars(a)
    todo = sorted(L for L in want if str(L) not in man["layers"] or
                  not np.array_equal(np.load(DIR / f"L{L:02d}.ids.npy"), placement[L]))
    if not todo:
        print("residents: every requested layer is done", flush=True)
        return 0
    cal = [s["token_ids"] for s in calib_data.load("calib")]
    held = [s["token_ids"] for s in calib_data.load("heldout")]
    w = Weights()
    t_start = time.time()
    st = Stream(w, {"calib": cal, "held": held}, a.batch)
    print(f"residents: {a.n} experts, layers {todo}; {len(cal)} calib + {len(held)} held-out sequences, inputs "
          f"{time.time() - t_start:.0f} s", flush=True)
    loader = ExpertLoader(str(w.model.files[0].path), workers=a.workers, ahead=1)
    try:
        for L in range(max(todo) + 1):
            tm = {}
            t0 = time.time()
            ex = loader.get(L)
            tm["load"] = time.time() - t0
            torch.cuda.reset_peak_memory_stats()
            acc = Acc(ex, placement[L]) if L in todo else None
            cap = []

            def hook(split, L_, x, ids, wt):
                if acc is None:
                    return
                if split == "calib":
                    acc.add(x, ids, wt)
                else:
                    cap.append((x.clone(), ids.clone(), wt.clone()))

            st.hook = hook
            t0 = time.time()
            st.layer(L, ex)
            torch.cuda.synchronize()
            tm["forward"] = time.time() - t0
            if acc is not None:
                heldd = {k: torch.cat([c[i] for c in cap]) for i, k in enumerate(("x", "ids", "wt"))}
                exr = {k: v[acc.sel] for k, v in ex.items()}       # the other experts' 5-6 GB leave the GPU
                acc.ex = ex = cap = None
                torch.cuda.empty_cache()
                t0 = time.time()
                rows, P = quantize_layer(L, exr, acc, heldd, a)
                del exr
                tm["gptq"] = time.time() - t0
                tm["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                tmp = DIR / f"L{L:02d}.tmp.npy"
                np.save(tmp, rows)
                os.replace(tmp, DIR / f"L{L:02d}.npy")
                np.save(DIR / f"L{L:02d}.ids.npy", placement[L])
                np.savez(DIR / f"L{L:02d}.proxy.npz", **P)
                man["layers"][str(L)] = {"n": len(placement[L]), "sha1": hashlib.sha1(rows.tobytes()).hexdigest()}
                (DIR / "manifest.tmp.json").write_text(json.dumps(man, indent=1))
                os.replace(DIR / "manifest.tmp.json", DIR / "manifest.json")
                hs, hg, f2 = P["held_search"].sum(), P["held_gptq"].sum(), P["held_f2"].sum()
                tm["held_rel_search"], tm["held_rel_gptq"] = hs / f2, hg / f2
                del heldd, rows
            del acc, ex
            torch.cuda.empty_cache()
            print(f"residents: layer {L} {'quantized' if L in todo else 'forward only'} "
                  f"{ {k: round(v, 4 if k.startswith('held') else 2) for k, v in tm.items()} } "
                  f"total {time.time() - t_start:.0f} s", flush=True)
    finally:
        loader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
