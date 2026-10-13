"""Data-free expert store: every routed expert of the model (FNX_MODEL) as BFP4 rows quantized from its GGUF values
with the per-block min-MSE exponent (fnx.quant.methods.search, the MTP drafter's method), and tiered plans whose
resident experts are the hottest of Strata's routing profile (fnx.paths.EXPERT_PROFILE).  No activations, no GPU: the
store for a variant without GPTQ calibration (ExpertQuant's pipeline needs CUDA), read by fnx.tt.model's device MoE
exactly like the GPTQ store (fnx.quant.store layout, under fnx.paths.STORE).

    python -m fnx.quant.datafree pool [--layers 0-47] [--workers 4] [--chunk 32]
        -> STORE/pool/L{LL}.b4.npy (uint8 [3 * N_EXPERT, 921600], row 3 e + m) and L{LL}.b4idx.npy ((e, m) per row)
    python -m fnx.quant.datafree plan --budget 2.4e10 [--name NAME]
        -> STORE/plans/<NAME> (default search4-tiered-<budget>): bfp4 for every matrix; resident in device DRAM = the
           first floor(budget / 2764800) (layer, expert) pairs of the profile (hottest first), the rest host tier.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths
from ..config import N_EMBD, N_EXPERT, N_FF, N_LAYERS

FMT = "bfp4"
MATS = ("gate", "up", "down")
EXPERT_BYTES = 3 * 921600            # one routed expert's three BFP4 rows (fnx.quant.store / moe_device.matrix_bytes)


def read_profile(path: Path | str = paths.EXPERT_PROFILE) -> np.ndarray:
    """Strata's expert profile ("STRP", u32 version, n_layers, n_expert, slots, n_ranked, then n_ranked (u16 layer, u16
    expert) pairs, hottest first) -> int32 [n_ranked, 2]; checked against the variant's geometry."""
    b = Path(path).read_bytes()
    if b[:4] != b"STRP":
        raise ValueError(f"{path}: not a Strata expert profile")
    _, n_layers, n_expert, _, n = struct.unpack("<5I", b[4:24])
    if (n_layers, n_expert) != (N_LAYERS, N_EXPERT):
        raise ValueError(f"{path}: a profile of {n_layers} x {n_expert} experts, FNX_MODEL={paths.MODEL} has "
                         f"{N_LAYERS} x {N_EXPERT}")
    return np.frombuffer(b, "<u2", count=2 * n, offset=24).reshape(n, 2).astype(np.int32)


def read_counts(stats_dir: Path | str | None = None) -> np.ndarray:
    """[(layer, expert)] int32 [N_LAYERS * N_EXPERT, 2] hottest first by measured routing frequency (routings per
    token of the layer): stats_dir/L{LL}.npz 'n' (tools/route_counts.py; the Coder's: fnx.quant.pipeline); default
    FNX_QUANT_DIR/stats when it holds counts, else the repository's fnx.paths.ROUTE_COUNTS ('calib')."""
    d = Path(stats_dir) if stats_dir is not None else paths.QUANT / "stats"
    if (d / "L00.npz").exists():
        freq = np.stack([np.load(d / f"L{L:02d}.npz")["n"] for L in range(N_LAYERS)]).astype(np.float64)
    elif stats_dir is None and paths.ROUTE_COUNTS.exists():
        d = paths.ROUTE_COUNTS
        freq = np.load(d)["calib"].astype(np.float64)
    else:
        raise FileNotFoundError(f"{d}: no routing counts (FNX_MODEL={paths.MODEL}: run tools/route_counts.py)")
    if freq.shape != (N_LAYERS, N_EXPERT):
        raise ValueError(f"{d}: counts of shape {freq.shape}, FNX_MODEL={paths.MODEL} has {(N_LAYERS, N_EXPERT)}")
    freq /= freq.sum(1, keepdims=True) / 10.0
    order = np.argsort(-freq.reshape(-1), kind="stable")
    return np.stack([order // N_EXPERT, order % N_EXPERT], 1).astype(np.int32)


# the cpu MoE's resident experts per variant (fnx.tt.moe, FNX_EXPERT_PLACEMENT unset): (N, format) = the N hottest
# (layer, expert) pairs by the variant's measured routing counts (read_counts) in that format: bfloat8_b, bfloat4_b
# (min-MSE BFP4 of the GGUF values), gptq4 (BFP4 from fnx.quant.residents' GPTQ store).  N = tools/expert_placement.py
# --measure --dtype: free DRAM after the model - 3 GB headroom - 0.5 GB MTP drafter.  The Coder's cpu MoE keeps every
# routed expert on the CPU (its default MoE is the device one).  iq3s: 8823 BFP4 experts (user's choice 2026-10-11:
# MTP +20% over bfp8-4671), GPTQ since 2026-10-13: KL .0279, perplexity -0.66%, GSM8K-50 45 (min-MSE bfloat4_b:
# KL .0437, -2.21%, 45; see CONTRACT); bfp8-4671 (KL .0066) via a placement file (tools/expert_placement.py --counts
# ... --dtype bfloat8_b).
DEFAULT_RESIDENT = {"coder": (0, "bfloat8_b"), "iq3s": (8823, "gptq4")}


def resident_layers(n: int) -> list[np.ndarray]:
    """Per-layer sorted expert ids (int32) of the n hottest (layer, expert) pairs (read_counts)."""
    top = read_counts()[:n]
    return [np.sort(top[top[:, 0] == L, 1]).astype(np.int32) for L in range(N_LAYERS)]


def quantize_layer(L: int, shard1: str, out_dir: str, chunk: int, threads: int) -> tuple[int, float]:
    """Layer L's N_EXPERT x 3 matrices -> pool files (written under a .tmp name, then renamed).  Returns (L, s)."""
    import torch

    from ..gguf_load import Model
    from . import bfp, methods
    torch.set_num_threads(threads)
    t0 = time.time()
    model = Model(shard1)
    out = Path(out_dir)
    tmp = out / f"L{L:02d}.b4.tmp.npy"
    rows = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                     shape=(3 * N_EXPERT, bfp.row_bytes(FMT, N_EMBD, N_FF)))
    for m, mat in enumerate(MATS):
        name = f"blk.{L}.ffn_{mat}_exps.weight"
        for e0 in range(0, N_EXPERT, chunk):
            e1 = min(e0 + chunk, N_EXPERT)
            W = torch.from_numpy(model.get_experts(name, e0, e1))               # [c, out, in]
            Wt = methods.blockq(W.transpose(-1, -2).contiguous(), FMT)          # = methods.search, as W^T [c, in, out]
            rows[3 * e0 + m:3 * e1:3] = bfp.pack_batch(Wt, FMT).numpy()
    rows.flush()
    del rows
    idx = np.stack([np.repeat(np.arange(N_EXPERT), 3), np.tile(np.arange(3), N_EXPERT)], 1).astype(np.int32)
    np.save(out / f"L{L:02d}.b4idx.npy", idx)
    os.replace(tmp, out / f"L{L:02d}.b4.npy")
    return L, time.time() - t0


def layer_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def cmd_pool(a) -> int:
    out = paths.STORE / "pool"
    out.mkdir(parents=True, exist_ok=True)
    todo = [L for L in layer_list(a.layers) if a.force or not (out / f"L{L:02d}.b4.npy").exists()]
    threads = max(1, (os.cpu_count() or 8) // a.workers)
    print(f"datafree pool: FNX_MODEL={paths.MODEL}, {len(todo)} layers x {N_EXPERT} experts from {a.gguf} -> {out} "
          f"({a.workers} workers x {threads} threads)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for L, s in ex.map(quantize_layer, todo, [str(a.gguf)] * len(todo), [str(out)] * len(todo),
                           [a.chunk] * len(todo), [threads] * len(todo)):
            print(f"  layer {L:2d}: {s:.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_plan(a) -> int:
    from . import store
    pairs = read_counts() if a.rank == "counts" else read_profile()
    n_res = int(a.budget // EXPERT_BYTES)
    if len(pairs) < N_LAYERS * N_EXPERT:
        raise ValueError(f"the ranking holds {len(pairs)} of {N_LAYERS * N_EXPERT} experts")
    res = np.zeros((N_LAYERS, N_EXPERT), bool)
    top = pairs[:n_res]
    res[top[:, 0], top[:, 1]] = True
    name = a.name or f"search4-tiered-{a.budget:.4g}"
    alloc = (f"measured routing counts {paths.QUANT / 'stats'} (tools/route_counts.py)" if a.rank == "counts" else
             f"Strata's routing profile {paths.EXPERT_PROFILE.name}") + ", hottest first"
    meta = {"type": "bfp4-tiered", "budget_resident": a.budget, "resident_bytes": int(res.sum()) * EXPERT_BYTES,
            "resident_experts": int(res.sum()), "per_layer_resident": res.sum(1).tolist(),
            "method": "data-free BFP4 (fnx.quant.datafree: methods.search of the GGUF values)",
            "alloc": alloc, "model": paths.MODEL}
    d = store.write_plan(name, np.full((N_LAYERS, N_EXPERT, 3), 4, np.int8), meta, resident=res)
    print(f"plan {d}: {int(res.sum())} of {N_LAYERS * N_EXPERT} experts resident "
          f"({int(res.sum()) * EXPERT_BYTES / 1e9:.2f} GB), per layer {res.sum(1).min()}..{res.sum(1).max()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--gguf", type=Path, default=paths.SHARD1)
    p.add_argument("--layers", default=f"0-{N_LAYERS - 1}")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk", type=int, default=32, help="experts per dequantization + quantization batch")
    p.add_argument("--force", action="store_true", help="rebuild layers whose pool file exists")
    p = sub.add_parser("plan")
    p.add_argument("--budget", type=float, required=True, help="bytes of resident experts")
    p.add_argument("--name", default=None)
    p.add_argument("--rank", choices=("counts", "profile"), default="counts",
                   help="resident = the hottest by measured counts (FNX_QUANT_DIR/stats) or by Strata's profile")
    a = ap.parse_args(argv)
    return {"pool": cmd_pool, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
