#!/usr/bin/env python3
"""Pick the routed experts fnx.tt.moe keeps on the P150, from measured routing counts (--counts) or Strata's profile.

    tools/expert_placement.py [--free-gb 27.29 | --measure] [--headroom-gb 3] [--mtp-gb 0.5] [--dtype bfloat8_b] [--out PATH]
                              [--golden chat,code,doc] [--curve] [--counts FNX_DATA/quant/stats]

--counts ranks the pairs by measured routing frequency (fnx.quant.pipeline's calibration counts) instead of the
profile; prefer it: the profile is the original 512-expert ranking mapped onto the kept experts (mapping verified
against the original routers) and carries no within-layer information for the Coder's routing.

Profile (Strata src/core/expert_cache.cpp read_expert_profile, tools/make_profile.py): "STRP", uint32 version,
n_layers, n_expert, slots, n_ranked, then n_ranked (uint16 layer, uint16 expert) pairs, hottest first.  The Coder's
data/expert-profile-coder.bin ranks all 48 x 256 pairs.  The placement is the first N pairs, N = budget / bytes per
expert, budget = free device DRAM after TtModel loads (KV for 2048 positions) - headroom - the MTP drafter's
reservation (--mtp-gb: head 289 MB + projections/HC ~180 MB + KV ~10 MB, loaded after TtModel); a layer holds however many of
its experts made the cut.  Bytes per expert: gate + up + down = 3 x 20 x 80 tiles (1088 B bfloat8_b, 576 B bfloat4_b).

--measure opens the device (run it through tools/devrun.sh, FNX_EXPERT_PLACEMENT unset), builds TtModel with
max_ctx 2048 and reads the free DRAM.  --golden scores the placement on Strata's routing dumps of the golden prompts
(golden/<name>/routing.bin: records of int32 layer, int32 k, k int32 ids, k f32 weights): the share of routed slots
whose expert is resident (the device hit rate), overall and at T=1 / T=4 as distinct CPU experts per call.
Writes JSON {"layers": [[ids] x 48], ...} for FNX_EXPERT_PLACEMENT / fnx.tt.moe.load_placement.  The device weight
caches (bfloat8_b, ~24 GB for the default placement) can be built without the device, in parallel layer ranges:
    TT_METAL_MOCK_CLUSTER_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/third_party/tt-cluster-descriptors/\\
    blackhole/p100_cluster_desc/p100_cluster_desc.yaml NODEV=1 tools/devrun.sh prep -m fnx.tt.moe --prepare OUT.json
(otherwise the first device run converts them, ~7 s per layer).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # tt/: import fnx
from fnx import paths  # noqa: E402
from fnx.config import N_EXPERT, N_LAYERS  # noqa: E402

GOLDEN = paths.GOLDEN
TILES = 3 * 20 * 80
TILE_BYTES = {"bfloat8_b": 1088, "bfloat4_b": 576, "bfloat16": 2048}


def read_profile(path: Path) -> np.ndarray:
    """[(layer, expert)] int32 [n, 2], hottest first."""
    b = path.read_bytes()
    if b[:4] != b"STRP":
        raise ValueError(f"{path}: does not start with STRP")
    _, nl, ne, slots, n = struct.unpack("<5I", b[4:24])
    if (nl, ne) != (N_LAYERS, N_EXPERT) or n > slots:
        raise ValueError(f"{path}: {nl}x{ne} with {n} pairs for {slots} slots, not a Flash-Next Coder profile")
    p = np.frombuffer(b, dtype="<u2", count=2 * n, offset=24).reshape(n, 2).astype(np.int32)
    if (p[:, 0] >= nl).any() or (p[:, 1] >= ne).any() or len({(int(l), int(e)) for l, e in p}) != n:
        raise ValueError(f"{path}: pairs out of range or repeated")
    return p


def read_counts(stats_dir: Path) -> np.ndarray:
    """[(layer, expert)] int32 [12288, 2] hottest first, by measured routing frequency (routings per token) from
    fnx.quant.pipeline's calibration statistics (stats_dir/L{LL}.npz 'n': 128 x 2048 balanced chat/code/instruct/docs
    tokens through the float model). Strata's profile carries no within-layer information for the Coder (its ranking
    is the original 512-expert ranking mapped onto the kept experts; Spearman vs the Coder's routing ~0)."""
    freq = np.stack([np.load(stats_dir / f"L{l:02d}.npz")["n"] for l in range(N_LAYERS)]).astype(np.float64)
    freq /= freq.sum(1, keepdims=True) / 10.0
    order = np.argsort(-freq.reshape(-1), kind="stable")
    return np.stack([order // N_EXPERT, order % N_EXPERT], 1).astype(np.int32)


def resident_mask(pairs: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((N_LAYERS, N_EXPERT), bool)
    m[pairs[:n, 0], pairs[:n, 1]] = True
    return m


def read_routing(name: str) -> tuple[np.ndarray, np.ndarray]:
    """golden/<name>/routing.bin -> (layer [R], ids [R, k]) in file order (per layer, consecutive tokens)."""
    raw = np.fromfile(GOLDEN / name / "routing.bin", dtype=np.int32)
    k = int(raw[1])
    rec = raw.reshape(-1, 2 + 2 * k)
    if (rec[:, 1] != k).any():
        raise ValueError(f"{name}: mixed k in routing.bin")
    return rec[:, 0], rec[:, 2:2 + k]


def score(mask: np.ndarray, layer: np.ndarray, ids: np.ndarray) -> dict:
    hit = mask[layer[:, None], ids]
    miss1 = (~hit).sum(1)
    w4 = []
    for l in range(N_LAYERS):
        sel = layer == l
        il, hl = ids[sel], hit[sel]
        for s in range(0, len(il) - 3, 4):
            w4.append((len(set(il[s:s + 4][~hl[s:s + 4]].ravel())), len(set(il[s:s + 4][hl[s:s + 4]].ravel()))))
    w4 = np.array(w4)
    return {"hit_rate": round(float(hit.mean()), 4), "t1_cpu_experts": round(float(miss1.mean()), 2),
            "t4_cpu_experts": round(float(w4[:, 0].mean()), 2), "t4_device_experts": round(float(w4[:, 1].mean()), 2)}


def measure_free_gb() -> float:
    import os
    os.environ.pop("FNX_EXPERT_PLACEMENT", None)
    import ttnn
    from fnx.cpu_experts import CpuExperts
    from fnx.tt.model import TtModel, open_mesh
    from fnx.weights import Weights
    w = Weights()
    mesh = open_mesh()
    model = None
    try:
        model = TtModel(mesh, w, CpuExperts(w.model, N_LAYERS, 4), N_LAYERS, max_ctx=2048, verbose=False)  # kept alive:
        m = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)                     # its buffers go with the object
        free = m.total_bytes_free_per_bank * m.num_banks / 1e9
        print(f"measured: {m.total_bytes_allocated_per_bank * m.num_banks / 1e9:.3f} GB allocated, {free:.3f} GB free "
              f"(largest contiguous {m.largest_contiguous_bytes_free_per_bank / 1e9:.3f} GB per bank x {m.num_banks})")
    finally:
        model = None
        ttnn.close_mesh_device(mesh)
    return free


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", type=Path, default=paths.EXPERT_PROFILE)
    ap.add_argument("--counts", type=Path, default=None,
                    help=f"rank by measured routing counts (fnx.quant stats dir, e.g. {paths.QUANT / 'stats'}) "
                         "instead of --profile")
    ap.add_argument("--free-gb", type=float, default=27.29,
                    help="free device DRAM after TtModel loads (measured 2026-10-06: 27.29 GB)")
    ap.add_argument("--measure", action="store_true", help="measure --free-gb on the device first")
    ap.add_argument("--headroom-gb", type=float, default=3.0)
    ap.add_argument("--mtp-gb", type=float, default=0.5, help="device DRAM kept for the MTP drafter")
    ap.add_argument("--dtype", default="bfloat8_b", choices=sorted(TILE_BYTES))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--golden", default="chat,code,doc")
    ap.add_argument("--curve", action="store_true", help="hit rate vs budget (no file written)")
    a = ap.parse_args()

    pairs = read_counts(a.counts) if a.counts else read_profile(a.profile)
    per_expert = TILES * TILE_BYTES[a.dtype]
    golden = {g: read_routing(g) for g in a.golden.split(",") if g}
    if a.curve:
        for gb in (4, 8, 12, 16, 20, 24, 28, 34, 40, 48, 64):
            n = min(len(pairs), int(gb * 1e9 // per_expert))
            m = resident_mask(pairs, n)
            print(f"{gb:3d} GB {a.dtype}: {n:5d} experts ({n / pairs.shape[0]:.0%})  " +
                  "  ".join(f"{g} {score(m, *r)['hit_rate']:.3f}" for g, r in golden.items()))
        return 0
    free = measure_free_gb() if a.measure else a.free_gb
    budget = free - a.headroom_gb - a.mtp_gb
    n = max(0, min(len(pairs), int(budget * 1e9 // per_expert)))
    mask = resident_mask(pairs, n)
    counts = mask.sum(1)
    res = {"profile": str(a.counts or a.profile), "dtype": a.dtype, "bytes_per_expert": per_expert, "free_gb": free,
           "headroom_gb": a.headroom_gb, "mtp_gb": a.mtp_gb, "budget_bytes": n * per_expert, "n_experts": int(n),
           "per_layer": {"min": int(counts.min()), "median": int(np.median(counts)), "max": int(counts.max())},
           "golden": {g: score(mask, *r) for g, r in golden.items()},
           "layers": [np.nonzero(mask[l])[0].tolist() for l in range(N_LAYERS)]}
    out = a.out or paths.CACHE / "placement" / f"{paths.MODEL}-{a.dtype}-{n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"{n} experts ({n * per_expert / 1e9:.2f} GB {a.dtype}; free {free:.2f} GB - headroom {a.headroom_gb} GB - MTP {a.mtp_gb} GB), "
          f"per layer {res['per_layer']}")
    for g, s in res["golden"].items():
        print(f"  {g}: {s}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
