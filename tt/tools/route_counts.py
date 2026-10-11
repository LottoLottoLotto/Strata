"""Measured routing counts of the model on the P150 (no GPU): the cpu MoE's top-10 ids over fnx.quant's calibration
sequences (FNX_QUANT_DIR/calib/seqs.json, disjoint from the eval set) and, separately, its held-out ones.

    FNX_MODEL=iq3s tools/devrun.sh counts tools/route_counts.py [--every 2] [--max-minutes 120]

Writes FNX_QUANT_DIR/stats/L{LL}.npz and stats-held/L{LL}.npz with n = int64 [N_EXPERT] routings per expert (the
format tools/expert_placement.py --counts and fnx.quant.datafree plan --counts read); sequences already counted are
skipped (stats/done.json), so a capture can be split over several device runs.  Routing is the cpu MoE's (the
variant's default resident set as bfloat8_b: its ids are the exact experts' but for near-tie flips).  Teacher forcing
in 128-row chunks without the head.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # tt/: import fnx
from fnx import paths  # noqa: E402
from fnx.config import N_EXPERT, N_LAYERS  # noqa: E402

CHUNK = 128


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=2, help="count every k-th calibration sequence (all held-out ones)")
    ap.add_argument("--max-minutes", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    seqs = json.loads((paths.QUANT / "calib" / "seqs.json").read_text())
    todo = [("stats", s) for s in seqs["calib"][::a.every]] + [("stats-held", s) for s in seqs["heldout"]]
    counts, done = {}, {}
    for kind in ("stats", "stats-held"):
        d = paths.QUANT / kind
        d.mkdir(parents=True, exist_ok=True)
        f = d / "done.json"
        done[kind] = set(json.loads(f.read_text())) if f.exists() else set()
        counts[kind] = np.stack([np.load(d / f"L{L:02d}.npz")["n"] if (d / f"L{L:02d}.npz").exists()
                                 else np.zeros(N_EXPERT, np.int64) for L in range(N_LAYERS)])
    todo = [(k, s) for k, s in todo if s["id"] not in done[k]]
    print(f"route_counts: FNX_MODEL={paths.MODEL}, {len(todo)} sequences to count", flush=True)
    if not todo:
        return 0

    import os
    os.environ.setdefault("FNX_MOE", "cpu")
    import ttnn

    from fnx.cpu_experts import CpuExperts
    from fnx.tt.model import TtModel, open_mesh
    from fnx.weights import Weights
    w = Weights()
    mesh = open_mesh()
    deadline = time.time() + 60 * a.max_minutes
    try:
        model = TtModel(mesh, w, CpuExperts(w.model, N_LAYERS, a.threads), max_ctx=4096, verbose=False, moe="cpu")
        for kind, s in todo:
            if time.time() > deadline:
                print("time budget reached", flush=True)
                break
            t0 = time.time()
            model.reset()
            toks = np.asarray(s["token_ids"], np.int64)
            for c0 in range(0, len(toks), CHUNK):
                model.forward(toks[c0:c0 + CHUNK], head=False)
                for L in range(N_LAYERS):
                    ids = model.layers[L][3].last["ids"]
                    counts[kind][L] += np.bincount(ids[ids >= 0].reshape(-1), minlength=N_EXPERT)
            done[kind].add(s["id"])
            d = paths.QUANT / kind
            for L in range(N_LAYERS):
                np.savez(d / f"L{L:02d}.npz", n=counts[kind][L])
            (d / "done.json").write_text(json.dumps(sorted(done[kind])))
            print(f"  {kind} {s['id']}: {len(toks)} tokens in {time.time() - t0:.0f} s", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
