"""The engine's record of the routing it served: per (layer, routed expert) how many token rows chose it, summed
over every request since the record began (prompt rows and decode rows alike: the work the MoE actually did), so
the card-resident experts can be ranked by real traffic (tools/usage_residents.py).

The cpu MoE counts its rows' top-10 ids as it routes them (fnx.tt.moe.Moe.plan: one bincount per call, after the
engine's warm-up only); the engine adds them to the file after every request (`flush`), atomically.  File
(fnx.paths.ROUTE_USAGE, FNX_ROUTE_USAGE; "0" turns recording off), npz: n int64 [N_LAYERS, N_EXPERT] routings,
rows int64 [N_LAYERS] token rows routed, requests int64, model str.  Verify-window rows whose drafts were rejected
count too (the experts ran for them).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import paths
from .config import N_EXPERT, N_LAYERS


def load(path: Path | str | None = None) -> dict:
    """The record at `path` (default fnx.paths.ROUTE_USAGE), or an empty one."""
    p = Path(path or paths.ROUTE_USAGE)
    if p.exists():
        with np.load(p) as f:
            rec = {k: f[k] for k in f.files}
        if str(rec.get("model", paths.MODEL)) != paths.MODEL or rec["n"].shape != (N_LAYERS, N_EXPERT):
            raise ValueError(f"{p}: a record of another model ({rec.get('model')}, {rec['n'].shape})")
        return rec
    return {"n": np.zeros((N_LAYERS, N_EXPERT), np.int64), "rows": np.zeros(N_LAYERS, np.int64),
            "requests": np.int64(0), "model": np.array(paths.MODEL)}


class Recorder:
    """Counts the layers' routing (attach) and adds it to the record file (flush)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or paths.ROUTE_USAGE)
        self.n = np.zeros((N_LAYERS, N_EXPERT), np.int64)
        self.rows = np.zeros(N_LAYERS, np.int64)

    def attach(self, model) -> None:
        """Every cpu-MoE layer of a TtModel counts into this recorder from now on."""
        for L, layer in enumerate(model.layers):
            if hasattr(layer[3], "usage"):
                layer[3].usage = (self.n[L], self.rows[L:L + 1])

    def flush(self, requests: int = 1) -> None:
        """Add the counts since the last flush to the file (written to a temporary name, then renamed)."""
        rec = load(self.path)
        rec["n"] = rec["n"] + self.n
        rec["rows"] = rec["rows"] + self.rows
        rec["requests"] = np.int64(rec["requests"] + requests)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, **rec)
        os.replace(tmp, self.path)
        self.n[:] = 0
        self.rows[:] = 0
