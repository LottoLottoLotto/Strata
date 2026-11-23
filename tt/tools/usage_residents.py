#!/usr/bin/env python3
"""Rank the card-resident routed experts by the routing the engine actually served (fnx.usage's record).

    cd tt; FNX_MODEL=iq3s python3 tools/usage_residents.py report [--record PATH]
    cd tt; FNX_MODEL=iq3s python3 tools/usage_residents.py build  [--record PATH] [--prior 20000] [--out DIR]

report: what the record holds (requests, token rows) and the share of its routed slots the current resident set
        serves on the card ("hit": the rest runs on the CPU, the prompt reads' bound) against the share a set of the
        same size ranked by the record would serve.  The CPU's expert work scales with 1 - hit.
build:  writes DIR (default FNX_QUANT_DIR/stats-usage) as per-layer L{LL}.npz 'n' - the ranking format of
        fnx.quant.datafree.read_counts - from the record's per-layer routing frequencies blended with the current
        ranking's: f = (1 - a) f_record + a f_current, a = prior / (prior + record rows), so a short record moves
        the set a little and a long one decides it.  Then (the engine stopped; RTX 3090, ~30 min):
            FNX_MODEL=iq3s FNX_ROUTE_RANK=DIR <venv python> -m fnx.quant.residents
        rebuilds the GPTQ values of the layers whose set changed (fnx.quant.residents), and FNX_ROUTE_RANK=DIR in the
        engine's environment (the run config's "env") makes it keep that set.  Unset it to go back (the residents
        then have to be rebuilt for the old set the same way).
Host only (numpy): no device, no torch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # tt/: import fnx
from fnx import paths, usage  # noqa: E402
from fnx.config import N_EXPERT, N_LAYERS  # noqa: E402
from fnx.quant.datafree import DEFAULT_RESIDENT, read_counts  # noqa: E402


def freq_of(n: np.ndarray) -> np.ndarray:
    """Per-layer routing frequencies (each layer's row sums to 1)."""
    n = n.astype(np.float64)
    return n / np.maximum(n.sum(1, keepdims=True), 1.0)


def current_freq() -> np.ndarray:
    """The current ranking's per-layer frequencies (read_counts' source, FNX_ROUTE_RANK included)."""
    d = paths.ROUTE_RANK or paths.QUANT / "stats"
    if (Path(d) / "L00.npz").exists():
        n = np.stack([np.load(Path(d) / f"L{L:02d}.npz")["n"] for L in range(N_LAYERS)])
    else:
        n = np.load(paths.ROUTE_COUNTS)["calib"]
    return freq_of(n)


def top_set(freq: np.ndarray, k: int) -> np.ndarray:
    """bool [N_LAYERS, N_EXPERT]: the k hottest pairs by per-layer routings per token (read_counts' order)."""
    order = np.argsort(-freq.reshape(-1), kind="stable")[:k]
    s = np.zeros(N_LAYERS * N_EXPERT, bool)
    s[order] = True
    return s.reshape(N_LAYERS, N_EXPERT)


def hit(n: np.ndarray, resident: np.ndarray) -> float:
    return float(n[resident].sum() / max(n.sum(), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("report", "build"))
    ap.add_argument("--record", default=None, help=f"the engine's record (default {paths.ROUTE_USAGE})")
    ap.add_argument("--n", type=int, default=DEFAULT_RESIDENT[paths.MODEL][0], help="resident (layer, expert) pairs")
    ap.add_argument("--prior", type=float, default=20000, help="token rows the current ranking weighs as")
    ap.add_argument("--out", default=None, help="build: the ranking directory (default FNX_QUANT_DIR/stats-usage)")
    a = ap.parse_args()
    rec = usage.load(a.record)
    n, rows = rec["n"], int(rec["rows"].max())
    if rows == 0:
        print(f"{a.record or paths.ROUTE_USAGE}: no routing recorded yet (serve requests with the engine first)")
        return 1
    cur = read_counts()[:a.n]
    now = np.zeros((N_LAYERS, N_EXPERT), bool)
    now[cur[:, 0], cur[:, 1]] = True
    alpha = a.prior / (a.prior + rows)
    blend = (1 - alpha) * freq_of(n) + alpha * current_freq()
    by_record, by_blend = top_set(freq_of(n), a.n), top_set(blend, a.n)
    h_now, h_rec, h_blend = hit(n, now), hit(n, by_record), hit(n, by_blend)
    print(f"record: {int(rec['requests'])} requests, {rows} token rows per layer, {int(n.sum())} routed slots")
    print(f"card-resident share of the recorded routing ({a.n} experts): current set {h_now:.1%}; "
          f"ranked by the record {h_rec:.1%}; blended (prior {a.prior:.0f} rows, weight {alpha:.2f}) {h_blend:.1%}")
    print(f"CPU expert work, blended set vs current: {(1 - h_blend) / max(1 - h_now, 1e-9):.2f}x "
          f"({len(np.flatnonzero(by_blend & ~now))} experts change)")
    if a.cmd == "report":
        return 0
    out = Path(a.out) if a.out else paths.QUANT / "stats-usage"
    out.mkdir(parents=True, exist_ok=True)
    for L in range(N_LAYERS):
        np.savez(out / f"L{L:02d}.npz", n=blend[L])
    print(f"wrote {out} (per-layer L??.npz 'n').  Next, with the engine stopped:\n"
          f"  cd {paths.TT_DIR} && FNX_MODEL={paths.MODEL} FNX_ROUTE_RANK={out} {paths.VENV_PY} -m fnx.quant.residents"
          f"\nthen add \"FNX_ROUTE_RANK\": \"{out}\" to the run config's env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
