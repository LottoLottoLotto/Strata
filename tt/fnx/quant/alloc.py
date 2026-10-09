"""Precision allocation for the routed experts: per (layer, expert) choose gate/up/down formats (bfp4 | bfp2) to
minimise the predicted end-to-end KL under a device DRAM budget (multiple-choice knapsack).

Predicted cost of expert (L, e) in option o (formats of gate, up, down, e.g. "442"):
    cost = a_L * u_Le * r_Le(o)
  u_Le   routings per token from the calibration corpus (262K tokens; Strata's Coder profile ranking is not used: it is
         uncorrelated with this model's routing, Spearman ~0 against the calibration counts at layers 2 and 11)
  r_Le   mean routing-weighted squared output error of the quantized expert per routing, from the held-out rows
         routed to it (proxy abs_o / n_held), shrunk toward its error on generic rows (pool_o / pool_f2 times the
         expert's calibration f2 / n) with weight 64 rows
  a_L    KL per unit of squared MoE-output error at layer L: gate^2 / ||R||^2 scale statistic of the layer (the error
         enters a stream that later layers RMS-normalise), times a per-group factor fitted to measured KL of
         layer-group RTN-bfp4 runs (fnx.quant.evalq rtn4:L0-L1) when present
Options are limited to the rows in the pool. Solver: per expert lower convex hull over (bytes, cost), then a global
greedy over hull segments by cost saved per byte (optimal at hull breakpoints; the last step may leave < 1 upgrade of
the budget unused).

bfp4-tiered-<budget> plans (--tiered): every expert GPTQ BFP4; the device-resident set is the globally most-routed
experts (calibration routings per token) that fit the budget (2,764,800 B per expert), the rest is flagged host
(store L{LL}.res.npy), maximising resident routing mass.
bfp4-tiered-<budget>-h2-<P> plans (--h2 P/100, with --tiered): the same resident set (BFP4); the host-tier matrices
are BFP4 or BFP2 per matrix so that the PCIe bytes per token (sum over host experts of routings per token x bytes)
drop by P% at the least predicted KL: the same knapsack as the budgets above with per-expert bytes = u_Le x option
bytes (host experts only; resident experts keep 444 and cost no PCIe bytes).

    /var/tmp/strata-tt/venv/bin/python -m fnx.quant.alloc [--tiered 24.4e9] [--h2 0.25,0.5] [--budgets 30e9,28.5e9]
        [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from ..config import N_EXPERT, N_LAYERS
from . import store
from .pipeline import COMBOS

QDIR = Path(os.environ.get("FNX_QUANT_DIR", "/var/tmp/strata-tt/quant"))
SIZE = {"4": 921600, "2": 512000}
K0 = 64.0


def option_bytes(o: str) -> int:
    return sum(SIZE[c] for c in o)


def load_tables():
    P = [dict(np.load(QDIR / "proxy" / f"L{L:02d}.npz")) for L in range(N_LAYERS)]
    S = [dict(np.load(QDIR / "stats" / f"L{L:02d}.npz")) for L in range(N_LAYERS)]
    return P, S


def usage(S) -> np.ndarray:
    """[48, 256] calibration routings per token (sums to 10 per layer)."""
    n = np.stack([s["n"] for s in S])
    return n / (n.sum(1, keepdims=True) / 10.0)


def per_routing_error(P, S, opt: str) -> np.ndarray:
    """[48, 256] r_Le(opt)."""
    key = opt
    out = np.zeros((N_LAYERS, N_EXPERT))
    for L in range(N_LAYERS):
        p, s = P[L], S[L]
        nh = p["n_held"]
        r_held = np.where(nh > 0, p[f"abs_{key}"] / np.maximum(nh, 1), 0.0)
        rel_gen = p[f"pool_{key}"] / np.maximum(p["pool_f2"], 1e-30)
        r_gen = rel_gen * s["f2"] / np.maximum(s["n"], 1)
        out[L] = (nh * r_held + K0 * r_gen) / (nh + K0)
    return out


def layer_scale(S) -> np.ndarray:
    a = np.zeros(N_LAYERS)
    for L, s in enumerate(S):
        n, r2, bo2, g2bo2 = s["scale"]
        a[L] = (g2bo2 / max(bo2, 1e-30)) / max(r2 / max(n, 1), 1e-30)
    return a


def group_factors(a_def, U, R444) -> tuple[np.ndarray, dict]:
    """Fit per-group KL factors from evalq.json rtn4:L0-L1 runs (if any)."""
    k = np.ones(N_LAYERS)
    ev = QDIR / "evalq.json"
    fits = {}
    if not ev.exists():
        return k, fits
    res = json.loads(ev.read_text())
    groups = []
    for cfg, r in res.items():
        if cfg.startswith("rtn4:"):
            lo, hi = (int(x) for x in cfg[5:].split("-"))
            groups.append((lo, hi, r["kl"]))
    for lo, hi, kl in groups:
        pred = float((a_def[lo:hi + 1, None] * U[lo:hi + 1] * R444[lo:hi + 1]).sum())
        k[lo:hi + 1] = kl / pred
        fits[f"{lo}-{hi}"] = {"kl": kl, "pred_unscaled": pred, "factor": kl / pred}
    return k, fits


def hull(bytes_, cost):
    """Lower convex hull of options sorted by bytes: returns indices along increasing bytes, decreasing cost."""
    order = np.lexsort((cost, bytes_))
    pts = []
    for i in order:
        if pts and bytes_[i] == bytes_[pts[-1]]:
            continue
        if pts and cost[i] >= cost[pts[-1]]:
            continue
        while len(pts) >= 2:
            i1, i2 = pts[-2], pts[-1]
            s1 = (cost[i2] - cost[i1]) / (bytes_[i2] - bytes_[i1])
            s2 = (cost[i] - cost[i2]) / (bytes_[i] - bytes_[i2])
            if s2 <= s1:
                pts.pop()
            else:
                break
        pts.append(i)
    return pts


def solve(opts, B, C, avail, budget):
    """opts: option names; B [n_opt] bytes; C [48, 256, n_opt] cost; avail [48, 256, n_opt] bool.
    Returns (choice [48, 256] option index, total bytes, total cost)."""
    choice = np.zeros((N_LAYERS, N_EXPERT), np.int64)
    segs = []
    base_b, base_c = 0, 0.0
    for L in range(N_LAYERS):
        for e in range(N_EXPERT):
            ok = np.flatnonzero(avail[L, e])
            h = [ok[j] for j in hull(B[ok].astype(np.float64), C[L, e, ok])]
            choice[L, e] = h[0]
            base_b += B[h[0]]
            base_c += C[L, e, h[0]]
            for j in range(1, len(h)):
                db = B[h[j]] - B[h[j - 1]]
                dc = C[L, e, h[j - 1]] - C[L, e, h[j]]
                segs.append((dc / db, L, e, h[j], db, dc))
    if base_b > budget:
        raise ValueError(f"budget {budget:.4g} below the cheapest feasible plan {base_b:.4g}")
    segs.sort(key=lambda s: -s[0])
    b, c = base_b, base_c
    for _, L, e, j, db, dc in segs:     # segments of one expert appear in hull order (decreasing slope)
        if b + db > budget:
            continue
        if B[choice[L, e]] + db != B[j]:
            continue                      # an earlier segment of this expert was skipped: keep the hull order
        choice[L, e] = j
        b += db
        c -= dc
    return choice, b, c


def solve_pe(Bpe, C, avail, budget):
    """solve() with per-expert option bytes: Bpe, C, avail [48, 256, n_opt].  Returns (choice, total bytes, cost)."""
    hulls = np.empty((N_LAYERS, N_EXPERT), object)
    pos = np.zeros((N_LAYERS, N_EXPERT), np.int64)
    segs = []
    b, c = 0.0, 0.0
    for L in range(N_LAYERS):
        for e in range(N_EXPERT):
            ok = np.flatnonzero(avail[L, e])
            h = [ok[j] for j in hull(Bpe[L, e, ok].astype(np.float64), C[L, e, ok])]
            hulls[L, e] = h
            b += Bpe[L, e, h[0]]
            c += C[L, e, h[0]]
            for j in range(1, len(h)):
                db = Bpe[L, e, h[j]] - Bpe[L, e, h[j - 1]]
                dc = C[L, e, h[j - 1]] - C[L, e, h[j]]
                segs.append((dc / db, L, e, j, db, dc))
    if b > budget:
        raise ValueError(f"budget {budget:.4g} below the cheapest feasible plan {b:.4g}")
    segs.sort(key=lambda t: -t[0])
    for _, L, e, j, db, dc in segs:
        if b + db > budget or pos[L, e] != j - 1:
            continue
        pos[L, e] = j
        b += db
        c -= dc
    choice = np.array([[hulls[L, e][pos[L, e]] for e in range(N_EXPERT)] for L in range(N_LAYERS)], np.int64)
    return choice, b, c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="30e9,28.5e9",
                    help="BFP4/BFP2 comparison plans (the pool holds bfp2 rows for each layer's coldest 40%% only, "
                         "so budgets below ~27.9e9 are infeasible)")
    ap.add_argument("--names", default="a-30.0,a-28.5")
    ap.add_argument("--tiered", default="24.4e9", help="bfp4-tiered plans: device-resident bytes, comma list")
    ap.add_argument("--h2", default="", help="with --tiered: also -h2-<P> plans cutting PCIe bytes by these fractions")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    P, S = load_tables()
    U = usage(S)
    summary = {"plans": {}}
    pool = store.Pool()
    has4 = np.stack([(pool.index(L, 4) >= 0).all(1) for L in range(N_LAYERS)])
    per_expert = option_bytes("444")
    opts = list(COMBOS)
    Bo = np.array([option_bytes(o) for o in opts])
    R = np.stack([per_routing_error(P, S, o) for o in opts], -1)          # [48, 256, 8]
    a_def = layer_scale(S)
    k, fits = group_factors(a_def, U, per_routing_error(P, S, "rtn444"))
    aL = a_def * k
    C = aL[:, None, None] * U[..., None] * R
    ix2 = np.stack([pool.index(L, 2) >= 0 for L in range(N_LAYERS)])     # [48, 256, 3] bfp2 row of each matrix
    has2 = ix2.all(-1)
    avail = np.stack([np.where(("4" in o) & ~has4, False, True) & np.where(("2" in o) & ~has2, False, True)
                      for o in opts], -1)
    i444 = opts.index("444")
    for tb in (float(x) for x in a.tiered.split(",") if x.strip()):
        # all experts GPTQ BFP4; the resident set maximises resident routing mass (routings per token, comparable
        # across layers: every layer routes 10 per token) under the device budget; the rest is read from host memory
        if not has4.all():
            raise ValueError("bfp4-tiered needs bfp4 rows of every expert in the pool (pipeline --pool4 1.0)")
        n_res = int(tb // per_expert)
        order = np.argsort(-U.reshape(-1), kind="stable")
        res = np.zeros(N_LAYERS * N_EXPERT, bool)
        res[order[:n_res]] = True
        res = res.reshape(N_LAYERS, N_EXPERT)
        name = f"bfp4-tiered-{tb:.4g}"
        per_layer = res.sum(1)
        info = {"type": "bfp4-tiered", "budget_resident": tb, "resident_bytes": int(res.sum() * per_expert),
                "host_bytes": int((~res).sum() * per_expert), "n_resident": int(res.sum()),
                "resident_routing_share": float((U * res).sum() / U.sum()),
                "resident_per_layer": {"min": int(per_layer.min()), "median": float(np.median(per_layer)),
                                       "max": int(per_layer.max())},
                "host_routings_per_token": float((U * ~res).sum())}
        summary["plans"][name] = info
        print(name, json.dumps(info))
        if a.write:
            store.write_plan(name, np.full((N_LAYERS, N_EXPERT, 3), 4, np.int8),
                             {**info, "method": "GPTQ-BFP4 (fnx.quant.pipeline)", "alloc": "fnx.quant.alloc tiered"},
                             resident=res)
        # -h2-<P>: host-tier matrices BFP4 / BFP2 (resident ones stay 444 with no PCIe bytes)
        pcie444 = float((U * ~res).sum() * per_expert)                     # PCIe bytes per token, all host BFP4
        Bpe = np.where(res[..., None], 0.0, U[..., None] * Bo[None, None, :])
        av = np.where(res[..., None], np.arange(len(opts)) == i444, avail)
        base_kl = float(C[..., i444].sum())
        for frac in (float(x) for x in a.h2.split(",") if x.strip()):
            choice, b, c = solve_pe(Bpe, C, av, (1.0 - frac) * pcie444)
            fmt = np.zeros((N_LAYERS, N_EXPERT, 3), np.int8)
            for i, o in enumerate(opts):
                fmt[choice == i] = [int(ch) for ch in o]
            nm = f"{name}-h2-{round(100 * frac)}"
            hinfo = {**info, "type": "bfp4-tiered-h2", "h2_fraction": frac, "pcie_bytes_per_token_bfp4": pcie444,
                     "pcie_bytes_per_token": float(b), "pred_kl": float(c), "pred_kl_bfp4_tiered": base_kl,
                     "host_bfp2_mats": int(((fmt == 2) & ~res[..., None]).sum()),
                     "host_bytes": int(sum(option_bytes("".join(map(str, fmt[L, e]))) for L, e in zip(*np.nonzero(~res)))),
                     "options_host": {o: int(((choice == i) & ~res).sum()) for i, o in enumerate(opts)}}
            summary["plans"][nm] = hinfo
            print(nm, json.dumps(hinfo))
            if a.write:
                store.write_plan(nm, fmt, {**hinfo, "method": "GPTQ-BFP (fnx.quant.pipeline)",
                                           "alloc": "fnx.quant.alloc tiered h2"}, resident=res)
    summary.update({"group_fits": fits, "a_L": aL.tolist()})
    all222 = float((aL[:, None] * U * R[..., opts.index("222")]).sum())
    all444 = float((aL[:, None] * U * R[..., opts.index("444")]).sum())
    summary["pred_all222"], summary["pred_all444"] = all222, all444
    for name, budget in zip(a.names.split(","), (float(x) for x in a.budgets.split(","))):
        try:
            choice, b, c = solve(opts, Bo, C, avail, budget)
        except ValueError as ex:
            print(name, "skipped:", ex)
            continue
        fmt = np.zeros((N_LAYERS, N_EXPERT, 3), np.int8)
        for i, o in enumerate(opts):
            m = choice == i
            fmt[m] = [int(ch) for ch in o]
        hot_share = float((U * (fmt == 4).all(-1)).sum() / U.sum())
        info = {"budget": budget, "bytes": int(b), "pred_kl": c, "pred_kl_all444": all444, "pred_kl_all222": all222,
                "n_bfp4_mats": int((fmt == 4).sum()), "n_bfp2_mats": int((fmt == 2).sum()),
                "options": {o: int((choice == i).sum()) for i, o in enumerate(opts)},
                "usage_share_all_bfp4": hot_share}
        summary["plans"][name] = info
        print(name, json.dumps(info))
        if a.write:
            store.write_plan(name, fmt, {**info, "method": "GPTQ-BFP (fnx.quant.pipeline)", "alloc": "fnx.quant.alloc"})
    (QDIR / "alloc.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
