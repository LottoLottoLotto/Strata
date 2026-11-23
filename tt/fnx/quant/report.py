"""Summaries of the quantization run: per-layer proxy-error table (RTN vs GPTQ x BFP4 vs BFP2), usage-weighted, and
a flat per-expert CSV.

    python -m fnx.quant.report                                       (host venv, cwd tt/)
Writes FNX_QUANT_DIR/report/{proxy_layers.md, proxy_layers.json, proxy_experts.csv}.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .. import paths
from ..config import N_EXPERT, N_LAYERS

QDIR = paths.QUANT
VARS = ("rtn4", "gptq4", "rtn2", "gptq2")
MATS = ("gate", "up", "down")


def main() -> int:
    out = QDIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    layers = [L for L in range(N_LAYERS) if (QDIR / "proxy" / f"L{L:02d}.npz").exists()]
    rows, table = [], {}
    csv = ["layer,expert,n_calib,n_held,usage," + ",".join(f"rel_x_{m}_{v}" for v in VARS for m in MATS) + ","
           + ",".join(f"rel_h_{m}_{v}" for v in VARS for m in MATS) + ",rel_expert_gptq444,rel_expert_gptq222,"
           "rel_expert_rtn444,rel_expert_rtn222"]
    for L in layers:
        p = dict(np.load(QDIR / "proxy" / f"L{L:02d}.npz"))
        s = dict(np.load(QDIR / "stats" / f"L{L:02d}.npz"))
        n = s["n"]
        u = n / n.sum()
        nh = p["n_held"]
        ok = nh >= 32
        rec = {"n_calib_min": int(n.min()), "n_calib_median": int(np.median(n)), "experts_held32": int(ok.sum())}
        for v in VARS:
            for m in MATS:
                x = p[f"rel_x_{m}_{v}"]
                rec[f"x_{m}_{v}"] = float(np.sum(u[ok] * x[ok]) / u[ok].sum())       # usage-weighted, held-out
                rec[f"h_{m}_{v}"] = float(np.sum(u * p[f"rel_h_{m}_{v}"]))             # usage-weighted, in-sample
        fh = np.maximum(p["fh2"], 1e-30)
        for c, key in (("gptq444", "abs_444"), ("gptq222", "abs_222"), ("rtn444", "abs_rtn444"),
                       ("rtn222", "abs_rtn222"), ("gptq442", "abs_442"), ("gptq224", "abs_224")):
            rel = p[key] / fh
            rec[f"expert_{c}"] = float(np.sum(u[ok] * rel[ok]) / u[ok].sum())
        table[L] = rec
        for e in range(N_EXPERT):
            vals = [p[f"rel_x_{m}_{v}"][e] for v in VARS for m in MATS] + \
                   [p[f"rel_h_{m}_{v}"][e] for v in VARS for m in MATS] + \
                   [p[k][e] / fh[e] if nh[e] else np.nan for k in ("abs_444", "abs_222", "abs_rtn444", "abs_rtn222")]
            csv.append(f"{L},{e},{int(n[e])},{int(nh[e])},{u[e]:.6g}," + ",".join(f"{x:.6g}" for x in vals))
    (out / "proxy_layers.json").write_text(json.dumps(table, indent=1))
    (out / "proxy_experts.csv").write_text("\n".join(csv) + "\n")
    md = ["Usage-weighted relative output MSE on held-out rows routed to each expert (experts with >= 32 held-out rows).",
          "Matrix columns: the matrix alone with exact inputs. Expert columns: the whole expert (all three matrices).", "",
          "| L | gate rtn4 | gate gptq4 | gate rtn2 | gate gptq2 | up gptq4 | up gptq2 | down rtn4 | down gptq4 | "
          "down rtn2 | down gptq2 | expert rtn4 | expert gptq4 | expert rtn2 | expert gptq2 | exp g/u4 d2 | exp g/u2 d4 |",
          "|" + "---|" * 17]
    for L, r in table.items():
        md.append(f"| {L} | " + " | ".join(f"{r[k]:.4f}" for k in (
            "x_gate_rtn4", "x_gate_gptq4", "x_gate_rtn2", "x_gate_gptq2", "x_up_gptq4", "x_up_gptq2",
            "x_down_rtn4", "x_down_gptq4", "x_down_rtn2", "x_down_gptq2", "expert_rtn444", "expert_gptq444",
            "expert_rtn222", "expert_gptq222", "expert_gptq442", "expert_gptq224")) + " |")
    means = {k: float(np.mean([r[k] for r in table.values()])) for k in next(iter(table.values()))}
    md.append("| mean | " + " | ".join(f"{means[k]:.4f}" for k in (
        "x_gate_rtn4", "x_gate_gptq4", "x_gate_rtn2", "x_gate_gptq2", "x_up_gptq4", "x_up_gptq2",
        "x_down_rtn4", "x_down_gptq4", "x_down_rtn2", "x_down_gptq2", "expert_rtn444", "expert_gptq444",
        "expert_rtn222", "expert_gptq222", "expert_gptq442", "expert_gptq224")) + " |")
    (out / "proxy_layers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-3:]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
