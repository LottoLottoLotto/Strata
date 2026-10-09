"""Settings experiments on captured MoE inputs (pipeline capture dumps calib/acts/L{LL}.npz):

  A. Hessian weighting: s_t = w_te^2 (routing weight squared) vs 1, for GPTQ bfp4/bfp2 of whole experts, scored by the
     held-out routing-weighted output error sum_t w^2 ||f_q(x_t) - f(x_t)||^2 (the expert's share of the MoE output
     error) and the unweighted relative error.
  B. Few calibration tokens: hot experts' Hessians rebuilt from n random calibration rows (n = 32..2560): GPTQ on the
     raw H_n, on H_n + (n_min - n) * prior (gate/up prior: the layer's pooled per-token x x^T; down prior: H_n's own
     diagonal / n), per-block exponent search (no data), RTN; scored on the expert's held-out rows.

    /var/tmp/strata-tt/venv/bin/python -m fnx.quant.experiments --layers 2,11
Writes FNX_QUANT_DIR/experiments.json (--only B2: experiments-B2.json, prior-strength sweep).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ..config import N_EXPERT, N_FF, N_USED
from ..eval.torchref import DEV, setup_torch, silu
from . import methods

QDIR = Path(os.environ.get("FNX_QUANT_DIR", "/var/tmp/strata-tt/quant"))


def ffn(Wg, Wu, Wd, x):
    return (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T


def rows_of(ids, wt, e):
    t, j = torch.nonzero(ids == e, as_tuple=True)
    return t, wt[t, j]


def hessians(x, s, Wg, Wu):
    h = silu(x @ Wg.T) * (x @ Wu.T)
    return (x * s[:, None]).T @ x, (h * s[:, None]).T @ h


def quant(W3, H3, fmt, method):
    Wg, Wu, Wd = W3
    if method == "rtn":
        return methods.rtn(Wg, fmt), methods.rtn(Wu, fmt), methods.rtn(Wd, fmt)
    if method == "search":
        return methods.search(Wg, fmt), methods.search(Wu, fmt), methods.search(Wd, fmt)
    Hgu, Hd = H3
    q = methods.gptq(torch.cat([Wg, Wu])[None], Hgu[None], fmt)[0]
    return q[:N_FF], q[N_FF:], methods.gptq(Wd[None], Hd[None], fmt)[0]


def score(W3, Q3, x, w):
    f = ffn(*W3, x)
    d = ffn(*Q3, x) - f
    e2 = (d * d).sum(-1)
    return {"werr": float((w * w * e2).sum() / (w * w * (f * f).sum(-1)).sum()),
            "rel": float(e2.sum() / (f * f).sum())}


def transformed(W3, mode):
    """methods.channel_transform/apply_transform on one expert: same function f, re-parametrised weights."""
    Wg, Wu, Wd = (w[None] for w in W3)
    perm, k = methods.channel_transform(Wg, Wu, mode)
    Wg2, Wu2, Wd2, _ = methods.apply_transform(Wg, Wu, Wd, torch.zeros(1, N_FF, N_FF, device=Wg.device), perm, k)
    return (Wg2[0], Wu2[0], Wd2[0]), perm[0], k[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="2,11")
    ap.add_argument("--n-a", type=int, default=24)
    ap.add_argument("--n-b", type=int, default=6)
    ap.add_argument("--only", default="", help="B2: prior strength sweep only")
    a = ap.parse_args(argv)
    setup_torch()
    from ..gguf_load import Model
    from ..weights import DEFAULT_SHARD1
    model = Model(DEFAULT_SHARD1)
    out = {}
    for L in (int(x) for x in a.layers.split(",")):
        z = np.load(QDIR / "calib" / "acts" / f"L{L:02d}.npz")
        cx, cid, cw = (torch.from_numpy(z[f"calib_{k}"]).to(DEV) for k in ("x", "ids", "wt"))
        hx, hid, hw = (torch.from_numpy(z[f"held_{k}"]).to(DEV) for k in ("x", "ids", "wt"))
        cx, hx, cid, hid = cx.float(), hx.float(), cid.long(), hid.long()
        cnt = torch.bincount(cid.reshape(-1), minlength=N_EXPERT).cpu().numpy()
        hcnt = torch.bincount(hid.reshape(-1), minlength=N_EXPERT).cpu().numpy()
        order = np.argsort(-cnt)
        # pooled per-token prior for gate/up (unweighted and weighted versions)
        Pu = (cx.T @ cx) / cx.shape[0]
        res = {"counts": {"min": int(cnt.min()), "median": int(np.median(cnt)), "max": int(cnt.max()),
                          "n_below_256": int((cnt < 256).sum()), "n_below_2560": int((cnt < 2560).sum())}}
        if a.only == "B2":
            # B2: how far to pad with the pooled prior: subsample n rows, pad to n_min (gate/up: pooled x x^T, down:
            # own diagonal), plus full data with an extra beta * N prior
            B2 = []
            for e in [x for x in order if hcnt[x] >= 256][:a.n_b]:
                W3 = tuple(torch.from_numpy(model.get_expert(f"blk.{L}.ffn_{r}_exps.weight", int(e))).to(DEV)
                           for r in ("gate", "up", "down"))
                t, w = rows_of(cid, cw, e)
                ht, hwe = rows_of(hid, hw, e)
                g = torch.Generator(device="cpu").manual_seed(int(e))
                for n in (2560, 8192, len(t)):
                    sub = torch.randperm(len(t), generator=g)[:n].to(DEV)
                    s = torch.ones_like(w[sub])          # unweighted: experiment A's choice
                    Hgu, Hd = hessians(cx[t[sub]], s, W3[0], W3[1])
                    sbar = float(s.mean())
                    rec = {"e": int(e), "n": int(n)}
                    for n_min in (0, 8192, 16384, 32768, -0.25, -1.0):
                        pad = max(n_min - n, 0) if n_min >= 0 else -n_min * n     # negative: beta * n
                        H3 = (Hgu + pad * sbar * Pu, Hd + torch.diag(pad * Hd.diagonal() / n))
                        for fmt in ("bfp4", "bfp2"):
                            rec[f"pad{n_min}_{fmt}"] = score(W3, quant(W3, H3, fmt, "gptq"), hx[ht], hwe)["werr"]
                    B2.append(rec)
                    print(f"L{L} B2 e{e} n={n}: " + "  ".join(f"{k} {v:.4f}" for k, v in rec.items()
                                                              if isinstance(v, float)), flush=True)
            out[L] = {"B2": B2}
            continue
        # A: weighting, experts spread over the usage ranking (with >= 64 held-out rows)
        cand = [e for e in order if hcnt[e] >= 64]
        pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, a.n_a)]
        A = []
        for e in pick:
            W3 = tuple(torch.from_numpy(model.get_expert(f"blk.{L}.ffn_{r}_exps.weight", int(e))).to(DEV)
                       for r in ("gate", "up", "down"))
            t, w = rows_of(cid, cw, e)
            ht, hwe = rows_of(hid, hw, e)
            rec = {"e": int(e), "n": int(cnt[e]), "n_held": int(hcnt[e])}
            for wn, s in (("w2", w * w), ("ones", torch.ones_like(w))):
                H3 = hessians(cx[t], s, W3[0], W3[1])
                for fmt in ("bfp4", "bfp2"):
                    rec[f"{wn}_{fmt}"] = score(W3, quant(W3, H3, fmt, "gptq"), hx[ht], hwe)
            for fmt in ("bfp4", "bfp2"):
                rec[f"rtn_{fmt}"] = score(W3, quant(W3, None, fmt, "rtn"), hx[ht], hwe)
            A.append(rec)
            print(f"L{L} A e{e} n={rec['n']}: " + "  ".join(
                f"{k} {rec[k]['werr']:.4f}" for k in rec if isinstance(rec[k], dict)), flush=True)
        res["A"] = A
        # B: few tokens, hot experts
        B = []
        for e in [x for x in order if hcnt[x] >= 256][:a.n_b]:
            W3 = tuple(torch.from_numpy(model.get_expert(f"blk.{L}.ffn_{r}_exps.weight", int(e))).to(DEV)
                       for r in ("gate", "up", "down"))
            t, w = rows_of(cid, cw, e)
            ht, hwe = rows_of(hid, hw, e)
            g = torch.Generator(device="cpu").manual_seed(int(e))
            for n in (32, 128, 512, 2560, len(t)):
                sub = torch.randperm(len(t), generator=g)[:n].to(DEV)
                ts, ws = t[sub], w[sub]
                s = ws * ws
                Hgu, Hd = hessians(cx[ts], s, W3[0], W3[1])
                rec = {"e": int(e), "n": int(n)}
                pad = max(2560 - n, 0)
                sbar = float(s.mean())
                Hgu_p = Hgu + pad * sbar * Pu
                Hd_p = Hd + torch.diag(pad * Hd.diagonal() / n)
                for fmt in ("bfp4", "bfp2"):
                    rec[f"raw_{fmt}"] = score(W3, quant(W3, (Hgu, Hd), fmt, "gptq"), hx[ht], hwe)["werr"]
                    rec[f"prior_{fmt}"] = score(W3, quant(W3, (Hgu_p, Hd_p), fmt, "gptq"), hx[ht], hwe)["werr"]
                    rec[f"search_{fmt}"] = score(W3, quant(W3, None, fmt, "search"), hx[ht], hwe)["werr"]
                    rec[f"rtn_{fmt}"] = score(W3, quant(W3, None, fmt, "rtn"), hx[ht], hwe)["werr"]
                B.append(rec)
                print(f"L{L} B e{e} n={n}: " + "  ".join(f"{k} {v:.4f}" for k, v in rec.items()
                                                         if isinstance(v, float)), flush=True)
        res["B"] = B
        # C: exact intermediate-channel transforms before GPTQ (w2-weighted H from all calibration rows)
        Cr = []
        for e in pick:
            W3 = tuple(torch.from_numpy(model.get_expert(f"blk.{L}.ffn_{r}_exps.weight", int(e))).to(DEV)
                       for r in ("gate", "up", "down"))
            t, w = rows_of(cid, cw, e)
            ht, hwe = rows_of(hid, hw, e)
            rec = {"e": int(e), "n": int(cnt[e])}
            for mode in ("none", "perm", "scale", "scale+perm"):
                W3t, _, _ = transformed(W3, mode)
                H3 = hessians(cx[t], w * w, W3t[0], W3t[1])
                for fmt in ("bfp4", "bfp2"):
                    rec[f"{mode}_{fmt}"] = score(W3, quant(W3t, H3, fmt, "gptq"), hx[ht], hwe)["werr"]
            Cr.append(rec)
            print(f"L{L} C e{e}: " + "  ".join(f"{k} {v:.4f}" for k, v in rec.items() if isinstance(v, float)),
                  flush=True)
        res["C"] = Cr
        for fmt in ("bfp4", "bfp2"):
            res[f"C_mean_{fmt}"] = {m: float(np.mean([x[f"{m}_{fmt}"] for x in Cr]))
                                    for m in ("none", "perm", "scale", "scale+perm")}
        # summaries
        for fmt in ("bfp4", "bfp2"):
            r = np.array([[x[f"w2_{fmt}"]["werr"], x[f"ones_{fmt}"]["werr"], x[f"rtn_{fmt}"]["werr"]] for x in A])
            res[f"A_mean_{fmt}"] = {"w2": r[:, 0].mean(), "ones": r[:, 1].mean(), "rtn": r[:, 2].mean(),
                                    "w2_wins": int((r[:, 0] < r[:, 1]).sum()), "of": len(A)}
        out[L] = res
        print(json.dumps({k: v for k, v in res.items() if k[:6] in ("A_mean", "C_mean") or k == "counts"},
                         default=float), flush=True)
    name = f"experiments-{a.only}.json" if a.only else "experiments.json"
    (QDIR / name).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
