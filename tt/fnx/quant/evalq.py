"""End-to-end effect of quantized routed experts on the held-out calibration sequences (16 x 2048 tokens, disjoint
from the eval set), float32 torch forward (fnx.eval.torchref.Ref), against the same forward with float experts.

    python -m fnx.quant.evalq CONFIG [CONFIG ...]                    (host venv, cwd tt/)
CONFIG:
  plan:<name>          a store plan (fnx.quant.store.Plan): every routed expert from its snapped pool rows
  rtn4:<L0>-<L1>       tt-metal RTN bfp4 of every expert in layers L0..L1 (float elsewhere): layer-group sensitivity
  rtn2:<L0>-<L1>, gptq-pool4 / gptq-pool2 (pool rows where present, float elsewhere)
Per config: mean KL(float || quant) over all positions (full vocabulary), top-1 agreement, and per corpus the
perplexity change of the next tokens. Appends to FNX_QUANT_DIR/evalq.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from .. import paths
from ..config import N_LAYERS
from ..eval.torchref import DEV, ExpertLoader, Ref, setup_torch
from . import bfp, calib_data, methods
from .pipeline import shm_hygiene
from .store import FMT, SHAPE, Plan, Pool

QDIR = paths.QUANT


def pool_experts(pool: Pool, L: int, fmt_of: np.ndarray, ex: dict) -> dict:
    """Replace ex's matrices by the pool's snapped rows where fmt_of[e, m] in (4, 2) (0 keeps float)."""
    out = ex          # in place: the loader hands out a fresh copy per layer (saves 5 GB of GPU memory)
    for bits in (4, 2):
        sel = np.argwhere(fmt_of == bits)
        if not len(sel):
            continue
        ix = pool.index(L, bits)[sel[:, 0], sel[:, 1]]
        if (ix < 0).any():
            raise ValueError(f"layer {L}: bfp{bits} rows missing")
        rows = pool.rows(L, bits)
        for m, name in enumerate(("gate", "up", "down")):
            k = sel[:, 1] == m
            if not k.any():
                continue
            order = np.argsort(ix[k])                       # pool rows ascending (mmap-friendly)
            ri, es_all = ix[k][order], sel[k, 0][order]
            K, N = SHAPE[m]
            for c in range(0, len(ri), 16):
                r = torch.from_numpy(np.ascontiguousarray(rows[ri[c:c + 16]])).to(DEV)
                vals = bfp.unpack_batch(r, FMT[bits], K, N).transpose(1, 2)   # [n, out, in]
                out[name][torch.as_tensor(es_all[c:c + 16], device=DEV)] = vals
    return out


class Swap:
    """torchref loader wrapper: float experts from ExpertLoader, transformed per layer."""

    def __init__(self, base: ExpertLoader, fn):
        self.base, self.fn = base, fn

    def request(self, L):
        self.base.request(L)

    def get(self, L):
        ex = self.base.get(L)
        return self.fn(L, ex)


def make_fn(cfg: str):
    kind, _, arg = cfg.partition(":")
    if kind == "float":
        return lambda L, ex: ex
    if kind in ("rtn4", "rtn2"):
        lo, hi = (int(x) for x in arg.split("-"))
        f = "bfp4" if kind == "rtn4" else "bfp2"

        def rtn_all(v):
            return torch.cat([methods.rtn(v[i:i + 32], f) for i in range(0, v.shape[0], 32)])
        return lambda L, ex: {k: rtn_all(v) for k, v in ex.items()} if lo <= L <= hi else ex
    if kind == "plan":
        plan = Plan(arg)
        return lambda L, ex: pool_experts(plan.pool, L, plan.fmt(L), ex)
    if kind in ("gptq-pool4", "gptq-pool2"):
        pool = Pool()
        bits = 4 if kind == "gptq-pool4" else 2

        def fn(L, ex):
            have = pool.index(L, bits) >= 0
            return pool_experts(pool, L, np.where(have, bits, 0), ex)
        return fn
    raise ValueError(cfg)


def run(w, shard1, seqs, fn, workers):
    loader = Swap(ExpertLoader(shard1, workers=workers), fn)
    heads = {}
    try:
        Ref(w, loader).run(seqs, lambda i, h: heads.__setitem__(i, h.cpu()))
    finally:
        loader.base.close()
    return [heads[i] for i in range(len(seqs))]


def compare(Wh, base, quant, seqs, corpora, chunk=512) -> dict:
    kl, top1, lp_b, lp_q, corp = [], [], [], [], []
    for hb, hq, s, c in zip(base, quant, seqs, corpora):
        tok = torch.as_tensor(np.asarray(s, np.int64), device=DEV)
        for a in range(0, hb.shape[0], chunk):
            lb = (hb[a:a + chunk].to(DEV) @ Wh.T).double()
            lq = (hq[a:a + chunk].to(DEV) @ Wh.T).double()
            pb, pq = torch.log_softmax(lb, -1), torch.log_softmax(lq, -1)
            kl.append((pb.exp() * (pb - pq)).sum(-1).cpu())
            top1.append((lb.argmax(-1) == lq.argmax(-1)).cpu())
            n = min(a + chunk, len(s) - 1) - a
            if n > 0:
                t = tok[a + 1:a + 1 + n, None]
                lp_b.append(torch.gather(pb[:n], 1, t)[:, 0].cpu())
                lp_q.append(torch.gather(pq[:n], 1, t)[:, 0].cpu())
                corp += [c] * n
    kl, top1 = torch.cat(kl), torch.cat(top1).float()
    lp_b, lp_q = torch.cat(lp_b), torch.cat(lp_q)
    corp = np.array(corp)
    res = {"kl": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)), "top1": float(top1.mean()),
           "dppl_all": float(torch.exp((lp_b - lp_q).mean()) - 1)}
    for c in sorted(set(corp)):
        m = torch.from_numpy(corp == c)
        res[f"dppl_{c}"] = float(torch.exp((lp_b[m] - lp_q[m]).mean()) - 1)
        res[f"ppl_float_{c}"] = float(torch.exp(-lp_b[m].mean()))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    from ..weights import Weights
    setup_torch()
    shm_hygiene()
    recs = calib_data.load("heldout")[:a.n_seq]
    seqs, corpora = [r["token_ids"] for r in recs], [r["corpus"] for r in recs]
    w = Weights()
    shard1 = str(w.model.files[0].path)
    cache = QDIR / "evalq-base.pt"
    t0 = time.time()
    if cache.exists():
        base = torch.load(cache)
    else:
        base = run(w, shard1, seqs, make_fn("float"), a.workers)
        torch.save(base, cache)
    print(f"evalq: float base {time.time() - t0:.0f} s", flush=True)
    Wh = Ref(w, None).head_weight()
    out_p = QDIR / "evalq.json"
    allres = json.loads(out_p.read_text()) if out_p.exists() else {}
    for cfg in a.configs:
        t0 = time.time()
        q = run(w, shard1, seqs, make_fn(cfg), a.workers)
        res = compare(Wh, base, q, seqs, corpora)
        res["seconds"] = round(time.time() - t0, 1)
        res["tokens"] = sum(len(s) for s in seqs)
        allres[cfg] = res
        out_p.write_text(json.dumps(allres, indent=1))
        print(f"evalq: {cfg}: " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
