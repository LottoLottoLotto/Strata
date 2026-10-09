"""Teacher-forced scores of candidate logits against the cached float32 reference (fnx.eval.torchref), and the tables.

Per scored target token i (predicted by position i-1):
    nll      -log q(token i)        (candidate), nll_ref from the reference's tgt_lp
    top1     argmax q == the reference's argmax (Strata's greedy pick: first maximal index)
    kl       KL(P_ref || Q) over the reference's top-K tokens plus one bucket for the rest of the vocabulary
             (a lower bound of the full KL by the data-processing inequality; with K = 256 the rest bucket holds on
             average 0.3% (code) to 1.2% (instruct) of the reference's mass, at most 54% at one book position)
Per corpus: PPL = exp(mean nll), dPPL = PPL_cand / PPL_ref - 1, top-1 agreement, mean / p99 KL.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import data

CORPORA = ("chat", "code", "instruct", "book")


def load_ref(rec_id: str) -> dict:
    with np.load(data.EVAL / "ref" / f"{rec_id}.npz") as z:
        return {k: z[k] for k in z.files}


def targets(rec: dict) -> np.ndarray:
    """Scored target token indices of a record (each >= 1)."""
    return np.concatenate([np.arange(a, b) for a, b in rec["score"]]) if rec["score"] else np.zeros(0, np.int64)


def score_rows(logits, pos: np.ndarray, tokens: np.ndarray, ref: dict) -> dict:
    """logits f32 [m, V] of positions `pos` (absolute, each predicting tokens[pos + 1]) -> per-position metrics."""
    x = torch.as_tensor(np.asarray(logits, np.float32))
    lse = torch.logsumexp(x.double(), -1)
    tgt = torch.as_tensor(np.asarray(tokens, np.int64)[pos + 1])
    nll = -(x.gather(1, tgt[:, None])[:, 0].double() - lse)
    am = torch.argmax(x, -1).numpy()
    rid = torch.as_tensor(ref["top_ids"][pos].astype(np.int64))
    lq = x.gather(1, rid).double() - lse[:, None]                                     # [m, K]
    lp = torch.as_tensor(ref["top_lp"][pos].astype(np.float64))
    p = lp.exp()
    kl = (p * (lp - lq)).sum(-1)
    pr = (1.0 - p.sum(-1)).clamp(min=0.0)
    qr = (1.0 - lq.exp().sum(-1)).clamp(min=1e-300)
    kl = kl + torch.where(pr > 0, pr * (pr.clamp(min=1e-300).log() - qr.log()), torch.zeros_like(pr))
    return {"pos": pos.astype(np.int32), "nll": nll.numpy(), "nll_ref": -ref["tgt_lp"][pos].astype(np.float64),
            "top1": am == ref["argmax"][pos], "kl": kl.clamp(min=0.0).numpy(), "argmax": am.astype(np.int32)}


def corpus_table(per: dict[str, list[dict]]) -> dict:
    """{corpus: [per-record metric dicts]} -> {corpus: summary}, plus "all"."""
    out = {}
    groups = {c: v for c, v in per.items() if v}
    groups["all"] = [m for v in per.values() for m in v]
    for c, ms in groups.items():
        if not ms:
            continue
        nll = np.concatenate([m["nll"] for m in ms])
        nr = np.concatenate([m["nll_ref"] for m in ms])
        kl = np.concatenate([m["kl"] for m in ms])
        t1 = np.concatenate([m["top1"] for m in ms])
        ppl, pr = float(np.exp(nll.mean())), float(np.exp(nr.mean()))
        out[c] = {"n": int(nll.size), "ppl_ref": pr, "ppl": ppl, "dppl": ppl / pr - 1, "top1": float(t1.mean()),
                  "kl_mean": float(kl.mean()), "kl_p99": float(np.quantile(kl, 0.99)), "records": len(ms)}
    return out


def fmt_table(rows: dict[str, dict], order=CORPORA + ("all",)) -> str:
    lines = ["| corpus | tokens | PPL ref | PPL TT | dPPL | top-1 | mean KL | p99 KL |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in order:
        r = rows.get(c)
        if r:
            lines.append(f"| {c} | {r['n']} | {r['ppl_ref']:.4f} | {r['ppl']:.4f} | {100 * r['dppl']:+.2f}% | "
                         f"{r['top1']:.4f} | {r['kl_mean']:.4f} | {r['kl_p99']:.3f} |")
    return "\n".join(lines)


def load_json(p: Path):
    return json.loads(Path(p).read_text())
