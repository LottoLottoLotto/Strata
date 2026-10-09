"""Flash-Next numpy reference model (spec §A.2): embedding, PLE, 48 HC-wrapped GDN/QSA + MoE layers, output_hc, head.

`RefModel(w, experts).forward(tokens, positions)` runs T consecutive tokens of one sequence through every layer
(each module is token-serial inside) and returns float32 logits [T, 248320].  Numerics: float32 activations with
dequantized weights (CONTRACT "Numerics target"); routed experts through Strata's CPU kernels (fnx.cpu_experts),
exactly as the reference engine computes its CPU rows.  STRATA_IQ_MT_MIN=1 (fnx.cpu_experts' default), as the golden
runs do: otherwise a row's expert outputs depend on how many rows share the expert, i.e. on the chunking.  Even
then chunkings differ at rounding level (BLAS blocking), which the quantized experts and the router's top-10
amplify: measured 0.6% residual rel-L2 after 13 layers between a 31-row call and a 20+11 split.

Per-layer residual ladder for bisection: `forward(..., dump=d)` fills `d` with
  R      [n_layers+1, T, 4, 2560]  slot 0 = input residual (embedding in all 4 streams), slot L+1 = R after layer L
                                   (= Strata's --dump-layers "C1" ladder layout; the PLE update sits inside slot 2)
  R_ple  [T, 4, 2560]              R after the PLE block (before layer 1's attention read)
  bo_mix, bo_moe [n_layers, T, 2560]; inj_mix, inj_moe [n_layers, T, 4]; head_in [T, 2560]

Golden comparison (tools/golden.sh writes /var/tmp/strata-tt/golden/<name>/): `read_logpos` parses Strata's
STRATA_LOGPOS TSV, `LogposScore` scores logits against it (argmax agreement, top-k overlap, KL over Strata's top-K
plus a rest bucket, target NLL).  CLI: `python -m fnx.ref.model golden NAME [--max-pos N]`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..config import HC, N_EMBD, N_LAYERS, N_VOCAB, PLE_LAYER, is_qsa
from .embed import embed, head
from .gdn import GdnState, gdn
from .hc import hc_read, hc_write
from .moe import moe
from .ple import PleState, PleTable, ple_block
from .qsa import QsaState, qsa

GOLDEN = Path(os.environ.get("FNX_GOLDEN", "/var/tmp/strata-tt/golden"))


def hc_weights(w, prefix: str, inject: bool = True):
    """(gamma [10240], down [320,10240], up [10240,320], inject [4,10240] | None) of one HC read."""
    return (w.get(f"{prefix}_norm.weight"), w.get(f"{prefix}_down.weight"), w.get(f"{prefix}_up.weight"),
            w.get(f"{prefix}_inject.weight") if inject else None)


def greedy(logits: np.ndarray) -> np.ndarray:
    """Strata's argmax: first maximal index, NaN never picked (sampler.cu:143-167). logits [T, V] -> [T] int64."""
    x = np.where(np.isnan(logits), -np.inf, logits)
    return x.argmax(axis=-1)


class RefModel:
    def __init__(self, w, experts, n_layers: int = N_LAYERS, max_ctx: int = 4096):
        self.w, self.experts, self.n_layers, self.max_ctx = w, experts, n_layers, max_ctx
        self.ple_table = PleTable(w.model)
        self.hc = [(hc_weights(w, f"blk.{L}.hc_attn"), hc_weights(w, f"blk.{L}.hc_ffn")) for L in range(n_layers)]
        self.hc_out = hc_weights(w, "output_hc", inject=False)
        self.reset()

    def reset(self) -> None:
        self.gdn_state = {L: GdnState() for L in range(self.n_layers) if not is_qsa(L)}
        self.qsa_state = {L: QsaState(max_cells=self.max_ctx) for L in range(self.n_layers) if is_qsa(L)}
        self.ple_state = PleState()
        self.pos = 0

    def forward(self, tokens, positions=None, dump: dict | None = None) -> np.ndarray:
        """T tokens following the ones already consumed -> logits f32 [T, 248320]."""
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        T = len(tokens)
        pos = np.arange(self.pos, self.pos + T) if positions is None else np.asarray(positions, dtype=np.int64)
        e = embed(self.w, tokens)                                  # f32 [T, 2560]
        e_ple = self.ple_table.gather(self.ple_state.rows(tokens))   # f32 [T, 2560]; advances ple prev
        R = np.ascontiguousarray(np.broadcast_to(e[:, None, :], (T, HC, N_EMBD)), dtype=np.float32)
        lad, bos = ([R.copy()], {k: [] for k in ("bo_mix", "bo_moe", "inj_mix", "inj_moe")}) if dump is not None \
            else (None, None)
        for L in range(self.n_layers):
            if L == PLE_LAYER:
                R = ple_block(self.w, e_ple, R, self.ple_state)
                if dump is not None:
                    dump["R_ple"] = R.copy()
            attn, ffn = self.hc[L]
            mixed, inj = hc_read(R, *attn)
            bo = qsa(self.w, L, mixed, pos, self.qsa_state[L]) if is_qsa(L) else gdn(self.w, L, mixed, self.gdn_state[L])
            R = hc_write(R, bo, inj)
            if bos is not None:
                bos["bo_mix"].append(bo)
                bos["inj_mix"].append(inj)
            mixed, inj2 = hc_read(R, *ffn)
            bo = moe(self.w, L, mixed, self.experts)
            R = hc_write(R, bo, inj2)
            if bos is not None:
                bos["bo_moe"].append(bo)
                bos["inj_moe"].append(inj2)
                lad.append(R.copy())
        head_in, _ = hc_read(R, *self.hc_out)
        logits = head(self.w, head_in)
        if dump is not None:
            dump["R"] = np.stack(lad)
            dump.update({k: np.stack(v) for k, v in bos.items()})
            dump["head_in"] = head_in
        self.pos = int(pos[-1]) + 1
        return logits

    def generate(self, prompt, max_new: int, chunk: int = 64, eos=()) -> list[int]:
        """Greedy continuation of `prompt` from a fresh sequence (one-token-at-a-time decode after the prompt)."""
        self.reset()
        prompt = list(prompt)
        for i in range(0, len(prompt) - 1, chunk):
            self.forward(prompt[i:min(i + chunk, len(prompt) - 1)])
        out, tok = [], prompt[-1]
        for _ in range(max_new):
            tok = int(greedy(self.forward([tok]))[0])
            out.append(tok)
            if tok in eos:
                break
        return out


# ---------------------------------------------------------------------------------------------------- golden data
def read_ids(path) -> list[int]:
    s = Path(path).read_text().replace(",", " ").split()
    return [int(x) for x in s]


def read_logpos(path) -> dict:
    """STRATA_LOGPOS TSV (spec Part 8 §8.2h) -> dict of arrays; topk_ids/topk_lp [n, K] (K may be 0)."""
    pos, tgt, lpt, am, lpa, top_i, top_l = [], [], [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        pos.append(int(f[0])); tgt.append(int(f[1])); lpt.append(float(f[2])); am.append(int(f[3]))
        lpa.append(float(f[4]))
        kv = [x.split(":") for x in f[8:]]
        top_i.append([int(a) for a, _ in kv]); top_l.append([float(b) for _, b in kv])
    k = min((len(t) for t in top_i), default=0)
    return {"pos": np.array(pos), "target": np.array(tgt), "logp_target": np.array(lpt), "argmax": np.array(am),
            "logp_argmax": np.array(lpa), "topk_ids": np.array([t[:k] for t in top_i], dtype=np.int64).reshape(-1, k),
            "topk_lp": np.array([t[:k] for t in top_l], dtype=np.float64).reshape(-1, k)}


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


class LogposScore:
    """Accumulates per-position agreement of our logits with Strata's teacher-forced LOGPOS rows."""

    def __init__(self, golden: dict):
        self.g = golden
        self.by_pos = {int(p): i for i, p in enumerate(golden["pos"])}
        self.rows = []

    def add(self, positions, logits: np.ndarray) -> None:
        lp = log_softmax64(logits)
        am = greedy(logits)
        for t, p in enumerate(np.asarray(positions).reshape(-1)):
            i = self.by_pos.get(int(p))
            if i is None:
                continue
            g_ids, g_lp = self.g["topk_ids"][i], self.g["topk_lp"][i]
            our_top = np.argsort(-lp[t], kind="stable")[:10]
            P = np.exp(g_lp)
            Q = np.exp(lp[t][g_ids])
            p_rest, q_rest = max(1.0 - P.sum(), 1e-12), max(1.0 - Q.sum(), 1e-12)
            kl = float((P * (g_lp - lp[t][g_ids])).sum() + p_rest * np.log(p_rest / q_rest))
            self.rows.append({
                "pos": int(p), "target": int(self.g["target"][i]),
                "argmax_ok": int(am[t] == self.g["argmax"][i]),
                "top5": len(set(our_top[:5]) & set(g_ids[:5].tolist())) / 5,
                "top10": len(set(our_top[:10]) & set(g_ids[:10].tolist())) / 10,
                "kl": kl, "nll_ours": float(-lp[t][self.g["target"][i]]), "nll_strata": float(-self.g["logp_target"][i]),
                "dlogp_argmax": float(lp[t][self.g["argmax"][i]] - self.g["logp_argmax"][i]),
                "margin_strata": float(g_lp[0] - g_lp[1]) if len(g_lp) > 1 else float("nan"),
            })

    def summary(self) -> dict:
        r = self.rows
        if not r:
            return {"n": 0}
        a = lambda k: np.array([x[k] for x in r], dtype=np.float64)  # noqa: E731
        kl = a("kl")
        miss = [x for x in r if not x["argmax_ok"]]
        f = lambda v: round(float(v), 6)  # noqa: E731
        return {"n": len(r), "argmax_agree": f(a("argmax_ok").mean()), "top5_overlap": f(a("top5").mean()),
                "top10_overlap": f(a("top10").mean()), "kl_mean": f(kl.mean()), "kl_p50": f(np.median(kl)),
                "kl_p99": f(np.quantile(kl, 0.99)), "kl_max": f(kl.max()), "ppl_ours": f(np.exp(a("nll_ours").mean())),
                "ppl_strata": f(np.exp(a("nll_strata").mean())),
                "argmax_misses": [(x["pos"], round(x["margin_strata"], 4)) for x in miss][:20]}


def first_logits_score(golden: np.ndarray, ours: np.ndarray) -> dict:
    """Full-vocabulary comparison against STRATA_DUMP_FIRST_LOGITS (last prompt position)."""
    lg, lo = log_softmax64(golden[None])[0], log_softmax64(ours[None])[0]
    P = np.exp(lg)
    tg, to = np.argsort(-golden, kind="stable")[:10], np.argsort(-ours, kind="stable")[:10]
    d = ours.astype(np.float64) - golden
    return {"argmax_strata": int(tg[0]), "argmax_ours": int(to[0]), "top10_overlap": len(set(tg) & set(to)) / 10,
            "kl": float((P * (lg - lo)).sum()), "max_abs_logit_diff": float(np.abs(d).max()),
            "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(golden))}


def run_golden(name: str, max_pos: int | None, chunk: int, n_threads: int) -> dict:
    """RefModel over a golden prompt (+ its greedy continuation when logpos_full.tsv exists): LOGPOS scores,
    first-logits score, and a greedy continuation compared with Strata's serve continuation."""
    from ..cpu_experts import CpuExperts
    from ..weights import Weights

    d = GOLDEN / name
    prompt = read_ids(GOLDEN / "prompts" / f"{name}.ids")
    full = d / "logpos_full.tsv"
    lp_path = full if full.exists() else d / "logpos.tsv"
    ids = read_ids(d / "full.ids") if full.exists() else prompt
    g = read_logpos(lp_path)
    w = Weights()
    m = RefModel(w, CpuExperts(w.model, N_LAYERS, n_threads))
    score = LogposScore(g)
    n = len(ids) - 1 if max_pos is None else min(len(ids) - 1, max_pos)
    first = None
    t0 = time.time()
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        lo = m.forward(ids[i:j])
        score.add(np.arange(i, j), lo)
        if i <= len(prompt) - 1 < j:
            first = lo[len(prompt) - 1 - i]
        print(f"  {name}: {j}/{n} positions, {time.time() - t0:.0f} s, running {score.summary().get('argmax_agree')}",
              flush=True)
    res = {"name": name, "logpos_file": str(lp_path), "positions": n, "logpos": score.summary(),
           "seconds": round(time.time() - t0, 1)}
    if first is not None and (d / "first_logits.f32").exists():
        res["first_logits_vs_oneshot"] = first_logits_score(np.fromfile(d / "first_logits.f32", dtype=np.float32), first)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden", help="score RefModel against golden LOGPOS / first-logits files")
    g.add_argument("names", nargs="+")
    g.add_argument("--max-pos", type=int, default=None)
    g.add_argument("--chunk", type=int, default=64)
    g.add_argument("--threads", type=int, default=16)
    gen = sub.add_parser("generate", help="greedy continuation of a golden prompt vs Strata's serve continuation")
    gen.add_argument("name")
    gen.add_argument("--max-new", type=int, default=64)
    gen.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    if a.cmd == "golden":
        for n in a.names:
            r = run_golden(n, a.max_pos, a.chunk, a.threads)
            print("RESULT", r, flush=True)
    else:
        from ..cpu_experts import CpuExperts
        from ..config import EOS_IDS
        from ..weights import Weights
        w = Weights()
        m = RefModel(w, CpuExperts(w.model, N_LAYERS, a.threads))
        prompt = read_ids(GOLDEN / "prompts" / f"{a.name}.ids")
        t0 = time.time()
        out = m.generate(prompt, a.max_new, eos=EOS_IDS)
        ref = read_ids(GOLDEN / a.name / "serve_gen.ids")
        k = next((i for i, (x, y) in enumerate(zip(out, ref)) if x != y), min(len(out), len(ref)))
        print("RESULT", {"name": a.name, "ours": out, "strata_serve": ref, "first_divergence": k,
                         "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
