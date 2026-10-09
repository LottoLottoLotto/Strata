"""End-to-end Flash-Next: TtModel (P150) vs RefModel (numpy f32) vs Strata golden data.

Run on the device:  tools/devrun.sh test-model fnx/tests/test_model.py [--layers N] [--prompt chat] [--gen 16]
                    [--golden chat,code] [--long code,doc] [--skip a,b,c,d]
A. Prompt forward (one call, T = prompt rows): per-layer residual error of TT vs RefModel (the ladder slot after
   every layer, the PLE output, the head input) and per-position logits agreement.
B. Greedy decode in lockstep: both models are fed TT's greedy token each step (T = 1 calls); per-step argmax
   agreement and KL, and the TT token stream.
C. Golden (only with all 48 layers): TT teacher-forced over golden/<name>/full.ids, scored against Strata's
   STRATA_LOGPOS top-256 rows (argmax agreement, top-k overlap, KL); TT greedy tokens vs Strata's serve greedy tokens.
D. Long context (--long): TT vs RefModel teacher-forced over golden/<name>/full.ids in chunks, per-position argmax
   agreement and KL, also per 512-position bucket.
--layers N < 48 runs a truncated model (first N layers, then output_hc + head) for A and B only; the default is
every layer whose tensors are already on disk.
Pass thresholds (A/B, TT vs RefModel): final residual rel-L2 <= 0.05, mean KL <= 0.05, argmax agreement >= 0.9
where a position also counts as agreeing when TT's pick is a near-tie in the reference (ref log-prob gap to the ref
argmax <= 0.1 nats; the strict rate is printed too); (C, TT vs Strata): teacher-forced argmax agreement >= 0.9.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from fnx.config import EOS_IDS, N_LAYERS, PLE_LAYER, is_qsa
from fnx.cpu_experts import CpuExperts
from fnx.ref.model import (GOLDEN, LogposScore, RefModel, first_logits_score, greedy, log_softmax64, read_ids,
                           read_logpos)
from fnx.weights import Weights

TH_RESID, TH_AGREE, TH_KL, TH_GOLDEN, NEAR_TIE = 0.05, 0.9, 0.05, 0.9, 0.1


def available_layers(model) -> int:
    n = 0
    while n < N_LAYERS and all(model.available(k) for k in model.index if k.startswith(f"blk.{n}.")):
        n += 1
    return n


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))


def logits_cmp(ref: np.ndarray, tt: np.ndarray) -> dict:
    lr, lt = log_softmax64(ref), log_softmax64(tt)
    kl = (np.exp(lr) * (lr - lt)).sum(-1)
    ar, at = greedy(ref), greedy(tt)
    gap = lr[np.arange(len(ar)), ar] - lr[np.arange(len(at)), at]      # ref nats between its pick and TT's
    top = [len(set(np.argsort(-ref[t])[:10]) & set(np.argsort(-tt[t])[:10])) / 10 for t in range(len(ref))]
    return {"strict": float((ar == at).mean()), "agree": float((gap <= NEAR_TIE).mean()), "top10": float(np.mean(top)),
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "rel_l2": rel(tt, ref),
            "ref_argmax": ar.tolist(), "tt_argmax": at.tolist(),
            "gaps": [(t, round(float(gap[t]), 4)) for t in np.nonzero(ar != at)[0]]}


def part_a(ref: RefModel, tt, tokens) -> dict:
    dr, dt = {}, {}
    t0 = time.time()
    lr = ref.forward(tokens, dump=dr)
    t1 = time.time()
    lt, ids = tt.forward(tokens, dump=dt)
    t2 = time.time()
    print(f"\n== A. prompt forward, T={len(tokens)} (ref {t1 - t0:.1f} s, tt {t2 - t1:.1f} s incl. per-layer readback)")
    print(f"{'slot':>4} {'after':>12} {'rel_l2':>9} {'max_abs':>10} {'ref_rms':>10}  per-stream rel_l2")
    worst = 0.0
    for s in range(dr["R"].shape[0]):
        a, b = dt["R"][s], dr["R"][s]
        what = "embedding" if s == 0 else f"L{s - 1} {'qsa' if is_qsa(s - 1) else 'gdn'}"
        r = rel(a, b)
        worst = max(worst, r)
        streams = " ".join(f"{rel(a[:, c], b[:, c]):.4f}" for c in range(a.shape[1]))
        print(f"{s:>4} {what:>12} {r:9.5f} {np.abs(a - b).max():10.3e} {np.sqrt((b ** 2).mean()):10.3e}  {streams}")
        if s == PLE_LAYER:
            print(f"{'':>4} {'(PLE out)':>12} {rel(dt['R_ple'], dr['R_ple']):9.5f}")
    print(f"head_in rel_l2 {rel(dt['head_in'], dr['head_in']):.5f}")
    c = logits_cmp(lr, lt)
    final = rel(dt["R"][-1], dr["R"][-1])
    print(f"logits: rel_l2 {c['rel_l2']:.4f} argmax agree strict {c['strict']:.3f} near-tie-tolerant {c['agree']:.3f} "
          f"top10 {c['top10']:.3f} KL mean {c['kl_mean']:.4g} max {c['kl_max']:.4g}")
    print(f"disagreements (position, ref gap in nats): {c['gaps']}")
    print(f"TT argmax per position:  {c['tt_argmax']}\nref argmax per position: {c['ref_argmax']}")
    ok = final <= TH_RESID and c["agree"] >= TH_AGREE and c["kl_mean"] <= TH_KL
    print(f"A {'PASS' if ok else 'FAIL'}: final residual rel_l2 {final:.4f} (<= {TH_RESID}), agree {c['agree']:.3f} "
          f"(>= {TH_AGREE}), KL {c['kl_mean']:.4g} (<= {TH_KL}); worst slot rel_l2 {worst:.4f}")
    return {"ok": ok, "final_rel": final, **c}


def part_b(ref: RefModel, tt, prompt, n_gen: int) -> dict:
    ref.reset()
    tt.reset()
    if len(prompt) > 1:
        ref.forward(prompt[:-1])
        tt.forward(prompt[:-1])
    tok, toks, agree, strict, gaps, kls = prompt[-1], [], [], [], [], []
    t_tt = t_ref = 0.0
    for _ in range(n_gen):
        t0 = time.time()
        lt, ids = tt.forward([tok])
        t1 = time.time()
        lr = ref.forward([tok])
        t_ref += time.time() - t1
        t_tt += t1 - t0
        c = logits_cmp(lr, lt)
        agree.append(c["agree"])
        strict.append(c["strict"])
        gaps.append(c["gaps"][0][1] if c["gaps"] else 0.0)
        kls.append(c["kl_mean"])
        if int(ids[0]) != int(greedy(lt)[0]):
            print(f"  note: device argmax {int(ids[0])} != host argmax {int(greedy(lt)[0])}")
        tok = int(ids[0])
        toks.append(tok)
        if tok in EOS_IDS:
            break
    a, k = float(np.mean(agree)), float(np.mean(kls))
    ok = a >= TH_AGREE and k <= TH_KL
    print(f"\n== B. greedy decode, {len(toks)} steps (T=1): tt {t_tt / len(toks) * 1e3:.0f} ms/token, "
          f"ref {t_ref / len(toks) * 1e3:.0f} ms/token")
    print(f"TT tokens: {toks}")
    print(f"per-step ref argmax == TT token: {''.join('.' if x else 'X' for x in strict)}; ref gap of TT's token "
          f"(nats) {gaps}; KL per step {[round(x, 4) for x in kls]}")
    print(f"B {'PASS' if ok else 'FAIL'}: agree near-tie-tolerant {a:.3f} strict {np.mean(strict):.3f} "
          f"(>= {TH_AGREE}), KL mean {k:.4g} (<= {TH_KL})")
    return {"ok": ok, "tokens": toks, "agree": a, "kl": k}


def part_c(tt, names, n_gen: int, chunk: int) -> bool:
    ok_all = True
    for name in names:
        d = GOLDEN / name
        if not (d / "logpos_full.tsv").exists():
            print(f"\n== C. {name}: no golden {d}/logpos_full.tsv yet - skipped")
            ok_all = False
            continue
        ids = read_ids(d / "full.ids")
        prompt = read_ids(GOLDEN / "prompts" / f"{name}.ids")
        score = LogposScore(read_logpos(d / "logpos_full.tsv"))
        tt.reset()
        t0 = time.time()
        n = len(ids) - 1
        first = None
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            lt, _ = tt.forward(ids[i:j])
            score.add(np.arange(i, j), lt)
            if i <= len(prompt) - 1 < j:
                first = lt[len(prompt) - 1 - i]
        s = score.summary()
        t1 = time.time()
        gen = tt.generate(prompt, n_gen, chunk=chunk, eos=EOS_IDS)
        strata = read_ids(d / "serve_gen.ids")
        k = next((i for i, (x, y) in enumerate(zip(gen, strata)) if x != y), min(len(gen), len(strata)))
        ok = s["argmax_agree"] >= TH_GOLDEN
        ok_all &= ok
        print(f"\n== C. {name}: {s['n']} teacher-forced positions in {t1 - t0:.0f} s (chunks of {chunk})")
        print({k2: (round(v, 5) if isinstance(v, float) else v) for k2, v in s.items()})
        if first is not None and (d / "first_logits.f32").exists():
            print("last prompt position vs Strata one-shot STRATA_DUMP_FIRST_LOGITS (prefill path):",
                  first_logits_score(np.fromfile(d / "first_logits.f32", dtype=np.float32), first))
        print(f"TT greedy ({time.time() - t1:.0f} s): {gen}\nStrata serve greedy: {strata[:len(gen)]}\n"
              f"first divergence at token {k}")
        print(f"C {name} {'PASS' if ok else 'FAIL'}: argmax agreement {s['argmax_agree']:.4f} (>= {TH_GOLDEN})")
    return ok_all


def part_d(ref: RefModel, tt, names, chunk: int) -> bool:
    """TT vs RefModel teacher-forced over golden/<name>/full.ids (long context), chunk by chunk."""
    ok_all = True
    for name in names:
        ids = read_ids(GOLDEN / name / "full.ids")
        ref.reset()
        tt.reset()
        n = len(ids) - 1
        strict, agree, kl, top10 = [], [], [], []
        t_ref = t_tt = 0.0
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            t0 = time.time()
            lr = ref.forward(ids[i:j])
            t1 = time.time()
            lt, _ = tt.forward(ids[i:j])
            t_tt += time.time() - t1
            t_ref += t1 - t0
            lr64, lt64 = log_softmax64(lr), log_softmax64(lt)
            ar, at = greedy(lr), greedy(lt)
            gap = lr64[np.arange(len(ar)), ar] - lr64[np.arange(len(at)), at]
            strict += (ar == at).tolist()
            agree += (gap <= NEAR_TIE).tolist()
            kl += (np.exp(lr64) * (lr64 - lt64)).sum(-1).tolist()
            top10 += [len(set(np.argsort(-lr[t])[:10]) & set(np.argsort(-lt[t])[:10])) / 10 for t in range(len(lr))]
        kl = np.array(kl)
        a = float(np.mean(agree))
        ok = a >= TH_AGREE and kl.mean() <= TH_KL
        ok_all &= ok
        print(f"\n== D. {name}: TT vs RefModel over {n} teacher-forced positions (chunks of {chunk}; ref {t_ref:.0f} s, "
              f"tt {t_tt:.0f} s)")
        print(f"argmax strict {np.mean(strict):.4f} near-tie-tolerant {a:.4f} top10 {np.mean(top10):.4f} KL mean "
              f"{kl.mean():.4g} p50 {np.median(kl):.4g} p99 {np.quantile(kl, 0.99):.4g} max {kl.max():.4g}")
        for b0 in range(0, n, 512):
            s = slice(b0, min(b0 + 512, n))
            print(f"  positions {b0:>4}-{min(b0 + 512, n) - 1:>4}: strict {np.mean(strict[s]):.4f} KL mean "
                  f"{kl[s].mean():.4g} max {kl[s].max():.4g}")
        print(f"D {name} {'PASS' if ok else 'FAIL'}: near-tie-tolerant agreement {a:.4f} (>= {TH_AGREE}), KL mean "
              f"{kl.mean():.4g} (<= {TH_KL})")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--prompt", default="chat")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--golden", default="chat", help="comma list of golden prompts for part C ('' = skip)")
    ap.add_argument("--golden-gen", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--long", default="", help="comma list of golden prompts for part D (TT vs RefModel, long)")
    ap.add_argument("--skip", default="", help="comma list of parts to skip (a,b,c,d)")
    a = ap.parse_args()
    from fnx.tt.model import DEFAULT_PLAN, TtModel, moe_mode, open_mesh
    import ttnn

    w = Weights()
    n_layers = a.layers or available_layers(w.model)
    print(f"layers: {n_layers} (on disk: {available_layers(w.model)}); PLE table complete: "
          f"{w.model.available('per_layer_token_embd.weight')}")
    experts = CpuExperts(w.model, n_layers, a.threads)
    prompt = read_ids(GOLDEN / "prompts" / f"{a.prompt}.ids")
    skip = set(a.skip.split(","))
    # The reference computes the same expert weights as the TT path under test, so parts A/B/D measure device
    # numerics: Strata's CPU kernels for FNX_MOE=cpu, the plan's exact device values (float math) for FNX_MOE=device.
    # Part C compares against Strata itself either way.
    mode = moe_mode()
    if mode == "device":
        from fnx.quant.store import Plan
        from fnx.ref.moe import PlanExperts
        plan_name = os.environ.get("FNX_MOE_PLAN", DEFAULT_PLAN)
        print(f"MoE: device, plan {plan_name}; RefModel experts = the plan's snapped device weights in float32")
        ref = RefModel(w, PlanExperts(Plan(plan_name)), n_layers)
    else:
        print("MoE: cpu (Strata's CPU kernels on both sides)")
        ref = RefModel(w, experts, n_layers)
    mesh = open_mesh()
    try:
        tt = TtModel(mesh, w, experts, n_layers)
        ok = True
        if "a" not in skip:
            ok &= part_a(ref, tt, prompt)["ok"]
        if "b" not in skip:
            ok &= part_b(ref, tt, prompt, a.gen)["ok"]
        if "c" not in skip and a.golden:
            if n_layers < N_LAYERS:
                print(f"\n== C. skipped: truncated model ({n_layers} layers)")
            else:
                ok &= part_c(tt, a.golden.split(","), a.golden_gen, a.chunk)
        if "d" not in skip and a.long:
            ok &= part_d(ref, tt, a.long.split(","), a.chunk)
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"\ntest_model {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
