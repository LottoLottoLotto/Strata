"""Quality gate: Flash-Next on the P150 (any TtModel configuration) vs the published model in float32.

    tools/devrun.sh eval-cpu  -m fnx.eval.run --config cpu  [--max-minutes 10] [--corpus chat,code] [--records ..]
                                                            [--gsm 50] [--no-tf] [--env KEY=VAL ..]
    tools/devrun.sh eval-bfp8 -m fnx.eval.run --config bfp8 --gsm 50       (repeat until "nothing to run")
    python -m fnx.eval.run --report cpu,bfp8        (host venv, cwd tt/, no device: the stored results, plus the
                                                    reference's own noise floor)

Configurations (environment set before TtModel is built; --env KEY=VAL adds or overrides, and a new NAME with --env
defines a new configuration, e.g. a device-expert build):
    cpu    every routed expert on the host CPU through Strata's kernels (fnx.cpu_experts; FNX_MOE=cpu)
    device-gptq  every routed expert on the device (FNX_MOE=device, fnx.tt.moe_device): GPTQ-BFP4 plan
           bfp4-tiered-2.2e+10 (7957 experts in device DRAM, the rest in the pinned host pool read over PCIe; same
           numerics as the 2.4e+10 default: placement only)
    bfp8   FNX_MOE=cpu, FNX_EXPERT_PLACEMENT = tools/expert_placement.py's default placement (4555 experts resident as bfloat8_b,
           the rest on the CPU)

Teacher forcing (fnx.eval.data's evalset, reference = fnx.eval.torchref's cache): each sequence from a fresh state
(TtModel.reset), eager TtModel.forward over 128-row chunks (the serve engine's prompt chunk) with head + logits for
the chunks holding scored positions (head=False for the others), scored by fnx.eval.score.  Per record the
per-position metrics go to FNX_EVAL/runs/NAME/tf/<id>.npz; a run skips records already there and stops starting new
ones after --max-minutes, so a gate can be split over several short device runs (the device is shared).
GSM8K (--gsm N): the first N prompts of gsm8k.jsonl, greedy, max 512 new tokens, through fnx.serve_engine.Engine
exactly as served (MTP verify windows, traced decode: bitwise the plain greedy tokens), after the teacher forcing;
answers to runs/NAME/gsm8k.jsonl.  The reference answers come from Strata's CUDA engine on the RTX 3090
(fnx.eval.strata_ref), not from the float reference (a token-by-token float32 decode is too slow).
Every run ends with runs/NAME/results.json + results.md over everything stored for NAME.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .. import paths
from . import data, score

CONFIGS = {
    "cpu": {"FNX_MOE": "cpu", "FNX_EXPERT_PLACEMENT": ""},
    "device-gptq": {"FNX_MOE": "device", "FNX_MOE_PLAN": "bfp4-tiered-2.2e+10"},
    "bfp8": {"FNX_MOE": "cpu", "FNX_EXPERT_PLACEMENT": str(paths.CACHE / "placement" / "coder-bfloat8_b-4555.json")},
}
RUNS = data.EVAL / "runs"
CHUNK = 128
MAX_CTX = 2048
GSM_MAX_NEW = 512
STRATA_GSM = data.EVAL / "gsm8k-strata.jsonl"


def log(msg: str) -> None:
    print(f"fnx.eval: {msg}", flush=True)


def config_env(name: str, extra: list[str]) -> dict:
    env = dict(CONFIGS.get(name, {}))
    if name not in CONFIGS and not extra:
        raise SystemExit(f"unknown config {name!r} (known: {', '.join(CONFIGS)}); define one with --env KEY=VAL")
    for kv in extra:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ------------------------------------------------------------------------------------------------ device phases
def teacher_force(model, recs, out: Path, deadline: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for r in recs:
        p = out / f"{r['id']}.npz"
        if p.exists():
            continue
        if time.time() > deadline:
            log("time budget reached: stopping before " + r["id"])
            return
        toks = np.asarray(r["token_ids"], np.int64)
        n = len(toks)
        need = np.zeros(n, bool)
        need[score.targets(r) - 1] = True
        ref = score.load_ref(r["id"])
        model.reset()
        t0 = time.time()
        parts = []
        for c0 in range(0, n, CHUNK):
            c1 = min(c0 + CHUNK, n)
            pos = np.nonzero(need[c0:c1])[0] + c0
            if len(pos) == 0:
                model.forward(toks[c0:c1], head=False)
                continue
            logits, _ = model.forward(toks[c0:c1])
            parts.append(score.score_rows(logits[pos - c0], pos, toks, ref))
        m = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
        dt = time.time() - t0
        np.savez(p, seconds=dt, load=os.getloadavg()[0], **m)
        log(f"{r['id']}: {n} tokens in {dt:.1f} s ({n / dt:.1f} tok/s), {len(m['nll'])} scored: "
            f"dNLL {m['nll'].mean() - m['nll_ref'].mean():+.4f}, top-1 {m['top1'].mean():.4f}, KL {m['kl'].mean():.4f}")


def gsm8k(mesh, w, model, n: int, out: Path, deadline: float, spec: int) -> None:
    from ..config import EOS_IDS
    from ..serve_engine import Engine, Sampling, make_drafter, warm_up
    probs = data.load("gsm8k")[:n]
    done = {r["idx"] for r in read_jsonl(out)}
    todo = [p for p in probs if p["idx"] not in done]
    if not todo or time.time() > deadline:
        return
    tok = data.tokenizer()
    t0 = time.time()
    drafter = make_drafter(mesh, w, MAX_CTX, spec - 1, model=model) if spec > 1 else None
    warm_up(model, drafter, CHUNK)
    model.enable_trace(Ts=(1, spec) if spec > 1 else (1,), verbose=False,
                       drafter=drafter.mtp if drafter is not None else None)
    log(f"gsm8k: drafter + warm-up + trace in {time.time() - t0:.0f} s")
    got: list[int] = []
    emit = lambda s: got.append(int(s[2:])) if s.startswith("T ") else None
    eng = Engine(model, emit, ctx=MAX_CTX, chunk=CHUNK, prompt_cache=0, eos=EOS_IDS, drafter=drafter)
    for p in todo:
        if time.time() > deadline:
            log(f"time budget reached: gsm8k stops before problem {p['idx']}")
            break
        got.clear()
        t1 = time.time()
        eng.generate(p["prompt_ids"], GSM_MAX_NEW, Sampling(), threading.Event())
        dt = time.time() - t1
        stop = "eos" if got and got[-1] in EOS_IDS else "length"
        text = tok.decode([t for t in got if t not in EOS_IDS])
        pred = data.extract_answer(text)
        rec = {"idx": p["idx"], "gold": p["gold"], "pred": pred, "correct": pred == p["gold"], "n_new": len(got),
               "stop": stop, "seconds": round(dt, 2), "load": round(os.getloadavg()[0], 1), "ids": got, "text": text}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"gsm8k {p['idx']}: pred {pred} gold {p['gold']} {'OK' if rec['correct'] else 'WRONG'}, "
            f"{len(got)} tokens in {dt:.1f} s ({len(got) / dt:.1f} tok/s)")
    model.disable_trace()


def device_run(a, env: dict) -> None:
    os.environ.update(env)
    import ttnn

    from ..config import N_LAYERS
    from ..cpu_experts import CpuExperts
    from ..tt.model import TtModel, open_mesh
    from ..weights import Weights

    deadline = time.time() + 60 * a.max_minutes
    run = RUNS / a.config
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"config": a.config, "env": env}, indent=1))
    recs = data.load()
    if a.corpus:
        recs = [r for r in recs if r["corpus"] in a.corpus.split(",")]
    if a.records:
        recs = [r for r in recs if r["id"] in a.records.split(",")]
    todo_tf = [] if a.no_tf else [r for r in recs if not (run / "tf" / f"{r['id']}.npz").exists()]
    todo_gsm = a.gsm and len({r["idx"] for r in read_jsonl(run / "gsm8k.jsonl")} & set(range(a.gsm))) < a.gsm
    if not todo_tf and not todo_gsm:
        log("nothing to run")
        return
    t0 = time.time()
    w = Weights()
    experts = CpuExperts(w.model, N_LAYERS, a.threads)
    mesh = open_mesh()
    try:
        from ..tt.mtp import reserve_items
        model = TtModel(mesh, w, experts, max_ctx=MAX_CTX, verbose=False,
                        host_reserve=reserve_items() if todo_gsm and a.spec > 1 else 0)   # the drafter's host items
        log(f"config {a.config} {env}: TtModel built in {time.time() - t0:.0f} s")
        if todo_tf:
            teacher_force(model, todo_tf, run / "tf", deadline)
        if todo_gsm:
            gsm8k(mesh, w, model, a.gsm, run / "gsm8k.jsonl", deadline, a.spec)
    finally:
        ttnn.close_mesh_device(mesh)


# ------------------------------------------------------------------------------------------------ results
def gsm_summary(rows: list[dict], ref: list[dict]) -> dict:
    if not rows:
        return {}
    rf = {r["idx"]: r for r in ref}
    both = [r for r in rows if r["idx"] in rf]
    out = {"n": len(rows), "acc": float(np.mean([r["correct"] for r in rows])),
           "truncated": int(sum(r["stop"] != "eos" for r in rows))}
    if both:
        same_tok = [r["ids"] == rf[r["idx"]]["ids"] for r in both]
        div = []
        for r in both:
            a, b = r["ids"], rf[r["idx"]]["ids"]
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            div.append(k)
        fin = [r for r in both if r["stop"] == "eos" and rf[r["idx"]]["stop"] == "eos"]
        out.update({"ref_n": len(both), "ref_acc": float(np.mean([rf[r["idx"]]["correct"] for r in both])),
                    "ref_truncated": int(sum(rf[r["idx"]]["stop"] != "eos" for r in both)),
                    "acc_on_ref_set": float(np.mean([r["correct"] for r in both])),
                    "answer_agree": float(np.mean([r["pred"] == rf[r["idx"]]["pred"] for r in both])),
                    "both_finished": len(fin),
                    "answer_agree_finished": round(float(np.mean([r["pred"] == rf[r["idx"]]["pred"] for r in fin])), 3)
                    if fin else "n/a",
                    "identical_text": float(np.mean(same_tok)), "first_divergence_median": float(np.median(div))})
    return out


def results(name: str) -> dict:
    run = RUNS / name
    per = {c: [] for c in score.CORPORA}
    secs, toks, loads = 0.0, 0, []
    for r in data.load():
        p = run / "tf" / f"{r['id']}.npz"
        if p.exists():
            with np.load(p) as z:
                m = {k: z[k] for k in z.files}
            per[r["corpus"]].append(m)
            secs += float(m["seconds"])
            toks += len(r["token_ids"])
            if "load" in m:
                loads.append(float(m["load"]))
    cfg = score.load_json(run / "config.json") if (run / "config.json").exists() else {}
    res = {"config": name, "env": cfg.get("env"), "tf": score.corpus_table(per),
           "tf_tok_per_s": round(toks / secs, 2) if secs else None,
           "tf_host_load": [round(min(loads), 1), round(max(loads), 1)] if loads else None,
           "gsm8k": gsm_summary(read_jsonl(run / "gsm8k.jsonl"), read_jsonl(STRATA_GSM)),
           "reference": score.load_json(data.EVAL / "ref" / "meta.json")}
    (run / "results.json").write_text(json.dumps(res, indent=1))
    md = [f"### {name}  ({cfg.get('env')})", "", score.fmt_table(res["tf"])]
    g = res["gsm8k"]
    if g:
        md += ["", f"GSM8K: {g['n']} problems, TT accuracy {g['acc']:.3f} ({g['truncated']} hit 512 tokens)" +
               (f"; Strata 3090 reference {g['ref_acc']:.3f} on the same {g['ref_n']} ({g['ref_truncated']} hit 512); "
                f"answers agree {g['answer_agree']:.3f} overall, {g['answer_agree_finished']} on the "
                f"{g['both_finished']} both finished; token-identical generations {g['identical_text']:.3f}, median "
                f"first divergence at token {g['first_divergence_median']:.0f}" if "ref_n" in g else "")]
    if res["tf_tok_per_s"]:
        md += ["", f"teacher-forcing throughput {res['tf_tok_per_s']} tok/s (eager, 128-row chunks; host 1-min load "
                   f"average {res['tf_host_load'] or 'not recorded'}; timings are only comparable on a quiet host)"]
    (run / "results.md").write_text("\n".join(md) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VAL environment for the configuration")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--records", default="")
    ap.add_argument("--no-tf", action="store_true")
    ap.add_argument("--gsm", type=int, default=0, help="GSM8K problems (the first N)")
    ap.add_argument("--spec", type=int, default=4, help="GSM8K decode: verify-window rows (1 = plain decode)")
    ap.add_argument("--max-minutes", type=float, default=10.0,
                    help="start no new record / problem after this (the device is shared: keep runs short)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--report", default="", help="comma list of configs: print their tables (no device)")
    a = ap.parse_args(argv)
    if a.report:
        for name in a.report.split(","):
            results(name)
            print((RUNS / name / "results.md").read_text())
        nf = data.EVAL / "ref" / "noise.json"
        if nf.exists():
            n = score.load_json(nf)
            print(f"### float32 reference vs itself (torchref noise: row chunks {n['chunk']} instead of 4096)\n\n"
                  + score.fmt_table(n["table"]))
        return 0
    if not a.config:
        ap.error("--config NAME (or --report)")
    env = config_env(a.config, a.env)
    device_run(a, env)
    res = results(a.config)
    print((RUNS / a.config / "results.md").read_text())
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
