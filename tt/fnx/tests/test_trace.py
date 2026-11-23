"""Traced decode (fnx.tt.trace, TtModel.enable_trace) against the eager path on the P150: bitwise.

Run on the device:  tools/devrun.sh test-trace -m fnx.tests.test_trace [plain] [spec] [sparse] [bench]
                    [--golden chat,code] [--gen 24] [--layers N] [--bench-golden chat,code,doc] [--bench-gen 64]
Every part runs its eager pass first (no trace exists yet), then the model captures its traces once
(enable_trace(Ts=(1, 4))) and the same calls run again traced; the prompts are read eagerly both times (after the
capture: the eager programs compiled then must not disturb the parked traces).
plain   per golden prompt: eager prompt chunks, then --gen greedy T=1 steps (read_logits=True): every step's logits and
        ids, and afterwards every non-positional state tensor (GDN S + conv history, PLE history, hash window,
        position), every QSA layer's K/V cells [0, pos), host indexer state (raw keys, pooled keys, norms) and device
        P2 rows of the completed blocks.
spec    per golden prompt: serve_engine.Engine with the MTP drafter (verify windows of 4 rows + commit(n_keep)), greedy
        and seeded sampling (logits read back): every window's logits / ids, the tokens, the final state.
sparse  beyond the dense limit (golden doc + code ids, prompt to 2040): 4-row windows to 2047 (dense), the window
        2048..2051 across the limit (eager under tracing), T=1 steps, then 4-row windows with wrong drafts and
        commit(1..4) (sparse traced), teacher-forced: logits, ids and state.
bench   Engine plain and MTP, greedy, --bench-golden prompts, eager vs traced: tokens identical, decode tok/s, the
        traced step's host time breakdown, and the device time of each segment kind (replayed alone).
Pass: everything bitwise / identical; bench also requires traced faster than eager on every prompt.
The MoE is TtModel's (FNX_MOE=device | cpu); the captured segments per forward are printed after the capture.
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from fnx.config import EOS_IDS, N_LAYERS, is_qsa
from fnx.cpu_experts import CpuExperts
from fnx.ref.model import GOLDEN, read_ids
from fnx.weights import Weights

SAMPLED = " temperature=0.7 top_p=0.95 top_k=20 seed=42"


def golden(name: str) -> list[int]:
    """The golden prompt's token ids (tokenizer-only: the same file serves every variant)."""
    return read_ids(GOLDEN / "prompts" / f"{name}.ids")


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(np.ascontiguousarray(a, np.float32).view(np.uint32),
                              np.ascontiguousarray(b, np.float32).view(np.uint32))
    return np.array_equal(a, b)


def prefill(tt, prompt: list[int], chunk: int, upto: int | None = None) -> None:
    tt.reset()
    n = len(prompt) - 1 if upto is None else upto
    for i in range(0, n, chunk):
        tt.forward(prompt[i:min(i + chunk, n)], head=False)


def state(tt) -> dict:
    """Non-positional state + positional QSA state up to the position (after the deferred indexer work)."""
    import ttnn
    if tt._trace is not None:
        tt._trace.flush()
    out = {f"s{i}": ttnn.to_torch(t).float().numpy() for i, t in
           enumerate(t for m in tt._state_modules() for t in m.state_tensors())}
    out["prev"], out["pos"] = np.array(tt.ple.prev), np.array([tt.pos])
    n = tt.pos
    for L in range(tt.n_layers):
        if not is_qsa(L):
            continue
        q = tt.layers[L][1]
        for nm, t in (("K", q.K), ("V", q.V)):
            out[f"{nm}{L}"] = ttnn.to_torch(t).float().numpy().reshape(2, q.cells, -1)[:, :n]
        nb = n // 4
        out[f"kr{L}"], out[f"pooled{L}"], out[f"pnorm{L}"] = q.kr[:n].copy(), q.pooled[:nb].copy(), q.pnorm[:nb].copy()
        out[f"P2{L}"] = ttnn.to_torch(q.P2).float().numpy().reshape(q.blocks, -1)[:nb]
    return out


def diff_state(a: dict, b: dict) -> list[str]:
    return [k for k in a if not same(a[k], b[k])]


class Rec:
    """A TtModel proxy recording every forward's (logits, ids) for the comparison."""

    def __init__(self, m):
        self.__dict__["m"], self.__dict__["out"] = m, []

    def __getattr__(self, k):
        return getattr(self.m, k)

    def forward(self, *a, **k):
        r = self.m.forward(*a, **k)
        if k.get("head", True):
            self.out.append((None if r[0] is None else r[0].copy(), r[1].copy()))
        return r


def run_engine(engine, ids: list[int], gen: int, keys: str = "") -> list[int]:
    from fnx.serve_engine import parse_gen
    toks: list[int] = []

    def emit(s: str) -> None:
        if s.startswith("T "):
            toks.append(int(s[2:]))
        elif s.startswith("ERR"):
            raise RuntimeError(s)

    engine.emit = emit
    _, sp, _ = parse_gen(f"GEN {gen}{keys} 1")
    engine.generate(ids, gen, sp, threading.Event())
    return toks


def moe_info(tt) -> str:
    """The model's MoE (FNX_MOE) and, for the all-device MoE, its expert bytes in device DRAM / the host pool."""
    if tt.moe != "device":
        return "MoE cpu (routed experts on the host CPU)"
    st = [lay[3].store for lay in tt.layers]
    return (f"MoE device: experts {sum(s.nbytes for s in st) / 1e9:.2f} GB device DRAM + "
            f"{sum(s.host_bytes for s in st) / 1e9:.2f} GB host pool "
            f"({(tt.host_pool.nbytes if tt.host_pool else 0) / 1e9:.2f} GB pinned)")


# ---- parts: each returns the record of one pass (eager before the traces exist, traced after)
def pass_plain(tt, names, gen: int, chunk: int) -> dict:
    rec = {}
    for name in names:
        prompt = golden(name)
        prefill(tt, prompt, chunk)
        steps, tok = [], prompt[-1]
        t0 = time.perf_counter()
        for _ in range(gen):
            lg, ids = tt.forward([tok], read_logits=True)
            steps.append((lg, ids))
            tok = int(ids[0])
        ms = 1e3 * (time.perf_counter() - t0) / gen
        rec[name] = dict(steps=steps, state=state(tt), ms=ms)
    return rec


def pass_spec(tt, drafter, names, gen: int, chunk: int) -> dict:
    from fnx.serve_engine import Engine
    rec = {}
    for name in names:
        prompt = golden(name)
        out = {}
        for keys in ("", SAMPLED):
            r = Rec(tt)
            eng = Engine(r, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS, drafter=drafter)
            toks = run_engine(eng, prompt, gen, keys)
            out[keys] = dict(toks=toks, calls=r.out, stats=dict(eng.stats), state=state(tt))
        rec[name] = out
    return rec


def sparse_ids() -> list[int]:
    return read_ids(GOLDEN / "doc" / "full.ids") + read_ids(GOLDEN / "code" / "full.ids")


def pass_sparse(tt, chunk: int) -> dict:
    S = sparse_ids()
    prefill(tt, S, chunk, upto=2040)
    calls = []

    def call(win, n_keep=None):
        lg, ids = tt.forward(win, read_logits=True, verify=n_keep is not None)
        calls.append((len(win), tt.pos - len(win), lg, ids))
        if n_keep is not None:
            tt.commit(n_keep)

    p = 2040
    while p < 2048:                                  # dense 4-row windows
        call(S[p:p + 4], 4)
        p += 4
    call(S[p:p + 4], 4)                              # 2048..2051: across the dense limit
    p += 4
    for _ in range(6):                               # sparse T=1
        call([S[p]])
        p += 1
    for n_keep in (1, 2, 3, 4, 2):                   # sparse windows, wrong drafts after n_keep rows
        win = [S[p + j] if j < n_keep else (S[p + j] + 17) % 150000 for j in range(4)]
        call(win, n_keep)
        p += n_keep
    return dict(calls=calls, state=state(tt))


def pass_conv(tt, drafter, chunk: int, gen: int, ctx: int = 4096) -> dict:
    """serve_engine.Engine with conversation reuse (as test_serve P3-P5): a request, a follow-up whose new suffix has
    row counts no earlier call had (checkpoint split: never-seen prompt chunk shapes after the traces exist), the
    first request again (from the turn checkpoint); then the doc prompt and its follow-up, which crosses the dense
    attention limit (2050) in its prompt and decodes at sparse positions."""
    from fnx.serve_engine import Engine
    follow = read_ids(GOLDEN / "prompts" / "chat_follow.ids")
    eng = Engine(tt, None, ctx=32768, chunk=chunk, prompt_cache=6, eos=EOS_IDS, drafter=drafter)
    rec, decode_recaptures = [], 0
    for name in ("chat", "doc") if ctx >= 2304 else ("chat",):
        prompt = golden(name)
        toks = []
        for ids in (prompt, None, prompt, prompt):     # the 4th: only shapes seen before (steady state)
            ids = prompt + toks[0] + follow if ids is None else ids
            tr = tt._trace
            e0 = len(tr.events) if tr is not None else 0
            toks.append(run_engine(eng, ids, gen))
            st = eng.stats
            ev = tr.events[e0:] if tr is not None else []
            bad = [e for e in ev if e[0] != "after a prompt read"]   # after new prompt chunk shapes: allowed
            decode_recaptures += len(bad)
            tps = 1e3 * st["tokens"] / st["decode_ms"]
            ttft = st["prompt_ms"]
            print(f"conv {name} request {len(toks)}: {len(ids)}-token prompt read in {ttft:.0f} ms, {st['tokens']} "
                  f"tokens at {tps:.2f} tok/s; recaptures: "
                  f"{[(w, n, round(s, 2)) for w, n, s in ev] or 'none'}"
                  f"{f'  (DURING DECODE: {bad})' if bad else ''}", flush=True)
        rec.append((name, *toks, len(eng.live)))
    return dict(rec=rec, state=state(tt), decode_recaptures=decode_recaptures)


def cmp_conv(a: dict, b: dict) -> bool:
    """a eager, b traced: equal tokens and state, and no recapture during a traced decode (only right after a
    prompt read whose chunk row counts were new)."""
    ok = a["rec"] == b["rec"]
    ds = diff_state(a["state"], b["state"])
    ok &= not ds and b["decode_recaptures"] == 0
    print(f"conv: tokens of every request equal {a['rec'] == b['rec']}; state differing {ds or 'none'}; "
          f"recaptures during decode {b['decode_recaptures']}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def pass_bench(tt, drafter, names, gen: int, chunk: int) -> dict:
    from fnx.serve_engine import Engine
    plain = Engine(tt, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS)
    spec = Engine(tt, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS, drafter=drafter)
    rec = {}
    for name in names:
        prompt = golden(name)
        r = {}
        for label, eng in (("plain", plain), ("mtp", spec)):
            if tt._trace is not None:
                tt._trace.prof = {}
            toks = run_engine(eng, prompt, gen)
            st = dict(eng.stats)
            r[label] = dict(toks=toks, tps=1e3 * st["tokens"] / st["decode_ms"], ms=st["decode_ms"] / st["tokens"],
                            rounds=st["rounds"], prof=tt._trace.prof if tt._trace is not None else None)
        rec[name] = r
    if tt._trace is not None:
        tt._trace.prof = None
    return rec


def segment_times(tt, T: int, span) -> dict:
    """Device ms of each segment kind of one sequence, every segment replayed alone (20x, then a sync)."""
    import ttnn
    seq = tt._trace.seqs[(T, span)]
    acc: dict = {}
    for (_, _, kind), tid in zip(seq.segs, seq.tids):
        ttnn.synchronize_device(tt.mesh)
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(tt.mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(tt.mesh)
        ms = 1e3 * (time.perf_counter() - t0) / 20
        n, s = acc.get(kind, (0, 0.0))
        acc[kind] = (n + 1, s + ms)
    if seq.verify:
        for n_keep, tid in seq.commit_tids.items():
            ttnn.synchronize_device(tt.mesh)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.execute_trace(tt.mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(tt.mesh)
            acc[f"commit{n_keep}"] = (1, 1e3 * (time.perf_counter() - t0) / 5)
    return acc


# ---- comparisons
def cmp_plain(A: dict, B: dict) -> bool:
    ok_all = True
    for name in A:
        a, b = A[name], B[name]
        bad = [i for i, (x, y) in enumerate(zip(a["steps"], b["steps"])) if not (same(x[0], y[0]) and same(x[1], y[1]))]
        ds = diff_state(a["state"], b["state"])
        ok = not bad and not ds and len(a["steps"]) == len(b["steps"])
        ok_all &= ok
        toks = [int(s[1][0]) for s in b["steps"]]
        print(f"plain {name}: {len(b['steps'])} T=1 steps, logits+ids bitwise: {not bad}"
              f"{f' (first differing step {bad[0]})' if bad else ''}; state tensors differing: {ds or 'none'} of "
              f"{len(a['state'])}; eager {a['ms']:.1f} -> traced {b['ms']:.1f} ms/token; tokens {toks}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok_all


def cmp_calls(ca, cb) -> int:
    """First differing call index (-1: all equal)."""
    if len(ca) != len(cb):
        return min(len(ca), len(cb))
    for i, (x, y) in enumerate(zip(ca, cb)):
        if not all(same(u, v) for u, v in zip(x, y)):
            return i
    return -1


def cmp_spec(A: dict, B: dict) -> bool:
    ok_all = True
    for name in A:
        for keys in A[name]:
            a, b = A[name][keys], B[name][keys]
            first = cmp_calls(a["calls"], b["calls"])
            ds = diff_state(a["state"], b["state"])
            ok = a["toks"] == b["toks"] and first < 0 and not ds
            ok_all &= ok
            st = b["stats"]
            print(f"spec {name}{keys or ' greedy'}: {len(b['toks'])} tokens equal {a['toks'] == b['toks']}, "
                  f"{len(b['calls'])} forwards ({st['rounds']} windows, {st['accepted']}/{st['drafted']} accepted) "
                  f"logits/ids bitwise {first < 0}{f' (first differing forward {first})' if first >= 0 else ''}; "
                  f"state differing: {ds or 'none'}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok_all


def cmp_sparse(a: dict, b: dict) -> bool:
    first = cmp_calls(a["calls"], b["calls"])
    ds = diff_state(a["state"], b["state"])
    ok = first < 0 and not ds
    calls = b["calls"]
    where = f" (first differing call {first}: T={calls[first][0]} at {calls[first][1]})" if first >= 0 else ""
    print(f"sparse: {len(calls)} calls (positions {calls[0][1]}..{int(b['state']['pos'][0]) - 1}) logits/ids "
          f"bitwise {first < 0}{where}; state differing: {ds or 'none'}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def cmp_bench(A: dict, B: dict, segs: dict) -> bool:
    ok_all = True
    print("\n| prompt | mode | eager tok/s | traced tok/s | speedup | eager ms/token | traced ms/token | tokens equal |"
          "\n|---|---|---|---|---|---|---|---|")
    for name in A:
        for mode in ("plain", "mtp"):
            a, b = A[name][mode], B[name][mode]
            eq = a["toks"] == b["toks"]
            ok = eq and b["tps"] > a["tps"]
            ok_all &= ok
            print(f"| {name} | {mode} | {a['tps']:.2f} | {b['tps']:.2f} | {b['tps'] / a['tps']:.2f}x | {a['ms']:.1f} | "
                  f"{b['ms']:.1f} | {eq} |", flush=True)
    print("\ntraced host time per forward (ms): inputs = embedding/PLE rows/QSA inputs; launch = execute_trace calls; "
          "readback = blocked on the device segment + transfer; route = routing + CPU-expert start; cpu_wait = blocked "
          "on the CPU experts after the device part; cpu = the CPU experts' own time (overlapped); indexer = deferred "
          "QSA host indexer (overlapped with segment 0); final = ids/logits read")
    keys = ("inputs", "launch", "readback", "route", "upload", "cpu_wait", "cpu", "indexer", "final", "total")
    print("| prompt | mode | forwards | " + " | ".join(keys) + " |\n|---|---|---|" + "---|" * len(keys))
    for name in B:
        for mode in ("plain", "mtp"):
            p = B[name][mode]["prof"]
            if not p:
                continue
            n = max(p.get("steps", 0), 1)
            print(f"| {name} | {mode} | {n} | " + " | ".join(f"{1e3 * p.get(k, 0.0) / n:.2f}" for k in keys) + " |")
    for (T, span), acc in segs.items():
        tot = sum(s for k, (n, s) in acc.items() if not k.startswith("commit"))
        parts = ", ".join(f"{k} {n} x {s / n:.3f} = {s:.2f}" for k, (n, s) in acc.items())
        print(f"device time, T={T} span {span}: {tot:.2f} ms per forward ({parts})")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["plain", "spec", "sparse", "conv", "bench"])
    ap.add_argument("--golden", default="chat,code")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--bench-golden", default="chat,code,doc")
    ap.add_argument("--bench-gen", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096, help="model / drafter context (cells)")
    a = ap.parse_args()
    import ttnn

    from fnx.serve_engine import make_drafter, warm_up
    from fnx.tt.model import TtModel, open_mesh
    names = a.golden.split(",")
    bnames = a.bench_golden.split(",")
    w = Weights()
    experts = CpuExperts(w.model, a.layers, a.threads)
    mesh = open_mesh()
    ok = True
    try:
        t0 = time.time()
        need_dr = any(p in a.parts for p in ("spec", "conv", "bench")) and a.layers == N_LAYERS
        from fnx.tt.mtp import reserve_items
        tt = TtModel(mesh, w, experts, n_layers=a.layers, max_ctx=a.ctx, verbose=False,
                     host_reserve=reserve_items() if need_dr else 0)
        print(f"TtModel ({a.layers} layers) built in {time.time() - t0:.0f} s: {moe_info(tt)}", flush=True)
        drafter = make_drafter(mesh, w, a.ctx, 3, model=tt) if need_dr else None
        t0 = time.time()
        warm_up(tt, drafter, a.chunk)
        print(f"model + drafter ready, eager warm-up {time.time() - t0:.0f} s", flush=True)
        runs = []
        if "plain" in a.parts:
            runs.append(("plain", lambda: pass_plain(tt, names, a.gen, a.chunk)))
        if "spec" in a.parts and drafter is not None:
            runs.append(("spec", lambda: pass_spec(tt, drafter, names, a.gen, a.chunk)))
        if "sparse" in a.parts:
            runs.append(("sparse", lambda: pass_sparse(tt, a.chunk)))
        if "conv" in a.parts and drafter is not None:
            # traced FIRST, right after the serve warm-up (as fnx.serve_engine does): its prompt chunks compile
            # shapes no call had before the capture; then the same requests eagerly (traces released)
            tt.enable_trace(Ts=(1, 4), drafter=drafter.mtp)
            t0 = time.time()
            conv_tr = pass_conv(tt, drafter, a.chunk, 64, a.ctx)
            print(f"traced conv pass {time.time() - t0:.0f} s", flush=True)
            tt.disable_trace()
            ok &= cmp_conv(pass_conv(tt, drafter, a.chunk, 64, a.ctx), conv_tr)
        if "bench" in a.parts and drafter is not None:
            runs.append(("bench", lambda: pass_bench(tt, drafter, bnames, a.bench_gen, a.chunk)))
        eager = {}
        for nm, f in runs:
            t0 = time.time()
            eager[nm] = f()
            print(f"eager {nm} pass {time.time() - t0:.0f} s", flush=True)
        tt.reset()
        tt.enable_trace(Ts=(1, 4), drafter=drafter.mtp if drafter is not None else None)
        print("captured segments per forward: " + ", ".join(
            f"T={T} {'sparse' if span is None else f'span {span}'}: {len(sq.tids)}"
            for (T, span), sq in tt._trace.seqs.items()), flush=True)
        traced, segs = {}, {}
        for nm, f in runs:
            t0 = time.time()
            traced[nm] = f()
            print(f"traced {nm} pass {time.time() - t0:.0f} s", flush=True)
        if "bench" in traced:
            for key in ((1, 512), (1, 2304), (4, 512)):
                if key in tt._trace.seqs:
                    segs[key] = segment_times(tt, *key)
            tt.reset()
        cmp = dict(plain=cmp_plain, spec=cmp_spec, sparse=cmp_sparse)
        for nm, _ in runs:
            if nm == "bench":
                ok &= cmp_bench(eager[nm], traced[nm], segs)
            else:
                ok &= cmp[nm](eager[nm], traced[nm])
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"\ntest_trace {' '.join(a.parts)} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
