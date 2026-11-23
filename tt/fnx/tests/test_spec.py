"""Speculative decoding on the P150: verify windows (TtModel.forward(verify=True) + commit) and the MTP drafter in
fnx.serve_engine, against plain T=1 decode.

Run on the device:  tools/devrun.sh test-spec -m fnx.tests.test_spec [qsa] [inv] [spec] [--golden chat,code,doc]
                    [--gen 64] [--spec 4] [--min-p 0]
qsa   One Qsa layer (fnx.tests.test_qsa inputs): a 4-row call vs 4 single-row calls, bitwise bo, at dense positions,
      across the dense limit and at sparse (indexer) positions.
inv   Invariance (before relying on it): after the golden prompt, the window [last prompt token, 3 golden tokens]
      as ONE T=4 call vs four T=1 calls: the residual after every layer, the PLE output, the head input, the logits
      and the device argmax, bitwise.  Rollback: windows whose drafts are wrong after the first n_keep - 1 rows,
      commit(n_keep) (GdnFused.rollback, PLE commit), for n_keep = 1..3, against the kept rows alone in one
      call: every GDN layer's S and conv history, the PLE history and hash window, the position, and the logits of
      a following 4-row window (QSA's rejected cells rewritten), bitwise.
spec  fnx.serve_engine.Engine with the drafter (verify windows of --spec rows) vs without, greedy, --gen tokens per
      golden prompt, conversation reuse off (every request reads its prompt): identical tokens (and equal to
      Strata's golden serve tokens where plain TT equals them); acceptance rate, tokens per window, decode tok/s.
Pass: everything bitwise / identical as stated; spec also requires MTP decode faster than plain on every prompt.
The MoE is TtModel's (FNX_MOE=device | cpu); --layers N (inv only) builds the first N layers.
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


def golden(name: str) -> tuple[list[int], list[int]]:
    return read_ids(GOLDEN / "prompts" / f"{name}.ids"), read_ids(GOLDEN / name / "serve_gen.ids")


def prefill(tt, prompt: list[int], chunk: int) -> None:
    tt.reset()
    n = len(prompt) - 1
    for i in range(0, n, chunk):
        tt.forward(prompt[i:min(i + chunk, n)], head=False)


def bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def same(a, b) -> bool:
    return np.array_equal(bits(a), bits(b))


def state(tt) -> list:
    """Every non-positional state tensor (GDN S + conv history per layer, PLE history) + hash window + position."""
    import ttnn
    out = [ttnn.to_torch(t).float().numpy() for m in tt._state_modules() for t in m.state_tensors()]
    return out + [np.array(tt.ple.prev, np.float32), np.array([tt.pos], np.float32)]


def part_qsa(mesh, w, L: int = 3, windows=(60, 1000, 2049, 2100, 2160), T: int = 4) -> bool:
    """One Qsa layer on realistic inputs (fnx.tests.test_qsa's): a T-row call at p0 vs T single-row calls at
    p0..p0+T-1 (re-run over the same cells: the state is positional), dense and sparse positions, bitwise bo."""
    import ttnn
    from fnx.tests.test_qsa import sparse_inputs
    from fnx.tt.hc import to_dev, to_np
    from fnx.tt.qsa import Qsa
    mixed = sparse_inputs(w, "sparse", L)[:windows[-1] + T].astype(np.float32)
    q = Qsa(mesh, w, L, 4096)

    def call(p, n):
        x = to_dev(mesh, mixed[p:p + n].reshape(1, 1, n, -1))
        bo = q.forward(x, list(range(p, p + n)))
        out = to_np(bo).reshape(-1, bo.shape[-1])[:n]
        ttnn.deallocate(x)
        ttnn.deallocate(bo)
        return out

    ok, at = True, 0
    for p0 in windows:
        for c in range(at, p0, 128):
            call(c, min(128, p0 - c))
        a = call(p0, T)
        b = np.concatenate([call(p0 + t, 1) for t in range(T)])
        res = [same(a[t], b[t]) for t in range(T)]
        ok &= all(res)
        print(f"qsa L{L}: {T}-row call at {p0}..{p0 + T - 1} ({'dense' if p0 + T - 1 <= 2050 else 'sparse' if p0 > 2050 else 'dense+sparse'}) "
              f"vs single rows: bitwise per row {res}, max |diff| {np.abs(a - b).max():.3e}  "
              f"{'PASS' if all(res) else 'FAIL'}", flush=True)
        at = p0 + T
    return ok


def part_inv(tt, names, chunk: int) -> bool:
    ok_all = True
    kinds = ["qsa" if is_qsa(L) else "gdn" for L in range(tt.n_layers)]
    for name in names:
        prompt, gold = golden(name)
        t0 = time.time()
        prefill(tt, prompt, chunk)
        s0 = tt.snapshot()
        win = [prompt[-1]] + gold[:3]
        dA: dict = {}
        lA, iA = tt.forward(win, dump=dA, verify=True)
        tt.commit(len(win))
        tt.restore(s0)
        dB = [{} for _ in win]
        lB, iB = [], []
        for t, x in enumerate(win):
            lt, it = tt.forward([x], dump=dB[t])
            lB.append(lt[0])
            iB.append(int(it[0]))
        RB = np.concatenate([d["R"] for d in dB], axis=1)             # [layers + 1, 4, 4, 2560]
        rows = [int((bits(dA["R"][L]) != bits(RB[L])).reshape(len(win), -1).any(1).sum()) for L in range(len(RB))]
        bad = [L for L in range(len(rows)) if rows[L]]
        first = (f"first differing ladder slot {bad[0]} (after layer {bad[0] - 1}, "
                 f"{kinds[bad[0] - 1] if bad[0] else 'embedding'}), max |diff| "
                 f"{np.abs(dA['R'][bad[0]] - RB[bad[0]]).max():.3e}") if bad else "every layer bitwise"
        ple = same(dA["R_ple"], np.concatenate([d["R_ple"] for d in dB]))
        head = same(dA["head_in"], np.concatenate([d["head_in"] for d in dB]))
        lg = same(lA, np.stack(lB))
        ids = [int(i) for i in iA] == iB
        ok = not bad and ple and head and lg and ids
        print(f"inv {name}: T=4 window vs 4 x T=1 at positions {len(prompt) - 1}..{len(prompt) + 2}: {first}; "
              f"rows differing per slot {rows if bad else 0}; PLE out {ple}, head in {head}, logits {lg}, argmax "
              f"{ids} ({[int(i) for i in iA]} vs {iB})  {'PASS' if ok else 'FAIL'}", flush=True)
        ok_all &= ok
        # rollback: window [x, d1, d2, d3] with d_j wrong for j >= n_keep, commit(n_keep) == the kept rows alone
        nxt_rows = 4
        for n_keep in (1, 2, 3):
            good = win[:n_keep]
            wrong = [t if j < n_keep else (t + 1) % 1000 + 100 for j, t in enumerate(win)]
            cont = (gold + [198] * 8)[n_keep - 1:n_keep - 1 + nxt_rows]
            tt.restore(s0)
            _, iw = tt.forward(wrong, read_logits=False, verify=True)
            tt.commit(n_keep)
            stA = state(tt)
            lA2, _ = tt.forward(cont)
            tt.restore(s0)
            tt.forward(good, head=False)
            stB = state(tt)
            lB2, _ = tt.forward(cont)
            diff = [i for i, (a, b) in enumerate(zip(stA, stB)) if not same(a, b)]
            ok = not diff and same(lA2, lB2)
            ok_all &= ok
            print(f"rollback {name} n_keep={n_keep}: window {wrong} -> commit({n_keep}) vs {good} alone: "
                  f"state tensors differing {diff if diff else 'none'} of {len(stA)} (pos {int(stA[-1][0])}); next "
                  f"{nxt_rows}-row window logits bitwise {same(lA2, lB2)}  {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"   ({time.time() - t0:.0f} s)", flush=True)
    return ok_all


SAMPLED = (" temperature=0.7 top_p=0.95 top_k=20 seed=42", " penalty_present=1.5 penalty_last_n=16",
           " temperature=1.0 top_k=40 min_p=0.05 penalty_repeat=1.3 penalty_last_n=32 seed=7")


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


def part_spec(model, drafter, names, gen: int, chunk: int) -> bool:
    from fnx.serve_engine import Engine, warm_up
    t0 = time.time()
    warm_up(model, drafter, chunk)
    print(f"warm-up {time.time() - t0:.1f} s; windows of {1 + drafter.k} rows, draft min_p {drafter.min_p}", flush=True)
    plain = Engine(model, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS)
    spec = Engine(model, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS, drafter=drafter)
    ok_all, rows = True, []
    for name in names:
        prompt, gold = golden(name)
        off = run_engine(plain, prompt, gen)
        s_off = plain.stats
        on = run_engine(spec, prompt, gen)
        s_on = spec.stats
        tps_off, tps_on = (1e3 * s["tokens"] / s["decode_ms"] for s in (s_off, s_on))
        k = next((i for i, (a, b) in enumerate(zip(off, gold)) if a != b), min(len(off), len(gold)))
        equal = on == off
        gold_ok = on[:k] == gold[:k]
        fast = tps_on > tps_off
        ok = equal and gold_ok and fast
        ok_all &= ok
        acc = s_on["accepted"] / max(s_on["drafted"], 1)
        rows.append((name, tps_off, tps_on, acc, s_on["tokens"] / max(s_on["rounds"], 1)))
        print(f"spec {name}: {len(on)} tokens, MTP on == off: {equal}"
              f"{'' if equal else f' (first divergence {next((i for i, (a, b) in enumerate(zip(on, off)) if a != b), min(len(on), len(off)))})'}"
              f"; plain == Strata golden for the first {k}/{len(gold)} tokens, MTP equal there: {gold_ok}; "
              f"acceptance {s_on['accepted']}/{s_on['drafted']} = {acc:.3f}, {s_on['rounds']} windows "
              f"({s_on['tokens'] / max(s_on['rounds'], 1):.2f} tokens/window); decode {tps_off:.2f} -> {tps_on:.2f} "
              f"tok/s ({tps_on / tps_off:.2f}x; {s_off['decode_ms'] / len(off):.1f} -> "
              f"{s_on['decode_ms'] / len(on):.1f} ms/token)  {'PASS' if ok else 'FAIL'}", flush=True)
        if not equal:
            print(f"   off: {off}\n   on:  {on}", flush=True)
    prompt, _ = golden(names[0])
    for keys in SAMPLED:                        # host chain on read-back logits: row t with counter p + t
        off, on = run_engine(plain, prompt, 32, keys), run_engine(spec, prompt, 32, keys)
        ok_all &= on == off
        print(f"sampled {names[0]}{keys}: MTP on == off: {on == off} ({len(on)} tokens, acceptance "
              f"{spec.stats['accepted']}/{spec.stats['drafted']})  {'PASS' if on == off else 'FAIL'}", flush=True)
        if on != off:
            print(f"   off: {off}\n   on:  {on}", flush=True)
    print("\n| prompt | plain tok/s | MTP tok/s | speedup | acceptance | tokens/window |\n|---|---|---|---|---|---|")
    for name, a, b, acc, tpw in rows:
        print(f"| {name} | {a:.2f} | {b:.2f} | {b / a:.2f}x | {acc:.3f} | {tpw:.2f} |")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["qsa", "inv", "spec"])
    ap.add_argument("--golden", default="chat,code,doc")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=N_LAYERS, help="model layers (inv; spec needs all)")
    a = ap.parse_args()
    import ttnn

    from fnx.serve_engine import make_drafter
    from fnx.tt.model import TtModel, open_mesh
    names = a.golden.split(",")
    w = Weights()
    experts = CpuExperts(w.model, N_LAYERS, a.threads)
    mesh = open_mesh()
    ok = True
    try:
        if "qsa" in a.parts:
            ok &= part_qsa(mesh, w)
        t0 = time.time()
        from fnx.tt.mtp import reserve_items
        model = TtModel(mesh, w, experts, n_layers=a.layers, max_ctx=4096, verbose=False,
                        host_reserve=reserve_items() if "spec" in a.parts else 0)
        print(f"TtModel ({a.layers} layers, MoE {model.moe}) built in {time.time() - t0:.0f} s", flush=True)
        if "inv" in a.parts:
            ok &= part_inv(model, names, a.chunk)
        if "spec" in a.parts and a.layers == N_LAYERS:
            drafter = make_drafter(mesh, w, 4096, a.spec - 1, a.min_p, model)
            ok &= part_spec(model, drafter, names, a.gen, a.chunk)
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"\ntest_spec {' '.join(a.parts)} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
