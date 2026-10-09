"""MTP draft layer and verify-window rollback checks.

    python -m fnx.tests.test_mtp strata [chat code doc]          # host (venv): numpy drafter vs Strata's drafts
    python -m fnx.tests.test_mtp ple                              # host: PLE rollback, numpy reference
    tools/devrun.sh mtp-tt -m fnx.tests.test_mtp tt [chat ...]    # device: PLE rollback (numpy + TT), TT drafter
    tools/devrun.sh mtp-bench -m fnx.tests.test_mtp bench [chat ...]   # device: traced serving loop, plain vs MTP
    tools/devrun.sh mtp-prof -m fnx.tests.test_mtp profile             # device: drafter routing counts (calibration)

strata: Strata's drafts come from /var/tmp/strata-tt/mtp/strata_drafts.sh (RTX 3090, gdb-traced MtpDrafter::draft on
the golden prompts with the continuation forced to the golden serve output; inputs = Strata's own final residuals of
every position, STRATA_PREFILL_DUMP_R_ALL, BF16).  The numpy drafter replays the same rounds: prompt K/V over cells
[0, n_prompt-1), then per traced round the catch-up (rows 0..a) and the 3-step chain.  Reported: per-draft agreement
with Strata's drafts, agreement of draft 0 (whose inputs equal Strata's up to its residual rounding), the acceptance
rate (drafts matching the continuation, chain stopped at the first miss) of both, per-round acceptance agreement.
PASS: draft-0 agreement >= 90% and |acceptance difference| <= 0.10 per prompt (Strata's decode residuals differ from the
prompt path's BF16 dump we feed, and draft-layer near-ties flip; the drafts are output-neutral by construction).
ple: a 4-row verify window [x, d1, d2, d3] whose drafts beyond the a accepted ones are wrong tokens, commit(a + 1), then
a 4-row continuation, against the accepted tokens alone + the same continuation: equal hash window, history and
continuation (TT: bitwise; numpy: <= 1e-6, BLAS blocking differs between a 1-row and a 4-row call), and a negative
control without the rollback that must differ.
tt: the TT drafter (fnx.tt.mtp.Mtp, device-tensor inputs as TtModel hands them over, eager rounds) and the numpy
drafter on the device's BFP4 expert values (MtpRef expert_fmt="bfp4") on the same traced rounds (thresholds at
TT_D0_AGREE below).
bench: fnx.serve_engine's traced decode loop (TtModel + drafter, warm_up, enable_trace) per golden prompt, greedy,
BENCH_GEN tokens: plain and MTP tokens must be equal; reports decode tok/s, acceptance, tokens per window and the
draft round's host-timed ms.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fnx.ref import mtp as M

GOLDEN = Path(os.environ.get("FNX_GOLDEN", "/var/tmp/strata-tt/golden"))
STRATA = Path(os.environ.get("FNX_MTP_GOLDEN", "/var/tmp/strata-tt/mtp/golden"))
MIN_D0_AGREE, MAX_ACC_DIFF = 0.90, 0.10


def read_ids(path) -> list[int]:
    return [int(x) for x in Path(path).read_text().replace(",", " ").split()]


def accepted(drafts, seq, start: int) -> int:
    """How many leading drafts equal the sequence from index `start` (the verify rule)."""
    a = 0
    while a < len(drafts) and start + a < len(seq) and drafts[a] == seq[start + a]:
        a += 1
    return a


def golden_case(name: str) -> dict:
    d = STRATA / name
    prompt = read_ids(GOLDEN / "prompts" / f"{name}.ids")
    seq = read_ids(GOLDEN / name / "full.ids")
    R = M.read_final_R(d / "final_R.bf16")
    rounds = M.read_strata_drafts(d / "drafts.txt")
    assert R.shape[0] == len(seq) - 1, (R.shape, len(seq))
    for r in rounds:     # the traced outv are the forced continuation (beyond its end: the engine's own picks)
        k = min(r["T"], len(seq) - r["p"] - 1)
        assert r["outv"][:k] == seq[r["p"] + 1:r["p"] + 1 + k], (name, r)
    return {"prompt": prompt, "seq": seq, "R": R, "rounds": rounds}


def replay(drafter, case: dict, on_round=None) -> list[dict]:
    """Run `drafter` (fnx.ref.mtp.MtpRef or fnx.tt.mtp.Mtp: .reset/.prefill/.draft) over Strata's traced rounds."""
    seq, R, n_p = case["seq"], case["R"], len(case["prompt"])
    drafter.reset()
    drafter.prefill(R[:n_p - 1], seq[1:n_p], 0)
    out = []
    for r in case["rounds"]:
        p, a = r["p"], r["a"]
        t0 = time.perf_counter()
        drafts, probs = drafter.draft(R[p:p + a + 1], r["outv"], p, a)
        res = dict(r, ours=drafts, ours_probs=probs, ms=(time.perf_counter() - t0) * 1e3)
        out.append(res)
        if on_round:
            on_round(res)
    return out


def score(case: dict, res: list[dict]) -> dict:
    seq = case["seq"]
    n = len(res)
    same = np.array([[x == y for x, y in zip(r["ours"], r["drafts"])] for r in res])
    acc_s = np.array([accepted(r["drafts"], seq, r["p"] + r["a"] + 2) for r in res])
    acc_o = np.array([accepted(r["ours"], seq, r["p"] + r["a"] + 2) for r in res])
    nd = np.array([min(3, len(seq) - (r["p"] + r["a"] + 2)) for r in res])     # drafts verifiable in the sequence
    return {"rounds": n, "draft_agree": round(float(same.mean()), 4),
            "draft_agree_by_step": [round(float(x), 4) for x in same.mean(0)],
            "accept_rate_strata": round(float(acc_s.sum() / nd.sum()), 4),
            "accept_rate_ours": round(float(acc_o.sum() / nd.sum()), 4),
            "round_accept_agree": round(float((acc_s == acc_o).mean()), 4),
            "ms_per_round": round(float(np.mean([r["ms"] for r in res])), 1)}


def strata_checks(names) -> bool:
    from fnx.cpu_experts import CpuExperts
    from fnx.weights import Weights
    w = Weights()
    mw = M.MtpWeights()
    cx = CpuExperts(w.model, 0, 16, capacity=M.MTP_LAYER + 1)
    M.register_experts(cx, mw)
    ok = True
    for name in names:
        case = golden_case(name)
        ref = M.MtpRef(w, mw, cx, max_cells=len(case["seq"]) + 8)
        t0 = time.time()
        res = replay(ref, case)
        s = score(case, res)
        good = s["draft_agree_by_step"][0] >= MIN_D0_AGREE and \
            abs(s["accept_rate_strata"] - s["accept_rate_ours"]) <= MAX_ACC_DIFF
        ok &= good
        print(f"strata {name}: {s} ({time.time() - t0:.0f} s) {'PASS' if good else 'FAIL'}", flush=True)
        for r in res:
            if r["ours"] != r["drafts"]:
                print(f"   p={r['p']} a={r['a']}: strata {r['drafts']} {r['probs']}  ours {r['ours']} "
                      f"{[round(x, 6) for x in r['ours_probs']]}")
    return ok


# ------------------------------------------------------------------------------------------------- PLE rollback
PREFIX, WIN, NEXT = 12, 4, 4          # trunk rows: prefix call, verify window, continuation


def ple_case(a: int, tr: dict):
    """Window rows 12..15 = [x, d1, d2, d3] where drafts beyond the a accepted ones are wrong tokens (rows from the
    end of the trunk, with their residuals); the continuation starts at the token after the a accepted drafts."""
    tok, R = tr["tokens"].astype(np.int64), tr["R_ple_in"]
    true = list(range(PREFIX, PREFIX + a + 1))
    wrong = list(range(31 - (WIN - 1 - a) + 1, 32)) if a < WIN - 1 else []
    win = true + wrong
    nxt = list(range(PREFIX + a + 1, PREFIX + a + 1 + NEXT))
    return tok, R, list(range(PREFIX)), win, true, nxt


def ple_numpy(tr: dict, w) -> bool:
    from fnx.ref import ple as P
    Wt, table = P.ple_weights(w), P.PleTable(w.model)

    def call(st, idx, tok, R):
        e = table.gather(st.rows(tok[idx]))
        return P.ple_block(Wt, e, R[idx], st)

    ok = True
    for a in range(WIN):
        tok, R, pre, win, true, nxt = ple_case(a, tr)
        sA, sB, sC = P.PleState(), P.PleState(), P.PleState()
        for s in (sA, sB, sC):
            call(s, pre, tok, R)
        call(sA, win, tok, R)
        sA.commit(a + 1)                                   # window + rollback to the a accepted drafts
        outA = call(sA, nxt, tok, R)
        call(sB, true, tok, R)                             # the accepted tokens alone
        outB = call(sB, nxt, tok, R)
        call(sC, win, tok, R)                              # negative control: the window without the rollback
        outC = call(sC, nxt, tok, R)
        d_out = float(np.abs(outA - outB).max() / np.abs(outB).max())
        d_hist = float(np.abs(sA.hist - sB.hist).max() / max(np.abs(sB.hist).max(), 1e-30))
        ctrl = float(np.abs(outC - outB).max() / np.abs(outB).max())
        good = sA.prev == sB.prev and d_out <= 1e-6 and d_hist <= 1e-6 and (a == WIN - 1 or ctrl > 1e-4)
        ok &= good
        print(f"ple numpy a={a}: prev {sA.prev}=={sB.prev}, continuation max/max {d_out:.1e} "
              f"(bitwise {np.array_equal(outA, outB)}), hist {d_hist:.1e} (bitwise {np.array_equal(sA.hist, sB.hist)}); "
              f"without rollback {ctrl:.2e}  {'PASS' if good else 'FAIL'}")
    return ok


def ple_device(tr: dict, w, mesh) -> bool:
    from fnx.ref import ple as P
    from fnx.tt.hc import to_dev, to_np
    from fnx.tt.ple import Ple
    ple = Ple(mesh, w)
    Wt = P.ple_weights(w)

    def call(idx, tok, R):
        e = ple.upload_rows(tok[idx])
        out = ple.forward(e, to_dev(mesh, R[idx].reshape(1, 1, len(idx), -1)))
        return to_np(out).reshape(len(idx), -1)

    ok = True
    for a in range(WIN):
        tok, R, pre, win, true, nxt = ple_case(a, tr)
        ple.reset()
        call(pre, tok, R)
        call(win, tok, R)
        ple.commit(a + 1)
        outA, hA, pA = call(nxt, tok, R), ple.hist_host(), ple.prev
        ple.reset()
        call(pre, tok, R)
        call(true, tok, R)
        outB, hB, pB = call(nxt, tok, R), ple.hist_host(), ple.prev
        st = P.PleState()                                  # numpy reference of the accepted sequence
        for idx in (pre, true):
            P.ple_block(Wt, ple.table.gather(st.rows(tok[idx])), R[idx], st)
        ref = P.ple_block(Wt, ple.table.gather(st.rows(tok[nxt])), R[nxt], st).reshape(NEXT, -1)
        Rn = R[nxt].reshape(NEXT, -1)
        e_upd = float(np.linalg.norm((outA - Rn) - (ref - Rn)) / np.linalg.norm(ref - Rn))
        good = np.array_equal(outA, outB) and np.array_equal(hA, hB) and pA == pB == st.prev and e_upd < 1e-3
        ok &= good
        print(f"ple device a={a}: window+commit == accepted-only: continuation bitwise {np.array_equal(outA, outB)}, "
              f"hist bitwise {np.array_equal(hA, hB)}, prev {pA}; vs numpy update rel {e_upd:.2e} (< 1e-3) "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------- TT drafter
# The numpy drafter's own subset logits move by 1.7-2.5% rel L2 (mean over steps; max 10%) under Strata's activation
# rounding (q8_1 at the attention projections) and by 6-7% with a bfloat16 K/V cache (measured on the three golden
# prompts): its attention logits reach ~600, so any arithmetic below float32-exact moves them.  The TT module (bfloat16
# projections of the Q8_0 weights, float32 K/V, split-operand attention matmuls, the all-device MoE with BFP4 experts)
# is compared with the numpy drafter on the device's own expert values (MtpRef expert_fmt="bfp4"): it must stay within
# 2x the q8_1 figure on average, agree with numpy on >= 90% of draft 0 (identical inputs) and with Strata at least as
# often as numpy - 0.05.
TT_D0_AGREE, TT_LOGIT_REL_MEAN, TT_STRATA_SLACK = 0.90, 5e-2, 0.05


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def tt_checks(names, mesh, w) -> bool:
    import ttnn
    from fnx.cpu_experts import CpuExperts
    from fnx.tt.hc import to_dev
    from fnx.tt.mtp import Mtp, _rows
    mw = M.MtpWeights()
    cx = CpuExperts(w.model, 0, 16, capacity=M.MTP_LAYER + 1)          # today's reference: Strata's Q2_0 experts
    M.register_experts(cx, mw)
    t0 = time.time()
    tt = Mtp(mesh, w, mw, max_ctx=2048)
    print(f"Mtp built in {time.time() - t0:.0f} s; thresholds: draft-0 agreement with numpy (BFP4 experts) >= "
          f"{TT_D0_AGREE}, mean subset-logits rel L2 vs numpy (same inputs) <= {TT_LOGIT_REL_MEAN}, agreement with "
          f"Strata >= numpy's - {TT_STRATA_SLACK}", flush=True)
    ok = True
    for name in names:
        case = golden_case(name)
        seq, R, n_p = case["seq"], case["R"], len(case["prompt"])
        ref = M.MtpRef(w, mw, None, float_experts=True, max_cells=len(seq) + 8, expert_fmt="bfp4")
        ref_q = M.MtpRef(w, mw, cx, max_cells=len(seq) + 8)
        t1 = time.time()
        ref.prefill(R[:n_p - 1], seq[1:n_p], 0)
        t2 = time.time()
        ref_q.prefill(R[:n_p - 1], seq[1:n_p], 0)
        tt.reset()
        Rd = to_dev(mesh, R.reshape(1, 1, R.shape[0], -1))       # device inputs, as TtModel will hand them over
        Rp = _rows(Rd, 0, n_p - 1)
        tt.prefill(Rp, seq[1:n_p], 0)
        ttnn.deallocate(Rp)
        ttnn.synchronize_device(mesh)
        t3 = time.time()
        same, same_s, same_rs, same_q, lrel, ms, res = [], [], [], [], [], [], []
        for r in case["rounds"]:
            p, a = r["p"], r["a"]
            trace_r: list = []
            dr, pr = ref.draft(R[p:p + a + 1], r["outv"], p, a, trace=trace_r)
            dq, _ = ref_q.draft(R[p:p + a + 1], r["outv"], p, a)
            trace_t: list = []
            t4 = time.perf_counter()
            Rw = _rows(Rd, p, p + a + 1)
            dt, pt = tt.draft(Rw, r["outv"], p, a, trace=trace_t)
            ms.append((time.perf_counter() - t4) * 1e3)
            ttnn.deallocate(Rw)
            for j in range(len(dr)):
                same.append(dr[j] == dt[j])
                same_s.append(dt[j] == r["drafts"][j])
                same_rs.append(dr[j] == r["drafts"][j])
                same_q.append(dq[j] == dt[j])
                if j == 0 or dr[:j] == dt[:j]:                 # same chain inputs so far
                    lrel.append(rel(trace_t[j]["logits"], trace_r[j]["logits"]))
            res.append(dict(r, ours=dt, ours_probs=pt, ms=ms[-1]))
        ttnn.deallocate(Rd)
        s = score(case, res)
        d0 = float(np.mean(same[0::3]))
        good = d0 >= TT_D0_AGREE and np.mean(lrel) <= TT_LOGIT_REL_MEAN and \
            np.mean(same_s) >= np.mean(same_rs) - TT_STRATA_SLACK
        ok &= good
        print(f"tt {name}: drafts == numpy {np.mean(same):.4f} (draft 0: {d0:.4f}), == Strata {np.mean(same_s):.4f} "
              f"(numpy {np.mean(same_rs):.4f}), == numpy with today's Q2_0 CPU experts {np.mean(same_q):.4f}; "
              f"logits rel L2 vs numpy mean {np.mean(lrel):.2e} max {max(lrel):.2e}; acceptance tt "
              f"{s['accept_rate_ours']} strata {s['accept_rate_strata']}; prompt K/V {len(seq[1:n_p])} cells: numpy "
              f"{t2 - t1:.1f} s, tt {t3 - t2:.1f} s; tt ms/round {np.median(ms):.1f} (eager, 3 steps)  "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------- traced bench
BENCH_GEN, BENCH_CTX, BENCH_CHUNK = 64, 4096, 128


def bench(names, gen: int = BENCH_GEN, ctx: int = BENCH_CTX, chunk: int = BENCH_CHUNK) -> bool:
    """The serving decode loop as fnx.serve_engine runs it (TtModel FNX_MOE default, the drafter, warm_up, traced
    T = 1 / 4 + drafter rounds), greedy, per golden prompt: plain vs MTP decode (tokens must be equal), acceptance,
    tokens per window, draft ms per round (Drafter.draft host-timed: the tail row read-back + Mtp.draft), host load."""
    import ttnn
    from fnx.config import EOS_IDS, N_LAYERS
    from fnx.cpu_experts import CpuExperts
    from fnx.serve_engine import Engine, make_drafter, warm_up
    from fnx.tests.test_spec import golden as golden_gen, run_engine
    from fnx.tt.model import TtModel, open_mesh
    from fnx.tt.mtp import reserve_items
    from fnx.weights import Weights
    w = Weights()
    cx = CpuExperts(w.model, N_LAYERS, 16)
    mesh = open_mesh()
    ok, rows = True, []
    try:
        t0 = time.time()
        model = TtModel(mesh, w, cx, max_ctx=ctx, verbose=False, host_reserve=reserve_items())

        def dram_free() -> float:
            m = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return m.total_bytes_free_per_bank * m.num_banks / 1e9
        f0 = dram_free()
        dr = make_drafter(mesh, w, ctx, 3, model=model)
        print(f"bench: drafter DRAM {f0 - dram_free():.3f} GB (ctx {ctx}); {dram_free():.2f} GB free", flush=True)
        warm_up(model, dr, chunk)
        model.enable_trace(Ts=(1, 4), verbose=False, drafter=dr.mtp)
        stat = {"n": 0, "ms": 0.0}
        inner = dr.draft

        def timed(R, picks, p):
            t1 = time.perf_counter()
            out = inner(R, picks, p)
            stat["n"] += 1
            stat["ms"] += 1e3 * (time.perf_counter() - t1)
            return out
        dr.draft = timed
        print(f"bench: TtModel (MoE {model.moe}) + drafter, warm-up and captures in {time.time() - t0:.0f} s; "
              f"{gen} tokens per prompt, ctx {ctx}", flush=True)
        plain = Engine(model, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS)
        spec = Engine(model, None, ctx=32768, chunk=chunk, prompt_cache=0, eos=EOS_IDS, drafter=dr)
        for name in names:
            prompt, _ = golden_gen(name)
            off = run_engine(plain, prompt, gen)
            s_off = dict(plain.stats)
            r0, ms0 = stat["n"], stat["ms"]
            on = run_engine(spec, prompt, gen)
            s_on = dict(spec.stats)
            n_r = max(stat["n"] - r0, 1)
            tps_off, tps_on = (1e3 * s["tokens"] / s["decode_ms"] for s in (s_off, s_on))
            acc = s_on["accepted"] / max(s_on["drafted"], 1)
            good = on == off
            ok &= good
            rows.append((name, tps_off, tps_on, acc, s_on["tokens"] / max(s_on["rounds"], 1), (stat["ms"] - ms0) / n_r,
                         s_on["decode_ms"] / max(s_on["rounds"], 1), os.getloadavg()[0]))
            print(f"bench {name}: MTP tokens == plain {good}; acceptance {s_on['accepted']}/{s_on['drafted']} = "
                  f"{acc:.3f}; decode {tps_off:.2f} -> {tps_on:.2f} tok/s; draft {rows[-1][5]:.2f} ms/round of "
                  f"{rows[-1][6]:.1f} ms/window  {'PASS' if good else 'FAIL'}", flush=True)
        dt = getattr(dr.mtp._trace, "rounds", None)            # one-segment traced rounds: replay alone (20x)
        for T in (1, 4):
            rd = dt.get((T, dr.mtp._trace.spans[0])) if dt else None
            if rd is None or rd.tid is None:
                continue
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, rd.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"bench: traced round T={T} span {rd.span}: {(time.perf_counter() - t1) / 20 * 1e3:.2f} ms device "
                  f"(replayed alone; load {os.getloadavg()[0]:.1f})", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    print("\n| prompt | plain tok/s | MTP tok/s | acceptance | tokens/window | draft ms/round | ms/window | load |\n"
          "|---|---|---|---|---|---|---|---|")
    for name, a, b, acc, tpw, dms, wms, ld in rows:
        print(f"| {name} | {a:.2f} | {b:.2f} | {acc:.3f} | {tpw:.2f} | {dms:.2f} | {wms:.1f} | {ld:.1f} |")
    return ok


CALIB = Path("/var/tmp/strata-tt/quant/calib/seqs.json")      # fnx.quant.calib_data's calibration corpus
PROFILE_SEQS = {"chat": 3, "code": 3, "instruct": 1, "docs": 1}     # calibration sequences per corpus
PROFILE_LEN, PROFILE_CHUNK = 1024, 128


def profile() -> bool:
    """Routing counts of the drafter's MoE in draft rounds on calibration text (fnx.quant.calib_data, CALIB: disjoint
    from the eval set and the golden prompts), on the device: TtModel reads PROFILE_LEN tokens of each picked sequence
    (keep_R), the drafter's K/V follows each prompt chunk, then an eager 3-step round (catch-up row + the drafter's
    own chain, as in serving) at every 2nd position of the chunk, last positions first (a round's speculative cells
    then lie past every cell a later round reads), and the chunk's K/V is written again; every MoE call's top-10 is
    counted -> MTP_DIR/PROFILE (fnx.tt.mtp.host_experts: the FNX_MTP_RESIDENT hottest stay in device DRAM)."""
    import ttnn
    from fnx.config import N_LAYERS
    from fnx.cpu_experts import CpuExperts
    from fnx.tt.model import TtModel, open_mesh
    from fnx.tt.mtp import Mtp, _rows, reserve_items
    from fnx.weights import Weights
    w = Weights()
    mw = M.MtpWeights()
    cx = CpuExperts(w.model, N_LAYERS, 16)
    calib = json.loads(CALIB.read_text())["calib"]
    seqs = []
    for corpus, n in PROFILE_SEQS.items():
        seqs += [s for s in calib if s["corpus"] == corpus][:n]
    counts = np.zeros(M.MTP_EXPERTS, np.int64)
    mesh = open_mesh()
    try:
        model = TtModel(mesh, w, cx, max_ctx=PROFILE_LEN + PROFILE_CHUNK, verbose=False,
                        host_reserve=reserve_items(M.MTP_EXPERTS))
        mtp = Mtp(mesh, w, mw, max_ctx=PROFILE_LEN + PROFILE_CHUNK, resident=M.MTP_EXPERTS, pool=model.host_pool,
                  pool_first=model.host_reserve_first)
        moe = mtp.moe
        moe.keep_route = True
        inner = moe.forward

        def counted(x, T=None):
            y = inner(x, T)
            np.add.at(counts, ttnn.to_torch(moe.last_route).numpy().reshape(32, 32)[0, :M.N_USED], 1)
            return y
        moe.forward = counted
        for s in seqs:
            t0 = time.time()
            ids = [int(x) for x in s["token_ids"][:PROFILE_LEN + 1]]
            model.reset()
            mtp.reset()
            for c0 in range(0, PROFILE_LEN, PROFILE_CHUNK):
                c1 = c0 + PROFILE_CHUNK
                _, _, R = model.forward(ids[c0:c1], head=False, keep_R=True)
                mtp.prefill(R, ids[c0 + 1:c1 + 1], c0)
                for t in range(PROFILE_CHUNK - 1, -1, -2):
                    Rt = _rows(R, t, t + 1)
                    mtp.draft(Rt, [ids[c0 + t + 1]], c0 + t, 0)
                    ttnn.deallocate(Rt)
                mtp.prefill(R, ids[c0 + 1:c1 + 1], c0)
                model.free_R(R)
            print(f"profile {s['id']} ({s['corpus']}): {PROFILE_LEN // 2} rounds in {time.time() - t0:.0f} s",
                  flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    np.save(mw.dir / M.PROFILE, counts)
    o = np.sort(counts)[::-1].cumsum() / counts.sum()
    print(f"profile: {counts.sum()} selections, unused experts {(counts == 0).sum()}; share of the hottest "
          f"64 / 128 / 192 / 256 / 384: {o[63]:.3f} / {o[127]:.3f} / {o[191]:.3f} / {o[255]:.3f} / {o[383]:.3f} -> "
          f"{mw.dir / M.PROFILE}", flush=True)
    return True


def main(argv) -> int:
    cmd = argv[0] if argv else "strata"
    names = argv[1:] or ["chat", "code", "doc"]
    if cmd == "strata":
        ok = strata_checks(names)
    elif cmd == "ple":
        from fnx.tests.test_hc import load_trunk
        from fnx.weights import Weights
        ok = ple_numpy(load_trunk(32), Weights())
    elif cmd == "tt":
        import ttnn
        from fnx.tests.test_hc import load_trunk
        from fnx.tt.model import open_mesh
        from fnx.weights import Weights
        w = Weights()
        tr = load_trunk(32)
        ok = ple_numpy(tr, w)
        mesh = open_mesh()
        try:
            ok = ple_device(tr, w, mesh) and ok
            ok = tt_checks(names, mesh, w) and ok
        finally:
            ttnn.close_mesh_device(mesh)
    elif cmd == "bench":
        ok = bench(names)
    elif cmd == "profile":
        ok = profile()
    else:
        raise SystemExit(f"unknown command {cmd}")
    print("test_mtp", cmd, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
