"""HC read/write on the P150 vs the numpy reference, on real Coder weights and realistic residuals.

Inputs ("trunk"): the first T tokens of the golden code prompt run through layers 0..3 with the numpy references
(embedding, HC, GDN, QSA, MoE with float-dequantized routed experts, PLE at layer 1); the residual entering every HC
read is recorded, plus each write's bo/inj. Built once without the device and cached:
    NODEV=1 tools/devrun.sh mk -m fnx.tests.test_hc --make-inputs
Device test (eager per-call time, and traced device time for T in 1, 32; `--reads`, `--Ts` to narrow the run):
    tools/devrun.sh hc -m fnx.tests.test_hc
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from fnx.ref.hc import HC, N, hc_read, hc_weights, hc_write

INPUTS = Path(os.environ.get("FNX_HCPLE_INPUTS", "/var/tmp/strata-tt/cache/hcple"))
PROMPT = Path("/var/tmp/strata-tt/golden/prompts/code.ids")
READS = ["blk.0.hc_attn", "blk.0.hc_ffn", "blk.1.hc_attn", "blk.1.hc_ffn", "blk.3.hc_attn", "blk.3.hc_ffn", "output_hc"]
# Pass thresholds for the default (precise) path. The floor is the FPU's fp32 accumulation (~3e-4 relative on a
# 10240-deep signed projection, see fnx/tt/hc.py); a wrong stream / sign / scale shows up as >= 1e-2.
TOL = {"mixed_rel": 6e-4, "mixed_max": 4e-3, "inj_rel": 3e-4, "write_max": 1e-5}


def prompt_tokens(T: int) -> np.ndarray:
    ids = [int(x) for x in PROMPT.read_text().replace(",", " ").split()]
    return np.array(ids[:T], np.int64)


def ple_rows_on_disk(table, rows: np.ndarray) -> tuple[np.ndarray, int]:
    """Table rows not downloaded yet are replaced by a real on-disk row (row % VOCAB[0], inside head 0's fully present
    range), so e_ple has the table's real value distribution. Returns (rows, number replaced)."""
    from fnx.ref.ple import VOCAB
    ok = table.row_available(rows)
    out = rows.astype(np.int64).copy()
    out[~ok] = out[~ok] % VOCAB[0]
    assert table.row_available(out).all()
    return out, int((~ok).sum())


def make_trunk(T: int) -> dict:
    """Run layers 0..3 (numpy refs) on T prompt tokens; record each HC read's input and each write's bo/inj."""
    from fnx.weights import Weights
    from fnx.ref.embed import embed
    from fnx.ref.gdn import GdnState, gdn
    from fnx.ref.moe import moe
    from fnx.ref.ple import PleState, PleTable, ple_block
    from fnx.ref.qsa import QsaState, qsa
    w = Weights()
    tok = prompt_tokens(T)
    e = embed(w, tok)
    table, st = PleTable(w.model), PleState()
    rows, n_sub = ple_rows_on_disk(table, st.rows(tok))
    e_ple = table.gather(rows)
    R = np.ascontiguousarray(np.broadcast_to(e[:, None, :], (T, HC, N)), np.float32)
    out = {"tokens": tok, "e": e, "e_ple": e_ple, "ple_rows": rows, "ple_substituted": np.int64(n_sub)}
    cache: dict = {}
    for L in range(4):
        t0 = time.time()
        if L == 1:
            out["R_ple_in"] = R.copy()
            R = ple_block(w, e_ple, R, st)
        for half in ("attn", "ffn"):
            pre = f"blk.{L}.hc_{half}"
            out[f"R:{pre}"] = R.copy()
            mixed, inj = hc_read(R, *hc_weights(w, pre))
            if half == "attn":
                bo = qsa(w, L, mixed, np.arange(T), QsaState(max_cells=256)) if L == 3 else gdn(w, L, mixed, GdnState())
            else:
                bo = moe(w, L, mixed, float_experts=True, cache=cache)
            out[f"bo:{pre}"], out[f"inj:{pre}"] = bo, inj
            R = hc_write(R, bo, inj)
        print(f"trunk layer {L}: {time.time() - t0:.1f} s, rms R {np.sqrt((R ** 2).mean()):.4g}", flush=True)
        cache.clear()
    out["R:output_hc"] = R.copy()
    out["head_in"] = hc_read(R, *hc_weights(w, "output_hc"))[0]
    return out


def load_trunk(T: int = 32) -> dict:
    p = INPUTS / f"trunk_T{T}.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing: build it first (NODEV=1 tools/devrun.sh mk -m fnx.tests.test_hc --make-inputs)")
    return dict(np.load(p))


def err(got, ref) -> tuple[float, float, float]:
    """(max abs, max abs / max |ref|, ||got-ref|| / ||ref||)."""
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    d = np.abs(got - ref)
    return float(d.max()), float(d.max() / np.abs(ref).max()), float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def eager_ms(mesh, fn, n: int = 10) -> float:
    """Host-dispatched time per call (outputs freed)."""
    import ttnn
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        for t in fn():
            if t is not None:
                ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n * 1e3


def traced_ms(mesh, fn, n: int = 50) -> float:
    """Device time per call: one captured trace of fn(), replayed n times back to back."""
    import ttnn
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) / n * 1e3
    ttnn.release_trace(mesh, tid)
    for t in outs:
        if t is not None:
            ttnn.deallocate(t)
    return ms


def run_device(precise_list=(True, False), Ts=(64, 32, 5, 1), reads=READS, trace_Ts=(1, 32)) -> bool:
    import ttnn
    from fnx.weights import Weights
    from fnx.tt.hc import HcRead, hc_write as tt_hc_write, to_dev, to_np
    w = Weights()
    tr = load_trunk(32)
    # T=64 (prefill-sized) case: rows 32..63 are rows 0..31 reversed (HC rows are independent per token)
    tr = {k: (np.concatenate([v, v[::-1]]) if k.split(":")[0] in ("R", "bo", "inj") else v) for k, v in tr.items()}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    ok = True
    try:
        print(f"trunk: tokens {tr['tokens'][:6]}..., PLE rows substituted {int(tr['ple_substituted'])}/{tr['ple_rows'].size}")
        print(f"thresholds (precise=True): mixed norm-rel < {TOL['mixed_rel']}, mixed max-abs/max|ref| < {TOL['mixed_max']}, "
              f"inj norm-rel < {TOL['inj_rel']}, write max-abs/max|R| < {TOL['write_max']}")
        for pre in reads:
            g, Wd, Wu, Wi = hc_weights(w, pre)
            mods = {p: HcRead(mesh, w, pre, precise=p) for p in precise_list}
            for T in Ts:
                R = tr[f"R:{pre}"][:T]
                ref_m, ref_i = hc_read(R, g, Wd, Wu, Wi)
                Rd = to_dev(mesh, R.reshape(1, 1, T, HC * N))
                for p, mod in mods.items():
                    mixed, inj = mod.forward(Rd)
                    ms = eager_ms(mesh, lambda: mod.forward(Rd))
                    got_m = to_np(mixed).reshape(T, N)
                    em = err(got_m, ref_m)
                    line = f"{pre:14s} T={T:2d} precise={p!s:5s} eager {ms:6.3f} ms"
                    if p and T in trace_Ts:
                        line += f" traced {traced_ms(mesh, lambda: mod.forward(Rd)):6.3f} ms"
                    line += f"  mixed max {em[0]:.3e} max/ref {em[1]:.3e} rel {em[2]:.3e}"
                    good = em[2] < TOL["mixed_rel"] and em[1] < TOL["mixed_max"]
                    if ref_i is not None:
                        got_i = to_np(inj).reshape(T, -1)
                        ei = err(got_i[:, :HC], ref_i)
                        line += f" | inj max {ei[0]:.3e} rel {ei[2]:.3e} (cols 4.. max {np.abs(got_i[:, HC:]).max():.1e})"
                        good = good and ei[2] < TOL["inj_rel"] and np.abs(got_i[:, HC:]).max() == 0
                    if p:
                        ok = ok and good
                        line += "  PASS" if good else "  FAIL"
                    print(line, flush=True)
                    ttnn.deallocate(mixed)
                    if inj is not None:
                        ttnn.deallocate(inj)
                ttnn.deallocate(Rd)
            if pre != "output_hc":
                # write: real bo/inj of this half; bo as f32 and as bf16; out of place and in place
                for T in Ts:
                    R, bo, inj = tr[f"R:{pre}"][:T], tr[f"bo:{pre}"][:T], tr[f"inj:{pre}"][:T]
                    ref = hc_write(R, bo, inj).reshape(T, HC * N)
                    inj32 = np.zeros((T, 32), np.float32)
                    inj32[:, :HC] = inj
                    Rd, injd = to_dev(mesh, R.reshape(1, 1, T, HC * N)), to_dev(mesh, inj32.reshape(1, 1, T, 32))
                    for bdt in (ttnn.float32, ttnn.bfloat16):
                        bod = to_dev(mesh, bo.reshape(1, 1, T, N), bdt)
                        out = tt_hc_write(Rd, bod, injd)
                        ms = eager_ms(mesh, lambda: [tt_hc_write(Rd, bod, injd)])
                        timing = f"eager {ms:6.3f} ms"
                        if bdt == ttnn.float32 and T in trace_Ts:
                            timing += f" traced {traced_ms(mesh, lambda: [tt_hc_write(Rd, bod, injd)]):6.3f} ms"
                        if bdt == ttnn.float32:
                            e_ = err(to_np(out).reshape(T, -1), ref)
                            good = e_[0] / np.abs(ref).max() < TOL["write_max"]
                            Rin = to_dev(mesh, R.reshape(1, 1, T, HC * N))
                            got_in = tt_hc_write(Rin, bod, injd, inplace=True)
                            e_in = err(to_np(Rin).reshape(T, -1), ref)
                            good = good and got_in is Rin and e_in[0] == e_[0]
                            ttnn.deallocate(Rin)
                            ok = ok and good
                            print(f"{pre:14s} T={T:2d} hc_write bo f32  {timing}  max {e_[0]:.3e} max/|R| "
                                  f"{e_[0] / np.abs(ref).max():.3e} rel {e_[2]:.3e} (in place: max {e_in[0]:.3e})  "
                                  f"{'PASS' if good else 'FAIL'}", flush=True)
                        else:
                            import torch
                            bo16 = torch.from_numpy(bo).to(torch.bfloat16).float().numpy()
                            e_ = err(to_np(out).reshape(T, -1), hc_write(R, bo16, inj).reshape(T, -1))
                            print(f"{pre:14s} T={T:2d} hc_write bo bf16 {timing}  vs ref(bf16(bo)): max {e_[0]:.3e} "
                                  f"rel {e_[2]:.3e}", flush=True)
                        ttnn.deallocate(out)
                        ttnn.deallocate(bod)
                    ttnn.deallocate(Rd)
                    ttnn.deallocate(injd)
    finally:
        ttnn.close_mesh_device(mesh)
    print("test_hc", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-inputs", action="store_true")
    ap.add_argument("-T", type=int, default=32)
    ap.add_argument("--reads", default=",".join(READS), help="comma list of HC read prefixes to test")
    ap.add_argument("--Ts", default="64,32,5,1", help="comma list of row counts")
    a = ap.parse_args()
    if a.make_inputs:
        INPUTS.mkdir(parents=True, exist_ok=True)
        tr = make_trunk(a.T)
        np.savez(INPUTS / f"trunk_T{a.T}.npz", **tr)
        print("saved", INPUTS / f"trunk_T{a.T}.npz", {k: v.shape for k, v in tr.items()})
        sys.exit(0)
    sys.exit(0 if run_device(Ts=tuple(int(t) for t in a.Ts.split(",")), reads=a.reads.split(",")) else 1)
