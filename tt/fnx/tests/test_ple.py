"""PLE: hash vs Strata's oracle vectors, table rows vs the oracle probes, reference history semantics, and the TT block
vs the numpy reference on real blk.1 weights and the trunk's real post-layer-0 residual (see fnx/tests/test_hc.py).

    NODEV=1 tools/devrun.sh ple-host -m fnx.tests.test_ple --host     # host checks only
    tools/devrun.sh ple -m fnx.tests.test_ple                          # host checks + device
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

from fnx.ref import ple as P
from fnx.ref.hc import HC, N, D

ORACLE = Path("/var/tmp/strata-tt/Strata/src/kernels/ple_oracle_vectors.inc")
# device pass thresholds on the PLE update (R_new - R) and the history; floor = FPU fp32 accumulation, ~3e-4
TOL = {"upd_rel": 1e-3, "upd_max": 5e-3, "hist_rel": 1e-3}


def _arr(src: str, name: str, conv=int):
    """Flat values of a C array initializer `name[...]... = {...};` (nested braces allowed)."""
    body = re.search(r"%s(?:\[[^\]]*\])+\s*=\s*\{(.*?)\};" % re.escape(name), src, re.S).group(1)
    body = body.replace("{", " ").replace("}", " ").replace("\n", " ")
    return [conv(x.strip().rstrip("f")) for x in body.split(",") if x.strip()]


def host_checks() -> bool:
    ok = True
    src = ORACLE.read_text()
    n_cases = int(re.search(r"kHashCaseCount = (\d+)", src).group(1))
    for c in range(n_cases):
        toks, prev = _arr(src, f"case{c}_tokens"), _arr(src, f"case{c}_prev")
        want = np.array(_arr(src, f"case{c}_rows"), np.int64).reshape(-1, 16)
        got = np.array([P.ngram_rows_one(t, (prev[2 * i], prev[2 * i + 1])) for i, t in enumerate(toks)])
        good = np.array_equal(got, want)
        ok &= good
        print(f"hash oracle case {c} ({len(toks)} tokens): {'PASS' if good else 'FAIL'}")
    # sequence form: the window advances (prev[1] -> prev[0], tok -> prev[1]); case0's 2nd..4th rows are a real window
    rows, prev = P.ngram_rows([11, 22, 33], (-1, -1))
    seq = np.array([P.ngram_rows_one(11, (-1, -1)), P.ngram_rows_one(22, (-1, 11)), P.ngram_rows_one(33, (11, 22))])
    good = np.array_equal(rows, seq) and prev == (22, 33)
    ok &= good
    print(f"ngram_rows window advance: {'PASS' if good else 'FAIL'}")

    from fnx.weights import Weights
    w = Weights()
    table = P.PleTable(w.model)
    probes = _arr(src, "kProbeRows")
    heads = np.array(_arr(src, "kProbeHead", float), np.float32).reshape(-1, 8)
    tails = np.array(_arr(src, "kProbeTail", float), np.float32).reshape(-1, 8)
    sums = _arr(src, "kProbeSum", float)
    for i, r in enumerate(probes):
        if not table.row_available([r])[0]:
            print(f"table probe row {r}: not downloaded yet, skipped")
            continue
        v = table.gather(np.full((1, 16), r))[0, :160]
        good = np.allclose(v[:8], heads[i], rtol=1e-6, atol=1e-9) and np.allclose(v[-8:], tails[i], rtol=1e-6, atol=1e-9) \
            and abs(np.abs(v.astype(np.float64)).sum() - sums[i]) <= 1e-6 * max(sums[i], 1)
        ok &= good
        print(f"table probe row {r}: L1 {np.abs(v.astype(np.float64)).sum():.9f} (oracle {sums[i]:.9f}) {'PASS' if good else 'FAIL'}")
    import gguf.quants as gq
    from gguf.constants import GGMLQuantizationType
    rr = np.array([[0, 1, 12345, 9663979, 26558231, 56120240, 74755659, 7, 99, 1000, 123456, 2000000, 3, 5, 77, 18]])
    rr = rr[:, :16]
    avail = table.row_available(rr[0])
    e = table.gather(rr)
    ref = np.concatenate([np.asarray(gq.dequantize(table.raw[r * 90:(r + 1) * 90], GGMLQuantizationType.IQ4_NL), np.float32)
                          for r in rr[0]])
    good = np.array_equal(e[0][np.repeat(avail, 160)], ref[np.repeat(avail, 160)])
    ok &= good
    print(f"gather (head-slowest) == gguf-py IQ4_NL dequant on {int(avail.sum())} on-disk rows: {'PASS' if good else 'FAIL'}")

    # history semantics of the reference: one call of T rows == T sequential calls (taps t-9/t-6/t-3 inside the call)
    from fnx.tests.test_hc import load_trunk
    tr = load_trunk(32)
    Wt = P.ple_weights(w)
    s1, s2 = P.PleState(), P.PleState()
    R1 = P.ple_block(Wt, tr["e_ple"], tr["R_ple_in"], s1)
    R2 = np.concatenate([P.ple_block(Wt, tr["e_ple"][t:t + 1], tr["R_ple_in"][t:t + 1], s2) for t in range(32)])
    # equal up to BLAS summation order (a batched sgemm may sum in another order than T=1 rows)
    d = np.abs(R1 - R2).max() / np.abs(R1).max()
    dh = np.abs(s1.hist - s2.hist).max() / np.abs(s1.hist).max()
    good = d < 1e-5 and dh < 1e-5
    ok &= good
    print(f"ref ple_block: one T=32 call vs 32 T=1 calls: R max diff/max {d:.1e}, hist max diff/max {dh:.1e} (< 1e-5) "
          f"{'PASS' if good else 'FAIL'}")
    parts: dict = {}
    P.ple_block(Wt, tr["e_ple"], tr["R_ple_in"], P.PleState(), parts)
    print("ref magnitudes: rms R {:.3g}, rms gated {:.3g}, rms conv (pre-silu) {:.3g}, gate range [{:.3f}, {:.3f}]".format(
        np.sqrt((tr["R_ple_in"] ** 2).mean()), np.sqrt((parts["gated"] ** 2).mean()),
        np.sqrt((parts["conv"] ** 2).mean()), parts["gate"].min(), parts["gate"].max()))
    return bool(ok)


def err(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    d = np.abs(got - ref)
    return float(d.max()), float(d.max() / np.abs(ref).max()), float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def device_checks() -> bool:
    import ttnn
    from fnx.weights import Weights
    from fnx.tests.test_hc import load_trunk
    from fnx.tt.hc import to_dev, to_np
    from fnx.tt.ple import Ple
    w = Weights()
    tr = load_trunk(32)
    Wt = P.ple_weights(w)
    # a 64-token sequence (the 32 trunk rows twice), so one call also covers prefill-sized T and taps across calls
    e_ple, R = np.concatenate([tr["e_ple"]] * 2), np.concatenate([tr["R_ple_in"]] * 2)
    n = R.shape[0]
    print(f"inputs: trunk e_ple ({int(tr['ple_substituted'])}/{tr['ple_rows'].size} rows not yet downloaded were replaced "
          f"by on-disk rows), R after layer 0 (rms {np.sqrt((R ** 2).mean()):.3g})")
    print(f"thresholds: update (R_new-R) norm-rel < {TOL['upd_rel']}, max/max|upd| < {TOL['upd_max']}; hist norm-rel < {TOL['hist_rel']}")
    ok = True
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        for precise in (True, False):
            ple = Ple(mesh, w, precise=precise)
            for split in ([64], [1, 1, 1, 5, 24, 32], [3, 3, 3, 3, 20, 32], [12, 1, 19, 2, 30]):
                ple.reset()
                st = P.PleState()
                t0, outs = 0, []
                ms_list = []
                for T in split:
                    Rd = to_dev(mesh, R[t0:t0 + T].reshape(1, 1, T, D))
                    ed = to_dev(mesh, e_ple[t0:t0 + T].reshape(1, 1, T, N))
                    ttnn.synchronize_device(mesh)
                    t1 = time.perf_counter()
                    out = ple.forward(ed, Rd)
                    ttnn.synchronize_device(mesh)
                    ms_list.append((time.perf_counter() - t1) * 1e3)
                    outs.append(to_np(out).reshape(T, D))
                    t0 += T
                ref = P.ple_block(Wt, e_ple, R, st).reshape(n, D)
                got = np.concatenate(outs)
                Rf = R.reshape(n, D)
                eu = err(got - Rf, ref - Rf)
                eh = err(ple.hist_host(), st.hist)
                good = eu[2] < TOL["upd_rel"] and eu[1] < TOL["upd_max"] and eh[2] < TOL["hist_rel"]
                if precise:
                    ok &= good
                print(f"precise={precise!s:5s} calls {split}: update max {eu[0]:.3e} max/max {eu[1]:.3e} rel {eu[2]:.3e} | "
                      f"R_new max/max|R| {err(got, ref)[1]:.2e} | hist rel {eh[2]:.3e} | ms/call {np.round(ms_list, 3).tolist()}"
                      + ("  " + ("PASS" if good else "FAIL") if precise else ""), flush=True)
            # steady-state timing, T=1 and T=32
            for T in (1, 32):
                Rd = to_dev(mesh, R[:T].reshape(1, 1, T, D))
                ed = to_dev(mesh, e_ple[:T].reshape(1, 1, T, N))
                ttnn.deallocate(ple.forward(ed, Rd))
                ttnn.synchronize_device(mesh)
                t1 = time.perf_counter()
                for _ in range(10):
                    ttnn.deallocate(ple.forward(ed, Rd))
                ttnn.synchronize_device(mesh)
                print(f"precise={precise!s:5s} T={T:2d}: {(time.perf_counter() - t1) / 10 * 1e3:.3f} ms per forward (eager)")
        # upload_rows: hash + gather + upload on the host side of the TT module
        ple = Ple(mesh, w)
        toks = tr["tokens"]
        rows_ref, _ = P.ngram_rows(toks, (-1, -1))
        avail = ple.table.row_available(rows_ref)
        ed = ple.upload_rows(toks)
        got = to_np(ed).reshape(32, N)
        want = ple.table.gather(rows_ref)
        mask = np.repeat(avail, 160, axis=1)
        good = np.array_equal(got[mask], want[mask]) and ple.prev == (int(toks[-2]), int(toks[-1]))
        ok &= good
        print(f"Ple.upload_rows: device e_ple == host gather on {int(avail.sum())}/{avail.size} on-disk rows, prev advanced: "
              f"{'PASS' if good else 'FAIL'}")
    finally:
        ttnn.close_mesh_device(mesh)
    return bool(ok)


if __name__ == "__main__":
    ok = host_checks()
    if "--host" not in sys.argv:
        ok = device_checks() and ok
    print("test_ple", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
