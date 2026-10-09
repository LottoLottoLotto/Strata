"""TileArgmax (fnx.tt.tile_argmax) against the stock device argmax it replaces (`stock` below, the former
fnx.tt.head.Head.argmax: ttnn.to_layout(ROW_MAJOR) + ttnn.argmax), bitwise: ids, and with keep_rm the row-major
logits' bits (in element order).

    tools/devrun.sh targ -m fnx.tests.test_tile_argmax [--no-real] [--bench]

Inputs, at the LM head width (248320) and the MTP draft width (draft_vocab.bin rounded up to 32 columns), rows
1, 2, 3, 4, 8, 16, 17, 32: real logits (the head on the trunk's real head inputs; the draft width = their draft-vocab
columns + the -1e30 vmask added on the device, as fnx.tt.mtp does), normal, heavy integer ties, ties at worker / tile /
face boundaries, all-equal rows, signed zeros, -1e30 masked columns, -inf, NaN (both signs), +inf, denormals, extremes.
Every case also checks the stock ids against a numpy emulation of ttnn.argmax's float32 order (first index of the
maximum, NaN never picked, -0 == +0, index 0 when nothing exceeds -inf).  Trace safety: captured TileArgmax programs
replayed on logits rewritten in place.  --bench: traced us/call, stock vs new, T=1 and T=4 (both widths; with and
without keep_rm).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROWS = (1, 2, 3, 4, 8, 16, 17, 32)
PNAN, NNAN, SNAN = (np.array([b], np.uint32).view(np.float32)[0] for b in (0x7FC00000, 0xFFC00000, 0x7F800001))


def stock_order(x: np.ndarray) -> np.ndarray:
    """ttnn.argmax (float32, ROW_MAJOR, last dim) semantics: per row the first index of the maximum over non-NaN
    values (float compare: -0 == +0); 0 when no value exceeds -inf."""
    out = np.zeros(x.shape[0], np.int64)
    for r, row in enumerate(x):
        ok = ~np.isnan(row)
        if not ok.any():
            continue
        m = row[ok].max()
        if m == -np.inf:
            continue
        out[r] = int(np.flatnonzero(ok & (row == m))[0])
    return out


def stock(logits, keep_rm: bool = False):
    """The replaced device argmax: logits [1,1,T,V] TILE -> uint32 [1,1,T,1] ROW_MAJOR (+ the row-major logits)."""
    import ttnn
    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ids = ttnn.argmax(rm, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if keep_rm:
        return ids, rm
    ttnn.deallocate(rm)
    return ids


def synthetic(rng, T: int, V: int, n_real: int, workers: int) -> dict[str, np.ndarray]:
    f32 = np.float32
    tiles = V // 32
    bounds = np.array([w * tiles // workers * 32 for w in range(1, workers)])   # first column of each worker > 0
    c = {}
    c["normal"] = (rng.standard_normal((T, V)) * 4).astype(f32)
    c["ties_int"] = rng.integers(0, 4, (T, V)).astype(f32)
    x = rng.integers(-8, 0, (T, V)).astype(f32)
    for r in range(T):
        x[r, np.sort(rng.choice(V, 3, replace=False))] = 5.0
    c["ties_sparse"] = x
    x = rng.integers(-8, 0, (T, V)).astype(f32)
    for r in range(T):
        b = int(bounds[rng.integers(len(bounds))])
        kind = r % 4
        cols = {0: [b - 1, b], 1: [b, b + 1], 2: [b + 15, b + 16], 3: [V - 1]}[kind]
        x[r, cols] = 7.0
    c["ties_boundary"] = x
    consts = [0.0, -0.0, -1e30, 2.5, -np.inf, np.inf, 1e-42, -3.0]
    c["equal"] = np.stack([np.full(V, consts[r % len(consts)], f32) for r in range(T)])
    x = -np.abs(rng.standard_normal((T, V))).astype(f32) - f32(1)
    for r in range(T):
        kind = r % 5
        if kind == 0:     # -0 first, +0 later (other worker)
            x[r, int(bounds[3]) + 5], x[r, int(bounds[40]) + 9] = -0.0, 0.0
        elif kind == 1:   # +0 first, -0 later
            x[r, int(bounds[3]) + 5], x[r, int(bounds[40]) + 9] = 0.0, -0.0
        elif kind == 2:   # -0 then +0 in the same face row
            x[r, 100], x[r, 101] = -0.0, 0.0
        elif kind == 3:   # all signed zeros
            x[r] = np.where(rng.random(V) < 0.5, f32(-0.0), f32(0.0))
        else:             # all -0
            x[r] = -0.0
    c["zeros_signed"] = x
    x = (rng.standard_normal((T, V)) * 4).astype(f32)
    x[:, n_real:] = -1e30
    x[T // 2:, :n_real:7] = -1e30
    if T > 1:
        x[1] = -1e30
    c["masked"] = x
    x = np.full((T, V), -np.inf, f32)
    for r in range(T):
        kind = r % 4
        if kind == 1:
            x[r, rng.integers(V)] = -1e30
        elif kind == 2:
            x[r, rng.random(V) < 0.3] = PNAN
        elif kind == 3:
            x[r, 0] = NNAN
            x[r, rng.integers(1, V)] = 1.0
    c["neginf"] = x
    x = (rng.standard_normal((T, V)) * 4).astype(f32)
    m = rng.random((T, V))
    x[m < 0.03] = PNAN
    x[(m >= 0.03) & (m < 0.05)] = NNAN
    x[(m >= 0.05) & (m < 0.06)] = SNAN
    x[:, 0] = PNAN
    if T > 1:
        x[1] = PNAN
    if T > 2:
        x[2] = NNAN
        x[2, 777] = -np.inf
    c["nan"] = x
    x = (rng.standard_normal((T, V)) * 4).astype(f32)
    for r in range(T):
        x[r, np.sort(rng.choice(V, 2, replace=False))] = np.inf
    c["inf"] = x
    x = (rng.standard_normal((T, V)) * 1e-39).astype(f32)                       # denormals of both signs
    x[:, ::3] = 0.0
    x[:, 1::5] = -0.0
    c["denormal"] = x
    x = (rng.standard_normal((T, V)) * 3e37).astype(f32)
    x[:, rng.integers(V, size=4)] = np.finfo(f32).max
    x[:, rng.integers(V, size=4)] = -np.finfo(f32).max
    c["extreme"] = x
    return c


def main() -> bool:
    import torch
    import ttnn
    from fnx.ref.embed import N_VOCAB
    from fnx.tt.head import Head
    from fnx.tt.tile_argmax import TileArgmax
    from fnx.ref.mtp import MTP_DIR

    real = "--no-real" not in sys.argv
    vocab = np.fromfile(MTP_DIR / "rt" / "draft_vocab.bin", dtype=np.int32)
    n_draft = len(vocab)
    v_draft = -(-n_draft // 32) * 32
    widths = {"head": (N_VOCAB, N_VOCAB - 37), "draft": (v_draft, n_draft)}
    print(f"load average {os.getloadavg()}; widths head {N_VOCAB}, draft {v_draft} ({n_draft} real columns)", flush=True)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    ok = True
    try:
        ta = TileArgmax(mesh)

        def up(x: np.ndarray):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(1, 1, *x.shape),
                                   dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)

        def ids_np(t) -> np.ndarray:
            return ttnn.to_torch(t).to(torch.int64).numpy().reshape(-1)

        def bits(t) -> np.ndarray:
            """A row-major tensor's elements, flattened, as raw float32 bits."""
            return np.ascontiguousarray(ttnn.to_torch(t).numpy()).view(np.uint32).reshape(-1)

        def check(name: str, lg, T: int, x: np.ndarray | None) -> bool:
            si, srm = stock(lg, keep_rm=True)
            ni, nrm = ta(lg, keep_rm=True)
            n2 = ta(lg)
            a, b, b2 = ids_np(si), ids_np(ni), ids_np(n2)
            rb_s, rb_n = bits(srm), bits(nrm)
            good = np.array_equal(a, b) and np.array_equal(a, b2) and np.array_equal(rb_s, rb_n)
            emu = "" if x is None else (" emu==stock" if np.array_equal(stock_order(x), a) else
                                        f" NOTE emu!=stock rows {np.flatnonzero(stock_order(x) != a).tolist()}")
            src = "" if x is None else (" rm==host bits" if np.array_equal(rb_s.reshape(T, -1), x.view(np.uint32))
                                        else " NOTE stock rm != host bits")
            if not good:
                print(f"  FAIL {name} T={T}: stock {a[:8].tolist()} new {b[:8].tolist()} new(no rm) {b2[:8].tolist()} "
                      f"rm bits equal {np.array_equal(rb_s, rb_n)}{emu}{src}", flush=True)
            elif emu.startswith(" NOTE") or src.startswith(" NOTE"):
                print(f"  {name} T={T}: ids equal{emu}{src}", flush=True)
            for t in (si, srm, ni, nrm, n2):
                ttnn.deallocate(t)
            return good

        rng = np.random.default_rng(7)
        for wname, (V, n_real) in widths.items():
            for T in ROWS:
                cs = synthetic(rng, T, V, n_real, ta.workers)
                res = []
                for name, x in cs.items():
                    lg = up(x)
                    res.append(check(f"{wname}/{name}", lg, T, x))
                    ttnn.deallocate(lg)
                print(f"{wname} V={V} T={T:2d}: {sum(res)}/{len(res)} synthetic cases bitwise (ids, ids without rm, "
                      f"rm bits)", flush=True)
                ok &= all(res)

        if real:
            from fnx.tests.test_hc import load_trunk
            from fnx.tt.hc import to_dev
            from fnx.weights import Weights
            hd = Head(mesh, Weights())
            hx = load_trunk(32)["head_in"]
            vmask = np.zeros((1, v_draft), np.float32)
            vmask[:, n_draft:] = -1e30
            vm = up(vmask)

            def logits(wname: str, T: int):
                """Real logits: the head on T real head inputs; the draft width = their draft-vocab columns (+ zero
                columns) + vmask on the device, as fnx.tt.mtp.Mtp.rest builds lgm."""
                xd = to_dev(mesh, hx[:T].reshape(1, 1, T, -1))
                lg = hd.forward(xd)
                ttnn.deallocate(xd)
                if wname == "head":
                    return lg
                host = ttnn.to_torch(lg).numpy().reshape(-1, N_VOCAB)[:T]
                ttnn.deallocate(lg)
                sub = np.zeros((T, v_draft), np.float32)
                sub[:, :n_draft] = host[:, vocab]
                ls = up(sub)
                lgm = ttnn.add(ls, vm)
                ttnn.deallocate(ls)
                return lgm

            for T in ROWS:
                res = []
                for wname in widths:
                    lg = logits(wname, T)
                    res.append(check(f"real/{wname}", lg, T, None))
                    ttnn.deallocate(lg)
                ok &= all(res)
                print(f"real logits T={T:2d}: head {'PASS' if res[0] else 'FAIL'}, draft (+vmask on device) "
                      f"{'PASS' if res[1] else 'FAIL'}", flush=True)
        else:
            def logits(wname: str, T: int):
                return up((rng.standard_normal((T, widths[wname][0])) * 4).astype(np.float32))

        # trace safety: one captured program per variant, replayed on logits rewritten in place
        res = []
        for wname, (V, n_real) in widths.items():
            for T in (1, 4):
                x0 = (rng.standard_normal((T, V)) * 4).astype(np.float32)
                lg = up(x0)
                outs = [ta(lg), ta(lg, keep_rm=True)]
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for o in outs:
                    for t in (o if isinstance(o, tuple) else (o,)):
                        ttnn.deallocate(t)
                outs = [ta(lg), ta(lg, keep_rm=True)]
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                for name, x in list(synthetic(rng, T, V, n_real, ta.workers).items())[:6]:
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(torch.from_numpy(x).reshape(1, 1, T, V), dtype=ttnn.float32,
                                        layout=ttnn.TILE_LAYOUT), lg)
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    si, srm = stock(lg, keep_rm=True)
                    want = ids_np(si)
                    res.append(np.array_equal(ids_np(outs[0]), want) and np.array_equal(ids_np(outs[1][0]), want)
                               and np.array_equal(bits(outs[1][1]), bits(srm)))
                    ttnn.deallocate(si)
                    ttnn.deallocate(srm)
                ttnn.release_trace(mesh, tid)
                for o in outs:
                    for t in (o if isinstance(o, tuple) else (o,)):
                        ttnn.deallocate(t)
                ttnn.deallocate(lg)
        ok &= all(res)
        print(f"trace replays on logits rewritten in place: {sum(res)}/{len(res)} bitwise (ids, ids + rm bits)",
              flush=True)

        if "--bench" in sys.argv:
            def free(r):
                for t in (r if isinstance(r, tuple) else (r,)):
                    ttnn.deallocate(t)

            def traced_us(fn, inner: int = 10, reps: int = 20):
                """us per call of fn over `reps` replays of a trace of `inner` calls; returns the last call's
                outputs (kept alive through the replays) too."""
                free(fn())
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for _ in range(inner - 1):
                    free(fn())
                last = fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(reps):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                us = (time.perf_counter() - t0) / (reps * inner) * 1e6
                ttnn.release_trace(mesh, tid)
                return us, last

            print(f"traced us/call (load average {os.getloadavg()}):", flush=True)
            for wname in widths:
                for T in (1, 4):
                    lg = logits(wname, T)
                    si = stock(lg)
                    want = ids_np(si)
                    ttnn.deallocate(si)
                    variants = {"stock": lambda: stock(lg), "stock+rm": lambda: stock(lg, keep_rm=True),
                                "new": lambda: ta(lg), "new+rm": lambda: ta(lg, keep_rm=True)}
                    out = []
                    for vname, fn in variants.items():
                        us, last = traced_us(fn)
                        same = np.array_equal(ids_np(last[0] if isinstance(last, tuple) else last), want)
                        ok &= same
                        free(last)
                        out.append(f"{vname} {us:7.1f}" + ("" if same else " (ids differ after replay!)"))
                    print(f"  {wname} T={T}: " + "  ".join(out), flush=True)
                    ttnn.deallocate(lg)
    finally:
        ttnn.close_mesh_device(mesh)
    print("test_tile_argmax", "PASS" if ok else "FAIL")
    return bool(ok)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
