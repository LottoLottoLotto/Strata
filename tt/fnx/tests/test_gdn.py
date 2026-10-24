"""TT Gdn vs the numpy reference on real Coder weights (GDN layers 0,1,2,4,5,6 when on disk).

Inputs are realistic layer inputs: the token embeddings (IQ4_XS rows, dequantized) of a tokenized code snippet,
broadcast to the 4 HC streams (the residual every layer starts from) and read through layer L's own
`blk.L.hc_attn_*` mixer (fnx.ref.hc.hc_read): exactly GDN layer 0's input, and layer-L-scaled `mixed` elsewhere.

Per layer and impl ("fused", "ops", "native"; see fnx.tt.gdn), consecutive tokens from a zero state, schedules: eager
calls of N tokens each (1 = decode, 8 = verify-window size, 128 = serve prompt chunk), one replay per token of a
captured T=1 trace (input copied into a fixed buffer, state in place; its replay time is the per-token device time),
and "win" (fused only): verify windows of 4 rows (GdnFused.forward(snap=True)) each rolled back to its first 1..4
rows (GdnFused.rollback), the next window starting at the first rejected row: the kept rows' bo and the final state
must be bitwise the T=1 decode of the same rows.
Compared per token (bo) and at the end (S, conv history) against:
  ref   - fnx.ref.gdn with the GGUF weights dequantized to f32 (the spec numerics);
  refq  - the same, with the projections replaced by the values read back from the device (bfloat8_b rounding):
          isolates the device arithmetic.
Errors: rel = ||tt - ref|| / ||ref|| per token (max over tokens) and for the state; printed with thresholds.

Run: tools/devrun.sh test_gdn fnx/tests/test_gdn.py [--layers 0,1,2] [--tokens 32] [--impls fused,ops,native]
     [--schedules 1,8,trace,win]  (long run: --layers 1,4 --tokens 1024 --impls fused --schedules trace,128)
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import torch
import ttnn

from fnx.gguf_load import gq, GGMLQuantizationType
from fnx.ref import gdn as rg
from fnx.ref.hc import HC, hc_read, hc_weights
from fnx.tt.gdn import GDN_TENSORS, Gdn
from fnx.weights import Weights

TEXT = ('def merge_sort(items):\n    """Sort a list with merge sort."""\n    if len(items) <= 1:\n'
        '        return items\n    mid = len(items) // 2\n    left = merge_sort(items[:mid])\n'
        '    right = merge_sort(items[mid:])\n    return merge(left, right)\n')
# rel-error thresholds per impl: (vs refq = device arithmetic only, vs ref = incl. bfloat8_b weight rounding).
# fused: FP32 state and exact FP32 elementwise math; FPU contractions (~10-bit inputs) set the floor (measured on
#   L1/L4: bo 2.6-3.0e-3, S 1.0-1.3e-3 after 32 and after 1024 tokens).
# ops: fp32 state and elementwise math; matmul / norm inputs are rounded to ~10 mantissa bits (~1e-3 per op, measured:
#   0.2-0.6% bo after 32 tokens, 0.4% after 1024). native: BF16 state, decay, q/k/v, conv history and output
#   (measured 1-2.5% bo after 32 tokens, growing with length: ~9% after 256 tokens on L1/L4).
TOL = {"fused": (5e-3, 5e-2), "ops": (1e-2, 5e-2), "native": (3e-2, 5e-2)}


def byte_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b), cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def tokenize(model, text: str) -> list[int]:
    """Greedy longest-match over the GGUF byte-level BPE vocabulary (realistic ids, not the exact BPE merge)."""
    vocab = {t: i for i, t in enumerate(model.metadata["tokenizer.ggml.tokens"])}
    bu = byte_unicode()
    ids = []
    for piece in re.findall(r" ?\w+|\s+|[^\w\s]+", text):
        s = "".join(bu[b] for b in piece.encode())
        while s:
            for j in range(len(s), 0, -1):
                if s[:j] in vocab:
                    ids.append(vocab[s[:j]])
                    s = s[j:]
                    break
            else:
                raise ValueError(f"untokenizable {s!r}")
    return ids


def embed(model, ids) -> np.ndarray:
    info = model.info("token_embd.weight")
    row = info.expected_bytes() // info.shape[1]
    raw = model.raw("token_embd.weight")
    return np.stack([np.asarray(gq.dequantize(raw[i * row:(i + 1) * row], GGMLQuantizationType.IQ4_XS),
                                np.float32).reshape(-1) for i in ids])


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def run_tt(mesh, gd, mixed: np.ndarray, chunk: int):
    gd.reset()
    outs, ms = [], []
    for t0 in range(0, len(mixed), chunk):
        x = ttnn.from_torch(torch.from_numpy(mixed[t0:t0 + chunk].reshape(1, 1, -1, rg.N)), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh)
        t = time.perf_counter()
        bo = gd.forward(x)
        ttnn.synchronize_device(mesh)
        ms.append((time.perf_counter() - t) * 1e3)
        outs.append(ttnn.to_torch(bo).float().numpy().reshape(-1, rg.N))
    return np.concatenate(outs), gd.state_numpy(), ms


def run_traced(mesh, gd, mixed: np.ndarray, n_time: int = 64):
    """Decode (T=1) by replaying one captured trace of gd.forward: per token the input is copied into a fixed
    device buffer and the state advances in place. Returns outputs, final state, device ms per replay."""
    host = lambda a: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a).reshape(1, 1, 1, rg.N)),
                                     dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    x = ttnn.from_torch(torch.zeros(1, 1, 1, rg.N), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gd.reset()
    gd.forward(x)                                           # compile outside the capture
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    bo = gd.forward(x)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    try:
        gd.reset()
        outs = []
        for t in range(len(mixed)):
            ttnn.copy_host_to_device_tensor(host(mixed[t]), x)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            outs.append(ttnn.to_torch(bo).float().numpy().reshape(-1))
        state = gd.state_numpy()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(n_time):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) * 1e3 / n_time
    finally:
        ttnn.release_trace(mesh, tid)
    return np.stack(outs), state, [ms]


def run_windows(mesh, gd, mixed: np.ndarray, T: int = 4, keeps=(1, 2, 3, 4, 2, 1, 3, 4, 1)):
    """Verify windows (GdnFused.forward(snap=True) of T rows, then rollback(n) to the first n rows; the next window
    starts at the first rejected row) over the rows of `mixed`.  Returns the kept rows' bo, the final state, the
    number of rows consumed: bitwise the T=1 decode of those rows (every row keeps the same arithmetic)."""
    gd.reset()
    outs, p = [], 0
    for n in keeps:
        rows = mixed[p:p + T]
        x = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(rows).reshape(1, 1, len(rows), rg.N)),
                            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bo = ttnn.to_torch(gd.forward(x, snap=True)).float().numpy().reshape(-1, rg.N)
        if n < len(rows):
            gd.rollback(n)
        outs.append(bo[:n])
        p += n
    return np.concatenate(outs), gd.state_numpy(), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2,4,5,6")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--schedules", default="1,8,trace,win,wide",
                    help="tokens per eager call, 'trace' (T=1 replay), 'win' (fused: verify windows + rollback) or "
                         "'wide' (fused: 64/128/256-row blocks vs the same rows in 32-row calls, bitwise)")
    ap.add_argument("--impls", default="fused,ops,native")
    args = ap.parse_args()
    w = Weights()
    m = w.model
    layers = [L for L in map(int, args.layers.split(","))
              if L % 4 != 3 and all(m.available(f"blk.{L}.{s}") for s in GDN_TENSORS)
              and all(m.available(f"blk.{L}.hc_attn_{s}.weight") for s in ("norm", "down", "up", "inject"))]
    ids = tokenize(m, TEXT * (1 + args.tokens // 40))[:args.tokens]   # the snippet is ~70 tokens
    assert len(ids) == args.tokens, len(ids)
    emb = embed(m, ids)
    R = np.repeat(emb[:, None, :], HC, axis=1)
    print(f"tokens {ids}\nemb rms {np.sqrt((emb ** 2).mean()):.4f}; layers {layers}; thresholds (refq, ref) {TOL}",
          flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    ok = True
    try:
        for L in layers:
            mixed, _ = hc_read(R, *hc_weights(w, f"blk.{L}.hc_attn"))
            W = rg.gdn_weights(w, L)
            st = rg.GdnState()
            ref = rg.gdn(W, L, mixed, st), st
            print(f"L{L}: mixed rms {np.sqrt((mixed ** 2).mean()):.4f}, bo rms {np.sqrt((ref[0] ** 2).mean()):.4f}",
                  flush=True)
            for impl in args.impls.split(","):
                t = time.perf_counter()
                gd = Gdn(mesh, w, L, impl)
                init_s = time.perf_counter() - t
                st = rg.GdnState()
                refs = {"refq": (rg.gdn(gd.uploaded_weights(W), L, mixed, st), st), "ref": ref}
                for sched in args.schedules.split(","):
                    same = lambda a, b: np.array_equal(np.asarray(a, np.float32).view(np.uint32),
                                                       np.asarray(b, np.float32).view(np.uint32))
                    if sched == "wide":                # GdnFused's one-pass prompt blocks (32-row multiples)
                        if impl != "fused":
                            continue
                        for T in (64, 128, 256):
                            if T > len(mixed):
                                continue
                            bo, (S, hist), _ = run_tt(mesh, gd, mixed[:T], T)
                            bo1, (S1, hist1), _ = run_tt(mesh, gd, mixed[:T], 32)
                            good = same(bo, bo1) and same(S, S1) and same(hist, hist1)
                            ok &= good
                            print(f"  L{L} {impl:6s} one {T}-row block: bo, S, hist bitwise {T // 32} 32-row calls: "
                                  f"{'PASS' if good else 'FAIL'}", flush=True)
                        continue
                    if sched == "win":
                        if impl != "fused":
                            continue
                        bo, (S, hist), n = run_windows(mesh, gd, mixed)
                        bo1, (S1, hist1), _ = run_tt(mesh, gd, mixed[:n], 1)
                        good = same(bo, bo1) and same(S, S1) and same(hist, hist1)
                        ok &= good
                        print(f"  L{L} {impl:6s} verify windows of 4 rows + rollback, {n} rows kept: bo, S, hist "
                              f"bitwise the T=1 decode: {'PASS' if good else 'FAIL'}", flush=True)
                        continue
                    if sched == "trace":
                        bo, (S, hist), ms = run_traced(mesh, gd, mixed)
                        line = [f"  L{L} {impl:6s} traced T=1 x{len(mixed)}: device ms/token {ms[0]:.3f}"]
                    else:
                        bo, (S, hist), ms = run_tt(mesh, gd, mixed, int(sched))
                        line = [f"  L{L} {impl:6s} eager T={sched:<2s} x{len(ms)} (init {init_s:.1f}s): ms/call first "
                                f"{ms[0]:.1f} median rest {np.median(ms[1:]) if len(ms) > 1 else float('nan'):.2f}"]
                    for (name, (rbo, rst)), tol in zip(refs.items(), TOL[impl]):
                        e_bo = max(rel(bo[i], rbo[i]) for i in range(len(bo)))
                        e_S, e_h = rel(S, rst.S), rel(hist, rst.conv)
                        good = max(e_bo, e_S, e_h) < tol
                        ok &= good
                        line.append(f"vs {name:4s}: bo rel max {e_bo:.2e} (last {rel(bo[-1], rbo[-1]):.2e}) S rel "
                                    f"{e_S:.2e} (max|d| {np.abs(S - rst.S).max():.2e}) hist rel {e_h:.2e} "
                                    f"{'PASS' if good else 'FAIL'} <{tol}")
                    print("\n    ".join(line), flush=True)
                del gd
    finally:
        ttnn.close_mesh_device(mesh)
    print("test_gdn", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
