"""Embedding + LM head: the numpy embedding vs gguf-py, and the TT head vs numpy on realistic head inputs (the trunk's
output_hc read, see fnx/tests/test_hc.py), with on-device argmax.  The head mode is FNX_HEAD (fnx.tt.head: bfp8 =
tt-metal's bfloat8_b of the Q6_K values, gptq4 = the GPTQ-BFP4 store rows as bfloat4_b).

    NODEV=1 tools/devrun.sh head-prep -m fnx.tests.test_head --prepare   # convert + cache the 80 head blocks
    tools/devrun.sh head -m fnx.tests.test_head [--mode gptq4]

Checks: (block-float store modes) the device weight equals the store exactly (one-hot inputs read rows of W^T);
the device logits against numpy float64 logits of the exact device weights (the matmul's own error); the device
argmax against numpy's on every row whose top-1/top-2 margin exceeds MARGIN_X * max error and against the argmax of
the device's own logits; and the logits against the float32 Q6_K head (the re-quantization error, bounded per mode).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from fnx.ref.embed import N, N_VOCAB, argmax_lowest, embed, head

# matmul floor (TF32/FPU, fp32 accumulation) against the device weights; re-quantization error against the float
# Q6_K head per weight format (bfp8: 16-value shared exponent, 7-bit mantissa; bfp4 GPTQ: rel_h 3.1e-3 held out ->
# ~5.5% rel L2)
TOL = {"mm_rel": 1e-3, "margin_x": 2.0, "quant_rel": {"bfp8": 1.5e-2, "bfp4": 8e-2}}


def host_checks(w) -> bool:
    import gguf.quants as gq
    from gguf.constants import GGMLQuantizationType
    toks = [0, 1, 11, 12345, 100000, 248043, 248044, 248319]
    e = embed(w, toks)
    raw = w.model.raw("token_embd.weight")
    ok = all(np.array_equal(e[i], np.asarray(gq.dequantize(raw[t * 1360:(t + 1) * 1360], GGMLQuantizationType.IQ4_XS),
                                                np.float32).reshape(-1)) for i, t in enumerate(toks))
    spec0 = np.array([-5.114e-05, -1.2785e-03, 5.114e-04], np.float32)          # spec Part 2 §1, token 0
    ok = ok and np.allclose(e[0, :3], spec0, rtol=1e-3) and abs(np.sqrt((e[0] ** 2).mean()) - 2.87e-3) < 1e-5
    print(f"embed (IQ4_XS rows) == gguf-py dequant for {toks}, token 0 = spec values: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def device_checks(w, mode: str) -> bool:
    import ttnn
    from fnx.tests.test_hc import load_trunk
    from fnx.tt.hc import to_dev, to_np
    from fnx.tt.head import Head, device_values, head_parts
    tr = load_trunk(32)
    x = tr["head_in"]                                                     # [32, 2560] f32
    x = np.concatenate([x, x[::-1] * np.float32(0.9)])                   # + 32 more rows for a T=64 (prefill) case
    t0 = time.time()
    ref_f = head(w, x)
    chunk, n_parts = head_parts()
    ref = np.empty((x.shape[0], N_VOCAB), np.float64)
    rows = {k0: np.empty((32, N_VOCAB), np.float32) for k0 in (0, 1264, 2528)}
    for i in range(n_parts):
        v0, v1 = i * chunk, min((i + 1) * chunk, N_VOCAB)
        Wd = device_values(w, mode, v0, v1)                              # [v1 - v0, 2560], the device's values
        ref[:, v0:v1] = x.astype(np.float64) @ Wd.T.astype(np.float64)
        for k0, r in rows.items():
            r[:, v0:v1] = Wd[:, k0:k0 + 32].T
    fmt = "bfp8" if mode == "bfp8" else "bfp4"
    print(f"mode {mode} ({fmt}): numpy heads (float Q6_K, float64 of the device values) on {x.shape[0]} rows: "
          f"{time.time() - t0:.1f} s; head_in rms {np.sqrt((x ** 2).mean()):.3g}; logits rms {np.sqrt((ref ** 2).mean()):.3g}")
    ok = True
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        t0 = time.time()
        hd = Head(mesh, w, mode)
        print(f"Head upload (80 cached blocks, exact page-copy join): {time.time() - t0:.1f} s, weight {hd.w.dtype} "
              f"{tuple(hd.w.shape)}")
        for k0, r in rows.items():                                       # e_k inputs: rows k0.. of W^T, exactly
            oh = np.zeros((32, N), np.float32)
            oh[np.arange(32), k0 + np.arange(32)] = 1
            lg = hd.forward(to_dev(mesh, oh.reshape(1, 1, 32, N)))
            got = to_np(lg).reshape(32, N_VOCAB)
            ttnn.deallocate(lg)
            good = np.array_equal(got, r)
            ok &= good
            print(f"device W^T rows {k0}..{k0 + 31} == host values: {'PASS' if good else 'FAIL'}")
        print(f"thresholds: logits norm-rel vs device-value float64 < {TOL['mm_rel']}, vs float Q6_K < "
              f"{TOL['quant_rel'][fmt]}; argmax equal on rows with margin > {TOL['margin_x']} x max|err|")
        for T in (64, 32, 5, 1):
            for xdt in (ttnn.float32, ttnn.bfloat16):
                xd = to_dev(mesh, x[:T].reshape(1, 1, T, N), xdt)
                logits = hd.forward(xd)
                ids_d = hd.argmax(logits)
                ttnn.synchronize_device(mesh)
                got = to_np(logits).reshape(T, N_VOCAB)
                ids = ttnn.to_torch(ids_d).to(__import__("torch").int64).numpy().reshape(-1)
                r = ref[:T]
                d = np.abs(got.astype(np.float64) - r)
                rel = float(np.linalg.norm(got - r) / np.linalg.norm(r))
                rel_f = float(np.linalg.norm(got - ref_f[:T]) / np.linalg.norm(ref_f[:T]))
                want = argmax_lowest(r)
                srt = np.sort(r, axis=1)
                margin = srt[:, -1] - srt[:, -2]
                host_ids = argmax_lowest(got)
                need = margin > TOL["margin_x"] * d.max()
                good = (rel < TOL["mm_rel"] and rel_f < TOL["quant_rel"][fmt] and np.array_equal(ids, host_ids)
                        and np.array_equal(ids[need], want[need]))
                # timing (eager, synchronized)
                n = 10
                ttnn.synchronize_device(mesh)
                t1 = time.perf_counter()
                for _ in range(n):
                    ttnn.deallocate(hd.forward(xd))
                ttnn.synchronize_device(mesh)
                ms_f = (time.perf_counter() - t1) / n * 1e3
                t1 = time.perf_counter()
                for _ in range(n):
                    ttnn.deallocate(hd.argmax(logits))
                ttnn.synchronize_device(mesh)
                ms_a = (time.perf_counter() - t1) / n * 1e3
                t1 = time.perf_counter()
                for _ in range(n):
                    g = hd.greedy(xd)
                ms_g = (time.perf_counter() - t1) / n * 1e3
                assert np.array_equal(g, ids)
                if xdt == ttnn.float32:
                    ok &= good
                print(f"T={T:2d} x={'f32 ' if xdt == ttnn.float32 else 'bf16'}: logits max {d.max():.3e} rel {rel:.3e} "
                      f"(vs float Q6_K {rel_f:.3e}, argmax agree {int((ids == argmax_lowest(ref_f[:T])).sum())}/{T}) | "
                      f"argmax dev==numpy {int((ids == want).sum())}/{T} (rows with margin > bound: {int(need.sum())}, "
                      f"min margin {margin.min():.3g}) dev==host(TT logits) {np.array_equal(ids, host_ids)} | "
                      f"forward {ms_f:.3f} ms, argmax {ms_a:.3f} ms, greedy+readback {ms_g:.3f} ms"
                      + ("  " + ("PASS" if good else "FAIL") if xdt == ttnn.float32 else ""), flush=True)
                ttnn.deallocate(logits)
                ttnn.deallocate(ids_d)
    finally:
        ttnn.close_mesh_device(mesh)
    return bool(ok)


if __name__ == "__main__":
    from fnx.weights import Weights
    w = Weights()
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else None
    if "--prepare" in sys.argv:
        # host-only conversion: ttnn's block-float packing needs a cluster descriptor, a mock one does without a device
        import os
        os.environ.setdefault("TT_METAL_MOCK_CLUSTER_DESC_PATH", "/home/container_app_user/tt-metal/tt_metal/third_party/"
                              "tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml")
        from fnx.tt.head import MODE, convert_parts
        t0 = time.time()
        parts = convert_parts(w, mode or MODE)
        print(f"converted/cached {len(parts)} head blocks ({mode or MODE}) in {time.time() - t0:.1f} s")
        sys.exit(0)
    from fnx.tt.head import MODE
    ok = host_checks(w)
    ok = device_checks(w, mode or MODE) and ok
    print("test_head", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
