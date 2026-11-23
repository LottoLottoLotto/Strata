"""CPU experts: correctness vs float experts on layers 0/1 and per-call latency at T=1 and T=4 (median of 200)."""
import sys
import time

import numpy as np

from fnx import paths
from fnx.cpu_experts import CpuExperts
from fnx.gguf_load import Model

threads = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "8,16").split(",")]
m = Model(paths.SHARD1)
rng = np.random.default_rng(1)
for nth in threads:
    cx = CpuExperts(m, 48, nth)
    for L in (0, 1):
        x = rng.standard_normal((5, 2560)).astype(np.float32)
        ids = np.stack([rng.choice(256, 10, replace=False) for _ in range(5)]).astype(np.int32)
        out = cx.moe(L, x, ids)
        worst = 0.0
        for t in range(5):
            for s in range(10):
                e = ids[t, s]
                g, u, d = (m.get_expert(f"blk.{L}.ffn_{r}_exps.weight", e) for r in ("gate", "up", "down"))
                a = g @ x[t]
                ref = d @ (a / (1 + np.exp(-a)) * (u @ x[t]))
                worst = max(worst, float(np.linalg.norm(out[t, s] - ref) / np.linalg.norm(ref)))
        print(f"threads {nth} layer {L}: worst rel err {worst:.4e}")
    for T in (1, 4):
        x = rng.standard_normal((T, 2560)).astype(np.float32)
        ts = []
        for i in range(200):
            ids = np.stack([rng.choice(256, 10, replace=False) for _ in range(T)]).astype(np.int32)
            t0 = time.perf_counter()
            cx.moe(i % 2, x, ids)
            ts.append(time.perf_counter() - t0)
        print(f"threads {nth} T={T}: median {np.median(ts) * 1e6:.0f} us, p90 {np.percentile(ts, 90) * 1e6:.0f} us")
