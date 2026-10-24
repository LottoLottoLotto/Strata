"""CpuExperts (libfnx_cpu) against float math on dequantized weights.

  python -m fnx.tests.test_cpu_experts      (CPU only; NODEV=1 tools/devrun.sh ... inside the TT image)

- Coder layers 0 (IQ3_XXS gate/up, IQ4_NL down) and 1 (IQ2_S gate/up, Q2_0 down): every slot of 5 tokens x 10 experts,
  with experts shared between tokens (multi-token groups) and a skipped slot (must come back zero: the output is not
  pre-filled).
- An extra layer registered with add_layer holding Q2_0 gate, up AND down (the MTP layer's format), built with
  Strata's own quantize_q2_0 from random weights, 512 experts.
- Grouping independence (STRATA_IQ_MT_MIN=1): a token's rows are bit-identical alone and in a group.
- The leader-thread path fnx.tt.moe uses (CpuExperts.submit / wait) is bit-identical to moe, also when a submitted
  call is never waited for and the next call has to finish it first.
Threshold: relative L2 <= 3e-2 per slot (the kernels quantize activations to 8 bits; measured ~1.5e-2).
"""
from __future__ import annotations

import sys

import numpy as np

from fnx import paths
from fnx.config import N_EXPERT
from fnx.cpu_experts import CpuExperts
from fnx.gguf_load import STRATA_TOOLS, Model, dequant_q2_0

sys.path.insert(0, str(STRATA_TOOLS))
from gguf_writer import quantize_q2_0  # noqa: E402

GGUF = str(paths.SHARD1)
H, FF, TOL = 2560, 640, 3e-2
fails = 0


def check(name: str, ok: bool, detail: str) -> None:
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def swiglu_ref(g, u, d, x):
    a = g @ x
    return d @ (a / (1 + np.exp(-a)) * (u @ x))


def slots(rng, n_expert, nt=5, k=10):
    ids = np.stack([rng.choice(n_expert, k, replace=False) for _ in range(nt)]).astype(np.int32)
    ids[1, :3] = ids[0, :3]        # shared experts -> multi-token groups
    ids[2, 4] = -1                 # skipped slot
    return ids


def worst_rel(out, ids, x, expert_fn):
    worst = 0.0
    for t in range(ids.shape[0]):
        for s in range(ids.shape[1]):
            e = ids[t, s]
            if e < 0:
                if out[t, s].any():
                    return float("inf")
                continue
            ref = swiglu_ref(*expert_fn(e), x[t])
            worst = max(worst, float(np.linalg.norm(out[t, s] - ref) / np.linalg.norm(ref)))
    return worst


def main() -> int:
    m = Model(GGUF)
    cx = CpuExperts(m, 48, 16, capacity=49)
    rng = np.random.default_rng(7)

    for L in (0, 1):
        x = rng.standard_normal((5, H)).astype(np.float32)
        ids = slots(rng, N_EXPERT)
        out = cx.moe(L, x, ids)
        names = [f"blk.{L}.ffn_{r}_exps.weight" for r in ("gate", "up", "down")]
        w = worst_rel(out, ids, x, lambda e: [m.get_expert(n, e) for n in names])
        check(f"{paths.MODEL} layer {L} ({'/'.join(m.type_of(n) for n in names)})", w <= TOL, f"worst rel {w:.3e}")

    # synthetic Q2_0 layer (gate/up [n_expert, FF, H], down [n_expert, H, FF]) in GGUF expert-stack layout
    n_expert = 512
    used = sorted(set(rng.choice(n_expert, 40, replace=False).tolist()))
    stacks, dense = {}, {r: {} for r in ("gate", "up", "down")}
    for r, (rows, cols) in (("gate", (FF, H)), ("up", (FF, H)), ("down", (H, FF))):
        per = rows * cols // 64 * 18
        buf = np.zeros(n_expert * per, dtype=np.uint8)
        for e in used:                       # only experts the test routes to get real weights
            wq = (rng.standard_normal((rows, cols)) * 0.03).astype(np.float32)
            raw = np.frombuffer(quantize_q2_0(wq.reshape(-1)), dtype=np.uint8)
            buf[e * per:(e + 1) * per] = raw
            dense[r][e] = dequant_q2_0(raw).reshape(rows, cols)
        stacks[r] = buf
    cx.add_layer(48, stacks["gate"], stacks["up"], stacks["down"], "Q2_0", "Q2_0")
    x = rng.standard_normal((5, H)).astype(np.float32)
    ids = np.stack([rng.choice(used, 10, replace=False) for _ in range(5)]).astype(np.int32)
    ids[1, :3] = ids[0, :3]
    ids[2, 4] = -1
    out = cx.moe(48, x, ids)
    w = worst_rel(out, ids, x, lambda e: [dense["gate"][e], dense["up"][e], dense["down"][e]])
    check("add_layer Q2_0 gate/up/down (512 experts)", w <= TOL, f"worst rel {w:.3e}")

    # grouping independence: token 0 alone == token 0 inside the 5-token call
    for L in (0, 1, 48):
        x = rng.standard_normal((5, H)).astype(np.float32)
        ids = slots(rng, N_EXPERT) if L < 48 else np.stack([rng.choice(used, 10, replace=False) for _ in range(5)]).astype(np.int32)
        ids[1, :3] = ids[0, :3]
        group = cx.moe(L, x, ids)[0]
        alone = cx.moe(L, x[:1], ids[:1])[0]
        check(f"layer {L} token rows independent of grouping", np.array_equal(group, alone),
              f"max |diff| {float(np.abs(group - alone).max()):.3e}")

    # wide groups (a prompt chunk: more than MAXT tokens of one expert, up to kIq512WideMax per gate/up call, the down
    # rows in pieces): every token's rows == the same token in 2-token calls (both multi-token paths)
    for L in (0, 1):
        x = rng.standard_normal((40, H)).astype(np.float32)
        ids = np.tile(rng.choice(N_EXPERT, 10, replace=False).astype(np.int32), (40, 1))
        wide = cx.moe(L, x, ids)
        pairs = np.concatenate([cx.moe(L, x[t:t + 2], ids[t:t + 2]) for t in range(0, 40, 2)])
        check(f"layer {L} 40 tokens of the same experts == 2-token calls", np.array_equal(wide, pairs),
              f"max |diff| {float(np.abs(wide - pairs).max()):.3e}")

    # the leader-thread path (submit / wait) == moe, bitwise
    calls = {}
    for L in (0, 1, 48):
        x = rng.standard_normal((4, H)).astype(np.float32)
        ids = slots(rng, 256, nt=4) if L < 48 else np.stack([rng.choice(used, 10, replace=False) for _ in range(4)]).astype(np.int32)
        sync = cx.moe(L, x, ids)
        a = cx.submit(L, x, np.ascontiguousarray(ids), np.full_like(sync, np.nan))
        us = cx.wait()
        calls[L] = (x, ids, sync)
        check(f"layer {L} submit/wait == moe", np.array_equal(a, sync) and us > 0, f"leader time {us:.0f} us")
    (x0, ids0, ref0), (x1, ids1, ref1) = calls[0], calls[1]
    a = cx.submit(0, x0, ids0)                    # never waited for: the next call finishes it first
    b = cx.moe(1, x1, ids1)
    check("a pending submit is finished before the next call", np.array_equal(a, ref0) and np.array_equal(b, ref1),
          "both bitwise")

    print("test_cpu_experts", "FAILED" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
