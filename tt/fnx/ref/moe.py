"""MoE block, numpy reference (spec Part 6 §2 / spec-moe.md §2: Strata's decode = verify-window semantics).

Per token, x = mixed (f32[2560], the FFN-half hyper-connection read):
    router   logits = W_r(256x2560, BF16) @ x  (f32; accumulated in f64 here, then rounded)            §2.2
             p = softmax(logits) in DOUBLE (max-shifted exp, serial ascending sum, f32(1/s), f32(ex*inv))
             ids = stable descending top-10 of p (ties -> lower id), in rank order
             w = f32(f64(p[ids]) / max(sum_rank_order(f64 p[ids]), 2^-14))
    shared   sh = Ws_d @ (silu(Ws_g @ x) * (Ws_u @ x));  sh = f32(sh * sigmoid(w_sg . x))           §2.3
    routed   parts[j] = expert(ids[j])(x), UNWEIGHTED f32[2560]                                     §2.4
    combine  out = f32(parts[0]*w[0]); out = fmaf(parts[j], w[j], out) for j=1..9; bo = out + sh   §2.5

Numerics: float32 with dequantized weights, no q8_1 activation emulation (CONTRACT.md).  Routed experts come
either from Strata's own CPU kernels (fnx.cpu_experts.CpuExperts: ggml Q8_K activation in, Q8_0 / ActQ between,
exactly the engine's CPU-pool contract, §2.4.2) or, for the all-float variant, from f32 dequantized expert
weights (model.get_expert) to measure the CPU kernels' activation-quantization error.
"""
from __future__ import annotations

import numpy as np

from ..config import N_EMBD, N_EXPERT, N_USED

F32, F64 = np.float32, np.float64
W_CLAMP = 2.0 ** -14          # router_top10.cu:183


def router_logits(w, L: int, x: np.ndarray) -> np.ndarray:
    """[T,2560] f32 -> [T,256] f32 router logits (f32 activation x BF16 weight; f64 accumulation, one rounding)."""
    Wr = w.layer(L, "ffn_gate_inp.weight")                       # (256, 2560), BF16 values
    assert Wr.shape == (N_EXPERT, N_EMBD), Wr.shape
    return (np.asarray(x, F32).astype(F64) @ Wr.astype(F64).T).astype(F32)


def route(logits: np.ndarray, k: int = N_USED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """router_top10 (spec §2.2) on [T,E] f32 logits -> (ids [T,k] int32 rank order, w [T,k] f32, p [T,E] f32)."""
    lg = np.asarray(logits, F32)
    mx = lg.max(axis=-1, keepdims=True)
    ex = np.exp(lg.astype(F64) - mx.astype(F64))
    s = np.add.accumulate(ex, axis=-1)[..., -1:]                 # serial, ascending e
    inv = (1.0 / s).astype(F32)
    p = (ex * inv.astype(F64)).astype(F32)
    ids = np.argsort(-p, axis=-1, kind="stable")[..., :k]       # descending, ties -> lower id
    ws = np.take_along_axis(p, ids, axis=-1).astype(F64)
    S = np.add.accumulate(ws, axis=-1)[..., -1:]                 # rank order
    wt = (ws / np.maximum(S, W_CLAMP)).astype(F32)
    return ids.astype(np.int32), wt, p


def silu(g: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (g / (F32(1) + np.exp(-g))).astype(F32)


def sigmoid(g: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (F32(1) / (F32(1) + np.exp(-np.asarray(g, F32)))).astype(F32)


def shared_gate_logit(w, L: int, x: np.ndarray) -> np.ndarray:
    """[T] f32 = w_sg . x (BF16 weight, f32 activation; f64 accumulation, one rounding)."""
    wsg = w.layer(L, "ffn_gate_inp_shexp.weight").reshape(-1)
    assert wsg.shape == (N_EMBD,), wsg.shape
    return (np.asarray(x, F32).astype(F64) @ wsg.astype(F64)).astype(F32)


def shared_expert(w, L: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T,2560] -> (sh [T,2560] f32 already scaled by the scalar gate, gs [T] f32 = sigmoid(w_sg . x))."""
    x = np.asarray(x, F32)
    g = x @ w.layer(L, "ffn_gate_shexp.weight").T               # (640, 2560) -> [T,640]
    u = x @ w.layer(L, "ffn_up_shexp.weight").T
    h = (silu(g) * u).astype(F32)
    sh = (h @ w.layer(L, "ffn_down_shexp.weight").T).astype(F32)  # (2560, 640) -> [T,2560]
    gs = sigmoid(shared_gate_logit(w, L, x))
    return (sh * gs[:, None]).astype(F32), gs


def expert_parts_float(model, L: int, x: np.ndarray, ids: np.ndarray, cache: dict | None = None) -> np.ndarray:
    """All-float routed experts: [T,2560], ids [T,k] -> [T,k,2560] UNWEIGHTED, from f32 dequantized weights.
    `cache` (optional dict) keeps dequantized experts across calls, keyed (L, e)."""
    x = np.asarray(x, F32)
    T, k = ids.shape
    out = np.zeros((T, k, x.shape[1]), F32)
    cache = {} if cache is None else cache
    for e in np.unique(ids[ids >= 0]):
        key = (L, int(e))
        if key not in cache:
            cache[key] = tuple(model.get_expert(f"blk.{L}.ffn_{r}_exps.weight", int(e)) for r in ("gate", "up", "down"))
        Wg, Wu, Wd = cache[key]
        t, j = np.nonzero(ids == e)
        xe = x[t]
        h = (silu(xe @ Wg.T) * (xe @ Wu.T)).astype(F32)
        out[t, j] = h @ Wd.T
    return out


class PlanExperts:
    """Routed experts in float32 from a quantization plan's exact device values (fnx.quant.store.Plan.snapped), with
    the same `moe(L, x, ids)` contract as fnx.cpu_experts.CpuExperts: the reference for FNX_MOE=device, so a TT-vs-ref
    comparison measures device numerics rather than the re-quantization itself."""

    class _Weights:
        def __init__(self, plan):
            self.plan = plan

        def get_expert(self, name: str, e: int) -> np.ndarray:
            L = int(name.split(".")[1])
            m = ("gate", "up", "down").index(name.split(".")[2].split("_")[1])
            return np.ascontiguousarray(self.plan.snapped(L, e, m).T, F32)     # snapped is W.T [in, out]

    def __init__(self, plan):
        self.plan = plan
        self._w = PlanExperts._Weights(plan)
        self._cache: dict = {}
        self._layer = -1

    def moe(self, L: int, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
        # one layer's experts at a time (<= 256 x ~20 MB in float32): a whole-model cache reached 80 GB and was
        # OOM-killed next to the device path's pinned host pool
        if L != self._layer:
            self._cache.clear()
            self._layer = L
        return expert_parts_float(self._w, L, x, np.asarray(ids), self._cache)


def combine(parts: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """parts [T,k,2560] unweighted, wt [T,k] -> [T,2560] f32: rounded first product, then fmaf in rank order
    (native_moe.cu:66-97).  The f64 product of two f32 is exact, so each step is one f32 rounding of an exact
    fma up to rare double rounding."""
    parts = np.asarray(parts, F32)
    wt = np.asarray(wt, F32)
    out = (parts[:, 0] * wt[:, 0:1]).astype(F32)
    for j in range(1, parts.shape[1]):
        out = (parts[:, j].astype(F64) * wt[:, j:j + 1].astype(F64) + out.astype(F64)).astype(F32)
    return out


def moe(w, L: int, mixed: np.ndarray, experts=None, *, float_experts: bool = False, cache: dict | None = None,
        trace: dict | None = None) -> np.ndarray:
    """MoE output bo [T,2560] f32 for mixed [T,2560].

    experts: fnx.cpu_experts.CpuExperts (Strata's CPU kernels).  float_experts=True uses f32 dequantized expert
    weights (w.model.get_expert) instead; `cache` then holds them across calls.  `trace` (dict) receives the
    intermediates: logits, ids, wt, p, sh, gs, parts, routed."""
    x = np.ascontiguousarray(mixed, F32)
    logits = router_logits(w, L, x)
    ids, wt, p = route(logits)
    sh, gs = shared_expert(w, L, x)
    if float_experts:
        parts = expert_parts_float(w.model, L, x, ids, cache)
    else:
        if experts is None:
            raise ValueError("moe: pass a CpuExperts (or float_experts=True)")
        parts = experts.moe(L, x, ids)
    routed = combine(parts, wt)
    bo = (routed + sh).astype(F32)
    if trace is not None:
        trace.update(logits=logits, ids=ids, wt=wt, p=p, sh=sh, gs=gs, parts=parts, routed=routed)
    return bo
