"""Gated DeltaNet mixer (36 layers, L % 4 != 3), numpy float32 reference (spec Part 4 §2, spec-gdn.md §2).

Strata's decode (verify window) math, sequential over the T tokens of a call, with dequantized GGUF weights and
float32 activations (no q8_1 activation emulation, CONTRACT "Numerics target"). Per token, x = mixed[t]:

    qkv  = Wqkv @ x  (10240)     z = Wz @ x  (6144)     a = Wa @ x, b = Wb @ x  (48; BF16 weights, f32 x)
    s[c] = hist[c,0]*w[c,0] + hist[c,1]*w[c,1] + hist[c,2]*w[c,2] + qkv[c]*w[c,3]   (hist oldest first, raw qkv)
    h    = silu(s); hist <- [hist[:,1], hist[:,2], qkv]
    q = h[0:2048], k = h[2048:4096] (16 heads x 128), each head / sqrt(sum(head^2) + 1e-6); v = h[4096:] (48 x 128)
    gate = softplus(a + dt) * ssm_a  (softplus threshold 20, ssm_a stored negative), g = exp(gate), beta = sigmoid(b)
    v-head h uses q/k head h % 16 (GGUF "tiled" order, NOT h // 3)
    kv = S_h^T k ; delta = (v - g*kv) * beta ; S_h <- g*S_h + k (x) delta ; o = S_h^T q / sqrt(128)
    y  = o * rsqrt(mean_j o^2 + 1e-6) * ssm_norm * sigmoid(z)     (per v-head, over its 128 values)
    bo = Wout @ y

State (per GDN layer, zeroed at sequence start): S f32[128 (i=key), 48 (h=v-head), 128 (j=value)],
conv f32[10240, 3] (oldest first).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

N = 2560
HK, HV, DK = 16, 48, 128
QK = HK * DK            # 2048
CONV_CH = 2 * QK + HV * DK  # 10240
VALUE_DIM = HV * DK     # 6144
D_CONV = 4
EPS = np.float32(1e-6)
f32 = np.float32


def _state_s() -> np.ndarray:
    return np.zeros((DK, HV, DK), np.float32)


def _state_conv() -> np.ndarray:
    return np.zeros((CONV_CH, D_CONV - 1), np.float32)


@dataclass
class GdnState:
    """One GDN layer's per-sequence state; `gdn` mutates it in place."""
    S: np.ndarray = field(default_factory=_state_s)        # [i=key 128, h=v-head 48, j=value 128]
    conv: np.ndarray = field(default_factory=_state_conv)  # [10240, 3] raw pre-conv qkv, oldest first

    def reset(self) -> None:
        self.S[...] = 0
        self.conv[...] = 0


@dataclass(frozen=True)
class GdnWeights:
    qkv: np.ndarray    # (10240, 2560)
    z: np.ndarray      # (6144, 2560)  attn_gate
    a: np.ndarray      # (48, 2560)    ssm_alpha (BF16 values)
    b: np.ndarray      # (48, 2560)    ssm_beta  (BF16 values)
    conv: np.ndarray   # (10240, 4)    tap 3 = current input
    dt: np.ndarray     # (48,)         ssm_dt.bias
    A: np.ndarray      # (48,)         ssm_a (negative)
    norm: np.ndarray   # (128,)        ssm_norm
    out: np.ndarray    # (2560, 6144)  ssm_out


@lru_cache(maxsize=48)
def gdn_weights(w, L: int) -> GdnWeights:
    """Layer L's GDN tensors from fnx.weights.Weights, f32, numpy (out, in) orientation, shape-checked."""
    g = lambda s: np.asarray(w.layer(L, s), np.float32)
    gw = GdnWeights(qkv=g("attn_qkv.weight"), z=g("attn_gate.weight"), a=g("ssm_alpha.weight"),
                    b=g("ssm_beta.weight"), conv=g("ssm_conv1d.weight"), dt=g("ssm_dt.bias"), A=g("ssm_a"),
                    norm=g("ssm_norm.weight"), out=g("ssm_out.weight"))
    want = dict(qkv=(CONV_CH, N), z=(VALUE_DIM, N), a=(HV, N), b=(HV, N), conv=(CONV_CH, D_CONV), dt=(HV,),
                A=(HV,), norm=(DK,), out=(N, VALUE_DIM))
    for k, s in want.items():
        if getattr(gw, k).shape != s:
            raise ValueError(f"blk.{L} GDN {k}: shape {getattr(gw, k).shape}, expected {s}")
    return gw


def sigmoid(x):
    return (f32(1) / (f32(1) + np.exp(-np.asarray(x, np.float32)))).astype(np.float32)


def softplus20(x):
    x = np.asarray(x, np.float32)
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, f32(20))))).astype(np.float32)


def gdn(w, L: int, mixed: np.ndarray, state: GdnState, dbg: dict | None = None) -> np.ndarray:
    """mixed [T, 2560] f32 (tokens in sequence order) -> bo [T, 2560] f32; advances `state` by T tokens.
    `w` is fnx.weights.Weights (or a GdnWeights, e.g. with device-rounded projections, for parity tests).
    `dbg`, if given, receives per-token arrays qkv, h (post conv/silu/l2), gate, beta, z, o (pre-norm), y."""
    W = w if isinstance(w, GdnWeights) else gdn_weights(w, L)
    x = np.asarray(mixed, np.float32).reshape(-1, N)
    T = x.shape[0]
    qkv = (x @ W.qkv.T).astype(np.float32)          # [T, 10240]
    z = (x @ W.z.T).astype(np.float32)              # [T, 6144]
    a = (x @ W.a.T).astype(np.float32)              # [T, 48]
    b = (x @ W.b.T).astype(np.float32)
    gate = (softplus20(a + W.dt) * W.A).astype(np.float32)
    decay = np.exp(gate).astype(np.float32)
    beta = sigmoid(b)
    kmap = np.arange(HV) % HK
    S, hist = state.S, state.conv
    ys = np.empty((T, VALUE_DIM), np.float32)
    rec = {k: [] for k in ("h", "o")} if dbg is not None else None
    for t in range(T):
        win = np.concatenate([hist, qkv[t][:, None]], axis=1)              # [10240, 4] x_{t-3..t}
        s = (win * W.conv).sum(axis=1, dtype=np.float32)
        h = (s * sigmoid(s)).astype(np.float32)
        hist[...] = win[:, 1:]
        qk = h[:2 * QK].reshape(2 * HK, DK)
        qk = (qk / np.sqrt((qk * qk).sum(axis=1, keepdims=True, dtype=np.float32) + EPS)).astype(np.float32)
        q, k, v = qk[:HK][kmap], qk[HK:][kmap], h[2 * QK:].reshape(HV, DK)   # q/k per v-head [48, 128]
        g = decay[t]
        kv = np.einsum("ihj,hi->hj", S, k)
        delta = ((v - g[:, None] * kv) * beta[t][:, None]).astype(np.float32)
        S[...] = g[None, :, None] * S + k.T[:, :, None] * delta[None]
        o = (np.einsum("ihj,hi->hj", S, q) / np.sqrt(f32(DK))).astype(np.float32)
        y = o / np.sqrt((o * o).mean(axis=1, keepdims=True, dtype=np.float32) + EPS)
        ys[t] = (y * W.norm[None] * sigmoid(z[t].reshape(HV, DK))).reshape(-1)
        if rec is not None:
            rec["h"].append(np.concatenate([qk.reshape(-1), h[2 * QK:]]))
            rec["o"].append(o.reshape(-1).copy())
    bo = (ys @ W.out.T).astype(np.float32)
    if dbg is not None:
        dbg.update(qkv=qkv, gate=gate, beta=beta, z=z, y=ys, h=np.stack(rec["h"]), o=np.stack(rec["o"]))
    return bo
