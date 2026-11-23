"""Numpy reference of a Qwen Sparse Attention (QSA) layer, Strata's decode (verify-window) semantics.

Spec: flashnext-spec.md Part 5 (= spec-qsa.md). float32 math with dequantized weights, no q8_1 activation emulation
(CONTRACT "Numerics target"); fp16 KV cache (Strata's default `--kv fp16`); RoPE angles from the float64 table
(spec §4.2, recommended reference; = Strata's STRATA_ROPE_TABLE=1).

One call takes T consecutive tokens (a verify window, T=1 included): every row's K/V and indexer key are appended
first, then each row t selects and attends with its own step n_kv = pos_t + 1 (spec §7, verify.cpp:945-1066).

Dense threshold (spec §4.6/§5): the selection width is min(n_kv, IDX_TOP_K + COMPRESS - 1 = 2051)
(qsa.hpp:133-136), and every top-k kernel returns the identity when n_kv <= width (qsa_select.cu:69-72, 541, 640,
898). So positions p <= 2050 attend densely to all cells 0..p; from p = 2051 on exactly 2051 cells are attended.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import COMPRESS, EPS, HEAD_DIM, IDX_DIM, IDX_HEADS, IDX_TOP_K, N_EMBD, N_HEAD, N_HEAD_KV, ROPE_BASE, ROPE_DIM

f32, f16 = np.float32, np.float16
GROUP = N_HEAD // N_HEAD_KV                  # q head h uses kv head h // 12 (qsa.hpp:98)
SEL_WIDTH = IDX_TOP_K + COMPRESS - 1        # 2051 (qsa.hpp:133-136)
DENSE_MAX_POS = SEL_WIDTH - 1                # 2050: last position whose selection is all cells
SCALE = f32(1.0 / 16.0)                      # 1/sqrtf(256) (qsa_decode_attn.cu:501)
TAIL_BIAS = f32(1e9)                         # qsa_select.cu:53


def rope_cos_sin(pos) -> tuple[np.ndarray, np.ndarray]:
    """float64 angle table pos * 1e7^(-2i/64), i < 32, cos/sin rounded to f32: shapes [len(pos), 32]."""
    i = np.arange(ROPE_DIM // 2, dtype=np.float64)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * ROPE_BASE ** (-2.0 * i / ROPE_DIM)
    return np.cos(ang).astype(f32), np.sin(ang).astype(f32)


def rms(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-row RMSNorm over the last dim: (rsqrt(ss/D + eps) * x) * gamma (native_rope.cu:99-106)."""
    ss = (x * x).sum(-1, keepdims=True, dtype=f32)
    return ((f32(1) / np.sqrt(ss / f32(x.shape[-1]) + f32(EPS))).astype(f32) * x) * g


def rope64(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """NEOX rotation of dims 0..63, pairs (i, i+32); x [..., D]; c, s broadcastable to [..., 32]."""
    y = x.copy()
    a, b = x[..., :32], x[..., 32:64]
    y[..., :32] = a * c - b * s
    y[..., 32:64] = a * s + b * c
    return y


class QsaWeights:
    """Dequantized f32 weights of one QSA layer (numpy order, (n_out, n_in))."""

    def __init__(self, w, L: int):
        g = lambda s: np.asarray(w.layer(L, s), dtype=f32)
        self.q = g("attn_q.weight")              # (12288, 2560): per head [q 256 | gate 256]
        self.k = g("attn_k.weight")              # (512, 2560)
        self.v = g("attn_v.weight")              # (512, 2560)
        self.o = g("attn_output.weight")         # (2560, 6144), input index h*256 + d
        self.qn = g("attn_q_norm.weight")        # (256,)
        self.kn = g("attn_k_norm.weight")
        self.iq = g("indexer.q_proj.weight")     # (512, 2560) BF16 values
        self.ik = g("indexer.k_proj.weight")     # (128, 2560)
        self.iqn = g("indexer.q_norm.weight")    # (128,)
        self.ikn = g("indexer.k_norm.weight")


@lru_cache(maxsize=16)
def layer_weights(w, L: int) -> QsaWeights:
    return QsaWeights(w, L)


class QsaState:
    """Per-layer, per-sequence state (spec §3): fp16 K/V cells and the indexer's tail / dead / pooled keys."""

    def __init__(self, max_cells: int = 4096):
        self.max_cells = max_cells
        self.K = np.zeros((max_cells, N_HEAD_KV, HEAD_DIM), f16)        # post-norm, post-RoPE
        self.V = np.zeros((max_cells, N_HEAD_KV, HEAD_DIM), f16)
        self.tail = np.zeros((COMPRESS - 1, IDX_DIM), f32)              # f16-rounded raw keys of the open block
        self.dead = np.zeros(IDX_DIM, f32)                               # normed key of cell 0
        self.pooled = np.zeros((max_cells // COMPRESS + 2, IDX_DIM), f32)
        self.last_ids: list[np.ndarray] = []                             # selected cells of the last call's rows

    def reset(self) -> None:
        self.__init__(self.max_cells)


def block_scores(keys: np.ndarray, qi: np.ndarray) -> np.ndarray:
    """Indexer block scores (spec §4.5): relu(key . qi[h]) summed over the 4 heads; keys [n, 128], qi [4, 128] ->
    f32 [n].  Accumulated in float64 and rounded to f32 once, so a score is a function of (key, query) alone: any
    subset of keys reproduces the same bits.  A float32 BLAS product is not (its rounding depends on the matrix size
    and the row's place in the blocking: 75% of the scores of a 300-key subset differ in the last bits from the full
    product), and Strata's own f32 order (qsa_select.cu: per-lane 4-product partials, xor-butterfly) is a third
    rounding.  The TT module's exact host recompute of near ties (fnx.tt.qsa) uses this function."""
    d = keys.astype(np.float64) @ qi.astype(np.float64).T
    return np.maximum(d, 0.0).sum(1).astype(f32)


def select(state: QsaState, qi: np.ndarray, p: int) -> np.ndarray:
    """Cells attended by the query at position p (spec §4.5-4.6); qi = normed+roped indexer query [4, 128]."""
    n = p + 1
    width = min(n, SEL_WIDTH)
    if n <= width:                                    # qsa_select.cu:69-72: the identity, ascending
        return np.arange(n)
    nb = n // COMPRESS
    keys = np.vstack([state.pooled[:nb], state.dead[None]])     # block nb (incomplete tail) scores `dead`
    bs = block_scores(keys, qi)                                 # relu per head, then summed
    if n % COMPRESS:
        bs[nb] += TAIL_BIAS
    cs = bs[np.minimum(np.arange(n) // COMPRESS, nb)]
    cs = np.where(np.isnan(cs), -np.inf, cs + f32(0))           # order_key: NaN lowest, -0 -> +0
    return np.sort(np.lexsort((np.arange(n), -cs))[:width])     # ties -> lowest index; ascending ids


def qsa(w, L: int, mixed: np.ndarray, pos, state: QsaState) -> np.ndarray:
    """QSA layer L on T consecutive tokens. mixed [T, 2560] f32, pos [T] absolute positions. Returns bo [T, 2560]
    f32 and appends the T tokens to `state`."""
    W = layer_weights(w, L)
    x = np.asarray(mixed, dtype=f32).reshape(-1, N_EMBD)
    pos = np.asarray(pos, dtype=np.int64).reshape(-1)
    T = x.shape[0]
    if pos.shape[0] != T:
        raise ValueError(f"pos has {pos.shape[0]} entries for {T} rows")
    if T > 1 and np.any(np.diff(pos) != 1):
        raise ValueError("a window's positions must be consecutive")
    if pos[0] < 0 or pos[-1] >= state.max_cells:
        raise ValueError(f"positions {pos[0]}..{pos[-1]} outside the state's {state.max_cells} cells")

    qf = (x @ W.q.T).reshape(T, N_HEAD, 2 * HEAD_DIM)          # [T, 24, 512]
    k = (x @ W.k.T).reshape(T, N_HEAD_KV, HEAD_DIM)
    v = (x @ W.v.T).reshape(T, N_HEAD_KV, HEAD_DIM)
    kr = (x @ W.ik.T).astype(f16).astype(f32)                  # raw indexer key, cached as f16
    qi = (x @ W.iq.T).reshape(T, IDX_HEADS, IDX_DIM)
    c, s = rope_cos_sin(pos)                                    # [T, 32]
    cb, sb = c[:, None, :], s[:, None, :]

    # appends for every row first (verify.cpp:945-998)
    state.K[pos] = rope64(rms(k, W.kn), cb, sb).astype(f16)
    state.V[pos] = v.astype(f16)
    for t, p in enumerate(pos):
        slot = p % COMPRESS
        if slot < COMPRESS - 1:
            state.tail[slot] = kr[t]
        if p == 0:                                              # spare key: block of 4 copies of cell 0, angle 0
            m = f32(0.25) * (((kr[t] + kr[t]) + kr[t]) + kr[t])
            state.dead = rms(m, W.ikn)
            state.pooled[0] = state.dead
        elif slot == COMPRESS - 1:                              # block b completes: pool, norm, rope at 4b
            b = p // COMPRESS
            m = f32(0.25) * (((state.tail[0] + state.tail[1]) + state.tail[2]) + kr[t])
            cb4, sb4 = rope_cos_sin([COMPRESS * b])
            state.pooled[b] = rope64(rms(m, W.ikn), cb4[0], sb4[0])
            state.pooled[b + 1] = state.dead

    q = rope64(rms(qf[:, :, :HEAD_DIM], W.qn), cb, sb)          # [T, 24, 256]
    gate = qf[:, :, HEAD_DIM:]                                  # raw projection
    qin = rope64(rms(qi, W.iqn), cb, sb)                        # [T, 4, 128]

    att = np.empty((T, N_HEAD, HEAD_DIM), f32)
    state.last_ids = []
    for t, p in enumerate(pos):
        ids = select(state, qin[t], int(p))
        state.last_ids.append(ids)
        Kf, Vf = state.K[ids].astype(f32), state.V[ids].astype(f32)        # [n, 2, 256]
        for j in range(N_HEAD_KV):
            qh = q[t, j * GROUP:(j + 1) * GROUP]                            # [12, 256]
            sc = (qh @ Kf[:, j].T) * SCALE                                  # [12, n]
            e = np.exp(sc - sc.max(1, keepdims=True))
            att[t, j * GROUP:(j + 1) * GROUP] = (e @ Vf[:, j]) / e.sum(1, keepdims=True, dtype=f32)
    a = att * (f32(1) / (f32(1) + np.exp(-gate)))                          # sigmoid gate
    return (a.reshape(T, N_HEAD * HEAD_DIM) @ W.o.T).astype(f32)
