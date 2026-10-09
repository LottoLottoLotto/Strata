"""Layer-1 per-layer n-gram embedding (PLE), numpy float32 reference (spec Part 2 §2-§7, spec-ple.md).

Host side (shared with the TT path):
  ngram_rows(tokens, prev)  uint64 n-gram hash -> 16 table rows per token (Strata src/kernels/ngram.cpp:71-97).
  PleTable(model).gather(rows) -> e_ple [T, 2560]: per_layer_token_embd rows (IQ4_NL, 160 values = 90 B per row,
                              split-half nibble order), head-slowest: e[h*160:(h+1)*160] = row[h].
Block (decode semantics, float32):
  key   = rms_streams(Wk @ e, g_key)       Wk (10240, 2560) BF16
  query = rms_streams(R, g_query)          R = un-normalized residual after layer 0's FFN write
  value = Wv @ e                           Wv (2560, 2560) BF16
  s[c]  = (sum_d key[c,d]*query[c,d]) / sqrt(2560)
  gate  = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
  gated = value[None, :] * gate[:, None]                     [4, 2560]
  normalized = rms_streams(gated, g_conv)                     conv input and the new history row
  conv  = Wc[:,0]*x(t-9) + Wc[:,1]*x(t-6) + Wc[:,2]*x(t-3) + Wc[:,3]*normalized    (dilation 3, per channel)
  R_new = R + (gated + silu(conv))
State: hist [10240, 9] float32 (column 8 newest; only columns 0, 3, 6 are read), prev = (two_back, one_back).
Verify-window rollback: PleState keeps the last call's history rows (ext = [9+T rows], the history after row t is
ext[t+1 : t+10]) and the hash window before it; commit(n_keep) rolls both back to after the first n_keep tokens.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .hc import EPS, HC, N, D, rms_streams, sigmoid

HIST = 9
PLE_DIM = 160
PLE_HEADS = 16
EOS = 248044
MULT = (23703573157769, 20109073645365, 8052911324071)
VOCAB = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
         20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171)
OFFS = tuple(int(sum(VOCAB[:h])) for h in range(PLE_HEADS))
U64 = (1 << 64) - 1
ROW_BYTES = 90                               # 5 IQ4_NL blocks of 18 B
TABLE = "per_layer_token_embd.weight"
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
SCALE = np.float32(1.0 / np.sqrt(np.float32(N)))   # 0.019764235f, computed on the host in fp32 by Strata


def ngram_rows_one(tok: int, prev) -> list[int]:
    """16 table rows of one token; prev = (two_back, one_back), -1 = no predecessor."""
    ctx, cut = [int(tok)], False
    for s in (1, 2):
        t = -1 if cut else int(prev[2 - s])
        cut = cut or t < 0 or t == EOS
        ctx.append(EOS if cut else t)
    rows = []
    for n in (2, 3):
        mixed = (ctx[0] * MULT[0]) & U64
        for j in range(1, n):
            mixed ^= (ctx[j] * MULT[j]) & U64
        for g in range(8):
            h = (n - 2) * 8 + g
            rows.append(mixed % VOCAB[h] + OFFS[h])
    return rows


def ngram_rows(tokens, prev=(-1, -1)):
    """Rows of consecutive tokens starting from window `prev` -> (uint32 [T, 16], new prev)."""
    prev = (int(prev[0]), int(prev[1]))
    out = np.empty((len(tokens), PLE_HEADS), np.uint32)
    for i, tok in enumerate(tokens):
        out[i] = ngram_rows_one(tok, prev)
        prev = (prev[1], int(tok))
    return out, prev


def iq4nl_rows(raw: np.ndarray) -> np.ndarray:
    """uint8 [n, 90] -> float32 [n, 160] (d * KV[nibble]; low nibbles -> elements 0..15, high -> 16..31)."""
    b = np.asarray(raw, np.uint8).reshape(-1, 5, 18)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)          # [n, 5, 1]
    q = b[:, :, 2:]
    return np.concatenate([d * KV_IQ4NL[q & 15], d * KV_IQ4NL[q >> 4]], axis=2).reshape(-1, PLE_DIM)


class PleTable:
    """Host row gather from the shard holding per_layer_token_embd (memory-mapped, never fully read)."""

    def __init__(self, model):
        self.model = model
        fi, t = model.index[TABLE]
        assert t.type_name == "IQ4_NL" and tuple(t.shape) == (PLE_DIM, 320001536), (t.type_name, t.shape)
        self.n_rows = int(t.shape[1])
        self.raw = model.raw(TABLE)
        self.path = Path(model.files[fi].path)
        self.base = model.files[fi].data_start + t.offset

    def row_available(self, rows) -> np.ndarray:
        """Whether each row's 90 bytes are on disk yet (the pdl `.done` sidecar lists finished 64 MiB chunks)."""
        rows = np.asarray(rows, np.int64)
        done_path = Path(str(self.path) + ".done")
        if not done_path.exists():
            return np.ones(rows.shape, bool)
        chunk = 64 << 20
        done = np.zeros((self.path.stat().st_size + chunk - 1) // chunk + 1, bool)
        done[[int(x) for x in done_path.read_text().split()]] = True
        lo = self.base + rows * ROW_BYTES
        return done[lo // chunk] & done[(lo + ROW_BYTES - 1) // chunk]

    def gather(self, rows) -> np.ndarray:
        """uint32 [T, 16] rows -> float32 [T, 2560] (head-slowest)."""
        rows = np.asarray(rows, np.int64)
        assert rows.ndim == 2 and rows.shape[1] == PLE_HEADS and rows.max() < self.n_rows
        idx = rows.reshape(-1, 1) * ROW_BYTES + np.arange(ROW_BYTES)
        return iq4nl_rows(self.raw[idx]).reshape(rows.shape[0], PLE_HEADS * PLE_DIM)


class PleState:
    """Per-sequence PLE state: conv history and the hash window, with the last call's per-row snapshots for a verify
    window's rollback (Strata verify.cpp:787-800 hist_snap_[t], commit verify.cpp:1595 + 2036-2040)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = np.zeros((D, HIST), np.float32)    # column 8 newest
        self.prev = (-1, -1)                            # (two_back, one_back)
        self.window = None                              # (prev before the last rows() call, its tokens)
        self.ext = None                                 # [9+T, D]: the last ple_block's history + its T rows

    def rows(self, tokens) -> np.ndarray:
        """Hash `tokens` from the current window and advance it."""
        self.window = ((int(self.prev[0]), int(self.prev[1])), [int(t) for t in tokens])
        self.ext = None
        rows, self.prev = ngram_rows(tokens, self.prev)
        return rows

    def snapshot(self, t: int) -> np.ndarray:
        """History after row t of the last call ([10240, 9], column 8 newest) = Strata's hist_snap_[t]."""
        return np.ascontiguousarray(self.ext[t + 1:t + 1 + HIST].T)

    def commit(self, n_keep: int) -> None:
        """Keep the first n_keep rows of the last call (rows() + ple_block()), dropping the rest: history <- the
        snapshot after row n_keep-1, hash window <- advanced by the kept tokens only."""
        if self.window is None or self.ext is None:
            raise RuntimeError("PleState.commit: no hashed and computed window to commit")
        prev0, toks = self.window
        if not 1 <= n_keep <= len(toks) or self.ext.shape[0] != HIST + len(toks):
            raise ValueError(f"commit({n_keep}) of a {len(toks)}-row window")
        if n_keep < len(toks):
            self.hist = self.snapshot(n_keep - 1)
            _, self.prev = ngram_rows(toks[:n_keep], prev0)
        self.window, self.ext = None, None


def ple_weights(w):
    """(Wk [10240,2560], Wv [2560,2560], g_key, g_query, g_conv [10240], Wc [10240,4]) as float32."""
    Wk = w.get("blk.1.ple_key.weight")
    Wv = w.get("blk.1.ple_value.weight")
    gk = w.get("blk.1.ple_norm_key.weight")
    gq = w.get("blk.1.ple_norm_query.weight")
    gc = w.get("blk.1.ple_norm_conv.weight")
    Wc = w.get("blk.1.ple_conv1d.weight")
    assert Wk.shape == (D, N) and Wv.shape == (N, N) and Wc.shape == (D, 4), (Wk.shape, Wv.shape, Wc.shape)
    return Wk, Wv, gk, gq, gc, Wc


def ple_block(w, e_ple, R, state: PleState, parts: dict | None = None):
    """e_ple [T, 2560], R [T, 4, 2560] (after layer 0) -> R_new [T, 4, 2560]; advances state.hist by T rows.

    `w` is a fnx.weights.Weights or the tuple from ple_weights(). Rows are processed as one causal sequence: row t's
    conv taps t-9, t-6, t-3 come from earlier rows of this call or from state.hist.  The call's history rows are kept
    in state.ext (per-row snapshots for PleState.commit).  If `parts` is a dict, the intermediates (key, query, value,
    gate, gated, normalized, conv) are stored in it.
    """
    Wk, Wv, gk, gq, gc, Wc = w if isinstance(w, tuple) else ple_weights(w)
    e = np.asarray(e_ple, np.float32).reshape(-1, N)
    R = np.asarray(R, np.float32).reshape(-1, HC, N)
    T = e.shape[0]
    key = rms_streams(e @ np.asarray(Wk, np.float32).T, gk)          # [T,4,N]
    query = rms_streams(R, gq)
    value = (e @ np.asarray(Wv, np.float32).T).astype(np.float32)     # [T,N]
    s = ((key * query).sum(axis=-1, dtype=np.float32) * SCALE).astype(np.float32)   # [T,4]
    mag = np.sqrt(np.maximum(np.abs(s), np.float32(1e-6))).astype(np.float32)
    gate = sigmoid(np.sign(s) * mag)
    gated = (value[:, None, :] * gate[:, :, None]).astype(np.float32)               # [T,4,N]
    normalized = rms_streams(gated, gc).reshape(T, D)
    ext = np.concatenate([np.asarray(state.hist, np.float32).T, normalized], axis=0)   # [9+T, D]
    Wc = np.asarray(Wc, np.float32)
    conv = ext[0:T] * Wc[:, 0]
    for k in (1, 2, 3):
        conv = (conv + ext[3 * k:3 * k + T] * Wc[:, k]).astype(np.float32)
    act = (conv * sigmoid(conv)).astype(np.float32)
    R_new = (R + (gated + act.reshape(T, HC, N))).astype(np.float32)
    state.hist = np.ascontiguousarray(ext[T:T + HIST].T)
    state.ext = ext
    if parts is not None:
        parts.update(key=key, query=query, value=value, gate=gate, gated=gated, normalized=normalized, conv=conv)
    return R_new
