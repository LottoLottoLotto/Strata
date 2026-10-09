"""Token embedding (IQ4_XS row gather) and LM head (Q6_K output.weight), numpy float32 reference.

embed: token_embd.weight ggml [2560, 248320] IQ4_XS; row `tok` = 10 blocks x 136 B = 1360 B, dequantized plainly to
    float32 (no scale, no norm; spec Part 2 §1). Every product is exact in float32.
head:  logits[t, v] = sum_d W[v, d] * head_in[t, d], W = dequantized output.weight (248320, 2560); no softcap, no scale
    (spec §B). Strata's decode quantizes head_in to q8_1 first; this reference does not (contract numerics target).
"""
from __future__ import annotations

import numpy as np

N = 2560
N_VOCAB = 248320
QK = 256
BLOCK = 136                                   # IQ4_XS: d f16 | scales_h u16 | scales_l u8[4] | qs u8[128]
ROW_BYTES = (N // QK) * BLOCK                 # 1360
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


def iq4xs_rows(raw: np.ndarray) -> np.ndarray:
    """uint8 [n, 1360] -> float32 [n, 2560] (ggml dequantize_row_iq4_xs)."""
    b = np.asarray(raw, np.uint8).reshape(-1, BLOCK)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)                    # [nb, 1]
    sh = b[:, 2:4].copy().view(np.uint16).astype(np.int32)                       # [nb, 1]
    sl = b[:, 4:8].astype(np.int32)                                              # [nb, 4]
    ib = np.arange(8)
    ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)   # [nb, 8], 0..63
    dl = d * (ls - 32).astype(np.float32)                                        # [nb, 8]
    qs = b[:, 8:].reshape(-1, 8, 16)                                             # [nb, 8 sub-blocks, 16]
    y = np.concatenate([KV_IQ4NL[qs & 0xF], KV_IQ4NL[qs >> 4]], axis=2) * dl[:, :, None]   # [nb, 8, 32]
    return y.reshape(-1, N).astype(np.float32)


def embed(w, tokens) -> np.ndarray:
    """tokens [T] -> float32 [T, 2560]. `w` is fnx.weights.Weights (or anything with `.model`)."""
    model = w.model if hasattr(w, "model") else w
    t = model.info("token_embd.weight")
    assert t.type_name == "IQ4_XS" and tuple(t.shape) == (N, N_VOCAB), (t.type_name, t.shape)
    raw = model.raw("token_embd.weight")
    tok = np.asarray(tokens, np.int64).reshape(-1)
    assert tok.min() >= 0 and tok.max() < N_VOCAB
    idx = tok[:, None] * ROW_BYTES + np.arange(ROW_BYTES)
    return iq4xs_rows(raw[idx])


def head_weight(w) -> np.ndarray:
    """Dequantized output.weight, float32 (248320, 2560) (memory-mapped from the fnx.weights cache)."""
    W = w.get("output.weight")
    assert W.shape == (N_VOCAB, N), W.shape
    return W


def head(w, head_in, chunk: int = 32768) -> np.ndarray:
    """head_in [T, 2560] -> logits float32 [T, 248320] (vocab-chunked to bound the memory-mapped reads)."""
    W = head_weight(w)
    x = np.asarray(head_in, np.float32).reshape(-1, N)
    out = np.empty((x.shape[0], N_VOCAB), np.float32)
    for v0 in range(0, N_VOCAB, chunk):
        out[:, v0:v0 + chunk] = x @ np.asarray(W[v0:v0 + chunk]).T
    return out


def argmax_lowest(logits) -> np.ndarray:
    """Greedy pick per row: highest logit, ties to the lowest index, NaN never picked."""
    x = np.where(np.isnan(logits), -np.inf, np.asarray(logits, np.float32))
    return np.argmax(x, axis=-1)
