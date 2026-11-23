"""Hyper-connection read / write, numpy float32 reference (spec Part 3 §3, spec-hc.md §3).

R is the 4-stream residual `[T, 4, 2560]` float32 (stream c, element d = flat index c*2560+d).

hc_read(R, g, Wd, Wu, Wi):
    rs    = 1/sqrt(mean_d R[c]^2 + 1e-6)            per stream (over 2560, not 10240)
    xn    = (R * g.reshape(4,2560)) * rs
    lo    = silu((Wd @ xn.ravel()) / 4)              Wd (320, 10240)
    gate  = (Wu @ lo).reshape(4,2560)                Wu (10240, 320)
    mixed = mean_c(xn[c] * sigmoid(gate[c]))         (2560,)
    inj   = Wi @ xn.ravel()                          Wi (4, 10240), raw (no activation)
hc_write(R, bo, inj) = R + (2*sigmoid(inj/4))[:, None] * bo[None, :]
"""
from __future__ import annotations

import numpy as np

N, HC, LR = 2560, 4, 320
D = N * HC
EPS = np.float32(1e-6)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    with np.errstate(over="ignore"):                       # exp(-x) = inf for x << 0 gives the exact limit 0
        return (np.float32(1) / (np.float32(1) + np.exp(-x))).astype(np.float32)


def rms_streams(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-stream RMSNorm: x [T, 4, 2560] (or [T, 10240]), g [10240] -> [T, 4, 2560], computed (x*g)*rs."""
    x = np.asarray(x, np.float32).reshape(-1, HC, N)
    rs = (np.float32(1) / np.sqrt((x * x).mean(axis=-1, dtype=np.float32) + EPS)).astype(np.float32)
    return ((x * np.asarray(g, np.float32).reshape(HC, N)) * rs[..., None]).astype(np.float32)


def hc_read(R, g, Wd, Wu, Wi=None):
    """R [T,4,2560] f32 -> (mixed [T,2560] f32, inj [T,4] f32 or None)."""
    R = np.asarray(R, np.float32).reshape(-1, HC, N)
    xn = rms_streams(R, g)                                   # [T,4,N]
    x = xn.reshape(-1, D)
    p = (x @ np.asarray(Wd, np.float32).T) / np.float32(HC)  # [T,320]
    lo = (p * sigmoid(p)).astype(np.float32)
    gate = (lo @ np.asarray(Wu, np.float32).T).reshape(-1, HC, N)
    mixed = ((xn * sigmoid(gate)).sum(axis=1, dtype=np.float32) / np.float32(HC)).astype(np.float32)
    inj = None if Wi is None else (x @ np.asarray(Wi, np.float32).T).astype(np.float32)
    return mixed, inj


def hc_write(R, bo, inj):
    """R [T,4,2560] + (2*sigmoid(inj/4))[T,4,1] * bo[T,1,2560]."""
    R = np.asarray(R, np.float32).reshape(-1, HC, N)
    w = (np.float32(2) * sigmoid(np.asarray(inj, np.float32) / np.float32(HC))).astype(np.float32)
    return (R + w[:, :, None] * np.asarray(bo, np.float32).reshape(-1, 1, N)).astype(np.float32)


def hc_weights(w, prefix: str):
    """(g, Wd, Wu, Wi|None) for prefix 'blk.{L}.hc_attn' / 'blk.{L}.hc_ffn' / 'output_hc' from fnx.weights.Weights."""
    g = w.get(f"{prefix}_norm.weight")
    Wd = w.get(f"{prefix}_down.weight")
    Wu = w.get(f"{prefix}_up.weight")
    Wi = w.get(f"{prefix}_inject.weight") if f"{prefix}_inject.weight" in w else None
    assert g.shape == (D,) and Wd.shape == (LR, D) and Wu.shape == (D, LR), (g.shape, Wd.shape, Wu.shape)
    assert Wi is None or Wi.shape == (HC, D)
    return g, Wd, Wu, Wi
