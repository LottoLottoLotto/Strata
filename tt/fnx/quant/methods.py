"""BFP quantizers for expert matrices (torch, any device), batched over experts.

Orientation: a matrix W is [out, in] (math/GGUF numpy order); ttnn holds W.T = [in, out], so a BFP block is 16
consecutive OUTPUT rows at one input column. GPTQ walks input columns: quantizing column j quantizes whole blocks
(W[:, j] is 640/16 or 2560/16 blocks), so act-order and lazy batches need no group bookkeeping (ttquant RECIPE §1).

Block quantizer `blockq`: per block, the shared exponent is chosen among E_max - c for c in `clips` (c = -1 lets a
saturating block maximum round up to the next power of two; c > 0 clips) to minimise the block's squared error. Every
candidate is a snapped block (|q| <= 2^W - 1 on that exponent's grid), so tt-metal reproduces it exactly; c = 0 uses
the tt-metal rounding itself (RTN). bfp2 is ternary per block ({-s, 0, s}, s a power of two), so its exponent choice
matters most.
"""
from __future__ import annotations

import torch

from . import bfp

CLIPS = {"bfp2": (-1, 0, 1, 2, 3), "bfp4": (-1, 0, 1, 2), "bfp8": (0,)}


def _flush_denormals(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(v), v)


def blockq(x: torch.Tensor, fmt: str, clips=None) -> torch.Tensor:
    """x [..., n] (blocks of 16 along the last dim) -> snapped fp32 of x's shape, min-MSE exponent per block."""
    clips = CLIPS[fmt] if clips is None else clips
    if tuple(clips) == (0,):
        return bfp.rtn(x, fmt)
    W = bfp.MANT_BITS[fmt]
    qmax = float((1 << W) - 1)
    xb = x.float().reshape(*x.shape[:-1], -1, bfp.BLOCK)
    Emax = bfp.shared_exponent(x)
    best = bfp.rtn(x, fmt).reshape(xb.shape)
    best_err = (best - xb).pow(2).sum(-1)
    for c in clips:
        if c == 0:
            continue
        E = (Emax - c).clamp(1, 254)
        step = bfp.step_for_exponent(E, fmt).unsqueeze(-1)
        v = torch.round(xb / step).clamp_(-qmax, qmax).mul_(step)
        err = (v - xb).pow(2).sum(-1)
        m = err < best_err
        best = torch.where(m.unsqueeze(-1), v, best)
        best_err = torch.where(m, err, best_err)
    best = torch.where((Emax > 0).unsqueeze(-1), best, torch.zeros_like(best))
    return _flush_denormals(best.reshape(x.shape))


def rtn(W: torch.Tensor, fmt: str) -> torch.Tensor:
    """tt-metal's own conversion of W [..., out, in] (blocks along out)."""
    return bfp.rtn(W.float().transpose(-1, -2).contiguous(), fmt).transpose(-1, -2).contiguous()


def search(W: torch.Tensor, fmt: str) -> torch.Tensor:
    """Per-block min-MSE exponent (no activation statistics): the fallback for experts without calibration data."""
    return blockq(W.float().transpose(-1, -2).contiguous(), fmt).transpose(-1, -2).contiguous()


def _cholesky_inv_upper(Hbox: list, damp: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched: upper Cholesky factor of (H + d I)^-1 with d = damp * mean(diag H), escalating d x4 per expert until
    H + d I is positive definite. Takes ownership of H (Hbox = [H], emptied; H's diagonal is modified in place) so
    that at most two [B, n, n] buffers are live at a time. Returns (U, final relative damping per expert)."""
    H = Hbox.pop()
    B, n, _ = H.shape
    diag = H.diagonal(dim1=-2, dim2=-1)
    mean_d = diag.mean(-1)
    rel = torch.full((B,), damp, dtype=H.dtype, device=H.device)
    added = torch.zeros_like(rel)
    for _ in range(8):
        diag.add_(((rel - added) * mean_d)[:, None])
        added = rel.clone()
        L, info = torch.linalg.cholesky_ex(H)
        bad = info != 0
        if not bool(bad.any()):
            break
        del L
        rel = torch.where(bad, rel * 4, rel)
    else:
        raise RuntimeError("H not positive definite even with heavy damping")
    del H, diag
    Hinv = torch.cholesky_inverse(L)
    del L
    U, info = torch.linalg.cholesky_ex(Hinv, upper=True)
    del Hinv
    if bool((info != 0).any()):
        raise RuntimeError("Cholesky of H^-1 failed")
    return U, rel


def gptq(W: torch.Tensor, H: torch.Tensor, fmt: str, damp: float = 0.01, blocksize: int = 128, act_order: bool = True,
         clips=None) -> torch.Tensor:
    """Batched GPTQ with the BFP block quantizer. W [B, out, in], H [B, in, in] (= sum x x^T over the expert's inputs).
    Returns snapped fp32 [B, out, in]. Runs in fp32 on W's device."""
    W = W.float().clone()
    H = H.float().clone()
    B, R, C = W.shape
    diag = H.diagonal(dim1=-2, dim2=-1)
    dead = diag == 0
    if bool(dead.any()):
        diag.masked_fill_(dead, 1.0)                    # view: writes H's diagonal
        W.masked_fill_(dead[:, None, :], 0.0)
    perm = None
    if act_order:
        perm = torch.argsort(H.diagonal(dim1=-2, dim2=-1), dim=-1, descending=True)   # [B, C]
        W = torch.gather(W, 2, perm[:, None, :].expand(B, R, C))
        H = torch.gather(H, 1, perm[:, :, None].expand(B, C, C))
        H = torch.gather(H, 2, perm[:, None, :].expand(B, C, C))
    box = [H]
    del H, diag
    U, _ = _cholesky_inv_upper(box, damp)
    Q = torch.empty_like(W)
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.empty_like(W1)
        U1 = U[:, i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, :, j]
            q = blockq(w, fmt, clips)
            Q[:, :, i1 + j] = q
            e = (w - q) / U1[:, j, j][:, None]
            W1[:, :, j:] -= e[:, :, None] * U1[:, j, j:][:, None, :]
            Err1[:, :, j] = e
        if i2 < C:
            W[:, :, i2:] -= torch.bmm(Err1, U[:, i1:i2, i2:])
    if perm is not None:
        Q = torch.gather(Q, 2, torch.argsort(perm, dim=-1)[:, None, :].expand(B, R, C))
    return Q


def block_step(x: torch.Tensor, fmt: str, clips=None) -> torch.Tensor:
    """x [..., n] (blocks of 16 along the last dim) -> per block the grid step of its min-MSE exponent among E_max - c
    (c in clips, round-to-nearest on each grid), [..., n/16]; 0 for an all-zero block."""
    clips = CLIPS[fmt] if clips is None else clips
    qmax = float((1 << bfp.MANT_BITS[fmt]) - 1)
    xb = x.float().reshape(*x.shape[:-1], -1, bfp.BLOCK)
    Emax = bfp.shared_exponent(x)
    best_step, best_err = None, None
    for c in clips:
        step = bfp.step_for_exponent((Emax - c).clamp(1, 254), fmt)
        err = (torch.round(xb / step.unsqueeze(-1)).clamp_(-qmax, qmax) * step.unsqueeze(-1) - xb).pow(2).sum(-1)
        if best_step is None:
            best_step, best_err = step, err
        else:
            m = err < best_err
            best_step, best_err = torch.where(m, step, best_step), torch.where(m, err, best_err)
    return torch.where(Emax > 0, best_step, torch.zeros_like(best_step))


def gptq_rows(W: torch.Tensor, H: torch.Tensor, fmt: str, damp: float = 0.01, blocksize: int = 128,
              group_order: bool = True, clips=None) -> torch.Tensor:
    """Batched GPTQ for W [B, out, in] held by ttnn AS IS (the cpu MoE's resident experts, fnx.tt.moe.DeviceExperts):
    a BFP block is 16 consecutive INPUT columns of one output row, so a block's exponent is fixed when GPTQ reaches
    its first column (block_step of the error-updated values) and its 16 columns are quantized on that grid in turn.
    group_order: the blocks in descending order of their summed H diagonal (act-order over whole blocks).  H [B, in,
    in].  Returns snapped fp32 [B, out, in] (tt-metal keeps it exactly: bfp.is_snapped)."""
    W = W.float().clone()
    H = H.float().clone()
    B, R, C = W.shape
    K = bfp.BLOCK
    assert C % K == 0 and blocksize % K == 0
    qmax = float((1 << bfp.MANT_BITS[fmt]) - 1)
    diag = H.diagonal(dim1=-2, dim2=-1)
    dead = diag == 0
    if bool(dead.any()):
        diag.masked_fill_(dead, 1.0)
        W.masked_fill_(dead[:, None, :], 0.0)
    perm = None
    if group_order:
        g = torch.argsort(H.diagonal(dim1=-2, dim2=-1).reshape(B, C // K, K).sum(-1), dim=-1, descending=True)
        perm = (g[:, :, None] * K + torch.arange(K, device=W.device)).reshape(B, C)
        W = torch.gather(W, 2, perm[:, None, :].expand(B, R, C))
        H = torch.gather(H, 1, perm[:, :, None].expand(B, C, C))
        H = torch.gather(H, 2, perm[:, None, :].expand(B, C, C))
    box = [H]
    del H, diag
    U, _ = _cholesky_inv_upper(box, damp)
    Q = torch.empty_like(W)
    step = safe = None
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.empty_like(W1)
        U1 = U[:, i1:i2, i1:i2]
        for j in range(i2 - i1):
            if j % K == 0:
                step = block_step(W1[:, :, j:j + K], fmt, clips)[..., 0]           # [B, out]
                safe = torch.where(step > 0, step, torch.ones_like(step))
            w = W1[:, :, j]
            q = torch.where(step > 0, torch.round(w / safe).clamp_(-qmax, qmax) * safe, torch.zeros_like(w))
            Q[:, :, i1 + j] = q
            e = (w - q) / U1[:, j, j][:, None]
            W1[:, :, j:] -= e[:, :, None] * U1[:, j, j:][:, None, :]
            Err1[:, :, j] = e
        if i2 < C:
            W[:, :, i2:] -= torch.bmm(Err1, U[:, i1:i2, i2:])
    if perm is not None:
        Q = torch.gather(Q, 2, torch.argsort(perm, dim=-1)[:, None, :].expand(B, R, C))
    return _flush_denormals(Q)


def rel_err_h(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Batched tr(dW H dW^T) / tr(W H W^T): relative output MSE under the second moment H. W, Q [B, out, in]."""
    dW = (W - Q).float()
    num = (torch.bmm(dW, H) * dW).sum((-1, -2))
    den = (torch.bmm(W.float(), H) * W.float()).sum((-1, -2))
    return num / den.clamp_min(1e-30)


def channel_transform(Wg: torch.Tensor, Wu: torch.Tensor, mode: str):
    """Exact re-parametrisation of experts [B, 640, 2560] gate/up (down [B, 2560, 640] follows): intermediate channel
    permutation `perm` [B, 640] and power-of-two up-row scales `k` [B, 640] (up row r *= 2^k_r, down column r
    *= 2^-k_r, so silu(g) * u -> 2^k silu(g) * u and the down product is unchanged). mode: "none", "scale" (equalise
    up-row RMS to the expert's median), "perm" (sort channels by gate-row RMS so a BFP block of 16 output rows holds
    similar magnitudes), "scale+perm". Returns (perm, k)."""
    B, R, _ = Wg.shape
    k = torch.zeros(B, R, device=Wg.device)
    perm = torch.arange(R, device=Wg.device).expand(B, R).clone()
    if "scale" in mode:
        r = Wu.pow(2).mean(-1).sqrt().clamp_min(1e-30)
        k = torch.round(torch.log2(r.median(-1, keepdim=True).values / r))
    if "perm" in mode:
        perm = torch.argsort(Wg.pow(2).mean(-1), dim=-1)
    return perm, k


def apply_transform(Wg, Wu, Wd, Hd, perm, k):
    """Apply channel_transform's (perm, k) to gate/up rows, down columns and the down-input Hessian."""
    s = torch.exp2(k)
    B, R, C = Wg.shape
    pr = perm[:, :, None]
    Wg2 = torch.gather(Wg, 1, pr.expand(B, R, C))
    Wu2 = torch.gather(Wu * s[:, :, None], 1, pr.expand(B, R, C))
    Wd2 = torch.gather(Wd / s[:, None, :], 2, perm[:, None, :].expand(B, Wd.shape[1], R))
    Hs = Hd * s[:, :, None] * s[:, None, :]
    Hd2 = torch.gather(torch.gather(Hs, 1, pr.expand(B, R, R)), 2, perm[:, None, :].expand(B, R, R))
    return Wg2, Wu2, Wd2, Hd2
