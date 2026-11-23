"""Bit-exact emulator of tt-metal's host BFP conversion (Bfp2_b / Bfp4_b / Bfp8_b) and the routed-expert store's
packed form.

Conversion (tt_metal/impl/data_format/blockfloat_common.cpp convert_u32_to_bfp, exp_b mode, round-to-nearest):
  * A block is 16 consecutive values along the LAST dim of the tensor given to ttnn (one face row). Expert matrices are
    handed over as W.T = [in, out], so a block is 16 consecutive OUTPUT channels at one input index.
  * Shared exponent E = max biased fp32 exponent of the block (zero/denormal inputs count 0).
  * m24 = (1<<23 | mantissa) >> (E - e)  (shifted-out bits dropped), then round-to-nearest-even to W bits
    (W = 1 bfp2, 3 bfp4, 7 bfp8), saturating at 2^W - 1 (no exponent bump). Zero/denormal -> 0; the sign is dropped
    when the magnitude is 0.
  * value = sign * q * 2^(E - 127 - (W - 1)).  bfp2 is therefore ternary per block: {-s, 0, +s}, s = 2^(E-127).
  * A code is sign-magnitude: (sign << W) | q. Codes are packed little-endian, element 0 in the low bits
    (create_packed_bfp_packed_as_u32: 16 / 8 / 4 codes per uint32).
Any block already on its grid (a "snapped" block) converts back to the same values, so snapped fp32 weights load
exactly through ttnn (bfp4/bfp8) and through tt-metal's Bfp2_b packer.

Store row (one matrix, fnx.quant.store): Wt = W.T [K=in, N=out] ->
    uint8 [K * N/16]   shared exponents, row-major [K, N/16]
    uint8 [K * N*W'/8] codes, row-major [K, N] (W' = W + 1 bits per code), element order along N, low bits first.
`to_tiles` reorders a row into tt-metal's TILE byte stream (tiles row-major over [K/32, N/32]; per tile 64 exponent
bytes - face 0 rows 0..15, face 1, face 2, face 3 - then the codes of face 0, 1, 2, 3, each face row-major).
"""
from __future__ import annotations

import numpy as np
import torch

MANT_BITS = {"bfp2": 1, "bfp4": 3, "bfp8": 7}
CODE_BITS = {f: w + 1 for f, w in MANT_BITS.items()}
BLOCK = 16
TILE = 32


def _blocks(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % BLOCK == 0, f"last dim {x.shape[-1]} not a multiple of {BLOCK}"
    return x.reshape(*x.shape[:-1], x.shape[-1] // BLOCK, BLOCK)


def shared_exponent(x: torch.Tensor) -> torch.Tensor:
    """Biased fp32 shared exponent per block (tt-metal rule), int32 [..., nb]."""
    b = _blocks(x.float().contiguous()).view(torch.int32)
    return ((b >> 23) & 0xFF).amax(-1)


def quantize_codes(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """tt-metal-exact conversion: (q int32 signed magnitudes [..., nb, 16], E int32 [..., nb])."""
    W = MANT_BITS[fmt]
    S = 24 - W
    b = _blocks(x.float().contiguous()).view(torch.int32)
    e = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    neg = b < 0
    E = e.amax(-1, keepdim=True)
    d = (E - e).clamp(max=31)
    m24 = (man | (1 << 23)) >> d
    m24 = torch.where(e == 0, torch.zeros_like(m24), m24)
    rv = m24 & ((1 << S) - 1)
    q = m24 >> S
    tie = 1 << (S - 1)
    up = (rv > tie) | ((rv == tie) & ((q & 1) == 1))
    q = (q + up.to(q.dtype)).clamp(max=(1 << W) - 1)
    q = torch.where(neg, -q, q)
    return q, E.squeeze(-1)


def step_for_exponent(E: torch.Tensor, fmt: str) -> torch.Tensor:
    W = MANT_BITS[fmt]
    return torch.ldexp(torch.ones_like(E, dtype=torch.float32), (E - 127 - (W - 1)).to(torch.int32))


def dequantize_codes(q: torch.Tensor, E: torch.Tensor, fmt: str, shape=None) -> torch.Tensor:
    W = MANT_BITS[fmt]
    step = torch.ldexp(torch.ones_like(E, dtype=torch.float64), (E - 127 - (W - 1)).to(torch.int64))
    v = (q.to(torch.float64) * step.unsqueeze(-1)).to(torch.float32)
    v = v.reshape(*v.shape[:-2], -1)
    return v if shape is None else v.reshape(shape)


def rtn(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Exactly the values tt-metal produces for x (fp32)."""
    q, E = quantize_codes(x, fmt)
    return dequantize_codes(q, E, fmt, x.shape)


def is_snapped(x: torch.Tensor, fmt: str) -> bool:
    return bool(torch.equal(rtn(x, fmt), x.float()))


# ------------------------------------------------------------------------------------------- packed store rows
def row_bytes(fmt: str, K: int, N: int) -> int:
    return K * N // BLOCK + K * N * CODE_BITS[fmt] // 8


def pack(Wt: torch.Tensor, fmt: str) -> np.ndarray:
    """Snapped Wt [K, N] (blocks along N) -> uint8 store row. Asserts that tt-metal reproduces Wt exactly."""
    K, N = Wt.shape
    q, E = quantize_codes(Wt, fmt)
    back = dequantize_codes(q, E, fmt, Wt.shape)
    if not torch.equal(back, Wt.float()):
        raise ValueError(f"pack: matrix is not {fmt}-snapped ({int((back != Wt.float()).sum())} values differ)")
    return pack_codes(q, E, fmt)


def pack_codes(q: torch.Tensor, E: torch.Tensor, fmt: str) -> np.ndarray:
    """(q [K, N/16, 16] signed, E [K, N/16]) -> uint8 store row."""
    W = MANT_BITS[fmt]
    cb = CODE_BITS[fmt]
    K = q.shape[0]
    code = (q.abs() | ((q < 0).to(q.dtype) << W)).reshape(K, -1).to(torch.int32).cpu().numpy().astype(np.uint8)
    per = 8 // cb
    code = code.reshape(K, -1, per).astype(np.uint16)
    byte = np.zeros(code.shape[:2], np.uint16)
    for i in range(per):
        byte |= code[:, :, i] << (cb * i)
    exps = E.to(torch.int32).cpu().numpy().astype(np.uint8).reshape(-1)
    return np.concatenate([exps, byte.astype(np.uint8).reshape(-1)])


def pack_batch(Wt: torch.Tensor, fmt: str) -> torch.Tensor:
    """Batched `pack` on Wt's device: snapped [B, K, N] -> uint8 [B, row_bytes] (same bytes as pack per matrix)."""
    B, K, N = Wt.shape
    W = MANT_BITS[fmt]
    cb = CODE_BITS[fmt]
    q, E = quantize_codes(Wt, fmt)
    if not torch.equal(dequantize_codes(q, E, fmt, Wt.shape), Wt.float()):
        raise ValueError(f"pack_batch: not {fmt}-snapped")
    code = (q.abs() | ((q < 0).to(q.dtype) << W)).reshape(B, K, -1, 8 // cb)
    byte = torch.zeros(code.shape[:3], dtype=torch.int32, device=Wt.device)
    for i in range(8 // cb):
        byte |= code[..., i] << (cb * i)
    return torch.cat([E.reshape(B, -1), byte.reshape(B, -1)], 1).to(torch.uint8)


def unpack_batch(rows: torch.Tensor, fmt: str, K: int, N: int) -> torch.Tensor:
    """Batched `unpack` on rows' device: uint8 [B, row_bytes] -> float32 [B, K, N]."""
    B = rows.shape[0]
    W = MANT_BITS[fmt]
    cb = CODE_BITS[fmt]
    ne = K * N // BLOCK
    E = rows[:, :ne].to(torch.int32).reshape(B, K, N // BLOCK)
    byte = rows[:, ne:].to(torch.int32).reshape(B, K, -1)
    c = torch.stack([(byte >> (cb * i)) & ((1 << cb) - 1) for i in range(8 // cb)], -1).reshape(B, K, N // BLOCK, BLOCK)
    mag = (c & ((1 << W) - 1)).to(torch.float32)
    q = torch.where((c >> W) != 0, -mag, mag)
    step = torch.ldexp(torch.ones_like(E, dtype=torch.float32), E - 127 - (W - 1))
    return (q * step.unsqueeze(-1)).reshape(B, K, N)


def split_row(row: np.ndarray, fmt: str, K: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """uint8 store row -> (exponents uint8 [K, N/16], code bytes uint8 [K, N*bits/8])."""
    ne = K * N // BLOCK
    assert row.size == row_bytes(fmt, K, N), (row.size, row_bytes(fmt, K, N))
    return row[:ne].reshape(K, N // BLOCK), row[ne:].reshape(K, N * CODE_BITS[fmt] // 8)


def unpack(row: np.ndarray, fmt: str, K: int, N: int) -> np.ndarray:
    """uint8 store row -> float32 [K, N], the exact values tt-metal holds."""
    W = MANT_BITS[fmt]
    cb = CODE_BITS[fmt]
    exps, codes = split_row(np.asarray(row, np.uint8), fmt, K, N)
    per = 8 // cb
    c = np.stack([(codes >> (cb * i)) & ((1 << cb) - 1) for i in range(per)], -1).reshape(K, N).astype(np.int32)
    mag = c & ((1 << W) - 1)
    q = np.where(c >> W, -mag, mag).astype(np.float64)
    step = np.ldexp(1.0, exps.astype(np.int64) - 127 - (W - 1))
    v = q.reshape(K, N // BLOCK, BLOCK) * step[:, :, None]
    return v.astype(np.float32).reshape(K, N)


def to_tiles(row: np.ndarray, fmt: str, K: int, N: int) -> np.ndarray:
    """uint8 store row -> tt-metal TILE byte stream (uint8 [K/32 * N/32 * tile_bytes]), tiles row-major over
    [K/32, N/32] (= pack_as_bfp_tiles of the tilized [K, N] tensor, exp_b)."""
    assert K % TILE == 0 and N % TILE == 0
    exps, codes = split_row(np.asarray(row, np.uint8), fmt, K, N)
    fb = BLOCK * CODE_BITS[fmt] // 8                                     # code bytes per face row
    te = exps.reshape(K // TILE, 2, 16, N // TILE, 2).transpose(0, 3, 1, 4, 2).reshape(K // TILE, N // TILE, 64)
    tc = codes.reshape(K // TILE, 2, 16, N // TILE, 2, fb).transpose(0, 3, 1, 4, 2, 5)
    tc = tc.reshape(K // TILE, N // TILE, 64 * fb)
    return np.concatenate([te, tc], -1).reshape(-1)


def tile_bytes(fmt: str) -> int:
    return 64 + 64 * BLOCK * CODE_BITS[fmt] // 8
