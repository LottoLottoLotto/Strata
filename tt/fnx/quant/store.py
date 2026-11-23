"""Quantized routed-expert store: one pool of GPTQ-BFP rows shared by every precision plan.

Layout under fnx.paths.STORE (FNX_QUANT_DIR/store, default FNX_DATA/quant/store):
  pool/L{LL}.b4.npy     uint8 [n4, 921600]  bfp4 matrices (fnx.quant.bfp store rows of W.T = [in, out])
  pool/L{LL}.b4idx.npy  int32 [n4, 2]       (expert, matrix) of each row; matrix 0 gate, 1 up, 2 down
  pool/L{LL}.b2.npy     uint8 [n2, 512000]  bfp2 matrices;  pool/L{LL}.b2idx.npy likewise
  plans/<name>/L{LL}.fmt.npy  int8  [E, 3]  4 or 2 per (expert, gate/up/down); E = fnx.config.N_EXPERT
  plans/<name>/L{LL}.row.npy  int32 [E, 3]  row of that matrix in the pool file of its format
  plans/<name>/L{LL}.res.npy  bool  [E]     tiered plans only (bfp4-tiered-*): True = resident in device DRAM,
                                               False = host pool (read over PCIe); Plan.resident(L)
  plans/<name>/plan.json      budget, bytes, predicted error, provenance
Row = exponents uint8 [in, out/16] then codes [in, out*bits/8] (fnx.quant.bfp docstring); gate/up: in 2560, out 640;
down: in 640, out 2560. tiles() gives tt-metal's TILE byte stream of W.T (576 B / 320 B per 32x32 tile, 1600 tiles).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import paths
from ..config import N_EXPERT
from . import bfp

STORE = paths.STORE
MATS = ("gate", "up", "down")
SHAPE = {0: (2560, 640), 1: (2560, 640), 2: (640, 2560)}       # W.T [in, out]
FMT = {4: "bfp4", 2: "bfp2"}
TAG = {4: "b4", 2: "b2"}


class Pool:
    def __init__(self, root: Path | str = STORE / "pool"):
        self.root = Path(root)
        self._rows, self._idx = {}, {}

    def rows(self, L: int, bits: int) -> np.ndarray:
        k = (L, bits)
        if k not in self._rows:
            self._rows[k] = np.load(self.root / f"L{L:02d}.{TAG[bits]}.npy", mmap_mode="r")
        return self._rows[k]

    def index(self, L: int, bits: int) -> np.ndarray:
        """[N_EXPERT, 3] int32 row of (e, m) in the bits pool, -1 when absent."""
        k = (L, bits)
        if k not in self._idx:
            ix = np.load(self.root / f"L{L:02d}.{TAG[bits]}idx.npy")
            out = np.full((N_EXPERT, 3), -1, np.int32)
            out[ix[:, 0], ix[:, 1]] = np.arange(len(ix), dtype=np.int32)
            self._idx[k] = out
        return self._idx[k]


class Plan:
    """A precision plan over the pool: Plan(name).fmt(L) [N_EXPERT, 3] (4|2), .row(L, e, m) uint8 store row,
    .tiles(L, e, m) tt-metal tile bytes, .snapped(L, e, m) float32 W.T [in, out] (the exact device values)."""

    def __init__(self, name: str, root: Path | str = STORE):
        self.root = Path(root)
        self.dir = self.root / "plans" / name
        self.meta = json.loads((self.dir / "plan.json").read_text())
        self.pool = Pool(self.root / "pool")
        self._fmt, self._row = {}, {}

    def fmt(self, L: int) -> np.ndarray:
        if L not in self._fmt:
            self._fmt[L] = np.load(self.dir / f"L{L:02d}.fmt.npy")
            self._row[L] = np.load(self.dir / f"L{L:02d}.row.npy")
        return self._fmt[L]

    def row(self, L: int, e: int, m: int) -> np.ndarray:
        bits = int(self.fmt(L)[e, m])
        return self.pool.rows(L, bits)[int(self._row[L][e, m])]

    def tiles(self, L: int, e: int, m: int) -> np.ndarray:
        bits = int(self.fmt(L)[e, m])
        K, N = SHAPE[m]
        return bfp.to_tiles(self.row(L, e, m), FMT[bits], K, N)

    def snapped(self, L: int, e: int, m: int) -> np.ndarray:
        bits = int(self.fmt(L)[e, m])
        K, N = SHAPE[m]
        return bfp.unpack(self.row(L, e, m), FMT[bits], K, N)

    def bytes(self) -> int:
        return int(sum(sum(bfp.row_bytes(FMT[int(b)], *SHAPE[m]) for (_, m), b in np.ndenumerate(self.fmt(L)))
                       for L in range(48)))

    def resident(self, L: int) -> np.ndarray:
        """bool [N_EXPERT]: device-resident experts of layer L (tiered plans; all True for plans without tiers)."""
        p = self.dir / f"L{L:02d}.res.npy"
        return np.load(p) if p.exists() else np.ones(N_EXPERT, bool)


def write_plan(name: str, fmt: np.ndarray, meta: dict, root: Path | str = STORE,
               resident: np.ndarray | None = None) -> Path:
    """fmt int8 [48, N_EXPERT, 3] (4|2) -> plans/<name>/ (row indices resolved against the pool; every row must exist).
    resident bool [48, N_EXPERT] (tiered plans): True = the expert lives in device DRAM, False = in the host pool read
    over PCIe; written as L{LL}.res.npy."""
    root = Path(root)
    pool = Pool(root / "pool")
    d = root / "plans" / name
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for L in range(fmt.shape[0]):
        row = np.full((N_EXPERT, 3), -1, np.int32)
        for bits in (4, 2):
            m = fmt[L] == bits
            if not m.any():                       # a format the layer does not use needs no pool file
                continue
            ix = pool.index(L, bits)
            if (ix[m] < 0).any():
                miss = np.argwhere(m & (ix < 0))
                raise ValueError(f"plan {name}: layer {L} needs bfp{bits} rows missing from the pool: {miss[:5]}")
            row[m] = ix[m]
            total += int(m.sum()) * bfp.row_bytes(FMT[bits], 2560, 640)
        np.save(d / f"L{L:02d}.fmt.npy", fmt[L].astype(np.int8))
        np.save(d / f"L{L:02d}.row.npy", row)
        if resident is not None:
            np.save(d / f"L{L:02d}.res.npy", resident[L].astype(bool))
    meta = dict(meta, bytes=total, name=name)
    (d / "plan.json").write_text(json.dumps(meta, indent=1))
    return d
