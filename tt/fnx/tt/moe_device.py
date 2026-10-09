"""All-device MoE block on the P150: router, top-10, routed experts (all 256 of the layer, in a block-float format
chosen per matrix, in device DRAM or - the host tier - in pinned host memory the kernels read over PCIe), shared
expert and combine, with no host round trip, so a decode step can run as one trace.  Same API as fnx.tt.moe.Moe:
`MoeDevice(mesh, w, L, ...).forward(mixed) -> bo`.

Per call (mixed [1, 1, T, 2560] f32 TILE; T <= 32 rows per device round, larger T in rounds of 32 rows):
  1. router  (kernels/moe_device/router_*.cpp, one program): column cores 0..8 compute one 32-column tile each of
             logits = x @ [W_r^T | w_sg | 0] (f32 x, BF16 weights in column-contiguous pages, HiFi4, FP32 DEST) and
             send each row's values to route core r < T, which takes the top-10 of the f32 logits (ties -> lower id),
             the softmax over the 10 selected logits and sigmoid(shared gate) in fixed point (topk.h, rel. error
             ~1e-7) into a [32, 32] int32 route buffer
  2. stage A (gu_*.cpp, every worker core): selection = union of the T rows' experts (ascending id) + the shared
             expert (entry 256, last); work unit (entry s, intermediate tile j): h[s][j] = SiLU(x G_e[:, j]) *
             (x U_e[:, j]) * route_s (per row; 0 for rows that did not pick e), BF16 x, FP32 DEST, one BF16 rounding;
             bfp4/bfp2 matmuls at HiFi3 (exact weights in SrcA; LoFi would truncate x: 8.8e-3 vs 1.0e-3 against the
             f64 emulation, same device time), bfp8/bf16 (shared expert) at HiFi4
  3. stage B (dn_*.cpp, one core per output tile n): y[n] = sum_s h[s] D_e[:, n] in one FP32 DEST accumulation over
             all entries (routed and shared); y is bo (f32 [1, 1, T, 2560])
A row's result depends only on its own row (rows that did not pick an expert multiply exact zeros, every sum runs
in ascending expert id), so a T-row verify window gives each row bitwise its T = 1 result.

Expert store (ExpertStore): per layer and format f in (bfp4, bfp2, bfp8, bf16) a store "A" of gate/up matrices and
a store "B" of down matrices, W^T [K, N] tiles (a block-float face row spans 16 output channels: the GPTQ-BFP
orientation of fnx.quant), one DRAM page per output column (KT = 80 tiles for A, JT = 20 for B) so a work unit
reads its weights as one contiguous burst (NCRISC on NOC0, up to DEPTH pages in flight), plus a table entry per
expert: per matrix a word tier << 28 | format << 24 | slot and a 64-bit host address.  Host tier (LayerMatrices.host,
the plan's non-resident experts): those matrices live in a pinned host pool (fnx.tt.hostmem.HostPool, one item per
matrix, its column pages back to back) and the readers fetch them over PCIe (kernels/hostmem/hostmem.h, ~27 GB/s)
into the same CBs in the same order, so the results are bitwise those of an all-resident store: a physically
contiguous item (tier 1) by its bus address, a paged one (tier 2: the pool's 4k region, no physical contiguity
beyond 4 KiB) through the store's page table (DRAM, one row per paged matrix: the bus address of each 4 KiB of it;
a reader fetches the few entries a column needs, then splits the read at the item's 4 KiB boundaries, merging
physically consecutive pages).  The pool lives as long as the store (release() closes it after a device sync), so
every captured trace may read it.
bfloat2_b has no ttnn DataType: the stores are raw int32 pages and the kernels program the unpacker per matrix
(kernels/moe_device/fmt.h).  Sources: `rtn_layer` (round-to-nearest of the GGUF dequantized experts, formats from a
per-layer plan) or `quant_layer` (ExpertQuant's GPTQ store, fnx.quant.store.Plan, read straight from its pool rows: no
converted copy on disk).  The shared expert is entry 256 (bf16 by default, always resident).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import ttnn

from .. import paths
from ..config import N_EMBD, N_EXPERT, N_FF, N_USED
from ..quant import bfp as qbfp

KDIR = Path(__file__).resolve().parent / "kernels" / "moe_device"
KT, JT, NT = N_EMBD // 32, N_FF // 32, N_EMBD // 32
SHARED = N_EXPERT                      # entry id of the shared expert (a store of n_exp routed experts: n_exp)
NE = N_EXPERT + 1
FMTS = ("bfp4", "bfp2", "bfp8", "bf16")   # format codes 0..3 (kernels/moe_device/common.h)
F_BFP4, F_BFP2, F_BFP8, F_BF16 = range(4)
TILE_BYTES = (576, 320, 1088, 2048)
CB_DTYPE = (ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16)   # bfp2: declared bfp4, set at run time
ROUTER_COLS = N_EXPERT + 32
MATS = ("gate", "up", "down")
TABLE_STRIDE = 9                       # table words per entry: gate, up, down word + 3 host addresses (lo, hi)
BT = 20                                # stage A weight block (tiles) per CB push
DEPTH = 4                              # weight block reads in flight per core in ring mode (kernels/moe_device/stream.h)
CAP_A = (3, 3, 3, 3)                   # stage A weight CB capacity per format (blocks of BT tiles)
CAP_B = (3, 3, 3, 3)                   # stage B weight CB capacity per format (blocks of JT tiles)
HCORES = 0                             # stage A cores reserved for host-tier units (kernels/moe_device/sched.h; 0: the
                                       # plain round-robin).  Measured 2026-10-07 (scratch/moespeed/abtest.py, L4/L20):
                                       # deeper rings, deep mode and 16 host cores gave no gain - the host tier is
                                       # PCIe-link bound (~26 GB/s) and already overlaps the DRAM stream
RB = 8                                 # router weight page (tiles)
TIER_HOST = 1 << 28                    # table word tier 1: the matrix is a contiguous host pool item (common.h)
TIER_PAGED = 2 << 28                   # tier 2: a paged host pool item, read through the store's page table
ENT_PAGE = 640                         # entry page bytes (common.h ENT_WORDS: route row + TOPK + 1 table entries)


# sizes that depend on the number of routed experts n_exp (common.h, FNX_N_EXP: 256 = the main layers, 512 = the MTP
# drafter's MoE; a multiple of 32: the router computes (n_exp + 32) / 32 logits column tiles)
def table_words(n_exp: int = N_EXPERT) -> int:
    """Expert table words (TABLE_STRIDE per entry, n_exp + 1 entries), padded to a 64 B multiple."""
    return -(-TABLE_STRIDE * (n_exp + 1) // 16) * 16


def sel_bytes(n_exp: int = N_EXPERT) -> int:
    """Selection page (common.h SEL_*): header, entry offsets and ids, then the 32 entry pages."""
    return -(-(16 + 2 * (n_exp + 1)) // 16) * 64 + 32 * ENT_PAGE


TMP_BYTES = 16384                      # page table scratch of paged host reads (common.h)


def _defines(n_exp: int) -> list:
    """Kernel defines of a store of n_exp routed experts (none for the default count: the kernels' own default), plus
    MD_PROF=1 (device-profiler zones, kernels/moe_device/prof.h) with FNX_MD_PROF=1 and TT_METAL_DEVICE_PROFILER=1."""
    d = [] if n_exp == N_EXPERT else [("FNX_N_EXP", str(n_exp))]
    return d + ([("MD_PROF", "1")] if os.environ.get("FNX_MD_PROF") == "1" else [])


TABLE_WORDS, SEL_BYTES = table_words(), sel_bytes()    # 2320, 22656 (256 experts)


def matrix_bytes(fmt: int) -> int:
    """Device bytes of one expert matrix (1600 tiles) in format `fmt`."""
    return KT * JT * TILE_BYTES[fmt]


# ------------------------------------------------------------------------------------------------ host tile bytes
def bf16_tiles(Wt: np.ndarray) -> np.ndarray:
    """f32 [K, N] -> BF16 (round to nearest even) TILE byte stream, tiles row-major over [K/32, N/32]."""
    K, N = Wt.shape
    b = torch.from_numpy(np.ascontiguousarray(Wt, np.float32)).to(torch.bfloat16).view(torch.int16).numpy()
    t = b.reshape(K // 32, 2, 16, N // 32, 2, 16).transpose(0, 3, 1, 4, 2, 5)     # kt, nt, face row, face col, r, c
    return np.ascontiguousarray(t).reshape(-1).view(np.uint8)


def rtn_tiles(Wt: np.ndarray, fmt: int) -> np.ndarray:
    """f32 W^T [K, N] -> its tt-metal TILE byte stream in format `fmt` (round to nearest, the ttnn conversion)."""
    if fmt == F_BF16:
        return bf16_tiles(Wt)
    K, N = Wt.shape
    q, E = qbfp.quantize_codes(torch.from_numpy(np.ascontiguousarray(Wt, np.float32)), FMTS[fmt])
    return qbfp.to_tiles(qbfp.pack_codes(q, E, FMTS[fmt]), FMTS[fmt], K, N)


def column_pages(tiles: np.ndarray, K: int, N: int, fmt: int) -> np.ndarray:
    """TILE byte stream (row-major over [K/32, N/32]) -> uint8 [N/32, K/32 * tile_bytes]: page n = output column n."""
    tb = TILE_BYTES[fmt]
    return tiles.reshape(K // 32, N // 32, tb).transpose(1, 0, 2).reshape(N // 32, (K // 32) * tb)


def rows_to_pages(rows: np.ndarray, K: int, N: int, fmt: int) -> np.ndarray:
    """Batched fnx.quant store rows (uint8 [n, row_bytes]: exponents [K, N/16] then codes [K, N * bits / 8], blocks
    along N) -> device pages uint8 [n * N/32, K/32 * tile_bytes] (= column_pages(to_tiles(row)) per row)."""
    rows = np.asarray(rows, np.uint8)
    n = rows.shape[0]
    cb = qbfp.CODE_BITS[FMTS[fmt]]
    fb = 16 * cb // 8                                                   # code bytes per face row
    ne = K * N // 16
    kt, nt = K // 32, N // 32
    te = rows[:, :ne].reshape(n, kt, 2, 16, nt, 2).transpose(0, 4, 1, 2, 5, 3).reshape(n, nt, kt, 64)
    tc = rows[:, ne:].reshape(n, kt, 2, 16, nt, 2, fb).transpose(0, 4, 1, 2, 5, 3, 6).reshape(n, nt, kt, 64 * fb)
    return np.concatenate([te, tc], -1).reshape(n * nt, kt * TILE_BYTES[fmt])


def tiles_to_float(tiles: np.ndarray, K: int, N: int, fmt: int) -> np.ndarray:
    """Inverse of rtn_tiles / a store's tiles: the exact f32 W^T [K, N] values the device holds (host reference)."""
    tb = TILE_BYTES[fmt]
    t = tiles.reshape(K // 32, N // 32, tb)
    if fmt == F_BF16:
        b = t.view(np.uint16).reshape(K // 32, N // 32, 2, 2, 16, 16).transpose(0, 2, 4, 1, 3, 5).reshape(K, N)
        return (b.astype(np.uint32) << 16).view(np.float32)
    name = FMTS[fmt]
    W = qbfp.MANT_BITS[name]
    cb = qbfp.CODE_BITS[name]
    exps = t[:, :, :64].reshape(K // 32, N // 32, 2, 2, 16).transpose(0, 2, 4, 1, 3).reshape(K, N // 16)
    fb = 16 * cb // 8
    codes = t[:, :, 64:].reshape(K // 32, N // 32, 2, 2, 16, fb).transpose(0, 2, 4, 1, 3, 5).reshape(K, N * cb // 8)
    per = 8 // cb
    c = np.stack([(codes >> (cb * i)) & ((1 << cb) - 1) for i in range(per)], -1).reshape(K, N).astype(np.int32)
    mag = c & ((1 << W) - 1)
    q = np.where(c >> W, -mag, mag).astype(np.float64)
    step = np.ldexp(1.0, exps.astype(np.int64) - 127 - (W - 1))
    return (q.reshape(K, N // 16, 16) * step[:, :, None]).astype(np.float32).reshape(K, N)


# ------------------------------------------------------------------------------------------------ layer sources
@dataclass
class LayerMatrices:
    """One layer's expert matrices for the device store: fmt [257, 3] format codes (entry 256 = shared expert;
    m = gate, up, down), tiles(e, m) -> the matrix's TILE byte stream (W^T [K, N], tiles row-major), optionally
    rows(e, m) -> its fnx.quant store row (block-float formats; ExpertStore then converts whole batches at once), and
    host: bool [257] (None: none) - the experts kept in pinned host memory (read over PCIe) instead of device DRAM."""
    fmt: np.ndarray
    tiles: Callable[[int, int], np.ndarray]
    rows: Callable[[int, int], np.ndarray] | None = None
    host: np.ndarray | None = None


def shared_tiles(w, L: int, m: int, fmt: int) -> np.ndarray:
    W = w.layer(L, f"ffn_{MATS[m]}_shexp.weight")                    # (n_out, n_in)
    return rtn_tiles(np.ascontiguousarray(W.T), fmt)


def rtn_layer(w, L: int, fmt: np.ndarray, shared_fmt: int = F_BF16) -> LayerMatrices:
    """Round-to-nearest store of layer L from the GGUF dequantized experts; fmt [256, 3] (or [256]) format codes."""
    fmt = np.asarray(fmt, np.int32)
    if fmt.ndim == 1:
        fmt = np.repeat(fmt[:, None], 3, 1)
    full = np.concatenate([fmt, np.full((1, 3), shared_fmt, np.int32)])

    def tiles(e: int, m: int) -> np.ndarray:
        if e == SHARED:
            return shared_tiles(w, L, m, int(full[e, m]))
        W = w.model.get_expert(f"blk.{L}.ffn_{MATS[m]}_exps.weight", int(e))
        return rtn_tiles(np.ascontiguousarray(W.T), int(full[e, m]))

    return LayerMatrices(full, tiles)


def quant_layer(w, L: int, plan, shared_fmt: int = F_BF16, host: np.ndarray | None = None) -> LayerMatrices:
    """ExpertQuant's GPTQ store (fnx.quant.store.Plan): its per-matrix bfp4/bfp2 choice and snapped rows; the host
    tier is the plan's non-resident experts (Plan.resident) unless `host` (bool [256]) overrides it."""
    fq = np.asarray(plan.fmt(L), np.int32)                            # [256, 3]: 4 | 2 bits
    code = np.where(fq == 4, F_BFP4, np.where(fq == 2, F_BFP2, -1))
    if code.shape != (N_EXPERT, 3) or (code < 0).any():
        raise ValueError(f"plan layer {L}: not [256, 3] bfp4/bfp2 formats")
    full = np.concatenate([code, np.full((1, 3), shared_fmt, np.int32)])
    hm = ~np.asarray(plan.resident(L), bool) if host is None else np.asarray(host, bool)

    def tiles(e: int, m: int) -> np.ndarray:
        if e == SHARED:
            return shared_tiles(w, L, m, int(full[e, m]))
        return np.asarray(plan.tiles(L, int(e), m), np.uint8)

    return LayerMatrices(full, tiles, lambda e, m: plan.row(L, int(e), m), np.concatenate([hm, [False]]))


def profile_plan(L: int, n_bfp4: int, profile=None) -> np.ndarray:
    """[256] format codes: layer L's n_bfp4 hottest experts (Strata's Coder profile) bfp4, the rest bfp2."""
    import struct
    path = Path(profile or paths.EXPERT_PROFILE)
    b = path.read_bytes()
    n = struct.unpack("<5I", b[4:24])[4]
    pairs = np.frombuffer(b, dtype="<u2", count=2 * n, offset=24).reshape(n, 2).astype(np.int32)
    order = pairs[pairs[:, 0] == L, 1]                                # hottest first
    fmt = np.full(N_EXPERT, F_BFP2, np.int32)
    fmt[order[:n_bfp4]] = F_BFP4
    return fmt


# ------------------------------------------------------------------------------------------------ device store
def _pages(lm: LayerMatrices, lst: list, K: int, N: int, f: int, shared: int = SHARED):
    """The device pages of matrices lst [(e, m)] (all format f, shape K x N; entry `shared` = the shared expert) in
    chunks: yields (first index, uint8 [n * N/32, K/32 * tile_bytes]).  Routed matrices with store rows convert 64
    per numpy pass."""
    npg = N // 32
    blk = lm.rows is not None and f in (F_BFP4, F_BFP2)
    i = 0
    while i < len(lst):
        if blk and lst[i][0] != shared:
            j = i
            while j < min(len(lst), i + 64) and lst[j][0] != shared:
                j += 1
            yield i, rows_to_pages(np.stack([lm.rows(*lst[k]) for k in range(i, j)]), K, N, f)
            i = j
        else:
            yield i, column_pages(lm.tiles(*lst[i]), K, N, f).reshape(npg, -1)
            i += 1


def host_matrices(lm: LayerMatrices) -> tuple[int, int]:
    """(number, largest bytes) of the host-tier matrices of a layer source (0, 0 without a host tier)."""
    if lm.host is None or not np.any(lm.host):
        return 0, 0
    f = np.asarray(lm.fmt)[np.asarray(lm.host, bool)]
    return int(f.size), max(matrix_bytes(int(x)) for x in np.unique(f))


class ExpertStore:
    """A layer's expert matrices (module docstring): A[f] / B[f] int32 DRAM page tensors (None when no resident matrix
    of that format), the host tier (lm.host experts: their matrices in items first.. of `pool`, a
    fnx.tt.hostmem.HostPool - the model's shared one, or the store's own when pool=None - one item per matrix = its
    pages back to back, read by the kernels over PCIe), `table` (int32 [1, 1, 1, table_words(n_exp)]: per entry the
    gate, up, down words tier << 28 | fmt << 24 | slot, then their host words lo, hi: a contiguous item's bus address,
    or a paged item's row of `ptab`), `ptab` (None without paged items: int32 [1, 1, n_paged, pt_row / 4], row r =
    lo, hi of the bus address of each 4 KiB of paged matrix r, padded to pt_row bytes, a 64 B multiple), `nbytes`
    (device DRAM, the page table included), `host_bytes`, `n_host`, `n_paged`.  The number of routed experts n_exp is
    lm.fmt's row count - 1 (entry n_exp = the shared expert).  The pool must outlive every trace that reads it:
    release() (after a device sync) closes the store's own pool; a shared one is its owner's (TtModel)."""

    def __init__(self, mesh, lm: LayerMatrices, pool=None, first: int = 0):
        self.mesh = mesh
        fmt = np.asarray(lm.fmt, np.int32)
        ne = fmt.shape[0]
        self.n_exp = shared = ne - 1
        assert fmt.shape == (ne, 3) and shared % 32 == 0, fmt.shape
        self.fmt = fmt
        host = np.zeros(ne, bool) if lm.host is None else np.asarray(lm.host, bool)
        assert host.shape == (ne,) and not host[shared], f"host tier: bool [{ne}], never the shared expert"
        self.host = host
        words = np.zeros((ne, 3), np.uint32)
        a_lists: list[list] = [[] for _ in FMTS]
        b_lists: list[list] = [[] for _ in FMTS]
        h_list: list = []
        for e in range(ne):
            for m in range(3):
                f = int(fmt[e, m])
                if host[e]:
                    words[e, m] = TIER_HOST | f << 24 | len(h_list)
                    h_list.append((e, m))
                    continue
                lst = b_lists[f] if m == 2 else a_lists[f]
                words[e, m] = f << 24 | len(lst)
                lst.append((e, m))
        table = np.zeros((table_words(shared),), np.uint32)
        ent = table[:TABLE_STRIDE * ne].reshape(ne, TABLE_STRIDE)
        ent[:, :3] = words
        self.pool, self._own_pool = pool, False
        self.host_bytes = 0
        if h_list:
            n, item = host_matrices(lm)
            if self.pool is None:
                from .hostmem import HostPool
                self.pool, self._own_pool, first = HostPool(n, item), True, 0
            if first + n > self.pool.n_items or item > self.pool.item_bytes:
                raise ValueError(f"host pool of {self.pool.n_items} x {self.pool.item_bytes} B: no room for items "
                                 f"{first}..{first + n - 1} of {item} B")
            from .hostmem import PAGE
            rows = []                       # the paged matrices' page tables
            for i, (e, m) in enumerate(h_list):
                if self.pool.contig[first + i]:
                    lo, hi = self.pool.lohi(first + i)
                    ent[e, 3 + 2 * m], ent[e, 4 + 2 * m] = lo, hi
                    continue
                ent[e, m] = TIER_PAGED | (int(ent[e, m]) & ~TIER_HOST)
                ent[e, 3 + 2 * m], ent[e, 4 + 2 * m] = len(rows), 0
                rows.append(self.pool.pages(first + i)[:-(-matrix_bytes(int(fmt[e, m])) // PAGE)])
            for f in range(len(FMTS)):
                for isb, K, N in ((False, N_EMBD, N_FF), (True, N_FF, N_EMBD)):
                    idx = [i for i, (e, m) in enumerate(h_list) if int(fmt[e, m]) == f and (m == 2) == isb]
                    for i0, pages in _pages(lm, [h_list[i] for i in idx], K, N, f, shared):
                        mats = pages.reshape(-1, matrix_bytes(f))
                        for k in range(mats.shape[0]):
                            self.pool.item(first + idx[i0 + k])[:matrix_bytes(f)] = mats[k]
                        self.host_bytes += mats.nbytes
        self.n_host = len(h_list)
        self.ptab, self.pt_row, self.n_paged = None, 64, len(rows) if h_list else 0
        if self.n_paged:
            self.pt_row = -(-max(len(r) for r in rows) * 8 // 64) * 64
            pt = np.zeros((self.n_paged, self.pt_row // 8), np.uint64)
            for r, pg in enumerate(rows):
                pt[r, :len(pg)] = pg
            self.ptab = ttnn.from_torch(torch.from_numpy(pt.view(np.int32)).reshape(1, 1, self.n_paged, -1),
                                        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.table = ttnn.from_torch(torch.from_numpy(table.view(np.int32)).reshape(1, 1, 1, -1), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.A, self.B = [], []
        self.nbytes = pt.nbytes if self.n_paged else 0
        for f in range(len(FMTS)):
            for lst, K, N, out in ((a_lists[f], N_EMBD, N_FF, self.A), (b_lists[f], N_FF, N_EMBD, self.B)):
                if not lst:
                    out.append(None)
                    continue
                npg = N // 32
                arr = np.empty((len(lst) * npg, (K // 32) * TILE_BYTES[f]), np.uint8)
                for i0, pages in _pages(lm, lst, K, N, f, shared):
                    arr[i0 * npg:i0 * npg + len(pages)] = pages
                out.append(ttnn.from_torch(torch.from_numpy(arr.view(np.int32)), dtype=ttnn.int32,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG))
                self.nbytes += arr.nbytes
                del arr

    def addrs(self, which: str) -> list[int]:
        """The reader's store and page table common args: A[f] / B[f] addresses (which "A" / "B"), ptab address and
        row bytes (the table's address when nothing is paged: never read)."""
        ts = self.A if which == "A" else self.B
        fallback = self.table.buffer_address()
        return [t.buffer_address() if t is not None else fallback for t in ts] + [
            self.ptab.buffer_address() if self.ptab is not None else fallback, self.pt_row]

    def tensors(self) -> list:
        return [t for t in self.A + self.B if t is not None] + [self.table] + (
            [self.ptab] if self.ptab is not None else [])

    def release(self) -> None:
        """Free the DRAM stores and unpin the store's own host pool (no kernel or trace may read them afterwards)."""
        if self._own_pool:
            ttnn.synchronize_device(self.mesh)
            self.pool.close()
        self.pool, self._own_pool = None, False
        for t in self.tensors():
            ttnn.deallocate(t)
        self.A, self.B, self.ptab = [None] * len(FMTS), [None] * len(FMTS), None


# ------------------------------------------------------------------------------------------------ programs
def _cb(crs, idx: int, dtype, page: int, n: int):
    return ttnn.CBDescriptor(total_size=page * n, core_ranges=crs,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype,
                                                                         page_size=page)])


def _acc(*ts) -> list:
    return [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]


def _mem(t) -> str:
    return "L1" if t.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"


class _Program:
    """generic_op kernels with per-call common runtime args (tensor addresses, T) and a fixed program hash."""

    def __init__(self, kernels: list, cbs: list, key: tuple, semaphores=()):
        self.kernels, self.cbs, self.semaphores = kernels, cbs, list(semaphores)
        self.hash = int.from_bytes(hashlib.sha1(repr(("fnx.tt.moe_device", key)).encode()).digest()[:8], "little")

    def __call__(self, io: list, common: list[list[int]]):
        """common: the common runtime args of the first len(common) kernels (reader, writer)."""
        for k, c in zip(self.kernels, common):
            k.common_runtime_args = c
        pd = ttnn.ProgramDescriptor(kernels=self.kernels, semaphores=self.semaphores, cbs=self.cbs)
        pd.custom_program_hash = self.hash
        ttnn.generic_op(io, pd)


_PROGRAMS: dict[tuple, _Program] = {}
FP = ttnn.KernelDescriptor.SourceType.FILE_PATH


def _dm(riscv: int, noc: int, dynamic: bool = False) -> ttnn.DataMovementConfigDescriptor:
    """RISC-V `riscv` (0 BRISC, 1 NCRISC) on NoC `noc` (NOC0 is the fast DRAM read path: fnx.tt.hc._dm); dynamic: the
    kernel may issue on both NoCs (DM_DYNAMIC_NOC: every data-movement kernel of the program must use it)."""
    kw = {"noc_mode": ttnn.NOC_MODE.DM_DYNAMIC_NOC} if dynamic else {}
    return ttnn.DataMovementConfigDescriptor(
        processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[riscv],
        noc=(ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default)[noc], **kw)


READER, WRITER = _dm(1, 0), _dm(0, 1)    # weight streaming on NCRISC + NOC0, the small reads / writes on BRISC + NOC1


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    cores = [(x, y) for y in range(g.y) for x in range(g.x)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i]
    return cores, crs, rt


def router_program(mesh, T: int, x, wr, logits, route, xb, table, ents, n_exp: int = N_EXPERT) -> _Program:
    """Router + top-10 in one program (kernels/moe_device/router_*.cpp): column cores 0..ncol-1 (ncol = (n_exp + 32) /
    32: 9 for 256 experts) compute one 32-column logits tile each (full K, f32 x, BF16 weights, HiFi4), send row r's
    values to route core r < T (gather page + semaphore), which runs route_row and writes the route buffer and its
    entry page (the row + its experts' expert-table entries: stage A / B's input); the logits tiles also go to
    `logits`, and x's BF16 copy (stage A's input, = ttnn.typecast) to `xb`."""
    key = ("router", T, _mem(x), _mem(logits), _mem(route), _mem(xb), _mem(ents), n_exp)
    p = _PROGRAMS.get(key)
    if p is None:
        g = mesh.compute_with_storage_grid_size()
        ncol = (n_exp + 32) // 32
        n = max(ncol, T)
        cores = [(i % g.x, i // g.x) for i in range(n)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(min(g.x, n - y * g.x) - 1, y))
                                 for y in range(-(-n // g.x))])
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(cores):
            rt[cx][cy] = [i]
        named, dfn = [("rb", RB)], _defines(n_exp)
        reader = ttnn.KernelDescriptor(kernel_source=str(KDIR / "router_reader.cpp"), source_type=FP, core_ranges=crs,
                                       compile_time_args=_acc(wr), named_compile_time_args=named, defines=dfn,
                                       runtime_args=rt, common_runtime_args=[0] * 8, config=READER)
        writer = ttnn.KernelDescriptor(kernel_source=str(KDIR / "router_writer.cpp"), source_type=FP,
                                       core_ranges=crs, compile_time_args=_acc(logits, route, x, wr, xb, table, ents),
                                       named_compile_time_args=named, defines=dfn, runtime_args=rt,
                                       common_runtime_args=[0] * (8 + 2 * T), config=WRITER)
        compute = ttnn.KernelDescriptor(kernel_source=str(KDIR / "router_compute.cpp"), source_type=FP,
                                        core_ranges=crs, named_compile_time_args=named, defines=dfn, runtime_args=rt,
                                        config=_compute_cfg(ttnn.MathFidelity.HiFi4))
        cbs = [_cb(crs, 0, ttnn.float32, 4096, KT), _cb(crs, 1, ttnn.bfloat16, 2048, KT // 2),
               _cb(crs, 2, ttnn.bfloat16, 2048, KT // 2), _cb(crs, 3, ttnn.bfloat16, 2048, -(-KT // ncol)),
               _cb(crs, 5, ttnn.uint32, table_words(n_exp) * 4, 1),
               _cb(crs, 7, ttnn.uint32, max(2048, ncol * 128 + ENT_PAGE), 1),
               _cb(crs, 16, ttnn.float32, 4096, 1)]
        p = _PROGRAMS[key] = _Program([reader, writer, compute], cbs, key,
                                      [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)])
        vc = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy)) for cx, cy in cores[:T]]
        p.noc_xy = [v for c in vc for v in (c.x, c.y)]
    return p


def _compute_cfg(fidelity, f32_cbs=()):
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True)
    if f32_cbs:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in f32_cbs:
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode.extend(modes)
    return cfg


def _weight_cbs(crs, caps, tiles: int) -> list:
    return [_cb(crs, 1 + f, CB_DTYPE[f], TILE_BYTES[f], caps[f] * tiles) for f in range(len(FMTS))]


def _caps(caps) -> list:
    return [(f"cap{f}", int(c)) for f, c in enumerate(caps)]


def gu_program(mesh, store: ExpertStore, ents, xb, h, fidelity) -> _Program:
    key = ("gu", _mem(ents), _mem(xb), _mem(h), str(fidelity), store.n_exp, CAP_A, DEPTH, HCORES)
    p = _PROGRAMS.get(key)
    if p is None:
        cores, crs, rt = _grid(mesh)
        nc = len(cores)
        anyA = next(t for t in store.A if t is not None)
        dfn = _defines(store.n_exp)
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gu_reader.cpp"), source_type=FP, core_ranges=crs,
            compile_time_args=_acc(ents, anyA, store.table),     # last: the page table (any DRAM tensor)
            named_compile_time_args=[("ncores", nc), ("bt", BT), ("depth", DEPTH), ("hcores", HCORES)] + _caps(CAP_A), defines=dfn,
            runtime_args=rt, common_runtime_args=[0] * 8, config=_dm(1, 0, True))
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gu_writer.cpp"), source_type=FP, core_ranges=crs,
            compile_time_args=_acc(xb, h), named_compile_time_args=[("ncores", nc), ("hcores", HCORES)], defines=dfn,
            runtime_args=rt, common_runtime_args=[0, 0], config=_dm(0, 1, True))
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gu_compute.cpp"), source_type=FP, core_ranges=crs,
            named_compile_time_args=[("ncores", nc), ("bt", BT), ("fid_lo", int(fidelity.value)), ("hcores", HCORES)],
            defines=dfn, runtime_args=rt, config=_compute_cfg(fidelity, f32_cbs=(5,)))
        cbs = [_cb(crs, 0, ttnn.bfloat16, 2048, KT)] + _weight_cbs(crs, CAP_A, BT) + [
            _cb(crs, 5, ttnn.float32, 4096, 2),
            _cb(crs, 6, ttnn.uint32, sel_bytes(store.n_exp), 1),
            _cb(crs, 7, ttnn.uint32, TMP_BYTES, 1),
            _cb(crs, 16, ttnn.bfloat16, 2048, 2)]
        p = _PROGRAMS[key] = _Program([reader, writer, compute], cbs, key)
    return p


def dn_program(mesh, store: ExpertStore, ents, h, y, fidelity) -> _Program:
    key = ("dn", _mem(ents), _mem(h), _mem(y), str(fidelity), store.n_exp, CAP_B, DEPTH)
    p = _PROGRAMS.get(key)
    if p is None:
        cores, crs, rt = _grid(mesh)
        nc = len(cores)
        anyB = next(t for t in store.B if t is not None)
        dfn = _defines(store.n_exp)
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "dn_reader.cpp"), source_type=FP, core_ranges=crs,
            compile_time_args=_acc(ents, anyB, store.table),     # last: the page table (any DRAM tensor)
            named_compile_time_args=[("ncores", nc), ("depth", DEPTH)] + _caps(CAP_B),
            defines=dfn, runtime_args=rt, common_runtime_args=[0] * 8, config=_dm(1, 0, True))
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "dn_writer.cpp"), source_type=FP, core_ranges=crs,
            compile_time_args=_acc(h, y), named_compile_time_args=[("ncores", nc)], defines=dfn,
            runtime_args=rt, common_runtime_args=[0, 0], config=_dm(0, 1, True))
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "dn_compute.cpp"), source_type=FP, core_ranges=crs,
            named_compile_time_args=[("ncores", nc), ("fid_lo", int(fidelity.value))], defines=dfn,
            runtime_args=rt, config=_compute_cfg(fidelity))
        cbs = [_cb(crs, 0, ttnn.bfloat16, 2048, 2 * JT)] + _weight_cbs(crs, CAP_B, JT) + [
            _cb(crs, 6, ttnn.uint32, sel_bytes(store.n_exp), 1),
            _cb(crs, 7, ttnn.uint32, TMP_BYTES, 1),
            _cb(crs, 16, ttnn.float32, 4096, 2)]
        p = _PROGRAMS[key] = _Program([reader, writer, compute], cbs, key)
    return p


def router_matrix(w, L: int) -> np.ndarray:
    """[W_r^T | w_sg | 0] f32 [2560, 288] (BF16 values: the GGUF router and shared-gate weights)."""
    R = np.zeros((N_EMBD, ROUTER_COLS), np.float32)
    R[:, :N_EXPERT] = np.asarray(w.layer(L, "ffn_gate_inp.weight"), np.float32).T
    R[:, N_EXPERT] = w.layer(L, "ffn_gate_inp_shexp.weight").reshape(-1)
    return R


def router_pages(mesh, R: np.ndarray):
    """The router matrix R (f32 [2560, n_exp + 32]: [W_r^T | w_sg | 0], router_matrix) for the fused router program:
    BF16 tiles, column-contiguous int32 pages of RB tiles (page c * KT / RB + p = k-tiles RB p .. RB p + RB - 1 of
    logits column tile c)."""
    t = bf16_tiles(R).reshape(KT, R.shape[1] // 32, 2048).transpose(1, 0, 2)
    pages = np.ascontiguousarray(t).reshape(-1, RB * 2048)
    return ttnn.from_torch(torch.from_numpy(pages.view(np.int32)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class MoeDevice:
    """The all-device MoE block of layer L (module docstring).  `lm`: the layer's matrices (rtn_layer / quant_layer;
    default: RTN with every routed matrix bfp4); its host tier goes to items pool_first.. of `pool` (a shared
    HostPool, e.g. TtModel's; None: the store's own).  fidelity: of the bfp4/bfp2 expert matmuls (their weights are exact in
    SrcA's high phase; HiFi3 adds the activation's low SrcB bits: measured 1.0e-3 vs 8.8e-3 at LoFi against the f64
    emulation; HiFi4 = HiFi3 here); bfp8/bf16 matrices (the shared expert) always run at HiFi4.
    `router` (f32 [2560, n_exp + 32] = [W_r^T | w_sg | 0], default router_matrix(w, L)) with `lm` (fmt [n_exp + 1, 3])
    makes a block of n_exp routed experts (a multiple of 32; e.g. the MTP drafter's 512: fnx.tt.mtp); w and L are
    then unused."""

    def __init__(self, mesh, w, L: int, lm: LayerMatrices | None = None, fidelity=ttnn.MathFidelity.HiFi3,
                 pool=None, pool_first: int = 0, router: np.ndarray | None = None):
        self.mesh, self.L, self.fidelity = mesh, L, fidelity
        R = router_matrix(w, L) if router is None else np.asarray(router, np.float32)
        self.n_exp = R.shape[1] - 32
        self.wr = router_pages(mesh, R)
        if lm is None:
            lm = rtn_layer(w, L, np.full(N_EXPERT, F_BFP4, np.int32))
        self.store = ExpertStore(mesh, lm, pool, pool_first)
        if R.shape[0] != N_EMBD or self.store.n_exp != self.n_exp:
            raise ValueError(f"router {R.shape} vs a store of {self.store.n_exp} experts")
        self.keep_route = False
        self.last_route = None             # with keep_route: the last call's route buffer (int32 [1, 1, 32, 32])
        self.last_logits = None            # and its router logits (f32 [1, 1, T, n_exp + 32])

    def reset(self) -> None:
        """The MoE block keeps no per-sequence state."""

    def state_tensors(self) -> list:
        return []

    # ---- stages of one device round (T <= 32 rows); each only dispatches device work
    def route(self, x, T: int):
        """x f32 [1, 1, T, 2560] -> (route int32 [1, 1, 32, 32] L1, logits f32 [1, 1, 32, n_exp + 32] L1, xb: x in
        BF16 [1, 1, T, 2560] L1, the stage A input, ents: the entry pages int32 [1, 1, 32, ENT_PAGE / 4] L1, the
        stage A / B selection input; rows < T valid)."""
        mesh, L1, st = self.mesh, ttnn.L1_MEMORY_CONFIG, self.store
        route = ttnn.empty([1, 1, 32, 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, mesh, L1)
        logits = ttnn.empty([1, 1, 32, self.n_exp + 32], ttnn.float32, ttnn.TILE_LAYOUT, mesh, L1)
        xb = ttnn.empty([1, 1, T, N_EMBD], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, L1)
        ents = ttnn.empty([1, 1, 32, ENT_PAGE // 4], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, mesh, L1)
        p = router_program(mesh, T, x, self.wr, logits, route, xb, st.table, ents, self.n_exp)
        args = [logits.buffer_address(), route.buffer_address(), x.buffer_address(), self.wr.buffer_address(), T,
                xb.buffer_address(), st.table.buffer_address(), ents.buffer_address()]
        p([x, self.wr, st.table, logits, route, xb, ents], [args, args + p.noc_xy])
        return route, logits, xb, ents

    def hidden(self, T: int):
        s_cap = min(self.n_exp + 1, N_USED * T + 1)
        return ttnn.empty([1, 1, 32 * s_cap, N_FF], ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                          ttnn.L1_MEMORY_CONFIG if T <= 16 else ttnn.DRAM_MEMORY_CONFIG)

    def _reader_args(self, which: str, ents, T: int) -> list[int]:
        return [T, ents.buffer_address()] + self.store.addrs(which)

    def stage_a(self, xb, ents, h, T: int) -> None:
        gu_program(self.mesh, self.store, ents, xb, h, self.fidelity)(
            [xb, ents, h], [self._reader_args("A", ents, T), [xb.buffer_address(), h.buffer_address()]])

    def stage_b(self, h, ents, y, T: int) -> None:
        dn_program(self.mesh, self.store, ents, h, y, self.fidelity)(
            [h, ents, y], [self._reader_args("B", ents, T), [h.buffer_address(), y.buffer_address()]])

    def _round(self, x, T: int):
        """One device round of T <= 32 rows: x f32 [1, 1, T, 2560] -> bo f32 [1, 1, T, 2560]."""
        route, logits, xb, ents = self.route(x, T)
        if self.keep_route:
            for t in (self.last_route, self.last_logits):
                if t is not None:
                    ttnn.deallocate(t)
            self.last_route, self.last_logits = route, logits
        else:
            ttnn.deallocate(logits)
        h = self.hidden(T)
        self.stage_a(xb, ents, h, T)
        ttnn.deallocate(xb)
        y = ttnn.empty([1, 1, T, N_EMBD], ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        self.stage_b(h, ents, y, T)
        ttnn.deallocate(h)
        ttnn.deallocate(ents)
        if not self.keep_route:
            ttnn.deallocate(route)
        return y

    def forward(self, mixed, T: int | None = None):
        """mixed [1, 1, rows, 2560] TILE (float32, or bfloat16), tokens in rows [0, T) (default T = rows) -> bo
        [1, 1, rows, 2560] float32 TILE (rows >= T: padding).  Device only: capture-safe for T <= 32."""
        rows = int(mixed.shape[-2])
        T = rows if T is None else int(T)
        x = mixed if mixed.dtype == ttnn.float32 else ttnn.typecast(mixed, ttnn.float32)
        if T <= 32:
            if T < rows:
                xs = ttnn.slice(x, [0, 0, 0, 0], [1, 1, T, N_EMBD])
                y = self._round(xs, T)
                ttnn.deallocate(xs)
                yp = ttnn.pad(y, [(0, 0), (0, 0), (0, rows - T), (0, 0)], 0.0)
                ttnn.deallocate(y)
                y = yp
            else:
                y = self._round(x, T)
        else:
            parts = []
            for r0 in range(0, T, 32):
                n = min(32, T - r0)
                xs = ttnn.slice(x, [0, 0, r0, 0], [1, 1, r0 + n, N_EMBD])
                parts.append(self._round(xs, n))
                ttnn.deallocate(xs)
            y = ttnn.concat(parts, dim=2)
            for p in parts:
                ttnn.deallocate(p)
            if T < rows:
                yp = ttnn.pad(y, [(0, 0), (0, 0), (0, rows - T), (0, 0)], 0.0)
                ttnn.deallocate(y)
                y = yp
        if x is not mixed:
            ttnn.deallocate(x)
        return y
