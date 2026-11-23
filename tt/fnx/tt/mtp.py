"""MTP draft layer on the P150 (TTNN): Strata's drafter (spec Part 7 §3; numpy twin fnx.ref.mtp.MtpRef).

`Mtp(mesh, w)` drafts tokens from the main model's final 4-stream residual (float32 [1,1,T,10240] TILE on the device,
after the last layer's write, before output_hc):
  prefill(R, next_tokens, cell0)       K/V of prompt cells cell0.. (front only), R rows = the main model's final R
  draft(R, outv, p, a, steps=3)        a round after a verify window at p with a accepted drafts: catch-up front over
                                       rows 0..a (cells p..p+a), the rest of the layer on row a -> draft 0, then
                                       chain steps j = 1.. on (R_out, previous draft, cell p+a+j) -> (drafts, probs)
`R` may be a device tensor or a host float32 array [T, 4, 2560] / [T, 10240].

A round is device work only (round_dev): the chain feeds each step's draft to the next on the device, so a traced
round (fnx.tt.trace.DraftTracer) is ONE captured segment; the host writes the round's inputs before it (RoundInputs)
and reads one 128 B result after it (the draft subset indices and probabilities).

Per cell (float32 activations; matmuls HiFi4 + fp32 accumulation, "precise": the float32 activation split exactly into
bfloat16 hi + lo, two matmuls, fnx.tt.hc.mm; the Q8_0 projections stored bfloat16):
  e       the cell's token embedding: catch-up rows from the host (IQ4_XS row, exact numpy, uploaded); chain steps
          gather the previous draft's raw IQ4_XS row from pinned host memory at the device argmax's index and
          dequantize it on the device, bitwise the host values (EmbedRows, kernels/mtp/row_gather.cpp)
  device  en = RMSNorm(e) * g_e (device, float32); e2 = en @ fc_e^T; hn = rms over all 10240 of Rin (pooled;
          hnorm="stream": per stream) * g_h; R = (hn per stream) @ fc_h^T + e2 (each stream);
          mixed, inj = HcRead(attn HC)(R)  (fnx.tt.hc kernels)
          qkvg = mixed @ [q | gate | k | v]^T (per-head [q|gate] rows of q_proj regrouped as fnx.tt.qsa);
          q/k per-head RMSNorm (q gamma * 1/16 = the softmax scale), NEOX RoPE (64 dims) at the cell; K/V appended to
          the layer's own cache (float32 row-major [1,1,2*cells,256], row j*cells + c = kv head j of cell c; a
          bfloat16 cache moved the draft logits by 6-7% rel L2 in the numpy reference and flipped 6% of the drafts;
          Strata keeps fp16)
  rest    (one row) dense attention over cells 0..c (a slice of the cache's prefix, -1e9 mask beyond c): the logits
          reach ~600 (the drafter's q/k gammas are ~3.7, the trunk's ~1.3), so q K^T and P V split BOTH operands into
          bfloat16 hi + lo (mm2) and the max is subtracted in float32 before the softmax; * sigmoid(gate), o_proj ->
          bo; R = hc_write(R, bo, inj); mixed2, inj2 = HcRead(mlp HC)(R)
  MoE     fnx.tt.moe_device.MoeDevice with the drafter's 512 routed experts (router [W_r^T | w_sg] BF16, fp32 top-10
          on the device, routed experts BFP4 quantized from their Q2_0 values: MtpWeights.bfp4_rows, the shared
          expert BF16 folded into the down sum with its sigmoid gate); `host` experts live in a pinned host pool read
          over PCIe by the kernels (the rest in device DRAM); R_out = hc_write(R, y, inj2)
  head    smix = HcRead(final mixer)(R_out); logits = smix @ W_out[draft vocab]^T (bfloat8_b, the main head's Q6_K
          rows of the subset, padded to a multiple of 32 columns that a -1e30 mask keeps out), one bank-packed matmul
          (fnx.tt.packed_matmul); device argmax (ties -> lowest subset index) and prob = 1 / sum exp(lg - max)
Measured against fnx.ref.mtp on the golden prompts (fnx/tests/test_mtp.py tt).
The K/V cache is positional (cell = position; speculative cells are rewritten by later catch-ups before any query
reads them, mtp.hpp:16-18), like fnx.tt.qsa's: `reset()` has nothing to clear, `state_tensors()` is empty, and a
TtModel.snapshot() of the first L tokens stays valid for the drafter as long as cells [0, L-1) are kept.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch
import ttnn

from .. import paths
from ..config import EPS, HC, HC_DIM, HEAD_DIM, N_EMBD, N_HEAD, N_HEAD_KV, ROPE_DIM
from ..quant import bfp as qbfp
from ..ref.embed import embed
from ..ref.mtp import BFP4_SHAPE, HC_PREFIX, MTP_EXPERTS, MTP_LAYER, PROFILE, MtpWeights
from ..ref.qsa import rope_cos_sin
from .hc import HcRead, consts, hc_write, mm
from .moe_device import (F_BF16, F_BFP4, FP, READER, LayerMatrices, MoeDevice, _acc, _cb, _Program, bf16_tiles,
                         host_matrices, matrix_bytes)
from .packed_matmul import PackedLayout, PackedMatmul, place_packed
from .qsa import KV_ROWS_MAX, kv_write
from .tile_argmax import TileArgmax

f32 = np.float32
F32, BF16, BFP8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
DRAM = ttnn.DRAM_MEMORY_CONFIG
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
TT_CACHE = str(paths.TT_CACHE)
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)
NORM_CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=False)
# attention q K^T and P V: sequential fp32 accumulation (no packer L1 accumulation), so the scores of the cells a row
# attends do not depend on how many masked cells pad the slice (a traced round attends a fixed span, fnx.tt.trace)
ATTN_CKC = NORM_CKC
QD, KD = N_HEAD * HEAD_DIM, N_HEAD_KV * HEAD_DIM          # 6144, 512
GROUP = N_HEAD // N_HEAD_KV
SPAN = 256                                                 # attended cells rounded up to this (program reuse)
MASKED = -1e9
MAX_PART = 8_000_000                                       # elements per head upload part (P150 staging limit)
# routed experts in device DRAM (FNX_MTP_RESIDENT): the 254 hottest of the calibration profile (88% of the routings)
# cost +0.3 ms per draft round vs all 512 and free 0.71 GB of DRAM for main-model experts (test_mtp bench, 2026-10-07)
DEFAULT_RESIDENT = 254
ITEM_BYTES = matrix_bytes(F_BFP4)  # host pool item: one BFP4 expert matrix (921600 B)
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "mtp")


def _cached(mesh, name: str, dtype, build, layout=TILE):
    """ttnn.as_tensor through the converted-tensor cache; build() (host float32 array) only runs on a miss."""
    os.makedirs(TT_CACHE, exist_ok=True)
    base = f"{TT_CACHE}/{name}_{dtype.name}"
    hit = os.path.isfile(f"{base}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")
    a = None
    if not hit:
        a = np.ascontiguousarray(build(), np.float32)
        while a.ndim < 4:
            a = a[None]
        a = torch.from_numpy(a)
    return ttnn.as_tensor(a, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, cache_file_name=base)


def _dev(mesh, a: np.ndarray, dtype=F32, layout=TILE):
    a = np.ascontiguousarray(a)
    while a.ndim < 4:
        a = a[None]
    t = torch.from_numpy(a)
    if dtype == BF16:
        t = t.to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM)


def _host_rm(a: np.ndarray, dtype=F32):
    """Host ROW_MAJOR tensor of `a` (for copy_host_to_device_tensor into a persistent buffer)."""
    return ttnn.Tensor(np.ascontiguousarray(a), data_type=dtype, layout=RM)


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _pos_dev(mesh, p: int):
    """kv_write's position page: uint32 [1, 1, 1, 32] row-major, word 0 = p."""
    t = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
    t[0, 0, 0, 0] = int(p)
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=RM, device=mesh, memory_config=DRAM)


def _rows(x, r0: int, r1: int):
    return ttnn.slice(x, [0, 0, r0, 0], [1, 1, r1, x.shape[-1]])


def _split(x):
    """float32 x -> (hi, lo) bfloat16 with hi + lo = x to ~2^-16 relative."""
    hi = ttnn.typecast(x, BF16)
    hi32 = ttnn.typecast(hi, F32)
    lo32 = ttnn.subtract(x, hi32)
    lo = ttnn.typecast(lo32, BF16)
    _free(hi32, lo32)
    return hi, lo


def mm2(a, b, transpose_b: bool = False, ckc=None):
    """float32 a @ b (b^T) -> float32 with both operands split into bf16 hi + lo (hi.hi + lo.hi + hi.lo).  The
    drafter's attention logits reach ~600 (its q/k gammas are ~3.7): reading float32 operands as TF32 moved them by
    ~1.5; split, what remains is the FPU's accumulation floor (~3e-4 of sum |a_i b_i|, contract)."""
    ah, al = _split(a)
    bh, bl = _split(b)
    parts = [ttnn.matmul(x, y, transpose_b=transpose_b, compute_kernel_config=ckc or HIFI4, dtype=F32)
             for x, y in ((ah, bh), (al, bh), (ah, bl))]
    s = ttnn.add(parts[0], parts[1])
    out = ttnn.add(s, parts[2])
    _free(ah, al, bh, bl, s, *parts)
    return out


# ------------------------------------------------------------------------------------------------ the MoE block
def router_matrix(mw: MtpWeights) -> np.ndarray:
    """[W_r^T | w_sg | 0] f32 [2560, 544] (BF16 values) for fnx.tt.moe_device.MoeDevice(router=...)."""
    R = np.zeros((N_EMBD, MTP_EXPERTS + 32), np.float32)
    R[:, :MTP_EXPERTS] = mw.get("mlp.gate.weight").T
    R[:, MTP_EXPERTS] = mw.get("mlp.shared_expert_gate.weight").reshape(-1)
    return R


def host_experts(mw: MtpWeights, resident: int) -> np.ndarray | None:
    """bool [512]: the routed experts outside the `resident` hottest of the routing profile (MTP_DIR/PROFILE, built
    by fnx.tests.test_mtp profile; ties -> lower id), i.e. those kept in pinned host memory; None: all resident."""
    if resident >= MTP_EXPERTS:
        return None
    order = np.argsort(-np.load(mw.dir / PROFILE), kind="stable")
    host = np.ones(MTP_EXPERTS, bool)
    host[order[:max(resident, 0)]] = False
    return host


def expert_matrices(mw: MtpWeights, host: np.ndarray | None = None) -> LayerMatrices:
    """The drafter's expert store for MoeDevice: entries 0..511 = the routed experts as BFP4 rows of their Q2_0
    values (MtpWeights.bfp4_rows), entry 512 = the shared expert (BF16, always resident); host: bool [512], the
    routed experts kept in pinned host memory (None: all in device DRAM)."""
    rows = mw.bfp4_rows()
    fmt = np.full((MTP_EXPERTS + 1, 3), F_BFP4, np.int32)
    fmt[MTP_EXPERTS] = F_BF16
    shared = [mw.get(f"mlp.shared_expert.{n}_proj.weight") for n in ("gate", "up", "down")]     # (n_out, n_in)

    def tiles(e: int, m: int) -> np.ndarray:
        if e == MTP_EXPERTS:
            return bf16_tiles(np.ascontiguousarray(shared[m].T))
        return qbfp.to_tiles(rows[e, m], "bfp4", *BFP4_SHAPE[m])

    hm = None if host is None or not host.any() else np.concatenate([np.asarray(host, bool), [False]])
    return LayerMatrices(fmt, tiles, lambda e, m: rows[e, m], hm)


# ------------------------------------------------------------------------------------------------ host row table
ROW_IQ4 = 1408                       # one IQ4_XS token embedding row (1360 B) padded to a 64 B multiple
ROWS_4K = 4096 // ROW_IQ4            # rows per 4 KiB page of an item: no row crosses a page (paged pool items)


class EmbedRows:
    """The draft vocabulary's token embedding rows (raw IQ4_XS, fnx.ref.embed) in pinned host memory: items first..
    of `pool` (a fnx.tt.hostmem.HostPool), ROWS_4K per 4 KiB page of an item, so a row never crosses a page (a
    pool's fallback items may be pinned 4 KiB pages without physical contiguity).  A call gathers the row of a device
    index and dequantizes it exactly to float32 on the device (kernels/mtp/row_gather.cpp: the row's bus address from
    a DRAM table, 8 per 64 B page, then a 1.4 KB read over PCIe; trace-safe).  The pool must outlive every captured
    round that gathers."""

    def __init__(self, mesh, w, vocab: np.ndarray, pool, first: int):
        from ..ref.embed import ROW_BYTES
        self.mesh, self.n = mesh, len(vocab)
        per = self.per_item(pool.item_bytes)
        if first + self.items(self.n, pool.item_bytes) > pool.n_items:
            raise ValueError(f"host pool of {pool.n_items} items: no room for the embedding rows from item {first}")
        raw = w.model.raw("token_embd.weight")
        addrs = np.empty(self.n, np.uint64)
        for i0 in range(0, self.n, per):
            rows = np.asarray(vocab[i0:i0 + per], np.int64)
            k = np.arange(len(rows))
            off = (k // ROWS_4K) * 4096 + (k % ROWS_4K) * ROW_IQ4
            data = pool.item(first + i0 // per)
            for j in range(len(rows)):
                data[off[j]:off[j] + ROW_BYTES] = raw[rows[j] * ROW_BYTES:(rows[j] + 1) * ROW_BYTES]
            addrs[i0:i0 + len(rows)] = pool.addr(first + i0 // per, off)
        tb = np.zeros((-(-self.n // 8) * 8, 2), np.uint32)
        tb[:self.n, 0], tb[:self.n, 1] = addrs & np.uint64(0xFFFFFFFF), addrs >> np.uint64(32)
        self.addrs = ttnn.from_torch(torch.from_numpy(tb.view(np.int32).reshape(1, 1, -1, 16)), dtype=ttnn.uint32,
                                     layout=RM, device=mesh, memory_config=DRAM)

    @staticmethod
    def per_item(item_bytes: int) -> int:
        return item_bytes // 4096 * ROWS_4K + min(item_bytes % 4096 // ROW_IQ4, ROWS_4K)

    @staticmethod
    def items(n: int, item_bytes: int) -> int:
        """Pool items holding n rows."""
        return -(-n // EmbedRows.per_item(item_bytes))

    def _program(self, idx, out) -> _Program:
        key = ("mtp_row_gather", self.n)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [0]
        k = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "row_gather.cpp"), source_type=FP,
                                  core_ranges=crs, compile_time_args=_acc(idx, self.addrs, out),
                                  named_compile_time_args=[("row", ROW_IQ4), ("n_rows", self.n)], runtime_args=rt,
                                  common_runtime_args=[0] * 3, config=READER)
        return _Program([k], [_cb(crs, 0, ttnn.uint32, 3 * 64 + ROW_IQ4 + 4 * N_EMBD, 1)], key)

    def __call__(self, idx):
        """idx: uint32 ROW_MAJOR DRAM tensor (word 0 = the row; an index >= n reads row n - 1) -> the row's float32
        embedding [1, 1, 1, N_EMBD] ROW_MAJOR DRAM (fnx.ref.embed.embed's values, bitwise)."""
        out = ttnn.empty([1, 1, 1, N_EMBD], F32, RM, self.mesh, DRAM)
        if not hasattr(self, "_prog"):
            self._prog = self._program(idx, out)
        self._prog([idx, self.addrs, out], [[idx.buffer_address(), self.addrs.buffer_address(),
                                              out.buffer_address()]])
        return out


def env_resident() -> int:
    """FNX_MTP_RESIDENT: routed experts the drafter keeps in device DRAM (default DEFAULT_RESIDENT)."""
    return int(os.environ.get("FNX_MTP_RESIDENT", DEFAULT_RESIDENT))


def reserve_items(resident: int | None = None, mw: MtpWeights | None = None) -> int:
    """TtModel(host_reserve=...) for the drafter (resident: default env_resident(); mw: its weights, default
    MtpWeights()): host_items of its vocabulary."""
    return host_items(env_resident() if resident is None else resident, len((mw or MtpWeights()).vocab))


def host_items(resident: int, n_vocab: int, item_bytes: int = ITEM_BYTES) -> int:
    """Items of item_bytes (>= one BFP4 expert matrix) the drafter needs in a host pool: its host-tier expert
    matrices ((512 - resident) * 3) then its embedding rows (TtModel(host_reserve=...): the shared pool)."""
    return max(MTP_EXPERTS - resident, 0) * 3 + EmbedRows.items(n_vocab, item_bytes)


# ------------------------------------------------------------------------------------------------ round inputs
def _zeros_rm(mesh, shape, dtype=F32):
    t = torch.zeros(shape, dtype=torch.int32 if dtype in (ttnn.int32, ttnn.uint32) else torch.float32)
    return ttnn.from_torch(t, dtype=dtype, layout=RM, device=mesh, memory_config=DRAM)


class RoundInputs:
    """Device inputs of a draft round with T catch-up rows and k steps, written by write() before the round (a traced
    round's exist before the captures: fnx.tt.trace.DraftTracer; an eager round allocates its own):
      e     f32 [1, 1, T, 2560] RM     the catch-up rows' token embeddings (host IQ4_XS rows, exact)
      C, S  f32 [1, T + k - 1, 1, 256] RM   RoPE tables of the cells p..p+a, then the chain cells p+a+1..p+a+k-1
      kv    u32 [1, 1, k, 32] RM       word 0 of row j: step j's K/V write position (p, then p+a+j)
      row   f32 [1, 1, 12, 1] TILE     the cell step 0 attends from (p + a; step j adds j on the device)"""

    def __init__(self, mesh, T: int, k: int):
        self.T, self.k = T, k
        self.e = _zeros_rm(mesh, (1, 1, T, N_EMBD))
        self.C, self.S = (_zeros_rm(mesh, (1, T + k - 1, 1, HEAD_DIM)) for _ in range(2))
        self.kv = _zeros_rm(mesh, (1, 1, k, 32), ttnn.uint32)
        self.row = ttnn.from_torch(torch.zeros(1, 1, 12, 1), dtype=F32, layout=TILE, device=mesh, memory_config=DRAM)

    def cells(self, p: int) -> list[int]:
        """The cell each step writes last and attends from: p + a + j (a = T - 1)."""
        return [p + self.T - 1 + j for j in range(self.k)]

    def write(self, mtp, tokens, p: int) -> None:
        """Round inputs for catch-up tokens tokens[:T] at cells p.. (tokens[t] = the token at p + t + 1)."""
        T, k = self.T, self.k
        e = embed(mtp.w, list(tokens)[:T])
        ttnn.copy_host_to_device_tensor(_host_rm(e.reshape(1, 1, T, N_EMBD)), self.e)
        cells = list(range(p, p + T)) + self.cells(p)[1:]
        c, s = rope_cos_sin(cells)
        C = np.ones((len(cells), HEAD_DIM), np.float32)
        S = np.zeros((len(cells), HEAD_DIM), np.float32)
        C[:, :ROPE_DIM] = np.concatenate([c, c], 1)
        S[:, :ROPE_DIM] = np.concatenate([-s, s], 1)
        ttnn.copy_host_to_device_tensor(_host_rm(C[None, :, None, :]), self.C)
        ttnn.copy_host_to_device_tensor(_host_rm(S[None, :, None, :]), self.S)
        kv = np.zeros((1, 1, k, 32), np.int32)
        kv[0, 0, :, 0] = [p] + self.cells(p)[1:]
        ttnn.copy_host_to_device_tensor(_host_rm(kv, ttnn.uint32), self.kv)
        row = np.full((1, 1, 12, 1), np.float32(p + T - 1), np.float32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(row), dtype=F32, layout=TILE), self.row)

    def free(self) -> None:
        _free(self.e, self.C, self.S, self.kv, self.row)


class _Step:
    """Step j of a round as Mtp.front / Mtp._attend read it: rows r0..r1 of the RoPE tables, row j of kv, the mask
    of the attending cell (row + j) over `span` cells (None: the eager round's own bucket, cells_row made here)."""

    def __init__(self, io: RoundInputs, j: int, span: int | None, cells_row):
        self.io, self.j, self.span, self.cells_row = io, j, span, cells_row
        self.r0, self.r1 = (0, io.T) if j == 0 else (io.T - 1 + j, io.T + j)
        self.kv = None

    @staticmethod
    def _slice(x, start, end):
        """ttnn.slice that never aliases: a full-range slice shares the buffer (freeing it would free x)."""
        if list(start) == [0] * 4 and list(end) == list(x.shape):
            return ttnn.clone(x, memory_config=DRAM)
        return ttnn.slice(x, start, end)

    def rope(self):
        out = []
        for t in (self.io.C, self.io.S):
            s = self._slice(t, [0, self.r0, 0, 0], [1, self.r1, 1, HEAD_DIM])
            out.append(ttnn.to_layout(s, TILE))
            ttnn.deallocate(s)
        return out

    def kv_pos(self):
        return self._slice(self.io.kv, [0, 0, self.j, 0], [1, 1, self.j + 1, 32])

    def mask(self, mesh, n_pad: int):
        """-1e9 where cell > the attending row's cell, over n_pad cells (f32 [1, 1, 12, n_pad] TILE)."""
        own = self.cells_row is None
        cr = _dev(mesh, np.arange(n_pad, dtype=np.float32).reshape(1, 1, 1, n_pad)) if own else self.cells_row
        row = self.io.row if self.j == 0 else ttnn.add(self.io.row, float(self.j))
        later = ttnn.gt(cr, row)                                # 1.0 where cell > the attending row's cell
        m = ttnn.multiply(later, MASKED)
        _free(later, *((cr,) if own else ()), *((row,) if row is not self.io.row else ()))
        return m


def draft_head_parts(mw: MtpWeights, w) -> tuple[int, list]:
    """(Vp, host tensors): the draft head W^T [2560, Vp] bfloat8_b (W_out rows of mw.vocab, Vp = n rounded up to 32,
    zero columns beyond n) as the converted (cached) tile-aligned column parts of at most MAX_PART elements."""
    vocab = mw.vocab
    n = len(vocab)
    vp = -(-n // 32) * 32
    tag = hashlib.sha1(vocab.tobytes()).hexdigest()[:12]
    chunk = (MAX_PART // N_EMBD) // 32 * 32
    rows = None
    parts = []
    for i, v0 in enumerate(range(0, vp, chunk)):
        def build(v0=v0):
            nonlocal rows
            if rows is None:
                rows = np.zeros((vp, N_EMBD), np.float32)
                rows[:n] = mw.head_rows(w)
            return rows[v0:v0 + chunk].T
        host = None
        name = f"{TT_CACHE}/mtp.draft_head.{tag}.p{vp}.T.part{chunk}_{i}_{BFP8.name}"
        if not os.path.isfile(f"{name}_dtype_{BFP8.name}_layout_{TILE.name}.tensorbin"):
            host = torch.from_numpy(np.ascontiguousarray(build(), np.float32))[None, None]
        parts.append(ttnn.as_tensor(host, dtype=BFP8, layout=TILE, cache_file_name=name))
    return vp, parts


# ------------------------------------------------------------------------------------------------ the drafter
class Mtp:
    def __init__(self, mesh, w, mw: MtpWeights | None = None, max_ctx: int = 2048, hnorm: str = "pooled",
                 steps: int = 3, window: int = 32768, proj_dtype=BF16, precise: bool = True,
                 resident: int = DEFAULT_RESIDENT, pool=None, pool_first: int = 0):
        """w: fnx.weights.Weights (main model: embedding, head rows); resident: the number of routed experts in
        device DRAM (the hottest of the routing profile, host_experts); the other experts' matrices and the
        draft-vocabulary embedding rows go to items pool_first.. of `pool` (a fnx.tt.hostmem.HostPool of >= 921600 B
        items with host_items(...) free items: TtModel's, built with host_reserve) or, pool=None, to the drafter's
        own pool."""
        if hnorm not in ("pooled", "stream"):
            raise ValueError(hnorm)
        self.mesh, self.w = mesh, w
        self.mw = mw = mw or MtpWeights()
        self.hnorm, self.steps, self.window = hnorm, steps, window
        self.precise = precise
        self.probs = True       # draft probabilities (Strata's min_p chain stop); False: skipped, reported as NaN
        self.cells = -(-(max_ctx + steps + 1) // SPAN) * SPAN
        if window < self.cells:
            raise ValueError("an attention window below the context is not implemented (Strata's default is 32768)")
        g = mw.get
        self.ge = _cached(mesh, "mtp.pre_fc_norm_embedding", F32,
                          lambda: g("pre_fc_norm_embedding.weight").reshape(1, 1, 1, N_EMBD))
        gh = np.asarray(g("pre_fc_norm_hidden.weight"), np.float64)
        # pooled: rsqrt(sum/10240 + eps) = sqrt(10240) rsqrt(sum + 10240 eps); stream: sqrt(2560), per stream
        self.gh = _cached(mesh, f"mtp.pre_fc_norm_hidden.x_sqrt.{hnorm}", F32, lambda: (
            gh * np.sqrt(HC_DIM if hnorm == "pooled" else N_EMBD)).astype(f32).reshape(1, HC_DIM))
        ps = np.zeros((32, 32), np.float32)                    # stream sums -> the sums the norm divides by
        ps[:HC, :HC] = 1.0 if hnorm == "pooled" else np.eye(HC)
        self.pool = _dev(mesh, ps, BF16) if mesh is not None else None
        self.fc_e = _cached(mesh, "mtp.fc_embedding.T", proj_dtype, lambda: g("fc_embedding.weight").T)
        self.fc_h = _cached(mesh, "mtp.fc_hidden.T", proj_dtype, lambda: g("fc_hidden.weight").T)
        self.hc_attn, self.hc_ffn, self.hc_out = (HcRead(mesh, mw, p) for p in HC_PREFIX)

        def qkvg():
            wq = g("self_attn.q_proj.weight").reshape(N_HEAD, 2, HEAD_DIM, N_EMBD)
            return np.concatenate([wq[:, 0].reshape(QD, N_EMBD), wq[:, 1].reshape(QD, N_EMBD),
                                   g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")]).T     # [2560, 13312]
        self.w_qkvg = _cached(mesh, "mtp.attn_qkvg.T", proj_dtype, qkvg)
        self.w_o = _cached(mesh, "mtp.attn_output.T", proj_dtype, lambda: g("self_attn.o_proj.weight").T)  # [6144, 2560]
        self.qn = _cached(mesh, "mtp.attn_q_norm.x0.0625", F32,
                          lambda: (g("self_attn.q_norm.weight") * f32(HEAD_DIM ** -0.5)).reshape(1, 1, 1, HEAD_DIM))
        self.kn = _cached(mesh, "mtp.attn_k_norm", F32, lambda: g("self_attn.k_norm.weight").reshape(1, 1, 1, HEAD_DIM))
        self._head(mesh)
        self.last: dict = {}
        self._trace = None                # fnx.tt.trace.DraftTracer (TtModel.enable_trace(drafter=this))
        if mesh is None:              # host-only cache preparation (main(): --prepare)
            return
        from .hostmem import HostPool
        lm = expert_matrices(mw, host_experts(mw, resident))
        n_mats = host_matrices(lm)[0]
        n_items = n_mats + EmbedRows.items(len(self.vocab), ITEM_BYTES)
        self.own_pool = None
        if pool is None:        # its own (chunk "auto"): it lives as long as this drafter, i.e. as long as its traces
            pool = self.own_pool = HostPool(n_items, ITEM_BYTES, "auto")
            pool_first = 0
        self.moe = MoeDevice(mesh, w, MTP_LAYER, lm, router=router_matrix(mw), pool=pool if n_mats else None,
                             pool_first=pool_first)
        self.en_rows = EmbedRows(mesh, w, self.vocab, pool, pool_first + n_mats)
        self.pinned = (self.own_pool.nbytes if self.own_pool is not None else       # pinned bytes it occupies
                       (n_mats + EmbedRows.items(len(self.vocab), pool.item_bytes)) * pool.item_bytes)
        self.K = ttnn.zeros([1, 1, N_HEAD_KV * self.cells, HEAD_DIM], dtype=F32, layout=RM, device=mesh,
                            memory_config=DRAM)
        self.V = ttnn.zeros([1, 1, N_HEAD_KV * self.cells, HEAD_DIM], dtype=F32, layout=RM, device=mesh,
                            memory_config=DRAM)
        self.K4 = ttnn.reshape(self.K, [1, N_HEAD_KV, self.cells, HEAD_DIM])
        self.V4 = ttnn.reshape(self.V, [1, N_HEAD_KV, self.cells, HEAD_DIM])
        # round result columns: step j's draft index at j, its probability at steps + j
        self.unit = [_dev(mesh, np.eye(32, dtype=np.float32)[i].reshape(1, 1, 1, 32)) for i in range(2 * steps)]

    # ---- setup
    def _head(self, mesh) -> None:
        """W_out rows of the draft vocabulary as one bank-packed W^T [2560, Vp] bfloat8_b (Vp = n rounded up to 32, zero
        columns beyond n, masked by vmask): its tile-aligned column parts placed by exact page copies
        (fnx.tt.packed_matmul.place_packed) for the PackedMatmul of head_logits."""
        self.vocab, n = self.mw.vocab, len(self.mw.vocab)
        vp, parts = draft_head_parts(self.mw, self.w)
        if mesh is None:
            return
        self.tile_argmax = TileArgmax(mesh)
        grid = mesh.compute_with_storage_grid_size()
        pcn = -(-(-(-vp // (grid.x * grid.y))) // 32)       # ttnn's auto 1D choice for [2560, Vp]: 31 columns
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=2, out_subblock_h=1,
            out_subblock_w=max(i for i in range(1, 5) if pcn % i == 0), per_core_M=1, per_core_N=pcn, fuse_batch=True,
            fused_activation=None, mcast_in0=True)
        layout = PackedLayout(N_EMBD // 32, vp // 32, pcn)
        self.head_w = ttnn.empty([1, 1, N_EMBD, vp], BFP8, TILE, mesh, DRAM)
        col = 0
        for t in parts:
            part = ttnn.to_device(t, mesh, memory_config=DRAM)
            place_packed(part, self.head_w, col // 32, layout)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(part)
            col += int(t.shape[-1])
        assert col == vp, (col, vp)
        self.head_mm = PackedMatmul(self.head_w, pc, HIFI4, F32, DRAM)
        vmask = np.zeros((1, 1, 1, vp), np.float32)
        vmask[..., n:] = -1e30
        self.vmask = _dev(mesh, vmask)

    def head_logits(self, smix):
        """smix [1,1,T,2560] f32 -> draft-vocabulary logits [1,1,T,Vp] f32 (before vmask): one packed dual-NoC matmul,
        bitwise the four auto-config column-group matmuls + concat it replaced (0.93 -> 0.71 ms traced, test_packed).
        The bfloat8_b rows' own error (0.64% rel L2 of the logits, measured) dominates the float32 activation's TF32
        rounding."""
        return self.head_mm(smix)

    def reset(self) -> None:
        """New sequence: nothing to clear (the K/V cache is positional, see the module docstring)."""
        self.last = {}

    def state_tensors(self) -> list:
        """Non-positional per-sequence device state: none."""
        return []

    # ---- pieces
    def _mm(self, x, w):
        """x float32 [1, 1, M, K] @ w (BF16) -> float32; precise: x split exactly into hi (bf16-valued) + lo, both
        as the batch of ONE matmul (the weights are read once instead of twice, fnx.tt.hc.mm's two), then hi + lo."""
        if not self.precise:
            return ttnn.linear(x, w, dtype=F32, compute_kernel_config=HIFI4, memory_config=DRAM)
        M, N = int(x.shape[-2]), int(w.shape[-1])
        hi = ttnn.typecast(x, BF16)
        hi32 = ttnn.typecast(hi, F32)
        lo = ttnn.subtract(x, hi32)
        xs = ttnn.concat([hi32, lo], dim=1)                                     # [1, 2, M, K]
        p = ttnn.matmul(xs, w, dtype=F32, compute_kernel_config=HIFI4)
        a, b = ttnn.slice(p, [0, 0, 0, 0], [1, 1, M, N]), ttnn.slice(p, [0, 1, 0, 0], [1, 2, M, N])
        out = ttnn.add(a, b)
        _free(hi, hi32, lo, xs, p, a, b)
        return out

    def _rope_tables(self, cells):
        c, s = rope_cos_sin(cells)                                          # [T, 32]
        C = np.ones((len(cells), HEAD_DIM), f32)
        S = np.zeros((len(cells), HEAD_DIM), f32)
        C[:, :ROPE_DIM] = np.concatenate([c, c], 1)
        S[:, :ROPE_DIM] = np.concatenate([-s, s], 1)
        return _dev(self.mesh, C[None, :, None, :]), _dev(self.mesh, S[None, :, None, :])

    @staticmethod
    def _rope(x, C, S):
        """x [1, T, H, 256] f32: NEOX pairs (i, i+32) on dims 0..63 (same construction as fnx.tt.qsa)."""
        T, H = x.shape[1], x.shape[2]
        lo = ttnn.slice(x, [0, 0, 0, 0], [1, T, H, ROPE_DIM // 2])
        hi = ttnn.slice(x, [0, 0, 0, ROPE_DIM // 2], [1, T, H, ROPE_DIM])
        rest = ttnn.slice(x, [0, 0, 0, ROPE_DIM], [1, T, H, HEAD_DIM])
        sw = ttnn.concat([hi, lo, rest], dim=-1)
        xc, ss = ttnn.multiply(x, C), ttnn.multiply(sw, S)
        out = ttnn.add(xc, ss)
        _free(lo, hi, rest, sw, xc, ss)
        return out

    def _hnorm(self, R):
        """rms of Rin (pooled over 10240, or per stream) times g_h: [1,1,T,10240] f32."""
        sq = ttnn.multiply(R, R)
        s = mm(sq, consts(self.mesh)[0])                                    # per-stream sums, cols 0..3
        tot = mm(s, self.pool)                                              # pooled: every col 0..3 = the total
        eps = float(np.float32(HC_DIM if self.hnorm == "pooled" else N_EMBD) * EPS)
        rs = ttnn.rsqrt(ttnn.add(tot, eps))
        rsb = mm(rs, consts(self.mesh)[1])
        xg = ttnn.multiply(R, self.gh)
        out = ttnn.multiply(xg, rsb)
        _free(sq, s, tot, rs, rsb, xg)
        return out

    def _upload_R(self, R):
        if isinstance(R, ttnn.Tensor):
            return R, False
        a = np.ascontiguousarray(np.asarray(R, f32).reshape(-1, HC_DIM))
        return _dev(self.mesh, a[None, None]), True

    def _embed_host(self, tokens):
        """Token embeddings (host IQ4_XS rows, exact) -> device f32 [1, 1, T, 2560] TILE."""
        return _dev(self.mesh, embed(self.w, list(tokens))[None, None])

    def front(self, R_in, e, cells, step: _Step | None = None) -> dict:
        """K/V part for T rows (R_in device [1,1,T,10240] f32; e = the tokens' embeddings, device f32 [1,1,T,2560]
        TILE, freed here) at consecutive cells.  Returns device tensors R [1,1,T,10240], inj, qr [1,T,24,256] (q
        normed+roped, scaled), gate [1,1,T,6144].  step (a round's _Step): RoPE tables and the K/V position from the round's device inputs
        (cells only give T)."""
        cells = [int(c) for c in cells]
        T = len(cells)
        if any(b - a != 1 for a, b in zip(cells, cells[1:])) or cells[0] < 0 or cells[-1] >= self.cells:
            raise ValueError(f"cells {cells[0]}..{cells[-1]} not consecutive within {self.cells}")
        en = ttnn.rms_norm(e, weight=self.ge, epsilon=EPS, compute_kernel_config=NORM_CKC)
        e2 = self._mm(en, self.fc_e)
        hn = self._hnorm(R_in)
        hs = ttnn.reshape(hn, [1, 1, T * HC, N_EMBD])
        h2 = self._mm(hs, self.fc_h)
        h2r = ttnn.reshape(h2, [1, 1, T, HC_DIM])
        e4 = ttnn.repeat(e2, ttnn.Shape([1, 1, 1, HC]))
        R = ttnn.add(h2r, e4)
        _free(e, en, e2, hn, *(() if hs is hn else (hs,)), h2, *(() if h2r is h2 else (h2r,)), e4)
        mixed, inj = self.hc_attn.forward(R)
        qkv = self._mm(mixed, self.w_qkvg)
        ttnn.deallocate(mixed)
        q = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, T, QD]), [1, T, N_HEAD, HEAD_DIM])
        gate = ttnn.slice(qkv, [0, 0, 0, QD], [1, 1, T, 2 * QD])
        k = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, 2 * QD], [1, 1, T, 2 * QD + KD]), [1, T, N_HEAD_KV, HEAD_DIM])
        v = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, 2 * QD + KD], [1, 1, T, 2 * QD + 2 * KD]),
                         [1, T, N_HEAD_KV, HEAD_DIM])
        ttnn.deallocate(qkv)
        C, S = self._rope_tables(cells) if step is None else step.rope()
        qn = ttnn.rms_norm(q, weight=self.qn, epsilon=EPS, compute_kernel_config=NORM_CKC)
        kn = ttnn.rms_norm(k, weight=self.kn, epsilon=EPS, compute_kernel_config=NORM_CKC)
        qr, kr = self._rope(qn, C, S), self._rope(kn, C, S)
        _free(q, k, qn, kn, C, S)
        pos = step.kv_pos() if step is not None else None
        G = KV_ROWS_MAX // N_HEAD_KV                    # tokens per kv_write call
        for src, dst in ((kr, self.K), (v, self.V)):
            hm = ttnn.reshape(src, [1, N_HEAD_KV, 1, HEAD_DIM]) if T == 1 else ttnn.permute(src, (0, 2, 1, 3))
            rm = ttnn.to_layout(hm, RM)
            if step is not None:
                kv_write(rm, dst, pos, self.cells)
            else:
                # eager (prompt cells, a request's committed cells): the position goes in as device data, so one
                # program serves every cell (a slice_write bakes its start cell into the program: a compile, and so
                # a trace recapture, at almost every request)
                for t0 in range(0, T, G):
                    t1 = min(T, t0 + G)
                    part = rm if t1 - t0 == T else ttnn.slice(rm, [0, 0, t0, 0], [1, N_HEAD_KV, t1, HEAD_DIM])
                    pd = _pos_dev(self.mesh, cells[0] + t0)
                    kv_write(part, dst, pd, self.cells)
                    _free(pd, *(() if part is rm else (part,)))
            _free(rm, *(() if hm is src else (hm,)))
        _free(kr, v, pos)
        return {"R": R, "inj": inj, "qr": qr, "gate": gate, "T": T}

    def _free_front(self, fr: dict) -> None:
        _free(fr["R"], fr["inj"], fr["qr"], fr["gate"])

    def _cells(self, cache4, n_pad: int):
        """Cells [0, n_pad) of every kv head as float32 TILE [1, 2, n_pad, 256] (a dense prefix: a slice, no
        gather; ttnn.embedding only takes bfloat16 tables).  All cells: the cache itself (a full-range slice would
        alias it, and freeing that would free the cache)."""
        rm = cache4 if n_pad == self.cells else ttnn.slice(cache4, [0, 0, 0, 0], [1, N_HEAD_KV, n_pad, HEAD_DIM])
        out = ttnn.to_layout(rm, TILE, memory_config=DRAM)
        if rm is not cache4:
            ttnn.deallocate(rm)
        return out

    def _attend(self, fr: dict, t: int, cell: int, step: _Step):
        """bo [1,1,1,2560] f32 of row t at `cell`: dense attention over cells 0..cell (over step.span cells when the
        round is traced; the padding cells change no bit: both matmuls accumulate sequentially)."""
        n_pad = step.span or -(-(cell + 1) // SPAN) * SPAN
        k, v = self._cells(self.K4, n_pad), self._cells(self.V4, n_pad)
        T = fr["T"]
        qt = fr["qr"] if T == 1 else ttnn.slice(fr["qr"], [0, t, 0, 0], [1, t + 1, N_HEAD, HEAD_DIM])
        qg = ttnn.reshape(qt, [1, N_HEAD_KV, GROUP, HEAD_DIM])
        mask = step.mask(self.mesh, n_pad)
        s = mm2(qg, k, transpose_b=True, ckc=ATTN_CKC)
        sm = ttnn.add(s, mask)
        mx = ttnn.max(sm, dim=-1, keepdim=True)
        z = ttnn.subtract(sm, mx)          # in float32 first: the logits reach ~600, where bfloat16 steps by 4
        pr = ttnn.softmax(z, dim=-1, numeric_stable=True, compute_kernel_config=NORM_CKC)
        _free(mx, z)
        o = mm2(pr, v, ckc=ATTN_CKC)                                                 # [1, 2, 12, 256]
        att = ttnn.reshape(o, [1, 1, 1, QD])
        gt = fr["gate"] if T == 1 else _rows(fr["gate"], t, t + 1)
        g = ttnn.sigmoid(gt)
        a = ttnn.multiply(att, g)
        bo = self._mm(a, self.w_o)
        _free(k, v, mask, s, sm, pr, o, *(() if att is o else (att,)), g, a, *(() if qt is fr["qr"] else (qt,)),
              *(() if qg is qt else (qg,)), *(() if gt is fr["gate"] else (gt,)))
        return bo

    def rest(self, fr: dict, t: int, cell: int, step: _Step, keep_logits: bool = False):
        """The rest of the layer for row t of front() at `cell` -> device (R_out [1,1,1,10240], draft subset index
        uint32 [1,1,1,1] RM, probability f32 [1,1,1,1], the row-major subset logits if keep_logits: fnx.tt.tile_argmax's
        [1,1,k,Vp/k] (k = 11), read it flattened)."""
        bo = self._attend(fr, t, cell, step)
        T = fr["T"]
        R = fr["R"] if T == 1 else _rows(fr["R"], t, t + 1)
        inj = fr["inj"] if T == 1 else _rows(fr["inj"], t, t + 1)
        R1 = hc_write(R, bo, inj)
        _free(bo, *(() if R is fr["R"] else (R,)), *(() if inj is fr["inj"] else (inj,)))
        mixed2, inj2 = self.hc_ffn.forward(R1)
        y = self.moe.forward(mixed2)
        R_out = hc_write(R1, y, inj2)
        _free(R1, mixed2, inj2, y)
        smix, _ = self.hc_out.forward(R_out)
        lg = self.head_logits(smix)
        lgm = ttnn.add(lg, self.vmask)
        _free(smix, lg)
        prob = None
        if self.probs:                     # 4 passes over the 106K logits: only for a min_p chain stop
            mx = ttnn.max(lgm, dim=-1, keepdim=True)
            z = ttnn.subtract(lgm, mx)
            e = ttnn.exp(z)
            s = ttnn.sum(e, dim=-1, keepdim=True)
            prob = ttnn.reciprocal(s)
            _free(mx, z, e, s)
        ids, lr = self.tile_argmax(lgm, keep_rm=True) if keep_logits else (self.tile_argmax(lgm), None)
        ttnn.deallocate(lgm)
        return R_out, ids, prob, lr

    # ---- rounds
    def prefill(self, R_rows, next_tokens, cell0: int, chunk: int = 32) -> None:
        """K/V of prompt cells cell0.. (R_rows = the final residuals at those positions, next_tokens = tokens at
        cell+1), in chunks of `chunk` rows.  R_rows stays the caller's: a full-range ttnn.slice shares its buffer, so
        only real sub-slices (and host uploads) are freed here."""
        n = len(next_tokens)
        dev = isinstance(R_rows, ttnn.Tensor)
        for c in range(0, n, chunk):
            c1 = min(c + chunk, n)
            alias = dev and c == 0 and c1 == R_rows.shape[-2]
            Rc = R_rows if alias else (_rows(R_rows, c, c1) if dev else self._upload_R(np.asarray(R_rows)[c:c1])[0])
            fr = self.front(Rc, self._embed_host(next_tokens[c:c1]), range(cell0 + c, cell0 + c1))
            self._free_front(fr)
            if not alias:
                ttnn.deallocate(Rc)

    def round_dev(self, io: RoundInputs, Rin, cells: list[int], span: int | None = None, cells_row=None,
                  keep_logits: bool = False):
        """A whole draft round on the device (eager, or captured: fnx.tt.trace.DraftTracer with a fixed attention
        span and its cells_row): io = the written RoundInputs, Rin = the window's final residual rows
        [1, 1, T, 10240] (the caller's), cells = io.cells(p) (only their count and order matter when captured).
        Step j > 0 embeds step j - 1's draft by gathering its row from the host table at the device argmax.
        Returns (res f32 [1, 1, 1, 32] TILE: column j = step j's draft subset index, column k + j its probability;
        the row-major subset logits of every step if keep_logits)."""
        T, k = io.T, io.k
        res, lrs, R, ids = None, [], Rin, None
        for j in range(k):
            st = _Step(io, j, span, cells_row)
            if j == 0:
                e = ttnn.to_layout(io.e, TILE)
            else:
                g = self.en_rows(ids)
                e = ttnn.to_layout(g, TILE)
                ttnn.deallocate(g)
            n = T if j == 0 else 1
            fr = self.front(R, e, range(cells[j] - n + 1, cells[j] + 1), step=st)
            if R is not Rin:
                ttnn.deallocate(R)
            _free(ids)
            R, ids, prob, lr = self.rest(fr, n - 1, cells[j], st, keep_logits)
            self._free_front(fr)
            idt = ttnn.to_layout(ids, TILE)
            idf = ttnn.typecast(idt, F32)
            ab = ttnn.multiply(idf, self.unit[j])
            if prob is not None:
                b = ttnn.multiply(prob, self.unit[k + j])
                a, ab = ab, ttnn.add(ab, b)
                _free(a, b, prob)
            _free(idt, idf)
            if res is None:
                res = ab
            else:
                r2 = ttnn.add(res, ab)
                _free(res, ab)
                res = r2
            lrs.append(lr)
        _free(R, ids)
        return res, lrs

    def decode(self, res: np.ndarray, steps: int, min_p: float) -> tuple[list[int], list[float]]:
        """A round's read-back result -> (drafts, probs): the first `steps`, cut after the first draft whose
        probability is below min_p (> 0), as Strata's chain stops."""
        k = self.steps
        drafts, probs = [], []
        for j in range(min(steps, k)):
            drafts.append(int(self.vocab[min(int(res[j]), len(self.vocab) - 1)]))
            probs.append(float(res[k + j]) if self.probs else float("nan"))
            if min_p > 0 and probs[-1] < min_p:
                break
        return drafts, probs

    def draft(self, R_rows, outv, p: int, a: int, steps: int | None = None, min_p: float = 0.0,
              trace: list | None = None):
        """A round after a verify window at p (see fnx.ref.mtp.MtpRef.draft): R_rows = final residuals of window rows
        0..a (device [1,1,>=a+1,10240] or host), outv = window picks.  Returns (drafts, probs); `trace` (list,
        eager only) receives each step's {"logits": subset logits}.  Traced (TtModel.enable_trace(drafter=self))
        the round replays its captured segment (device R_rows, a traced row count, cells in the cache; else
        eagerly)."""
        t0 = time.perf_counter()
        steps = self.steps if steps is None else steps
        if steps > self.steps:
            raise ValueError(f"{steps} draft steps: the drafter was built for {self.steps}")
        if self._trace is not None and isinstance(R_rows, ttnn.Tensor) and trace is None:
            out = self._trace.draft(R_rows, outv, p, a, steps, min_p)
            if out is not None:
                self.last["ms"] = (time.perf_counter() - t0) * 1e3
                return out
        T = a + 1
        if isinstance(R_rows, ttnn.Tensor):
            Rin = R_rows if R_rows.shape[-2] == T else _rows(R_rows, 0, T)
            own = Rin is not R_rows
        else:
            Rin, own = self._upload_R(np.asarray(R_rows, f32).reshape(-1, HC_DIM)[:T])
        io = RoundInputs(self.mesh, T, self.steps)
        io.write(self, outv, p)
        res, lrs = self.round_dev(io, Rin, io.cells(p), keep_logits=trace is not None)
        out = ttnn.to_torch(res).float().numpy().reshape(-1)
        _free(res)
        io.free()
        if own:
            ttnn.deallocate(Rin)
        drafts, probs = self.decode(out, steps, min_p)
        for lr in lrs:
            if trace is not None:
                trace.append({"logits": ttnn.to_torch(lr).float().numpy().reshape(-1)[:len(self.vocab)]})
            _free(lr)
        self.last["ms"] = (time.perf_counter() - t0) * 1e3
        return drafts, probs


def main() -> int:
    """Host-only conversion of the drafter's device weight caches (no device; ttnn's block-float packing needs a
    cluster descriptor, a mock one does) and its BFP4 expert rows:
        TT_METAL_MOCK_CLUSTER_DESC_PATH=<p100_cluster_desc.yaml> NODEV=1 tools/devrun.sh mtp-prep -m fnx.tt.mtp"""
    from ..weights import Weights
    t0 = time.time()
    m = Mtp(None, Weights())
    m.mw.bfp4_rows()
    print(f"fnx.tt.mtp: device weight caches ready under {TT_CACHE} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
