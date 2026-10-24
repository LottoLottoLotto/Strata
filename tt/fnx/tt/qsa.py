"""TTNN QSA layer (Qwen Sparse Attention, layers 3, 7, ..., 47) for one P150, exact to Strata's cell selection at every
position (dense up to 2050, the indexer's 2051 cells beyond).

Math: fnx/ref/qsa.py (spec Part 5). Per call, T consecutive tokens (the logical rows of `mixed`):
  1. one fused projection [q(24x256) | gate(24x256) | k(2x256) | v(2x256)] (bfloat8_b weight, HiFi4, f32 out). The
     GGUF's per-head [q|gate] rows of attn_q are regrouped at load time so q and gate are tile-aligned column slices.
  2. per-head RMSNorm of q and k (ttnn.rms_norm in f32, GGUF gammas as-is) and partial NEOX RoPE on dims 0..63 from
     the reference's float64 angle table, in f32 (q and k heads in one pass). The softmax scale 1/16 (exact power of
     two) is folded into q's gamma.
  3. K/V of every row are written into the cache first (kv_append: one program rounds kr and v to bfloat16 and writes
     the rows), then every row attends over its selected cells (below): K/V of those cells are gathered from the cache
     with ttnn.embedding, then scores = q K^T (f32), + additive mask (-1e9 on padding / later cells), softmax, @ V (f32).
     Both matmuls accumulate sequentially in fp32 DEST (ATTN_CKC: no packer L1 accumulation), so a row's attention is
     bitwise the same in calls of any row count (verify windows == T=1 decode).
  4. attn * sigmoid(gate) (one multiply with the sigmoid fused on its input) -> attn_output (bfloat8_b) -> bo float32
     [1, 1, T, 2560].
  Decode calls (T <= 32) run both projections with explicit 1D program configs that keep ttnn's K blocking.

KV cache: two bfloat16 ROW_MAJOR tensors [1, 1, 2 * cells, 256] (K, V), row j * cells + c = kv head j of cell c, so
one embedding call gathers both heads of a cell list straight into the [1, 2, n, 256] TILE layout the matmuls read.

Cell selection (spec §4.4-4.6; qsa.hpp:133-136):
  - position p <= 2050 (n_kv = p + 1 <= 2051): every cell 0..p. The row group gathers cells 0..n_pad-1 (n_pad = the
    last row's n_kv rounded up to 256) and masks cells > p.
  - p >= 2051: exactly 2051 cells = the tail (n_kv % 4 cells, forced by the +1e9 bias) + the 512 best complete
    4-cell blocks by score, then the first 3 - n_kv % 4 cells of the 513th, ties to the lowest block index. Each row
    gathers its own 2051 cells padded to S_SEL = 2304 (padding = cell 0, masked); rows are batched GMAX per group.
    Measured (scratch probes, T=1, traced): this costs 0.25 ms per layer at any context, against 0.18 / 0.92 / 3.63 ms
    for dense attention with a selection mask over 2K / 8K / 32K cells.
Block scores score[b] = sum_h relu(pooled[b] . qi[h]) (qi = normed + roped indexer query [4, 128]):
  - device: the indexer query (indexer.q_proj BF16 weight, HiFi4, f32 out; q_norm; RoPE at the row position) times
    the device pooled-key table P2 [blocks, 256] bfloat16 = [hi | lo] with hi + lo = the reference's f32 pooled key
    to 2^-16 (the query is fed twice, [qi | qi]), relu, summed over the heads: s~[t, b] for every block completed
    before the call (b < pos[0] // 4).
  - host: the exact indexer state, positional (f16 raw keys per cell, f32 pooled keys per block, dead key), computed
    from the read-back `mixed` rows with the reference's own expressions (fnx.ref.qsa: kr = f16(x Wk^T) per call,
    pool -> norm -> rope at 4b), so it equals QsaState bit for bit; blocks completed in this call are scored exactly
    on the host (fnx.ref.qsa.block_scores), and their P2 rows are uploaded (paged_fill_cache by 32-row pages) for
    later calls.  The single host round trip of the layer reads [mixed | s~] back once (one untilize, one transfer).
    Why the keys are not projected on the device: the exact rescoring needs the reference's keys anyway, and a device
    k_proj (relative error ~3e-4 at HiFi4, the f16 half-ulp is 2.4e-4) would round many raw keys to the neighbouring
    f16 value, i.e. a second, approximate key state that only widens the bound; the host keys cost ~10 us per row.
  - exact ranks: |s~_b - s_b| <= SCORE_ERR * sum_h |qi_h| * |pooled_b|, SCORE_ERR = 2^-8: by Cauchy-Schwarz the
    device query's relative error (measured max 2.4e-3 over 12410 sparse rows of layers 3 and 7) plus the matmul's
    TF32 rounding of the f32 query (<= 2^-10) is 3.4e-3 < 3.9e-3; the largest measured ratio is 7.6e-4 (5.2x below,
    fnx/tests/test_qsa.py sparse checks every device score).  As in fnx.tt.moe, a block is surely in the top K if its
    lower bound exceeds the (K+1)-th largest upper bound, surely out if its upper bound is below the K-th largest lower
    bound; the undecided blocks are rescored exactly on the host (block_scores: the reference's arithmetic, the same
    bits) and fill the remaining slots by (score desc, block asc).  K = 512 and, when 3 - n_kv % 4 > 0, K = 513 (the
    513th block = the difference).  The selected ids therefore equal the reference's (state.last_ids <-> last_ids).
  Measured on layers 3 and 7 (fnx/tests/test_qsa.py sparse / sparse32k): every one of 12410 (8K) + 30749 (32K)
  sparse rows selects exactly the reference's cells, ~300-420 blocks rescored per row; T=1 step 0.62 ms device at
  2K (dense), 0.76 ms at 2K/8K and 0.79 ms at 32K (sparse, traced), 2.0-3.6 ms eager incl. the round trip.
The device and host indexer state are positional (cell / block indexed, never cleared): a snapshot of the first L
tokens stays valid while cells [0, L) are kept, like the KV cache (fnx.tt.model.snapshot).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

from .. import paths
from ..config import COMPRESS, EPS, HEAD_DIM, IDX_DIM, IDX_HEADS, N_EMBD, N_HEAD, N_HEAD_KV, ROPE_DIM
from ..cpu_experts import assemble_selection
from ..ref.qsa import DENSE_MAX_POS, GROUP, SEL_WIDTH, block_scores, rms, rope64, rope_cos_sin
from .head import create_matmul_1d_decode_progcfg

SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
ROPE2 = True
CACHE = str(paths.TT_CACHE)
QD = N_HEAD * HEAD_DIM            # 6144
KD = N_HEAD_KV * HEAD_DIM         # 512
MASKED = -1e9
SPAN = 256                       # dense scored cells are rounded up to a multiple of this (padding masked): program
                                 # shapes change every 256 positions instead of every tile (each change recompiles)
S_SEL = -(-SEL_WIDTH // SPAN) * SPAN     # 2304 gathered cells per sparse row (2080 measured 28% slower: 65 tiles)
GMAX = 16                        # sparse rows per attention group (per row: 0.25 ms alone, 0.134 ms in groups of 16)
BSPAN = 1024                     # device block-score columns: blocks rounded up to this (recompile every 4096 pos)
TOPK_BLOCKS = (SEL_WIDTH - (COMPRESS - 1)) // COMPRESS       # 512 complete blocks always taken
SCORE_ERR = 2.0 ** -8            # device block-score error bound per unit of sum_h |qi_h| |pooled_b| (docstring)
f32, f16 = np.float32, np.float16
DRAM = ttnn.DRAM_MEMORY_CONFIG
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)
NORM_CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=False)
# q K^T and P V: fp32 accumulation in DEST, K blocks spilled/reloaded exactly (no packer L1 accumulation), i.e. one
# sequential sum per output element whatever K blocking ttnn's auto program picks.  With packer_l1_acc=True the
# blocking (which differs between one and two M tiles, d*12 <= 32 vs > 32 query rows) regroups the sum, and rows of a
# 3..5-row verify window differed from the same rows alone (2.9e-5 after layer 7; fnx/tests/test_spec.py qsa / inv).
ATTN_CKC = NORM_CKC
KDIR = Path(__file__).resolve().parent / "kernels" / "qsa"
POS_PAGE = 128                   # the uint32 position tensor of kv_write / kv_append: [1, 1, 1, 32] row-major, word 0
KV_ROWS_MAX = 32                 # rows per kv_write call (n_kv_heads * T, T <= GMAX)


def cached_tensor(mesh, name: str, dtype, build, layout=ttnn.TILE_LAYOUT):
    """ttnn.as_tensor with the contract's cache name; `build()` (host torch tensor) only runs on a cache miss."""
    base = f"{CACHE}/{name}_{dtype.name}"
    hit = os.path.isfile(f"{base}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")
    return ttnn.as_tensor(None if hit else build(), dtype=dtype, layout=layout, device=mesh, memory_config=DRAM,
                          cache_file_name=base)


def _dev(mesh, a: np.ndarray, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype == ttnn.bfloat16:
        t = t.to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM)


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _hi_lo(p: np.ndarray) -> torch.Tensor:
    """f32 rows [n, 128] -> bfloat16 [n, 256] = [hi | lo], hi = bf16(p), lo = bf16(p - hi) (hi + lo = p to 2^-16)."""
    hi = torch.from_numpy(np.ascontiguousarray(p, f32)).to(torch.bfloat16)
    lo = (torch.from_numpy(np.ascontiguousarray(p, f32)) - hi.float()).to(torch.bfloat16)
    return torch.cat([hi, lo], 1)


def kv_write(src, dst, pos_dev, cells: int):
    """Trace-safe KV append (kernels/qsa/kv_write.cpp): rows of src [1, H, T, D] row-major (row j*T + t) -> rows
    j*cells + p0 + t of dst [1, 1, H*cells, D] row-major, p0 = word 0 of pos_dev (uint32 [1, 1, 1, 32] row-major).
    The same bytes as ttnn.experimental.slice_write at [0, 0, p0, 0] of the [1, H, cells, D] view, but the position
    is read on the device, so a captured trace writes where the refreshed pos_dev says."""
    H, T, D = (int(s) for s in tuple(src.shape)[1:])
    size_of = {ttnn.bfloat16: 2, ttnn.float32: 4}
    if src.dtype != dst.dtype or src.dtype not in size_of:
        raise ValueError("kv_write: bfloat16 or float32 rows of the cache's dtype")
    row = D * size_of[src.dtype]
    if H * T > KV_ROWS_MAX:
        raise ValueError(f"kv_write: {H * T} rows > {KV_ROWS_MAX}")
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    size = POS_PAGE + KV_ROWS_MAX * row
    cb = ttnn.CBDescriptor(total_size=size, core_ranges=core, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=size)])
    acc = [a for t in (src, dst, pos_dev) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "kv_write.cpp"), core_ranges=core, compile_time_args=acc,
        named_compile_time_args=[("row_bytes", row), ("pos_page", POS_PAGE)],
        common_runtime_args=[src.buffer_address(), dst.buffer_address(), pos_dev.buffer_address(), H, T, cells],
        config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([src, pos_dev, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))


def _pos_page(mesh, p: int):
    """kv_write's position page for an eager call: uint32 [1, 1, 1, 32] row-major, word 0 = p."""
    t = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
    t[0, 0, 0, 0] = int(p)
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)


KV_GROUP = 8                     # tokens per kv_append staging group (kernels/qsa/kv_append.cpp G)
KV_STAGE = POS_PAGE + 2 * N_HEAD_KV * KV_GROUP * HEAD_DIM * (4 + 2)


def kv_append(kr, qkv, K, V, cells: int, pos_dev=None, p0: int = 0) -> None:
    """Fused KV append (kernels/qsa/kv_append.cpp): the roped keys kr (float32 TILE [1, T, 2, 256]) and the values
    (columns 2*QD + KD .. of qkv, float32 TILE [1, 1, T, 13312]) rounded to bfloat16 like ttnn.typecast and written
    to rows j*cells + p0 + t of K / V; p0 = word 0 of pos_dev (traced steps) or the argument (eager).  The same bytes
    as typecast + permute + to_layout + kv_write / slice_write of each tensor, in one program."""
    T = int(kr.shape[1])
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    cb = ttnn.CBDescriptor(total_size=KV_STAGE, core_ranges=core, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=KV_STAGE)])
    pacc = pos_dev if pos_dev is not None else K          # accessor args only (DRAM interleaved either way)
    acc = [a for t in (kr, qkv, K, V, pacc) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "kv_append.cpp"), core_ranges=core, compile_time_args=acc,
        named_compile_time_args=[("pos_page", POS_PAGE), ("qkv_wt", int(qkv.shape[-1]) // 32),
                                 ("v0t", (2 * QD + KD) // 32)],
        common_runtime_args=[kr.buffer_address(), qkv.buffer_address(), K.buffer_address(), V.buffer_address(),
                             pos_dev.buffer_address() if pos_dev is not None else 0, T, cells, int(p0)],
        config=ttnn.ReaderConfigDescriptor())
    io = [kr, qkv] + ([pos_dev] if pos_dev is not None else []) + [K, V]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))


class TraceInputs:
    """Per-step device inputs of a traced QSA step (fnx.tt.trace), shared by the 12 layers: fixed addresses, written
    by `write(pos)` (copy_host_to_device_tensor) before the replay; the TILE forms and the dense mask are made on the
    device by `prepare()` inside the first captured segment and freed by `release()` in the last one.
      rope  C/S tables [1, T, 1, D] for D = 256 (q, k) and 128 (indexer q), uploaded row-major
      kv    uint32 [1, 1, 1, 32]: word 0 = pos[0] (kv_append)
      rows  f32 [1, 1, T*12, 1]: the position of query row t*12 + h' (the dense mask, eager _StepInputs.dense_mask)
      span  dense cells scored per row (a multiple of SPAN covering pos[-1] + 1; the masked padding cells do not
            change a bit of the result, fnx/tests/test_trace.py), ids [1, 1, 1, T*2*S_SEL] uint32 the sparse cells"""

    def __init__(self, mesh, T: int, span: int, cells: int, sparse: bool):
        self.mesh, self.T, self.span, self.cells, self.sparse = mesh, T, span, cells, sparse
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.rope_rm = {f"{k}{D}": _dev(mesh, np.zeros((1, T, 1, D), f32), layout=rm)
                        for D in (HEAD_DIM, IDX_DIM) for k in "CS"}
        self.kv = _dev(mesh, np.zeros((1, 1, 1, POS_PAGE // 4), np.int32), ttnn.uint32, rm)
        self.rows = None if sparse else _dev(mesh, np.zeros((1, 1, T * GROUP, 1), f32))
        self.cells_row = None if sparse else _dev(mesh, np.arange(span, dtype=f32)[None, None, None, :])
        self.ids = _dev(mesh, np.zeros((1, 1, 1, T * N_HEAD_KV * S_SEL), np.int32), ttnn.uint32, rm) if sparse \
            else None
        self.aux, self.mask = None, None

    def write(self, pos: list[int]) -> None:
        c, s = rope_cos_sin(pos)
        rm = ttnn.ROW_MAJOR_LAYOUT
        for D in (HEAD_DIM, IDX_DIM):
            C = np.ones((len(pos), D), f32)
            S = np.zeros((len(pos), D), f32)
            C[:, :ROPE_DIM] = np.concatenate([c, c], 1)
            S[:, :ROPE_DIM] = np.concatenate([-s, s], 1)
            for k, a in (("C", C), ("S", S)):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(a[None, :, None, :]), dtype=ttnn.float32,
                                                                layout=rm), self.rope_rm[f"{k}{D}"])
        kv = np.zeros((1, 1, 1, POS_PAGE // 4), np.int32)
        kv[..., 0] = pos[0]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(kv), dtype=ttnn.uint32, layout=rm), self.kv)
        if self.rows is not None:
            r = np.repeat(np.asarray(pos, f32), GROUP)[None, None, :, None]
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(r)),
                                                            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.rows)

    def write_ids(self, ids: np.ndarray) -> None:
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(ids.astype(np.int32).reshape(1, 1, 1, -1)),
                                                        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.ids)

    def prepare(self) -> None:
        """Device (captured): TILE rope tables and the dense mask -1e9 where cell > row position (the eager ops)."""
        self.aux = {k: ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=DRAM) for k, v in self.rope_rm.items()}
        if self.rows is not None:
            later = ttnn.gt(self.cells_row, self.rows)
            self.mask = ttnn.multiply(later, MASKED)
            ttnn.deallocate(later)

    def release(self) -> None:
        _free(*self.aux.values(), self.mask)
        self.aux, self.mask = None, None


class _Const:
    """Position-independent device inputs, shared by every Qsa of a mesh: gather ids of the dense prefix, the mask
    of a sparse row (padding cells)."""
    tensors: dict = {}

    @classmethod
    def dense_ids(cls, mesh, cells: int, n_pad: int):
        key = ("ids", id(mesh), cells, n_pad)
        if key not in cls.tensors:
            ids = np.concatenate([np.arange(n_pad), cells + np.arange(n_pad)]).astype(np.int32)
            cls.tensors[key] = _dev(mesh, ids.reshape(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return cls.tensors[key]

    @classmethod
    def sparse_mask(cls, mesh):
        key = ("mask", id(mesh))
        if key not in cls.tensors:
            m = np.where(np.arange(S_SEL) < SEL_WIDTH, 0, MASKED).astype(f32)
            cls.tensors[key] = _dev(mesh, np.repeat(m.reshape(1, 1, 1, S_SEL), GROUP, axis=2))
        return cls.tensors[key]

    @classmethod
    def warm(cls, mesh, cells: int, spans) -> None:
        """Create every lazily cached constant up front (fnx.tt.trace: a device tensor first allocated after a trace
        capture could sit where a replay writes its temporaries): the dense ids of `spans` and the sparse mask."""
        for n_pad in spans:
            cls.dense_ids(mesh, cells, n_pad)
        cls.sparse_mask(mesh)


class _StepInputs:
    """Position-only device inputs of one call (RoPE tables, causal mask of the dense rows), shared by every QSA layer
    of a step: the latest KEEP (mesh, positions) entries are kept, oldest created first out, so the 12 layers of one
    decode step upload them once and a pipelined prompt block's entry outlives the next block's start (Qsa.start)."""
    KEEP = 2
    entries: dict = {}           # key -> {name: tensor}, in creation order

    @classmethod
    def get(cls, mesh, pos: list[int]) -> dict:
        key = (id(mesh), pos[0], len(pos))
        if key not in cls.entries:
            while len(cls.entries) >= cls.KEEP:
                _free(*cls.entries.pop(next(iter(cls.entries))).values())
            c, s = rope_cos_sin(pos)                                        # [T, 32] f32
            tensors = {}
            for D in (HEAD_DIM, IDX_DIM):                                   # [1, T, 1, D], broadcast over heads:
                C = np.ones((len(pos), D), f32)                             # y = x*C + swap_halves(x)*S, C = 1 and
                S = np.zeros((len(pos), D), f32)                            # S = 0 beyond the 64 rotated dims
                C[:, :ROPE_DIM] = np.concatenate([c, c], 1)
                S[:, :ROPE_DIM] = np.concatenate([-s, s], 1)
                tensors[f"C{D}"] = _dev(mesh, C[None, :, None, :])
                tensors[f"S{D}"] = _dev(mesh, S[None, :, None, :])
            cls.entries[key] = tensors
        return cls.entries[key]

    @classmethod
    def clear(cls) -> None:
        """Free the cached entries (fnx.tt.trace: nothing allocated after a capture may stay alive across a replay)."""
        for tensors in cls.entries.values():
            _free(*tensors.values())
        cls.entries = {}

    @classmethod
    def dense_mask(cls, mesh, pos: list[int], d: int):
        """Additive mask [1, 1, d*12, n_pad] of the first d rows (row t*12+h' -> pos[t]): -1e9 where cell > pos[t]."""
        tensors = cls.get(mesh, pos)
        key = ("mask", d)
        if key not in tensors:
            n_pad = -(-(pos[d - 1] + 1) // SPAN) * SPAN
            cells = _dev(mesh, np.arange(n_pad, dtype=f32)[None, None, None, :])
            rows = _dev(mesh, np.repeat(np.asarray(pos[:d], f32), GROUP)[None, None, :, None])
            later = ttnn.gt(cells, rows)                                    # 1.0 where cell > row position
            tensors[key] = ttnn.multiply(later, MASKED)
            _free(cells, rows, later)
        return tensors[key]


class Qsa:
    def __init__(self, mesh, w, L: int, max_ctx: int):
        self.mesh, self.L = mesh, L
        self.max_ctx = max_ctx
        p = f"blk.{L}."

        def qkvg():
            wq = np.asarray(w.layer(L, "attn_q.weight"), f32).reshape(N_HEAD, 2, HEAD_DIM, N_EMBD)
            rows = [wq[:, 0].reshape(QD, N_EMBD), wq[:, 1].reshape(QD, N_EMBD),
                    np.asarray(w.layer(L, "attn_k.weight"), f32),
                    np.asarray(w.layer(L, "attn_v.weight"), f32)]
            return torch.from_numpy(np.ascontiguousarray(np.concatenate(rows).T))      # [2560, 13312]

        self.w_qkvg = cached_tensor(mesh, p + "attn_qkvg_T", ttnn.bfloat8_b, qkvg)
        self.w_o = cached_tensor(mesh, p + "attn_output_T", ttnn.bfloat8_b, lambda: torch.from_numpy(
            np.ascontiguousarray(np.asarray(w.layer(L, "attn_output.weight"), f32).T)))   # [6144, 2560]
        # decode-row program configs (T <= 32): 1D mcast_in0 over more cores with ttnn's own K blocking for these
        # shapes (in0_block_w 2, packer L1 accumulation), so every output element sums in the same order: bitwise
        # the auto-config result, faster (qkvg 131 -> 116 us, o 142 -> 132 us at T=1, measured)
        grid = mesh.compute_with_storage_grid_size()
        self.pc_qkvg = create_matmul_1d_decode_progcfg(32, N_EMBD, 2 * QD + 2 * KD, 110, grid_w=grid.x)
        self.pc_o = create_matmul_1d_decode_progcfg(32, QD, N_EMBD, 64, grid_w=grid.x)
        self.pc_qkvg.in0_block_w = self.pc_o.in0_block_w = 2
        self.qn = cached_tensor(mesh, p + "attn_q_norm_x0.0625", ttnn.float32, lambda: torch.from_numpy(  # 1/sqrt(256)
            (np.asarray(w.layer(L, "attn_q_norm.weight"), f32) * f32(HEAD_DIM ** -0.5)).reshape(1, 1, 1, HEAD_DIM)))
        self.kn = cached_tensor(mesh, p + "attn_k_norm", ttnn.float32, lambda: torch.from_numpy(
            np.asarray(w.layer(L, "attn_k_norm.weight"), f32).reshape(1, 1, 1, HEAD_DIM).copy()))
        # indexer: device query side; host keeps the reference's arrays (same objects as fnx.ref.qsa.QsaWeights uses)
        self.h_ik = np.asarray(w.layer(L, "indexer.k_proj.weight"), dtype=f32)          # (128, 2560) BF16 values
        self.h_iq = np.asarray(w.layer(L, "indexer.q_proj.weight"), dtype=f32)          # (512, 2560) BF16 values
        self.h_ikn = np.asarray(w.layer(L, "indexer.k_norm.weight"), dtype=f32)
        self.h_iqn = np.asarray(w.layer(L, "indexer.q_norm.weight"), dtype=f32)
        self.w_iq = cached_tensor(mesh, p + "indexer_q_proj_T", ttnn.bfloat16,
                                  lambda: torch.from_numpy(np.ascontiguousarray(self.h_iq.T)))    # [2560, 512] exact
        self.iqn = cached_tensor(mesh, p + "indexer_q_norm", ttnn.float32,
                                 lambda: torch.from_numpy(self.h_iqn.reshape(1, 1, 1, IDX_DIM).copy()))

        self.cells = -(-max_ctx // SPAN) * SPAN
        self.blocks = -(-(self.cells // COMPRESS) // 32) * 32
        # device state, allocated once (fixed addresses), zero-filled so that masked gathers stay finite
        self.K = ttnn.zeros([1, 1, N_HEAD_KV * self.cells, HEAD_DIM], dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)
        self.V = ttnn.zeros([1, 1, N_HEAD_KV * self.cells, HEAD_DIM], dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)
        self.K4 = ttnn.reshape(self.K, [1, N_HEAD_KV, self.cells, HEAD_DIM])        # views for slice_write
        self.V4 = ttnn.reshape(self.V, [1, N_HEAD_KV, self.cells, HEAD_DIM])
        self.P2 = ttnn.zeros([1, 1, self.blocks, 2 * IDX_DIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=mesh, memory_config=DRAM)
        self.P2pages = ttnn.reshape(self.P2, [self.blocks // 32, 1, 32, 2 * IDX_DIM])  # view for paged_fill_cache
        # host indexer state (positional; exact = fnx.ref.qsa.QsaState)
        self.kr = np.zeros((self.cells, IDX_DIM), f16)                  # f16 raw key per cell
        self.pooled = np.zeros((self.blocks, IDX_DIM), f32)             # pooled+normed+roped key per block
        self.pnorm = np.zeros(self.blocks, np.float64)                  # |pooled[b]|
        self.dead = np.zeros(IDX_DIM, f32)                              # normed key of cell 0
        self.debug = False                                              # check every device score against exact
        self.last_ids: list[np.ndarray] = []
        self.last: dict = {}
        self.pending: set[int] = set()   # P2 pages whose upload a traced step deferred (index_rows / flush_pooled)

    def reset(self) -> None:
        """New sequence. Cells, blocks and the dead key are only read once their position has been written in this
        sequence (positional state), so nothing needs clearing; the device tensors keep their addresses."""
        self.last_ids = []
        self.last = {}

    def state_tensors(self) -> list:
        """Per-sequence device state that a snapshot must carry: none (all positional, see the module docstring)."""
        return []

    @staticmethod
    def _rope(x, aux):
        """x [1, T, H, D] f32: y = x*C + [x(32:64) | x(0:32) | x(64:)]*S, i.e. NEOX pairs (i, i+32) on dims 0..63 and
        dims 64.. times 1 plus 0 (exactly x)."""
        T, H, D = x.shape[1], x.shape[2], x.shape[3]
        lo = ttnn.slice(x, [0, 0, 0, 0], [1, T, H, ROPE_DIM // 2])
        hi = ttnn.slice(x, [0, 0, 0, ROPE_DIM // 2], [1, T, H, ROPE_DIM])
        rest = ttnn.slice(x, [0, 0, 0, ROPE_DIM], [1, T, H, D])
        sw = ttnn.concat([hi, lo, rest], dim=-1)
        xc, ss = ttnn.multiply(x, aux[f"C{D}"]), ttnn.multiply(sw, aux[f"S{D}"])
        out = ttnn.add(xc, ss)
        _free(lo, hi, rest, sw, xc, ss)
        return out

    # ---- device, before the host round trip ----------------------------------------------------------------------
    def _project(self, mixed, pos: list[int], sparse: bool, step: TraceInputs | None = None) -> dict:
        """Projections, q/k norm + RoPE, KV append of every row, and (sparse calls) the device block scores.
        step (fnx.tt.trace): the captured form, position-independent: RoPE tables and the KV position from the step's
        device inputs (kv_append), block scores over every block (any wider score row selects the same cells)."""
        T = len(pos)
        aux = _StepInputs.get(self.mesh, pos) if step is None else step.aux
        fast = T <= 32       # explicit decode program configs, fused rope/kv_append: bitwise == the general path
        qkv = ttnn.linear(mixed, self.w_qkvg, compute_kernel_config=HIFI4, dtype=ttnn.float32, memory_config=DRAM,
                          program_config=self.pc_qkvg if fast else None)
        q = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, T, QD]), [1, T, N_HEAD, HEAD_DIM])
        gate = ttnn.slice(qkv, [0, 0, 0, QD], [1, 1, T, 2 * QD])
        k = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, 2 * QD], [1, 1, T, 2 * QD + KD]), [1, T, N_HEAD_KV, HEAD_DIM])
        qn = ttnn.rms_norm(q, weight=self.qn, epsilon=EPS, compute_kernel_config=NORM_CKC)
        kn = ttnn.rms_norm(k, weight=self.kn, epsilon=EPS, compute_kernel_config=NORM_CKC)
        _free(q, k)
        if fast and ROPE2:                              # q and k heads roped in one pass (the same elementwise ops)
            qk = ttnn.concat([qn, kn], dim=2)
            qkr = self._rope(qk, aux)
            qr = ttnn.slice(qkr, [0, 0, 0, 0], [1, T, N_HEAD, HEAD_DIM])          # carries the 1/16 softmax scale
            kr = ttnn.slice(qkr, [0, 0, N_HEAD, 0], [1, T, N_HEAD + N_HEAD_KV, HEAD_DIM])
            _free(qk, qkr)
        else:
            qr = self._rope(qn, aux)                    # carries the 1/16 softmax scale (gamma)
            kr = self._rope(kn, aux)
        _free(qn, kn)
        # KV append, every row before any row attends
        if fast:                                        # one program: kr and the values of qkv -> bf16 cache rows
            kv_append(kr, qkv, self.K, self.V, self.cells, None if step is None else step.kv, pos[0])
            _free(kr, qkv)
        else:                                           # [1, T, 2, 256] -> [1, 2, T, 256] bf16 row-major -> rows p0..
            v = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, 2 * QD + KD], [1, 1, T, 2 * QD + 2 * KD]),
                             [1, T, N_HEAD_KV, HEAD_DIM])
            ttnn.deallocate(qkv)
            G = KV_ROWS_MAX // N_HEAD_KV                # tokens per kv_write call
            for src, dst in ((kr, self.K), (v, self.V)):
                s16 = ttnn.typecast(src, ttnn.bfloat16)
                hm = (ttnn.reshape(s16, [1, N_HEAD_KV, 1, HEAD_DIM]) if T == 1 else ttnn.permute(s16, (0, 2, 1, 3)))
                rm = ttnn.to_layout(hm, ttnn.ROW_MAJOR_LAYOUT)
                if step is None:                        # the position as device data: one program for every cell
                    for t0 in range(0, T, G):           # (a slice_write bakes its start cell into the program)
                        t1 = min(T, t0 + G)
                        part = rm if t1 - t0 == T else ttnn.slice(rm, [0, 0, t0, 0], [1, N_HEAD_KV, t1, HEAD_DIM])
                        pd = _pos_page(self.mesh, pos[0] + t0)
                        kv_write(part, dst, pd, self.cells)
                        _free(pd, *(() if part is rm else (part,)))
                else:
                    kv_write(rm, dst, step.kv, self.cells)
                _free(s16, rm, *(() if hm is s16 else (hm,)))
            _free(kr, v)
        out = {"qr": qr, "gate": gate, "scores": None, "qi": None}
        if sparse:
            out["scores"], out["qi"] = self._device_scores(mixed, aux, T, pos[0] // COMPRESS,
                                                           None if step is None else self.blocks)
        return out

    def _device_scores(self, mixed, aux, T: int, nb_old: int, nbs: int | None = None):
        """s~ [1, 1, T, nbs] f32: relu(qi . P2[b]) summed over the 4 indexer heads, columns b < nbs (>= nb_old)."""
        qi = ttnn.linear(mixed, self.w_iq, compute_kernel_config=HIFI4, dtype=ttnn.float32, memory_config=DRAM)
        q4 = ttnn.reshape(qi, [1, T, IDX_HEADS, IDX_DIM])
        qn = ttnn.rms_norm(q4, weight=self.iqn, epsilon=EPS, compute_kernel_config=NORM_CKC)
        qr = self._rope(qn, aux)                         # [1, T, 4, 128]
        _free(q4, qn)
        hq = qr if T == 1 else ttnn.permute(qr, (0, 2, 1, 3))          # T=1: heads as rows [1, 1, 4, 128]
        q2 = ttnn.concat([hq, hq], dim=-1)                              # [qi | qi] against [hi | lo]
        nbs = min(-(-max(nb_old, 1) // BSPAN) * BSPAN, self.blocks) if nbs is None else nbs
        P = self.P2 if nbs == self.blocks else ttnn.slice(self.P2, [0, 0, 0, 0], [1, 1, nbs, 2 * IDX_DIM])
        s = ttnn.matmul(q2, P, transpose_b=True, compute_kernel_config=HIFI4, dtype=ttnn.float32)
        r = ttnn.relu(s)
        sc = ttnn.sum(r, dim=2 if T == 1 else 1, keepdim=True)          # [1, 1, T, nbs]
        _free(q2, s, r, *(() if P is self.P2 else (P,)), *(() if hq is qr else (hq,)))
        if not self.debug:
            _free(qr)
            qr = None
        return sc, qr

    def _readback(self, mixed, dev: dict, T: int):
        """[mixed f32 | s~] rows -> host (one untilize, one transfer): (x [T, 2560], s~ [T, nbs] or None)."""
        both = self._readback_dev(mixed, dev)
        host = ttnn.to_torch(both).float().numpy().reshape(T, -1)
        ttnn.deallocate(both)
        return self._split_host(host)

    @staticmethod
    def _readback_dev(mixed, dev: dict):
        """The device half of _readback: [mixed f32 | s~] untilized (frees the scores)."""
        x = mixed if mixed.dtype == ttnn.float32 else ttnn.typecast(mixed, ttnn.float32)
        cat = ttnn.concat([x, dev["scores"]], dim=-1) if dev["scores"] is not None else x
        both = ttnn.untilize(cat)
        _free(*(() if cat is x else (cat,)), *(() if x is mixed else (x,)), dev["scores"])
        dev["scores"] = None
        return both

    @staticmethod
    def _split_host(host: np.ndarray):
        xh = np.ascontiguousarray(host[:, :N_EMBD])
        return xh, (host[:, N_EMBD:] if host.shape[1] > N_EMBD else None)

    # ---- host indexer -------------------------------------------------------------------------------------------
    def _index(self, xh: np.ndarray, pos: list[int]) -> list[int]:
        """Append the rows' indexer keys to the host state with the reference's arithmetic (fnx.ref.qsa.qsa:
        kr = f16(x Wk^T) for the call's rows, pool -> norm -> rope at 4b). Returns the blocks completed."""
        kr = (xh @ self.h_ik.T).astype(f16).astype(f32)
        done = []
        for t, p in enumerate(pos):
            self.kr[p] = kr[t]
            if p == 0:                                           # spare key: block of 4 copies of cell 0, angle 0
                m = f32(0.25) * (((kr[t] + kr[t]) + kr[t]) + kr[t])
                self.dead = rms(m, self.h_ikn)
                self.pooled[0] = self.dead
            elif p % COMPRESS == COMPRESS - 1:                   # block b completes: pool, norm, rope at 4b
                b = p // COMPRESS
                tail = self.kr[COMPRESS * b:COMPRESS * b + 3].astype(f32)
                m = f32(0.25) * (((tail[0] + tail[1]) + tail[2]) + kr[t])
                c4, s4 = rope_cos_sin([COMPRESS * b])
                self.pooled[b] = rope64(rms(m, self.h_ikn), c4[0], s4[0])
                self.pnorm[b] = np.linalg.norm(self.pooled[b].astype(np.float64))
                done.append(b)
        return done

    def _upload_pooled(self, done: list[int]) -> None:
        """P2 rows of the completed blocks (exact [hi | lo] of the host keys), written as whole 32-row pages."""
        self._upload_pages({b // 32 for b in done})

    def _upload_pages(self, pages) -> None:
        if not pages:
            return
        pages = sorted(pages)
        rows = np.concatenate([self.pooled[g * 32:(g + 1) * 32] for g in pages])
        inp = ttnn.from_torch(_hi_lo(rows).reshape(1, 1, -1, 2 * IDX_DIM), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)
        pt = ttnn.from_torch(torch.tensor([pages], dtype=torch.int32), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM)
        ttnn.experimental.paged_fill_cache(self.P2pages, inp, pt, batch_idx=0)
        _free(inp, pt)

    # ---- traced steps (fnx.tt.trace): the host indexer of the rows, P2 uploads deferred to the next step boundary
    def index_rows(self, xh: np.ndarray, pos: list[int]) -> None:
        """_index of rows that a traced step attended (dense rows: their cells for last_ids); the P2 pages of the
        completed blocks are uploaded by flush_pooled() before any call reads the device block scores again."""
        n_dense = sum(p <= DENSE_MAX_POS for p in pos)
        self.pending |= {b // 32 for b in self._index(xh, pos)}
        if n_dense:
            self.last_ids = [np.arange(p + 1) for p in pos[:n_dense]]

    def flush_pooled(self) -> None:
        """Upload the P2 pages deferred by index_rows / sparse_host (eager: between traced steps only)."""
        pages, self.pending = self.pending, set()
        self._upload_pages(pages)

    def sparse_host(self, host: np.ndarray, pos: list[int]) -> np.ndarray:
        """Host half of a traced sparse call (every row sparse): read-back [x | s~] -> indexer update (P2 deferred) and
        the exact selection -> gather ids uint32 [T * 2 * S_SEL] (one group: T <= GMAX), as _sparse_ids lays them."""
        xh, sd = self._split_host(host)
        self.last = {}
        self.pending |= {b // 32 for b in self._index(xh, pos)}
        sel = self._select(xh, pos, sd, 0)
        self.last_ids = sel
        cells = np.zeros((len(sel), S_SEL), np.int32)
        for i, c in enumerate(sel):
            cells[i, :SEL_WIDTH] = c
        return np.stack([cells, cells + self.cells], 1).reshape(-1)

    def _select(self, xh: np.ndarray, pos: list[int], sd: np.ndarray, n_dense: int, qi_dev=None) -> list[np.ndarray]:
        """Ascending selected cells of the sparse rows pos[n_dense:] (module docstring: exact ranks)."""
        T = len(pos)
        st = self.last
        qi = (xh @ self.h_iq.T).reshape(T, IDX_HEADS, IDX_DIM)        # = the reference (all rows: same bits)
        c, s = rope_cos_sin(pos)
        qin = rope64(rms(qi, self.h_iqn), c[:, None, :], s[:, None, :])
        nb_old = pos[0] // COMPRESS
        ts = np.arange(n_dense, T)
        n = np.asarray(pos)[ts] + 1
        nbt = n // COMPRESS
        nbm = int(nbt[-1])
        qn = np.linalg.norm(qin[ts].astype(np.float64), axis=2).sum(1)                 # sum_h |qi_h|
        lo = np.full((len(ts), nbm), -np.inf)
        hi = np.full((len(ts), nbm), -np.inf)
        bnd = SCORE_ERR * qn[:, None] * self.pnorm[None, :nb_old]
        dv = sd[ts, :nb_old].astype(np.float64)
        lo[:, :nb_old], hi[:, :nb_old] = dv - bnd, dv + bnd
        ex_new = np.stack([block_scores(self.pooled[nb_old:nbm], qin[t]) for t in ts]) if nbm > nb_old else None
        if ex_new is not None:
            lo[:, nb_old:] = hi[:, nb_old:] = ex_new
        lo[np.arange(nbm)[None, :] >= nbt[:, None]] = -np.inf                         # not complete for the row
        hi[np.arange(nbm)[None, :] >= nbt[:, None]] = -np.inf
        K = TOPK_BLOCKS
        e = COMPRESS - 1 - n % COMPRESS                                               # cells from the 513th block
        lo_p = np.partition(lo, [nbm - K - 1, nbm - K], axis=1)
        hi_p = np.partition(hi, [max(nbm - K - 2, 0), nbm - K - 1], axis=1)
        sure, cand = {}, {}
        for k, tau_lo, tau_hi in ((K, lo_p[:, nbm - K], hi_p[:, nbm - K - 1]),       # K-th largest lower bound,
                                  (K + 1, lo_p[:, nbm - K - 1],                      # (K+1)-th largest upper bound
                                   hi_p[:, nbm - K - 2] if nbm >= K + 2 else np.full(len(ts), -np.inf))):
            sure[k] = lo > tau_hi[:, None]
            cand[k] = ~sure[k] & (hi >= tau_lo[:, None])
        cand[K + 1][e == 0] = False                                                   # rows without a 513th
        need = cand[K] | cand[K + 1]
        ex = np.zeros((len(ts), nbm), f32)                 # exact scores of every candidate block (fnx_qsa_assemble)
        n_cand, n_ref = 0, 0
        for i, t in enumerate(ts):
            nd = np.flatnonzero(need[i])
            if len(nd):
                old = nd < nb_old
                ex[i, nd[old]] = block_scores(self.pooled[nd[old]], qin[t])
                if not old.all():
                    ex[i, nd[~old]] = ex_new[i, nd[~old] - nb_old]
            n_cand += len(nd)
            n_ref += bool(len(nd))
        out = assemble_selection(np.stack([sure[K], sure[K + 1]]), np.stack([cand[K], cand[K + 1]]), ex, e, nbt, n)
        st.update(cand=n_cand, refined_rows=n_ref, sparse_rows=len(ts))
        if self.debug:                                          # every device score against its exact value
            ratios = []
            for i, t in enumerate(ts):
                ex = block_scores(self.pooled[:nb_old], qin[t]).astype(np.float64)
                ratios.append(float((np.abs(dv[i] - ex) / np.maximum(qn[i] * self.pnorm[:nb_old], 1e-30)).max()))
            st["err_ratio"] = max(ratios)
            if qi_dev is not None:
                qd = ttnn.to_torch(qi_dev).float().numpy().reshape(T, IDX_HEADS, IDX_DIM)
                st["qi_rel"] = float((np.linalg.norm(qd - qin, axis=2) / np.linalg.norm(qin, axis=2)).max())
        return out

    # ---- device, after the host round trip ----------------------------------------------------------------------
    def _sparse_ids(self, sel: list[np.ndarray]) -> list:
        """Gather-id tensors of the sparse rows, one per group of <= GMAX rows: [rows, 2 heads, S_SEL] uint32."""
        out = []
        for g0 in range(0, len(sel), GMAX):
            grp = sel[g0:g0 + GMAX]
            cells = np.zeros((len(grp), S_SEL), np.int32)
            for i, c in enumerate(grp):
                cells[i, :SEL_WIDTH] = c
            ids = np.stack([cells, cells + self.cells], 1).reshape(1, 1, 1, -1)
            out.append(_dev(self.mesh, ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        return out

    def _gather(self, ids, batch: int, n: int):
        k = ttnn.reshape(ttnn.embedding(ids, self.K, layout=ttnn.TILE_LAYOUT, memory_config=DRAM), [1, batch, n,
                                                                                                    HEAD_DIM])
        v = ttnn.reshape(ttnn.embedding(ids, self.V, layout=ttnn.TILE_LAYOUT, memory_config=DRAM), [1, batch, n,
                                                                                                    HEAD_DIM])
        return k, v

    @staticmethod
    def _attn(qg, k, v, mask):
        s = ttnn.matmul(qg, k, transpose_b=True, compute_kernel_config=ATTN_CKC, dtype=ttnn.float32)
        sm = ttnn.add(s, mask)
        pr = ttnn.softmax(sm, dim=-1, numeric_stable=True, compute_kernel_config=NORM_CKC)
        o = ttnn.matmul(pr, v, compute_kernel_config=ATTN_CKC, dtype=ttnn.float32)
        _free(s, sm, pr, k, v)
        return o

    def _attend(self, dev: dict, pos: list[int], n_dense: int, ids: list, free_inputs: bool = True,
                step: TraceInputs | None = None):
        """Attention of every row over its cells (dense prefix rows, then the sparse groups), gate, output projection.
        free_inputs=False keeps dev's tensors (traced timing replays this part alone).  step (fnx.tt.trace): the dense
        rows score step.span cells with step.mask (the step's device inputs) instead of the position's rounding."""
        T = len(pos)
        qr, parts = dev["qr"], []
        if n_dense:
            d = n_dense
            n_pad = -(-(pos[d - 1] + 1) // SPAN) * SPAN if step is None else step.span
            k, v = self._gather(_Const.dense_ids(self.mesh, self.cells, n_pad), N_HEAD_KV, n_pad)
            qd = qr if d == T else ttnn.slice(qr, [0, 0, 0, 0], [1, d, N_HEAD, HEAD_DIM])
            if d == 1:                                   # rows t*12 + h' of kv head j = q head j*12 + h'
                qg = ttnn.reshape(qd, [1, N_HEAD_KV, GROUP, HEAD_DIM])
            else:
                qg = ttnn.reshape(ttnn.permute(ttnn.reshape(qd, [d, N_HEAD_KV, GROUP, HEAD_DIM]), (1, 0, 2, 3)),
                                  [1, N_HEAD_KV, d * GROUP, HEAD_DIM])
            mask = _StepInputs.dense_mask(self.mesh, pos, d) if step is None else step.mask
            o = self._attn(qg, k, v, mask)                                          # [1, 2, d*12, 256]
            if d == 1:
                parts.append(ttnn.reshape(o, [1, 1, 1, QD]))
            else:
                parts.append(ttnn.reshape(ttnn.permute(ttnn.reshape(o, [N_HEAD_KV, d, GROUP, HEAD_DIM]),
                                                       (1, 0, 2, 3)), [1, 1, d, QD]))
            _free(o, qg, *(() if qd is qr else (qd,)))
        g0 = n_dense
        for gid in ids:
            G = min(GMAX, T - g0)
            k, v = self._gather(gid, N_HEAD_KV * G, S_SEL)                         # batch t*2 + j
            qs = qr if G == T else ttnn.slice(qr, [0, g0, 0, 0], [1, g0 + G, N_HEAD, HEAD_DIM])
            qg = ttnn.reshape(qs, [1, N_HEAD_KV * G, GROUP, HEAD_DIM])
            o = self._attn(qg, k, v, _Const.sparse_mask(self.mesh))                 # [1, 2G, 12, 256]
            parts.append(ttnn.reshape(o, [1, 1, G, QD]))
            _free(o, qg, *(() if qs is qr else (qs,)))
            g0 += G
        att = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        fast = T <= 32
        if fast:
            a = ttnn.multiply(att, dev["gate"], input_tensor_b_activations=[SIGMOID])     # att * sigmoid(gate)
            _free(att, *(parts if len(parts) > 1 else ()), *((qr, dev["gate"]) if free_inputs else ()))
        else:
            g = ttnn.sigmoid(dev["gate"])
            a = ttnn.multiply(att, g)
            _free(att, g, *(parts if len(parts) > 1 else ()), *((qr, dev["gate"]) if free_inputs else ()))
        bo = ttnn.linear(a, self.w_o, compute_kernel_config=HIFI4, dtype=ttnn.float32, memory_config=DRAM,
                         program_config=self.pc_o if fast else None)
        ttnn.deallocate(a)
        return bo

    def forward(self, mixed, pos):
        """mixed: [1, 1, T, 2560] TILE (bf16 or f32), T logical rows = consecutive positions `pos` (host ints).
        Returns bo [1, 1, T, 2560] float32 TILE. Appends the T tokens to the KV cache and the indexer state;
        `last_ids` holds every row's attended cells (ascending), `last` the call's selection statistics and times.
        The phases below in sequence (TtModel.forward_prompt interleaves them over row blocks)."""
        t0 = time.perf_counter()
        c = self.start(mixed, pos)
        self.index(c)
        t1 = time.perf_counter()
        self.select(c)
        t2 = time.perf_counter()
        bo = self.finish(c)
        self.last.update(us=dict(device_a=(t1 - t0) * 1e6, host=(t2 - t1) * 1e6,
                                 device_b_dispatch=(time.perf_counter() - t2) * 1e6))
        return bo

    # ---- forward() in phases.  start(k + 1) may be dispatched after index(k) and before select(k) / finish(k): it
    # needs the indexer pages index(k) uploads and appends K/V cells that block k's rows only see masked (dense
    # padding: exact zeros after the softmax) or never select, so every block computes what forward() computes.
    def start(self, mixed, pos) -> dict:
        """Device: projections, K/V append, block scores; the [mixed | s~] read-back is queued, not awaited."""
        pos = [int(p) for p in pos]
        T = len(pos)
        if mixed.shape[-2] != T:
            raise ValueError(f"mixed has {mixed.shape[-2]} rows for {T} positions")
        if any(b - a != 1 for a, b in zip(pos, pos[1:])):
            raise ValueError("positions of one call must be consecutive")
        if pos[0] < 0 or pos[-1] >= self.max_ctx:
            raise ValueError(f"positions {pos[0]}..{pos[-1]} outside max_ctx {self.max_ctx}")
        n_dense = sum(p <= DENSE_MAX_POS for p in pos)
        self.flush_pooled()                  # P2 pages a traced step deferred (fnx.tt.trace)
        dev = self._project(mixed, pos, n_dense < T)
        both = self._readback_dev(mixed, dev)
        host = ttnn.from_device(both, blocking=False)
        return dict(pos=pos, T=T, n_dense=n_dense, dev=dev, both=both, host=host,
                    event=ttnn.record_event(self.mesh, 0))

    def index(self, c: dict) -> None:
        """Host: wait for the read-back, append the rows to the host indexer, upload the completed blocks' pages."""
        ttnn.event_synchronize(c.pop("event"))
        host = ttnn.to_torch(c.pop("host")).float().numpy().reshape(c["T"], -1)
        ttnn.deallocate(c.pop("both"))
        c["xh"], c["sd"] = self._split_host(host)
        self._upload_pooled(self._index(c["xh"], c["pos"]))

    def select(self, c: dict) -> None:
        """Host: the sparse rows' exact cell selection, uploaded as gather ids."""
        pos, n_dense, dev = c["pos"], c["n_dense"], c["dev"]
        self.last = {}
        sel = self._select(c.pop("xh"), pos, c.pop("sd"), n_dense, dev["qi"]) if n_dense < c["T"] else []
        _free(dev["qi"])
        c["ids"] = self._sparse_ids(sel)
        self.last_ids = [np.arange(p + 1) for p in pos[:n_dense]] + sel

    def finish(self, c: dict):
        """Device: attention over the selected cells, gate, output projection -> bo [1, 1, T, 2560] float32 TILE."""
        bo = self._attend(c["dev"], c["pos"], c["n_dense"], c["ids"])
        _free(*c["ids"])
        return bo
