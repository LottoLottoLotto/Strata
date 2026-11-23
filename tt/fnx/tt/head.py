"""LM head on the P150 (TTNN): logits = head_in @ W^T, W = output.weight (Q6_K in the GGUF) re-quantized for the
device: `FNX_HEAD=bfp8` tt-metal's bfloat8_b conversion of the dequantized Q6_K values, or a store method of
fnx.quant.head (`gptq4`: GPTQ-BFP4 rows, FNX_QUANT_DIR/store/head/output.weight.T.gptq4.npy) as bfloat4_b.

`Head(mesh, w[, mode]).forward(x [1,1,T,2560] f32|bf16) -> logits [1,1,T,248320] float32 TILE`;
`Head.argmax(logits) -> uint32 [1,1,T,1] ROW_MAJOR` (on-device greedy; ties -> lowest index as ttnn.argmax scans);
`Head.greedy(x) -> np.ndarray [T]` (forward + argmax + 4-byte-per-row readback).

Upload follows qwen36 model.py (image path models/demos/blackhole/qwen36/tt/model.py:257-288): the P150 staging
path cannot take the ~0.7 GB matrix in one transfer, so W^T [2560, 248320] is converted and uploaded in tile-aligned
column blocks of at most 8M elements (80 blocks of 97 tiles = 3104 vocab columns), each block's TILE pages copied to
their bank-packed places in the preallocated weight by fnx.tt.packed_matmul.place_packed (exact; ttnn.concat +
typecast re-blocks through BF16 and is not exact for bfloat8_b nor bfloat4_b).  fnx/tests/test_head.py checks the
device weight against the host values exactly.  Converted blocks are cached under FNX_TT_CACHE.
Matmul: qwen36 tp_common.create_matmul_1d_decode_progcfg (copied below) on 110 cores (71 columns each), in0_block_w 2,
run as fnx.tt.packed_matmul.PackedMatmul (the stock program with the bank-packed dual-NoC weight reader: bitwise the
stock ttnn.linear, 2.00 -> 1.65 ms traced, fnx/tests/test_packed.py) on every 32-row slice.  place_cols is the stock
(unpacked) layout's exact join, test_packed's reference.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import ttnn

from ..ref.embed import N, N_VOCAB, head_weight
from .hc import _DT_NAME, DRAM, F32, HIFI4, TT_CACHE
from .packed_matmul import PackedLayout, PackedMatmul, place_packed
from .tile_argmax import TileArgmax

TILE = 32
MAX_PART = 8_000_000
MODE = os.environ.get("FNX_HEAD", "bfp8")        # bfp8 | a fnx.quant.head store method (gptq4, rtn4, search4)


def mode_dtype(mode: str):
    return ttnn.bfloat8_b if mode == "bfp8" else ttnn.bfloat4_b


def mode_tag(mode: str) -> str:
    """Cache-name infix of a mode's converted blocks ('' keeps the bfp8 names of earlier caches)."""
    return "" if mode == "bfp8" else f".{mode}"


def _part_name(mode: str, chunk: int, i: int) -> str:
    return f"{TT_CACHE}/output.weight.T{mode_tag(mode)}.part{chunk}_{i}_{_DT_NAME[mode_dtype(mode)]}"


def _cached(name: str, dtype) -> bool:
    return os.path.isfile(f"{name}_dtype_{dtype.name}_layout_TILE.tensorbin")


def head_values(w, mode: str, v0: int, v1: int) -> np.ndarray:
    """W^T columns [v0, v1) as float32 [2560, v1 - v0]: the dequantized Q6_K values (bfp8, converted by tt-metal) or
    the store's snapped values (exactly the device's bfloat4_b values)."""
    if mode == "bfp8":
        return np.ascontiguousarray(np.asarray(head_weight(w)[v0:v1], np.float32).T)
    from ..quant.head import store_rows, unpack_cols
    return unpack_cols(store_rows(mode), "bfp4", v0, v1)


def device_values(w, mode: str, v0: int, v1: int) -> np.ndarray:
    """The values the device head holds for vocab rows [v0, v1): float32 [v1 - v0, 2560] (host emulation)."""
    vals = head_values(w, mode, v0, v1)
    if mode == "bfp8":
        from ..quant.bfp import rtn
        vals = rtn(torch.from_numpy(vals), "bfp8").numpy()
    return np.ascontiguousarray(vals.T)


def _find_largest_divisor(n, max_div=8):
    """qwen36 tt/tp_common.py:75-79."""
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def create_matmul_1d_decode_progcfg(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
    """qwen36 tt/tp_common.py:137-166: explicit-grid 1D (mcast_in0) decode matmul config, wide-first grid."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE)
    k_tiles = math.ceil(k / TILE)
    n_tiles = math.ceil(n / TILE)
    per_core_k = _find_largest_divisor(k_tiles)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows), in0_block_w=per_core_k, out_subblock_h=sub_h,
        out_subblock_w=sub_w, per_core_M=m_tiles, per_core_N=per_core_n, fuse_batch=True,
        fused_activation=fused_activation, mcast_in0=True)


def head_parts():
    """(chunk columns, number of parts) of the chunked W^T upload (qwen36 model.py:262-267)."""
    part_count = max(1, (N * N_VOCAB + MAX_PART - 1) // MAX_PART)
    tile_cols = N_VOCAB // TILE
    chunk = ((tile_cols + part_count - 1) // part_count) * TILE
    return chunk, (N_VOCAB + chunk - 1) // chunk


def _host_parts(w, mode: str):
    """Yield (first vocab column, host tensor) of the converted (cached) W^T column blocks."""
    dtype = mode_dtype(mode)
    chunk, n_parts = head_parts()
    for i in range(n_parts):
        v0 = i * chunk
        name = _part_name(mode, chunk, i)
        part = None if _cached(name, dtype) else torch.from_numpy(head_values(w, mode, v0, min(v0 + chunk, N_VOCAB)))
        yield v0, ttnn.as_tensor(None if part is None else part[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                 cache_file_name=name)


def convert_parts(w, mode: str = MODE):
    """Convert and cache the W^T column blocks on the host (no device needed); returns the host tensors."""
    return [host for _, host in _host_parts(w, mode)]


KDIR = Path(__file__).resolve().parent / "kernels" / "head"
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def place_cols(part, dst, col0: int) -> None:
    """Copy the TILE pages of part [.., K, W] into tile columns col0 .. col0 + W/32 of dst [.., K, N] (same block-float
    dtype; kernels/head/place_cols.cpp): exact, no format conversion."""
    mesh = dst.device()
    rows, pw, dw = int(part.shape[-2]) // TILE, int(part.shape[-1]) // TILE, int(dst.shape[-1]) // TILE
    page = TILE_BYTES[dst.dtype]
    g = mesh.compute_with_storage_grid_size()
    n = min(rows, g.x * g.y)
    cores = [(i % g.x, i // g.x) for i in range(n)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i]
    cb = ttnn.CBDescriptor(total_size=pw * page, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=pw * page)])
    acc = [a for t in (part, dst) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "place_cols.cpp"), core_ranges=crs, compile_time_args=acc,
        named_compile_time_args=[("page_bytes", page), ("part_cols", pw), ("dst_cols", dw)],
        runtime_args=rt, common_runtime_args=[part.buffer_address(), dst.buffer_address(), rows, col0, n],
        config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([part, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))


class Head:
    def __init__(self, mesh, w, mode: str | None = None, num_cores: int = 110):
        self.mesh, self.mode = mesh, mode or MODE
        self.dtype = dtype = mode_dtype(self.mode)
        grid = mesh.compute_with_storage_grid_size()
        self.pc = create_matmul_1d_decode_progcfg(TILE, N, N_VOCAB, num_cores=min(num_cores, grid.x * grid.y),
                                                  grid_w=grid.x)
        self.pc.in0_block_w = 2                      # qwen36 model.py:329: bound the in0 circular buffers
        layout = PackedLayout(N // TILE, N_VOCAB // TILE, self.pc.per_core_N)
        # Each converted block's pages copied to their packed places in the preallocated weight: exact.  (ttnn.concat of
        # block-float parts returns BF16 and the typecast back re-blocks it: NOT exact for bfloat4_b nor bfloat8_b.)
        self.w = ttnn.empty([1, 1, N, N_VOCAB], dtype, ttnn.TILE_LAYOUT, mesh, DRAM)
        for v0, host in _host_parts(w, self.mode):
            part = ttnn.to_device(host, mesh, memory_config=DRAM)
            place_packed(part, self.w, v0 // TILE, layout)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(part)
        assert tuple(self.w.shape)[-2:] == (N, N_VOCAB), self.w.shape
        self.mm = PackedMatmul(self.w, self.pc, HIFI4, F32, DRAM)
        self.tile_argmax = TileArgmax(mesh)

    def reset(self):
        pass

    def forward(self, x):
        """x [1,1,T,2560] float32 or bfloat16 -> logits [1,1,T,248320] float32: the packed decode matmul on each 32-row
        slice, joined (each row's logits are bitwise its T <= 32 value; for bfp8 also bitwise the ttnn auto program
        config the T > 32 calls used before, test_packed)."""
        T = x.shape[-2]
        if T <= TILE:
            return self.mm(x)
        outs = []
        for r0 in range(0, T, TILE):
            xs = ttnn.slice(x, [0, 0, r0, 0], [1, 1, min(r0 + TILE, T), N])
            outs.append(self.mm(xs))
            ttnn.deallocate(xs)
        out = ttnn.concat(outs, dim=2, memory_config=DRAM)
        for o in outs:
            ttnn.deallocate(o)
        return out

    def argmax(self, logits, keep_rm: bool = False):
        """logits [1,1,T,V] float32 TILE -> uint32 [1,1,T,1] ROW_MAJOR (device), ttnn.argmax's order bitwise (first
        maximum, NaN never, -0 == +0); keep_rm=True also returns the T rows row-major, the same bits (T <= 32:
        fnx.tt.tile_argmax's [1,1,T*k,V/k], read it flattened; T > 32: ttnn.to_layout's [1,1,T,V])."""
        if logits.shape[-2] <= TILE:
            return self.tile_argmax(logits, keep_rm=keep_rm)
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)   # prompt-sized forwards (T > 32)
        ids = ttnn.argmax(rm, dim=-1, keepdim=True, memory_config=DRAM)
        if keep_rm:
            return ids, rm
        ttnn.deallocate(rm)
        return ids

    def greedy(self, x) -> np.ndarray:
        logits = self.forward(x)
        ids = self.argmax(logits)
        ttnn.deallocate(logits)
        out = ttnn.to_torch(ids).to(torch.int64).numpy().reshape(-1)
        ttnn.deallocate(ids)
        return out
