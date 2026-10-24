"""Bank-packed dual-NoC decode matmuls on the P150: out[1,1,T,N] = x[1,1,T<=32,K] @ W for one tile row of activations.

Port of the Qwen3.6-35B x6 PackedDecodeMatmul (/var/tmp/fuse35/src qwen36/tt/packed_matmul.py).  The program is the
stock 1D-mcast matmul of the caller's program config (MatmulMultiCoreReuseMcast1DProgramFactory.create_descriptor with
the same compute kernel config, output dtype and memory config: same compute kernel, compile args, CBs, in0
multicast); only the in1 weight reader is swapped for kernels/packed/packed_in1_reader.cpp (the image's packed LM-head
reader with the K-group size generalized: K tiles % (8 * group) == 0, K = 80 tiles takes groups of 1, 2, 5 or 10; the
image's fixed 4 needs K % 32 == 0), reading a weight whose TILE pages were permuted worker-major / bank-rotating
(PackedLayout, placed on the device by place_packed: exact page copies).  Each worker streams its own columns as
contiguous K-group runs from one DRAM bank at a time, every read split over both NoCs (all data-movement kernels in
DM_DYNAMIC_NOC).  The compute kernel receives the identical in1 blocks in the identical order, so the outputs are
bitwise the stock matmul's (fnx/tests/test_packed.py).  Defaults group 2, 2 in1 CB slots, 1 block outstanding: the
other variants (groups 5/10, 3-4 slots, 2-3 outstanding) were within 1% on the LM head, draft head and GDN mega.

Requirements: weight bfloat8_b/bfloat4_b, DRAM interleaved, in packed order for (K tiles, N tiles, per_core_N, group);
K tiles % (8 * group) == 0; program config 1D mcast_in0, out block = per_core_N, in0_block_w | K tiles,
2 <= workers <= grid; input per_core_M tile rows (one for decode: <= 32 rows).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import ttnn

TILE = 32
BANKS = 8                                # P150 DRAM banks (interleaved page p -> bank p % 8)
GROUP = 2                                # default K-group (tiles per bank run): K = 80 tiles -> 5 runs per bank
KDIR = Path(__file__).resolve().parent / "kernels" / "packed"
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_STOCK_READER = "reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
_FACTORY = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMcast1DProgramFactory


@dataclass(frozen=True)
class PackedLayout:
    """Page order of a packed [K, N] weight: worker w (columns [w * cols, ...), row-major on the program grid) holds
    its K-groups g (`group` K tiles) in bank (w + g) % 8, worker-major and K-major within a bank."""
    k_tiles: int
    n_tiles: int
    cols: int                            # per_core_N of the program config
    group: int = GROUP

    def __post_init__(self):
        if self.k_tiles % (BANKS * self.group) or self.workers < 2:
            raise ValueError(f"PackedLayout: unsupported {self.k_tiles}x{self.n_tiles} tiles / cols {self.cols} / "
                             f"group {self.group}")

    @property
    def workers(self) -> int:
        return -(-self.n_tiles // self.cols)

    @property
    def last_cols(self) -> int:
        return self.n_tiles - (self.workers - 1) * self.cols

    def pages(self) -> np.ndarray:
        """int64 [k_tiles, n_tiles]: the packed page index of source tile (k, n) (the place_packed.cpp formula)."""
        k = np.arange(self.k_tiles, dtype=np.int64)[:, None]
        n = np.arange(self.n_tiles, dtype=np.int64)[None, :]
        w, j = n // self.cols, n % self.cols
        valid = np.where(w == self.workers - 1, self.last_cols, self.cols)
        g, h = k // self.group, k % self.group
        bank_page = (w * (self.k_tiles // BANKS) * self.cols + (g // BANKS) * self.group * valid + h * valid + j)
        return (g + w) % BANKS + BANKS * bank_page

    def _tiles(self, a: torch.Tensor) -> torch.Tensor:
        return a.reshape(self.k_tiles, TILE, self.n_tiles, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)

    def _untiles(self, t: torch.Tensor, shape) -> torch.Tensor:
        return t.reshape(self.k_tiles, self.n_tiles, TILE, TILE).permute(0, 2, 1, 3).reshape(shape)

    def unpack(self, a: torch.Tensor) -> torch.Tensor:
        """Host values [.., K, N] read back from a packed weight (ttnn.to_torch) -> the same values in source order."""
        pages = torch.from_numpy(self.pages().reshape(-1))
        return self._untiles(self._tiles(a)[pages], a.shape)


def place_packed(part, dst, col0: int, layout: PackedLayout, chunk: int = 128) -> None:
    """Copy the TILE pages of part [.., K, W] (DRAM interleaved, dst's block-float dtype) to their packed positions
    for tile columns col0 .. col0 + W/32 of dst [.., K, N] (kernels/packed/place_packed.cpp): exact page copies."""
    mesh = dst.device()
    pw = int(part.shape[-1]) // TILE
    if part.dtype != dst.dtype or int(part.shape[-2]) != layout.k_tiles * TILE or col0 + pw > layout.n_tiles:
        raise ValueError("place_packed: part does not fit the packed weight")
    page = TILE_BYTES[dst.dtype]
    chunk = min(chunk, pw)
    g = mesh.compute_with_storage_grid_size()
    n = min(layout.k_tiles, g.x * g.y)
    cores = [(i % g.x, i // g.x) for i in range(n)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i]
    cb = ttnn.CBDescriptor(total_size=chunk * page, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=chunk * page)])
    acc = [a for t in (part, dst) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "place_packed.cpp"), core_ranges=crs, compile_time_args=acc,
        named_compile_time_args=[("page_bytes", page), ("chunk", chunk), ("part_cols", pw), ("k_tiles", layout.k_tiles),
                                 ("cols", layout.cols), ("group", layout.group), ("last_worker", layout.workers - 1),
                                 ("last_cols", layout.last_cols)],
        runtime_args=rt, common_runtime_args=[part.buffer_address(), dst.buffer_address(), col0, n],
        config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([part, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))


class PackedMatmul:
    """x [1,1,T,K] @ packed weight -> [1,1,T,N] of `dtype` in `memory_config`: one generic_op, bitwise the stock
    ttnn.linear(x, W, program_config=program_config, dtype=dtype, compute_kernel_config=..., memory_config=...).
    T <= 32 * per_core_M: one tile row (decode) or, with per_core_M = M, M tile rows of a prompt block (each output
    tile accumulates its K blocks in the same order, so its rows are bitwise the one-tile-row calls': the weight is
    read once for all of them)."""

    def __init__(self, weight, program_config, compute_kernel_config, dtype=ttnn.bfloat16,
                 memory_config=ttnn.DRAM_MEMORY_CONFIG, group: int = GROUP, cb_slots: int = 2, outstanding: int = 1):
        pc = type(program_config).from_json(program_config.to_json())     # private copy: allowed_worker_cores
        grid = pc.compute_with_storage_grid_size
        if pc.allowed_worker_cores is None:      # = the factory's own auto-population (ttnn normalize_program_config)
            pc.allowed_worker_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        k_tiles, n_tiles = int(weight.padded_shape[-2]) // TILE, int(weight.padded_shape[-1]) // TILE
        self.layout = PackedLayout(k_tiles, n_tiles, pc.per_core_N, group)
        mc = weight.memory_config()
        if (weight.dtype not in TILE_BYTES or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                or mc.buffer_type != ttnn.BufferType.DRAM or math.prod(list(weight.padded_shape)[:-2]) != 1
                or not pc.mcast_in0 or pc.gather_in0 or pc.untilize_out
                or pc.out_block_w != pc.per_core_N or k_tiles % pc.in0_block_w
                or self.layout.workers > grid.x * grid.y):
            raise ValueError("PackedMatmul: unsupported weight/program geometry")
        if not 1 <= outstanding <= 4 or outstanding >= cb_slots or cb_slots > 8:
            raise ValueError("PackedMatmul: need 1 <= outstanding < cb_slots <= 8")
        self.weight, self.pc, self.ckc, self.dtype, self.mc = weight, pc, compute_kernel_config, dtype, memory_config
        self.cb_slots, self.outstanding = cb_slots, outstanding
        self.block_bytes = pc.in0_block_w * pc.per_core_N * TILE_BYTES[weight.dtype]
        self._progs: dict = {}            # descriptors by (input, output) addresses: eager calls skip the rebuild

    def _program(self, x, out):
        params = ttnn._ttnn.operations.matmul.MatmulParams()
        params.program_config = self.pc
        params.compute_kernel_config = self.ckc
        params.output_dtype = self.dtype
        params.output_mem_config = self.mc
        params.bcast_batch = True
        params.transpose_a = False
        params.transpose_b = False
        params.untilize_out = False
        inputs = ttnn._ttnn.operations.matmul.MatmulInputs()
        inputs.input_tensors = [x, self.weight]
        inputs.optional_input_tensors = [None]
        inputs.optional_output_tensors = [None]
        desc = _FACTORY.create_descriptor(params, inputs, [out])
        kernels = list(desc.kernels)
        readers = [k for k in kernels if Path(k.kernel_source).name == _STOCK_READER]
        if len(readers) != 1:
            raise ValueError("PackedMatmul: stock in1 reader not found")
        reader = readers[0]
        defines = [(name, value) for name, value in reader.defines]
        if any(name.startswith("QWEN_") for name, _ in defines):
            raise ValueError("PackedMatmul: stock reader already carries a packed layout")
        lay = self.layout
        defines += [
            ("QWEN_LMHEAD_PACKED_LAYOUT", "1"),
            ("QWEN_LMHEAD_CB_SLOTS", str(self.cb_slots)),
            ("QWEN_LMHEAD_OUTSTANDING", str(self.outstanding)),
            ("QWEN_PACKED_LAST_WORKER", str(lay.workers - 1)),
            ("QWEN_PACKED_PAGES_PER_WORKER_PER_BANK", str((lay.k_tiles // BANKS) * lay.cols)),
            ("QWEN_PACKED_GROUP_K_TILES", str(lay.group)),
            ("QWEN_PACKED_GRID_X", str(self.pc.compute_with_storage_grid_size.x)),
            ("QWEN_PACKED_DUAL_NOC", "1"),
        ]
        if self.outstanding > 1:
            defines.append(("QWEN_LMHEAD_PIPELINED", "1"))
        for kernel in kernels:
            cfg = kernel.config
            if isinstance(cfg, ttnn.DataMovementConfigDescriptor):
                kernel.config = ttnn.DataMovementConfigDescriptor(processor=cfg.processor, noc=cfg.noc,
                                                                  noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)
            elif not isinstance(cfg, ttnn.ComputeConfigDescriptor):
                raise ValueError(f"PackedMatmul: unexpected kernel config {type(cfg).__name__}")
        reader.defines = defines
        reader.kernel_source = str(KDIR / "packed_in1_reader.cpp")
        desc.kernels = kernels
        cbs = list(desc.cbs)
        found = 0
        for cb in cbs:
            if {f.buffer_index for f in cb.format_descriptors} == {1}:
                if cb.total_size != 2 * self.block_bytes:
                    raise ValueError(f"PackedMatmul: unexpected stock in1 CB size {cb.total_size}")
                cb.total_size = self.cb_slots * self.block_bytes
                found += 1
        if found != 1:
            raise ValueError("PackedMatmul: stock in1 CB not found")
        desc.cbs = cbs
        return desc

    def __call__(self, x):
        rows, m = int(x.shape[-2]), TILE * self.pc.per_core_M
        if (rows > m or int(x.padded_shape[-2]) != m or math.prod(list(x.padded_shape)[:-2]) != 1
                or int(x.shape[-1]) != self.layout.k_tiles * TILE):
            raise ValueError(f"PackedMatmul: input must be [1,1,<={m},K] ({self.pc.per_core_M} tile rows)")
        out = ttnn.empty([1, 1, rows, self.layout.n_tiles * TILE], self.dtype, ttnn.TILE_LAYOUT, x.device(), self.mc)
        key = (int(x.buffer_address()), int(out.buffer_address()), tuple(x.shape), x.dtype,
               x.memory_config().buffer_type)
        try:
            desc = self._progs.get(key)
            if desc is None:
                if len(self._progs) >= 64:
                    self._progs.clear()
                desc = self._progs[key] = self._program(x, out)
            ttnn.generic_op([x, self.weight, out], desc)
        except BaseException:
            ttnn.deallocate(out)
            raise
        return out
