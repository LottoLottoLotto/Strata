"""Row argmax of float32 TILE logits without the untilize: a drop-in for `ttnn.to_layout(ROW_MAJOR)` +
`ttnn.argmax(dim=-1, keepdim=True)` (fnx.tt.head.Head.argmax, the MTP draft head), ported from the Qwen3.6-35B x6
TileArgmax (/var/tmp/fuse35/src-b6 qwen36/tt/tile_argmax.py) to float32 logits and 1-32 rows.

`TileArgmax(mesh)(logits[, keep_rm=True])`: logits [1, 1, T, V] float32 TILE, interleaved DRAM (one tile row, T <= 32,
V % 32 == 0, V / 32 >= 2 x grid cores) -> ids uint32 [1, 1, T, 1] ROW_MAJOR DRAM (ttnn.argmax's keepdim tensor);
keep_rm=True also returns the T logits rows row-major (the bits ttnn.to_layout produces, in the same element order) as
a float32 ROW_MAJOR [1, 1, T * k, V / k] tensor, k = rm_pages(V / 32): each row spread over k DRAM pages / banks
instead of one 1 MB page in one bank; read it flattened (`.reshape(T, -1)`).  One program
(kernels/argmax/tile_argmax.cpp) on the whole grid rectangle, BRISC + NCRISC of each core scanning a contiguous tile
range of only the T face rows (128 B per row and tile instead of the 4 KB tile), row r gathered on core r / 2.
Trace-safe: no host sync; the compiled program depends on (T, V, keep_rm) only (tensor addresses are runtime args).

Exactness: ids equal ttnn.argmax's on every float32 input: the first index of the row maximum, NaN never picked,
-0 == +0, index 0 when no value exceeds -inf (fnx/tests/test_tile_argmax.py checks it against the stock path).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import ttnn

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "argmax" / "tile_argmax.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
CHUNK_BYTES = 32 << 10          # L1 per read chunk and lane (two chunks when a lane's range needs more than one)
FP = ttnn.KernelDescriptor.SourceType.FILE_PATH


def _dm(riscv: int, noc: int) -> ttnn.DataMovementConfigDescriptor:
    return ttnn.DataMovementConfigDescriptor(
        processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[riscv],
        noc=(ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default)[noc])


def rm_pages(tiles: int) -> int:
    """Pages per row of the keep_rm copy: the smallest divisor of the tile columns from 8 (the DRAM banks) to 64, else
    1 (one page per row)."""
    return next((k for k in range(8, 65) if tiles % k == 0), 1)


class TileArgmax:
    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.gy = g.x, g.y
        self.workers = 2 * g.x * g.y
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        # row r's gatherer is lane r % 2 of core r / 2 (row-major): its NoC coordinates, per core
        self.gather_xy = []
        for c in range(16):
            v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(c % g.x, c // g.x))
            self.gather_xy += [v.x, v.y]
        self.programs: dict[tuple, tuple] = {}

    def _program(self, rows: int, tiles: int, keep_rm: bool, io: list):
        key = (rows, tiles, keep_rm)
        p = self.programs.get(key)
        if p is not None:
            return p
        span = -(-tiles // self.workers)                   # tiles of the largest worker range
        tb = 128 * rows                                     # L1 bytes per tile (rows x (2 face rows of 64 B))
        chunk = min(span, max(1, CHUNK_BYTES // tb))
        nbuf = 1 if chunk >= span else 2
        acc = [a for t in io for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        defines = [("KEEP_RM", "1")] if keep_rm else []
        kernels = []
        for lane, cfg in ((0, _dm(0, 1)), (1, _dm(1, 0))):  # BRISC on NOC1, NCRISC on NOC0: one NoC per RISC
            named = [("rows", rows), ("tiles", tiles), ("workers", self.workers), ("lane", lane),
                     ("grid_x", self.gx), ("chunk", chunk), ("rm_pages", rm_pages(tiles))]
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=KERNEL, source_type=FP, core_ranges=self.crs, compile_time_args=acc,
                named_compile_time_args=named, defines=defines,
                common_runtime_args=[0, 0, 0] + self.gather_xy[:2 * -(-rows // 2)], config=cfg))

        def cb(idx: int, size: int):
            return ttnn.CBDescriptor(total_size=size, core_ranges=self.crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=size)])

        data = nbuf * chunk * tb + 64
        cbs = [cb(0, data), cb(1, data), cb(2, 2 * self.workers * 16 + 64), cb(3, 2 * 64 * (rows + 1) + 64)]
        if keep_rm:                                         # rows > 1: each lane's chunk joined row by row
            cbs.append(cb(4, (2 * chunk * tb if rows > 1 else 0) + 64))
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in (0, 1)]
        h = int.from_bytes(hashlib.sha1(repr(("fnx.tt.tile_argmax", key, self.gx, self.gy)).encode()).digest()[:8],
                           "little")
        p = self.programs[key] = (kernels, cbs, sems, h)
        return p

    def __call__(self, logits, keep_rm: bool = False):
        shape = list(logits.shape)
        pad = list(logits.padded_shape)
        rows, width = shape[-2], shape[-1]
        if (logits.dtype != ttnn.float32 or logits.layout != ttnn.TILE_LAYOUT or not 1 <= rows <= 32
                or width % 32 or pad[-2] != 32 or any(d != 1 for d in pad[:-2]) or logits.memory_config() != DRAM
                or width // 32 < self.workers):
            raise ValueError(f"TileArgmax needs interleaved DRAM float32 TILE logits with 1-32 rows in one tile row and "
                             f">= {self.workers} tiles: {logits.dtype} {logits.layout} {shape} {pad}")
        ids = ttnn.empty(shape[:-1] + [1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM)
        io = [logits, ids]
        if keep_rm:
            k = rm_pages(width // 32)
            io.append(ttnn.empty(shape[:-2] + [rows * k, width // k], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.mesh,
                                 DRAM))
        kernels, cbs, sems, h = self._program(rows, width // 32, keep_rm, io)
        common = [t.buffer_address() for t in io] + [0] * (3 - len(io)) + self.gather_xy[:2 * -(-rows // 2)]
        for kd in kernels:
            kd.common_runtime_args = common
        pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        pd.custom_program_hash = h
        ttnn.generic_op(io, pd)
        return (ids, io[2]) if keep_rm else ids
