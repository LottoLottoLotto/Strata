"""Hyper-connection read / write on the P150 (TTNN), matching fnx.ref.hc (decode semantics: f32 activations, BF16
weights, f32 accumulation).

Layouts: residual R `[1, 1, T, 10240]` float32 TILE in DRAM (column = stream*2560 + d; T = logical rows, tile-padded
to 32).  `HcRead(mesh, w, prefix).forward(R) -> (mixed, inj)`: mixed `[1,1,T,2560]` (float32 by default, `out_dtype`
to change), inj `[1,1,T,32]` float32 with columns 0..3 valid (None for `output_hc`).
`hc_write(R, bo, inj) -> R + (2*sigmoid(inj/4))[stream] * bo`, bo float32 or bfloat16 `[1,1,T,2560]`.

Both are single fused device programs (ttnn.generic_op; kernels in fnx/tt/kernels/hc, see hc_common.h for the core
roles), one launch per 32-row tile of R:
- HcRead, 107 cores of the 11x10 grid.  PREP (88 cores, 3-4 R tiles each): y = R*gamma*sqrt(2560), its exact
  bf16 split hi + lo, bf16(R^2).  DOWN (8 K-slices of 40 tiles x 11 N-tiles): bf16 matmuls (hi + lo) @ [Wd^T/4 |
  Wi^T] per slice with BF16 weights streamed from DRAM.  SSQ/RS: per-stream sums of squares (bf16(R^2) @ ones) and
  rs = rsqrt(sum + 2560*eps).  RED (11 cores): p = sum_c rs_c * slice partials, silu, exact bf16 split (the
  inject column is written out).  MIX (80 cores, one output column tile each): gate = silu(p) @ Wu^T,
  mixed = 1/4 sum_c y_c * rs_c * sigmoid(gate_c).
  Data movement (the program is bound by DRAM reads and inter-core traffic, not math; FNX_HC_PROF zones): the MAIN
  cores' NCRISC streams the down weights on NOC0 from the first cycle (4 chunks, DOWN consumes each as it lands),
  then 24 of its column's 40 up-weight tiles; their BRISC reads PREP's R tiles and gamma rows on NOC1 (a gamma row
  is replicated over its tile by local NoC copies: 128 B instead of 4 KB from DRAM), does every MAIN remote write on
  NOC1 and then reads the other 16 up-weight tiles.  Many-to-many multicasts stall the NoC (11 cores of a row
  multicasting into the same row: up to 37 us; 10 RED cores into the 80 MIX cores: up to 22 us), so both go
  through one core: each row's PREP output is gathered on core (10, s) and multicast once, RED's silu tiles on
  the RS core.  Traced T=1: 0.095 ms with every read on NOC1 and those multicasts, 0.054 ms now.
- hc_write, 80 cores: w = 2*sigmoid(inj/4), broadcast per stream by an exact bf16 matmul, out = R + w * bo.

Precision. Matmuls see only bf16 operands: a float32 activation x is split exactly into hi = bf16(x) and lo =
bf16(x - hi) (nearest-even, |x - hi - lo| <= 2^-18 |x|) and both halves are multiplied, so the FPU's TF32 input
truncation never applies; products are exact at HiFi4, accumulation is fp32 in DST.  The norm's sum of squares uses
bf16(R^2) (unbiased rounding of positive terms: ~2e-5 relative on the sum).  Everything elementwise (norm scale,
sigmoid, silu, rsqrt, the stream mean, R + w*bo) runs in fp32 on the SFPU.  `precise=False` drops the lo halves.

The older op-by-op helpers below (mm, stream_sums, stream_bcast, rms_streams, gamma_dev) are used by fnx.tt.ple.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import torch
import ttnn

from .. import paths
from ..ref.hc import D, EPS, HC, LR, N, hc_weights

F32, BF16 = ttnn.float32, ttnn.bfloat16
TT_CACHE = str(paths.TT_CACHE)
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                         packer_l1_acc=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
INJ_W = 32                                    # inj columns on device (0..3 valid)
KDIR = Path(__file__).resolve().parent / "kernels" / "hc"

_DT_NAME = {F32: "f32", BF16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}


def upload(mesh, name: str, arr: np.ndarray, dtype, layout=ttnn.TILE_LAYOUT):
    """Weight upload through the converted-tensor cache (fnx.paths.TT_CACHE/<name>_<dtype>)."""
    os.makedirs(TT_CACHE, exist_ok=True)
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    while a.ndim < 4:
        a = a[None]
    return ttnn.as_tensor(torch.from_numpy(a), dtype=dtype, layout=layout, device=mesh, memory_config=DRAM,
                          cache_file_name=f"{TT_CACHE}/{name}_{_DT_NAME[dtype]}")


def to_dev(mesh, arr: np.ndarray, dtype=F32, layout=ttnn.TILE_LAYOUT):
    """Uncached host -> device upload of a (≤4-D) array, padded to 4-D."""
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    while a.ndim < 4:
        a = a[None]
    return ttnn.from_torch(torch.from_numpy(a), dtype=dtype, layout=layout, device=mesh, memory_config=DRAM)


def to_np(t) -> np.ndarray:
    return ttnn.to_torch(t).float().numpy()


_CONSTS: dict = {}


def consts(mesh):
    """Per-device 0/1 stream matrices: BSUM [10240,32] (column c sums stream c), EXP [32,10240] (row c -> stream c)."""
    key = id(mesh)
    if key not in _CONSTS:
        bsum = np.zeros((D, INJ_W), np.float32)
        for c in range(HC):
            bsum[c * N:(c + 1) * N, c] = 1
        _CONSTS[key] = (to_dev(mesh, bsum, BF16), to_dev(mesh, np.ascontiguousarray(bsum.T), BF16))
    return _CONSTS[key]


def mm(x, w, precise: bool = True):
    """x float32 [.., M, K] @ w (BF16-valued) -> float32. precise: exact hi/lo split of x (two matmuls)."""
    if not precise or x.dtype != F32:
        return ttnn.matmul(x, w, dtype=F32, compute_kernel_config=HIFI4)
    hi = ttnn.typecast(x, BF16)
    hi32 = ttnn.typecast(hi, F32)
    lo = ttnn.subtract(x, hi32)
    ttnn.deallocate(hi32)
    a = ttnn.matmul(hi, w, dtype=F32, compute_kernel_config=HIFI4)
    ttnn.deallocate(hi)
    b = ttnn.matmul(lo, w, dtype=F32, compute_kernel_config=HIFI4)
    ttnn.deallocate(lo)
    out = ttnn.add(a, b)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return out


def stream_sums(x, precise: bool = True):
    """x float32 [1,1,T,10240] -> [1,1,T,32]: column c = sum over stream c (columns 4.. are 0)."""
    return mm(x, consts(x.device())[0], precise)


def stream_bcast(v, precise: bool = True):
    """v float32 [1,1,T,32] -> [1,1,T,10240]: stream c filled with v[:, c]."""
    return mm(v, consts(v.device())[1], precise)


def gamma_dev(mesh, name: str, g: np.ndarray):
    """Per-stream RMSNorm gamma for rms_streams: g * sqrt(2560) as float32 [1,1,1,10240] (the 1/2560 of the mean
    is folded into the gamma: rsqrt(s/2560 + eps) = sqrt(2560) * rsqrt(s + 2560*eps))."""
    gs = (np.asarray(g, np.float64) * np.sqrt(N)).astype(np.float32).reshape(1, D)
    return upload(mesh, f"{name}.x_sqrt{N}", gs, F32)


def rms_streams(x, gs, precise: bool = True):
    """Per-stream RMSNorm of x float32 [1,1,T,10240]: (x*g)*rsqrt(mean_c(x^2) + eps), gs from gamma_dev()."""
    sq = ttnn.multiply(x, x)
    s = stream_sums(sq, precise)
    ttnn.deallocate(sq)
    s_e = ttnn.add(s, float(np.float32(N) * EPS))
    ttnn.deallocate(s)
    rs = ttnn.rsqrt(s_e)
    ttnn.deallocate(s_e)
    rsb = stream_bcast(rs, precise)
    ttnn.deallocate(rs)
    xg = ttnn.multiply(x, gs)
    out = ttnn.multiply(xg, rsb)
    ttnn.deallocate(xg)
    ttnn.deallocate(rsb)
    return out


# ---------------------------------------------------------------------------------------------- fused programs

GRID_X, GRID_Y = 11, 10
_TILE = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
_F32_TILE, _BF16_TILE = 4096, 2048
_PAGE = {F32: _F32_TILE, BF16: _BF16_TILE}
# hc_common.h: CB indices / semaphores
(CB_R, CB_GS, CB_Y, CB_HL, CB_X2, CB_WD, CB_ACT, CB_PART, CB_WU, CB_LO, CB_RS, CB_YMIX, CB_ACC, CB_OUT, CB_X2IN,
 CB_ONES, CB_SSQ, CB_SSQIN, CB_PIN, CB_RED, CB_INJ, CB_XN) = range(22)
N_SEM = 9
_EPS_BITS = struct.unpack("<I", struct.pack("<f", float(np.float32(N) * EPS)))[0]


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _virtual_axes(mesh) -> tuple[list[int], list[int]]:
    """NOC coordinates of logical column x / row y; the kernels assume virtual(x, y) = (VX[x], VY[y])."""
    g = mesh.compute_with_storage_grid_size()
    if g.x < GRID_X or g.y < GRID_Y:
        raise RuntimeError(f"fused HC needs an {GRID_X}x{GRID_Y} worker grid, device has {g.x}x{g.y}")
    v = {(x, y): mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(GRID_X) for y in range(GRID_Y)}
    vx = [v[x, 0].x for x in range(GRID_X)]
    vy = [v[0, y].y for y in range(GRID_Y)]
    if any((c.x, c.y) != (vx[x], vy[y]) for (x, y), c in v.items()):
        raise RuntimeError("worker NOC coordinates are not separable in x / y")
    return vx, vy


def _tile_list(tiles: np.ndarray) -> np.ndarray:
    """[n, 32, 32] tiles -> [32, n*32]: as a one-tile-row TILE tensor, page i is tiles[i] (DRAM-interleaved: a
    core's consecutive pages spread over every bank, the fastest pattern measured for 88 concurrent readers)."""
    return np.ascontiguousarray(tiles.transpose(1, 0, 2).reshape(32, -1))


def _cb(cores, total: int, fmts) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(total_size=total, core_ranges=cores, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=_PAGE[dt], tile=_TILE) for i, dt in fmts])


def _compute_cfg(f32_cbs, full_sync: bool = False) -> ttnn.ComputeConfigDescriptor:
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, dst_full_sync_en=full_sync)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in f32_cbs:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode.extend(modes)
    return cfg


def _dm(riscv: int, noc: int) -> ttnn.DataMovementConfigDescriptor:
    """RISC-V `riscv` (0 BRISC, 1 NCRISC) on NoC `noc`.  NOC0 is the fast DRAM read path: 88 cores reading the
    HcRead weights interleaved take 18.5 us for 7.2 MB on NOC0 vs 33.5 us on NOC1; R + gamma + down weights on NOC0
    with the up weights on NOC1 take 36.3 us for all 16.4 MB, vs 41.8 us on NOC0 alone and 73.7 us on NOC1 alone."""
    return ttnn.DataMovementConfigDescriptor(
        processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[riscv],
        noc=(ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default)[noc])


def _profile_defines():
    """FNX_HC_PROF=1: device-profiler zones (PREP / DOWN / MIX ...) in the HcRead kernels (TT_METAL_DEVICE_PROFILER=1)."""
    return [("HC_PROF", "1")] if os.environ.get("FNX_HC_PROF") == "1" else []


class _Program:
    """A cached ProgramDescriptor whose kernels all take the same common runtime args (buffer addresses + row
    tile); generic_op caches the compiled program by structure, so a call only rewrites those args."""

    def __init__(self, kernels, cbs, semaphores, n_args: int):
        self.desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
        self.n_args = n_args

    def run(self, io_tensors, args):
        assert len(args) == self.n_args
        for k in self.desc.kernels:
            k.common_runtime_args = args
        ttnn.generic_op(io_tensors, self.desc)


_PROGRAMS: dict = {}


def _read_program(mesh, nn: int, precise: bool, out_dtype, r_dram: bool) -> _Program:
    key = ("read", id(mesh), nn, precise, out_dtype, r_dram)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    vx, vy = _virtual_axes(mesh)
    ct = [nn, int(precise), int(r_dram), _PAGE[out_dtype], _EPS_BITS, *vx, *vy]
    boxes = {1: (0, 0, GRID_X - 1, 7), 2: (0, 8, 7, 8), 3: (8, 8, 8, 8), 4: (0, 9, nn - 1, 9)}
    roles = {r: _rect(*b) for r, b in boxes.items()}
    every = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
                               for a, b, c, d in boxes.values()})
    f32_cbs = [CB_R, CB_GS, CB_Y, CB_PART, CB_RS, CB_YMIX, CB_ACC, CB_SSQ, CB_SSQIN, CB_PIN, CB_INJ, CB_XN]
    if out_dtype == F32:
        f32_cbs.append(CB_OUT)
    kernels = []
    for role, cores in roles.items():
        d = [("ROLE", str(role))] + _profile_defines()
        kernels += [
            # MAIN: NCRISC on NOC0 (the weight stream), BRISC on NOC1; the other roles: BRISC on NOC0 (_dm)
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "hcr_reader.cpp"), core_ranges=cores, compile_time_args=ct,
                                  defines=d, common_runtime_args=[0] * 7, config=_dm(1, int(role != 1))),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "hcr_writer.cpp"), core_ranges=cores, compile_time_args=ct,
                                  defines=d, common_runtime_args=[0] * 7, config=_dm(0, int(role == 1))),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "hcr_compute.cpp"), core_ranges=cores, compile_time_args=ct,
                                  defines=d, common_runtime_args=[0] * 7, config=_compute_cfg(f32_cbs, role in (1, 4))),
        ]
    F = _F32_TILE
    B = _BF16_TILE
    cbs = [
        _cb(every, 4 * F, [(CB_R, F32)]), _cb(every, 4 * F, [(CB_GS, F32)]), _cb(every, 4 * F, [(CB_Y, F32)]),
        _cb(every, 8 * B, [(CB_HL, BF16)]), _cb(every, 8 * B, [(CB_X2, BF16)]), _cb(every, 40 * B, [(CB_WD, BF16)]),
        # one region, a different user per role: MAIN act, SSQ x2in, RS ssqin, RED pin
        _cb(every, 80 * B, [(CB_ACT, BF16), (CB_X2IN, BF16), (CB_SSQIN, F32), (CB_PIN, F32)]),
        _cb(every, 1 * F, [(CB_PART, F32)]), _cb(every, 40 * B, [(CB_WU, BF16)]), _cb(every, 20 * B, [(CB_LO, BF16)]),
        _cb(every, 4 * F, [(CB_RS, F32)]), _cb(every, 4 * F, [(CB_YMIX, F32)]), _cb(every, 4 * F, [(CB_ACC, F32)]),
        _cb(every, _PAGE[out_dtype], [(CB_OUT, out_dtype)]), _cb(every, 1 * B, [(CB_ONES, BF16)]),
        _cb(every, 1 * F, [(CB_SSQ, F32)]), _cb(every, 2 * B, [(CB_RED, BF16)]), _cb(every, 1 * F, [(CB_INJ, F32)]),
        _cb(every, 4 * F, [(CB_XN, F32)]),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=every, initial_value=0) for i in range(N_SEM)]
    _PROGRAMS[key] = _Program(kernels, cbs, sems, 7)
    return _PROGRAMS[key]


def _write_program(mesh, bo_dtype, precise: bool, r_dram: bool) -> _Program:
    key = ("write", id(mesh), bo_dtype, precise, r_dram)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    _virtual_axes(mesh)
    cores = _rect(0, 0, 9, 7)
    ct = [int(r_dram), _PAGE[bo_dtype], int(precise)]
    f32_cbs = [CB_R, CB_INJ, CB_OUT] + ([CB_GS] if bo_dtype == F32 else [])
    kernels = [
        # DRAM reads on NOC0, writes on NOC1 (_dm)
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "hcw_reader.cpp"), core_ranges=cores, compile_time_args=ct,
                              common_runtime_args=[0] * 5, config=_dm(0, 0)),
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "hcw_writer.cpp"), core_ranges=cores, compile_time_args=ct,
                              common_runtime_args=[0] * 5, config=_dm(1, 1)),
        ttnn.KernelDescriptor(kernel_source=str(KDIR / "hcw_compute.cpp"), core_ranges=cores, compile_time_args=ct,
                              common_runtime_args=[0] * 5, config=_compute_cfg(f32_cbs)),
    ]
    F, B = _F32_TILE, _BF16_TILE
    cbs = [_cb(cores, 4 * F, [(CB_R, F32)]), _cb(cores, _PAGE[bo_dtype], [(CB_GS, bo_dtype)]),
           _cb(cores, 2 * B, [(CB_HL, BF16)]), _cb(cores, 4 * B, [(CB_X2, BF16)]), _cb(cores, 1 * F, [(CB_INJ, F32)]),
           _cb(cores, 4 * F, [(CB_OUT, F32)])]
    _PROGRAMS[key] = _Program(kernels, cbs, [], 5)
    return _PROGRAMS[key]


def _check_rows(t, width: int, dtypes, name: str) -> int:
    """Logical-row count of a [1,1,T,width] TILE tensor; validates dtype / layout / interleaving."""
    shape = tuple(t.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != width:
        raise ValueError(f"{name} must be [1,1,T,{width}], got {shape}")
    if t.dtype not in dtypes or t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} must be a {dtypes} TILE tensor, got {t.dtype} {t.layout}")
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"{name} must be interleaved")
    return shape[2]


def _is_dram(t) -> bool:
    return t.memory_config().buffer_type == ttnn.BufferType.DRAM


class HcRead:
    """One HC read (prefix 'blk.{L}.hc_attn' / 'blk.{L}.hc_ffn' / 'output_hc') with its weights on the device.

    Device weights (DRAM-interleaved, in the order the fused program reads them): gamma*sqrt(2560) float32 row-major
    [320, 32] (one 128 B page per K-tile; PREP core (x, s) reads K-tiles 40s + x + 11i and replicates each row over
    a tile in L1); down|inject [Wd^T/4 | Wi^T] ([10240, 352], the silu's /4 folded in exactly; output_hc:
    [10240, 320]) as a tile list ordered (slice s, N-tile n, K-tile k), K-tile = 40s + k; up Wu^T [320, 10240] as a
    tile list ordered (column j, stream c, K-tile n), column tile 80c + j.
    Stateless (`reset()` is a no-op).  `precise=False` drops the lo halves of the activation splits.
    """

    def __init__(self, mesh, w, prefix: str, out_dtype=F32, precise: bool = True):
        g, Wd, Wu, Wi = hc_weights(w, prefix)
        if out_dtype not in (F32, BF16):
            raise ValueError(f"out_dtype must be float32 or bfloat16, got {out_dtype}")
        self.mesh, self.prefix, self.out_dtype, self.precise = mesh, prefix, out_dtype, precise
        self.inject = Wi is not None
        self.nn = LR // 32 + (1 if self.inject else 0)              # N-tiles of the down|inject projection
        wdi = np.zeros((D, self.nn * 32), np.float32)
        wdi[:, :LR] = np.asarray(Wd).T * np.float32(1.0 / HC)       # exact for BF16 values
        if self.inject:
            wdi[:, LR:LR + HC] = np.asarray(Wi).T
        t = wdi.reshape(D // 32, 32, self.nn, 32).transpose(0, 2, 1, 3)              # [kt, n, r, c]
        t = t.reshape(8, 40, self.nn, 32, 32).transpose(0, 2, 1, 3, 4)              # [s, n, k, r, c]
        self.wd = upload(mesh, f"{prefix}_down4_inject.tiles", _tile_list(t.reshape(-1, 32, 32)), BF16)
        u = np.asarray(Wu).T.reshape(LR // 32, 32, D // 32, 32).transpose(2, 0, 1, 3)  # [col, n, r, c]
        u = u.reshape(HC, 80, LR // 32, 32, 32).transpose(1, 0, 2, 3, 4)            # [j, c, n, r, c]
        self.wu = upload(mesh, f"{prefix}_up.tiles", _tile_list(u.reshape(-1, 32, 32)), BF16)
        gs = (np.asarray(g, np.float64) * np.sqrt(N)).astype(np.float32).reshape(D // 32, 32)
        self.gs = upload(mesh, f"{prefix}_norm.weight.x_sqrt{N}.krows", gs, F32, ttnn.ROW_MAJOR_LAYOUT)

    def reset(self):
        pass

    def forward(self, R):
        T = _check_rows(R, D, (F32,), "R")
        prog = _read_program(self.mesh, self.nn, self.precise, self.out_dtype, _is_dram(R))
        mixed = ttnn.empty([1, 1, T, N], self.out_dtype, ttnn.TILE_LAYOUT, self.mesh, DRAM)
        inj = ttnn.empty([1, 1, T, INJ_W], F32, ttnn.TILE_LAYOUT, self.mesh, DRAM) if self.inject else None
        io = [R, self.gs, self.wd, self.wu, mixed] + ([inj] if inj is not None else [])
        args = [R.buffer_address(), self.gs.buffer_address(), self.wd.buffer_address(), self.wu.buffer_address(),
                mixed.buffer_address(), inj.buffer_address() if inj is not None else 0]
        for rt in range((T + 31) // 32):
            prog.run(io, args + [rt])
        return mixed, inj


def hc_write(R, bo, inj, precise: bool = True, inplace: bool = False):
    """R + (2*sigmoid(inj/4))[stream] * bo (fused, see the module docstring); `inplace=True` updates R itself."""
    T = _check_rows(R, D, (F32,), "R")
    if _check_rows(bo, N, (F32, BF16), "bo") != T or _check_rows(inj, INJ_W, (F32,), "inj") != T:
        raise ValueError("R, bo and inj must have the same number of rows")
    if not (_is_dram(bo) and _is_dram(inj)):
        raise ValueError("bo and inj must be in DRAM")
    mesh = R.device()
    prog = _write_program(mesh, bo.dtype, precise, _is_dram(R))
    out = R if inplace else ttnn.empty([1, 1, T, D], F32, ttnn.TILE_LAYOUT, mesh, R.memory_config())
    io = [bo, inj, R] if inplace else [R, bo, inj, out]
    args = [R.buffer_address(), bo.buffer_address(), inj.buffer_address(), out.buffer_address()]
    for rt in range((T + 31) // 32):
        prog.run(io, args + [rt])
    return out
