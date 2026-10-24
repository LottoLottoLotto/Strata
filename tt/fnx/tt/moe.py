"""MoE block on the P150: router and shared expert on the device, a resident subset of the routed experts on the device
(two JIT kernels, fnx/tt/kernels/moe), the other routed experts on the host CPU (spec Part 6 §2, decode semantics).

Per call (mixed: [1,1,rows,2560] TILE, float32 preferred or bfloat16; the first T rows are tokens):
  device  logits = mixed @ [W_r^T | w_sg | 0] (BF16 [2560, E + 32], E = N_EXPERT: columns 0..E-1 router, E shared gate
          (the router layout of qwen36's TTLocalMoEGate, models/common/modules/moe/tt_moe_local.py:98-107)
  host    read back [mixed | logits] once, row-major, exactly T rows (one untilize, one transfer): the one host round
          trip of the layer.  Top-10 of the device logits (stable, ties -> lower id) with near ties refined exactly
          (below); weights = softmax over the 10 selected logits in f64, which is the spec's renormalized f64 softmax
          p[ids]/sum(p[ids]) up to p's f32 rounding (the 2^-14 clamp cannot bind: sum >= max p >= 1/256).
          Slots whose expert is RESIDENT on the device (DeviceExperts.local >= 0) go to the device, the others to
          CpuExperts.submit (Strata's CPU kernels on libfnx_cpu's spinning leader thread + pinned workers: the hand-off
          is one atomic store, the call returns at once; negative id = skip)
  device  meanwhile: shared expert sh = W_d @ (silu(W_g @ x) * (W_u @ x)) (ttnn.linear, f32 activations; bfloat8_b
          weights by default, bfloat16 via shared_dtype), sh *= sigmoid(gate); and, when a slot is resident
          (DeviceExperts.run): upload one metadata row (entries, their 32 route weights), then
            stage A  h[s] = SiLU(x G_e^T) * (x U_e^T) * route_s           for every entry s = (32-row group, expert)
            stage B  y[g] = sum over the group's entries of h[s] D_e^T     (f32, the expert sum inside the matmul)
  host    CpuExperts.wait (spins until the CPU experts are done), weighted sum of the CPU parts (f32), upload;
          bo = sh + y + routed_cpu (terms absent when empty)

Device experts: weights stored per expert in their GGUF orientation [n_out, n_in] (TILE; the matmul transposes in1, so
a block-float face row shares one exponent along n_in), routed experts bfloat8_b (measured: 1.2% parts error vs the
all-float dequantized experts, the CPU kernels 1.6%; bfloat4_b 18%).  x and h are BF16, accumulation FP32 in DEST at
HiFi4 (HiFi2 measured no faster: the kernels are bound by data movement).  Device time at T=1: ~0.1-0.3 ms for 1-10
resident experts, below the CPU experts' time it overlaps.  The shared expert stays an f32-activation ttnn.linear: in
the kernel (BF16 x) its error rose 1.7e-3 -> 3.9e-3 and golden doc agreement fell (0.9392 -> 0.9333, KL 0.029 -> 0.046).  The resident set is per layer (`resident=`, or the placement file named by
FNX_EXPERT_PLACEMENT, written by tools/expert_placement.py from Strata's expert profile, else the variant's default,
`default_resident`); without one every routed expert runs on the CPU.

Why the top-k runs on the host: qwen36's TopKRouting kernel (models/common/modules/moe/kernels/routing_math.hpp)
selects on BF16-rounded logits (bf16-equal logits fall to the lower id, not the larger f32 value), returns BF16
weights in ascending-id order and handles 1 or 2 rows only; that is not the spec's f64 router_top10.

Near ties: the device matmul is not exact even at HiFi4 + fp32 accumulation and bf16 operands (relative error
~3e-4 per output for K = 32..2560, measured), so router logits are off by up to ~4e-3.  Measured on layers 0-9:
|err| <= 3.0e-4 * sum_k |x_k W_ek| <= 8.8e-5 * ||x|| ||W_e||.  The module bounds each device logit by
b_e = 2^-11 * ||x|| * ||W_e|| (a per-product error of 2^-11 under Cauchy-Schwarz; 5.5x the largest measured), so
the exact logit lies in [lo_e, hi_e] = device +- b_e.  An expert is surely in the top 10 if lo_e exceeds the 11th
largest hi, surely out if hi_e is below the 10th largest lo.  When fewer than 10 are surely in, the undecided
experts' logits are recomputed exactly on the host (f32 x, BF16 rows, f64 accumulation: the reference's
arithmetic) and the best of them complete the set, ties to the lower id.  Weights use the device logits
(absolute error <~5e-4).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import ttnn

from .. import paths
from ..config import N_EMBD, N_EXPERT, N_FF, N_LAYERS
from ..cpu_experts import route

TT_CACHE = str(paths.TT_CACHE)
ROUTER_COLS = N_EXPERT + 32        # N_EXPERT router logits, column N_EXPERT = shared-expert gate, 31 zero columns
PRODUCT_ERR = 2.0 ** -11           # device logit error bound per unit of ||x|| ||W_e|| (module docstring)

KDIR = Path(__file__).resolve().parent / "kernels" / "moe"
KT, JT, NT = N_EMBD // 32, N_FF // 32, N_EMBD // 32     # 80 hidden tiles, 20 intermediate tiles, 80 output tiles
HDR = 64                           # metadata header words: [0] entries, [1] groups, [2] first group, [3] rows T,
                                   # [4 + g] group starts
GMAX = HDR - 5                     # 32-row groups per device round (group starts [4, 4 + GMAX] fit the header)
BK = 40                            # stage A weight block (tiles)


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=True)


def _weight(mesh, name: str, a, dtype):
    """Device DRAM TILE tensor of `a` (f32 array, or a function building it, called only when the cache misses),
    cached as ttnn.as_tensor does.  mesh None: host-only cache preparation (returns None when already cached)."""
    os.makedirs(TT_CACHE, exist_ok=True)
    base = f"{TT_CACHE}/{name}_{dtype.name}"
    if callable(a):
        path = f"{base}_dtype_{dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"
        if os.path.isfile(path):
            return None if mesh is None else ttnn._ttnn.tensor.load_tensor_flatbuffer(path, device=mesh)
        a = a()
    return ttnn.as_tensor(torch.from_numpy(np.ascontiguousarray(a, np.float32)), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                          device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name=base)


def load_placement(path: str | os.PathLike) -> list[np.ndarray]:
    """tools/expert_placement.py's JSON -> per-layer sorted resident expert ids (int32)."""
    d = json.loads(Path(path).read_text())
    layers = [np.asarray(sorted(v), np.int32) for v in d["layers"]]
    if len(layers) != N_LAYERS or any(len(v) and (v[0] < 0 or v[-1] >= N_EXPERT) for v in layers):
        raise ValueError(f"{path}: not a {N_LAYERS}-layer placement of {N_EXPERT} experts")
    return layers


_PLACEMENT: dict[str, tuple[list[np.ndarray], str]] = {}
RESIDENT_FORMATS = ("bfloat8_b", "bfloat4_b", "gptq4")   # gptq4: bfloat4_b holding fnx.quant.residents' GPTQ values


def _placement() -> tuple[list[np.ndarray], str]:
    """(per-layer sorted resident ids, format in RESIDENT_FORMATS): the placement file FNX_EXPERT_PLACEMENT (its
    "dtype", default bfloat8_b; set but empty: none), else the variant's fnx.quant.datafree.DEFAULT_RESIDENT."""
    p = os.environ.get("FNX_EXPERT_PLACEMENT")
    if p is None:
        from ..quant.datafree import DEFAULT_RESIDENT, resident_layers
        n, dt = DEFAULT_RESIDENT[paths.MODEL]
        p = f"counts:{n}:{dt}"
        if p not in _PLACEMENT:
            _PLACEMENT[p] = (resident_layers(n) if n else [np.zeros(0, np.int32)] * N_LAYERS, dt)
    elif not p:
        return [np.zeros(0, np.int32)] * N_LAYERS, "bfloat8_b"
    if p not in _PLACEMENT:
        _PLACEMENT[p] = (load_placement(p), json.loads(Path(p).read_text()).get("dtype", "bfloat8_b"))
    if _PLACEMENT[p][1] not in RESIDENT_FORMATS:
        raise ValueError(f"resident expert format {_PLACEMENT[p][1]!r}: one of {', '.join(RESIDENT_FORMATS)}")
    return _PLACEMENT[p]


def default_resident(L: int) -> np.ndarray:
    """Layer L's resident experts (_placement)."""
    return _placement()[0][L]


def resident_format() -> str:
    """The resident experts' format (_placement): bfloat8_b, bfloat4_b (min-MSE-snapped BFP4) or gptq4."""
    return _placement()[1]


def snap_bfp4(a: np.ndarray, rows: int = 32768) -> np.ndarray:
    """a f32 [R, C] (C % 16 == 0) -> its min-MSE BFP4 values over ttnn's blocks (16 consecutive elements of a row;
    fnx.quant.methods.blockq), in place, row chunks (bounded temporaries).  tt-metal's bfloat4_b conversion keeps
    them exactly."""
    from ..quant import methods
    for r0 in range(0, a.shape[0], rows):
        a[r0:r0 + rows] = methods.blockq(torch.from_numpy(a[r0:r0 + rows]), "bfp4").numpy()
    return a


def _tile_bytes(dtype) -> int:
    return ttnn.Tile([32, 32]).get_tile_size(dtype)


def _cb(crs, idx: int, dtype, page: int, n: int):
    return ttnn.CBDescriptor(total_size=page * n, core_ranges=crs,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype,
                                                                         page_size=page)])


class _Stage:
    """One device-experts kernel set (reader + writer + compute on every worker core) for one tensor configuration;
    tensor addresses are common runtime args, set per call, so all layers share the compiled program."""

    def __init__(self, mesh, stage: str, a, meta, w, out, s_cap: int, fidelity, key: tuple):
        g = mesh.compute_with_storage_grid_size()
        cores = [(x, y) for y in range(g.y) for x in range(g.x)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(cores):
            rt[x][y] = [i]
        nc = len(cores)
        meta_bytes = (HDR + s_cap) * 4
        meta_page = int(meta.shape[-1]) * 4
        acc = []
        for t in (a, meta, w):
            acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wd = w.dtype
        common = [("jt", JT), ("ncores", nc)]
        if stage == "A":
            named = common + [("kt", KT), ("bk", BK)]
            src = ("gateup_reader.cpp", "gateup_compute.cpp")
            cbs = [_cb(crs, 0, ttnn.bfloat16, _tile_bytes(ttnn.bfloat16), KT),
                   _cb(crs, 1, wd, _tile_bytes(wd), 3 * BK),
                   _cb(crs, 3, ttnn.float32, _tile_bytes(ttnn.float32), 2),
                   _cb(crs, 4, ttnn.uint32, meta_bytes, 1),
                   _cb(crs, 16, ttnn.bfloat16, _tile_bytes(ttnn.bfloat16), 2)]
            wnamed = [("ncores", nc), ("count_word", 0), ("per", JT)]
        else:
            named = common + [("nt", NT)]
            src = ("down_reader.cpp", "down_compute.cpp")
            cbs = [_cb(crs, 0, ttnn.bfloat16, _tile_bytes(ttnn.bfloat16), 2 * JT),
                   _cb(crs, 1, wd, _tile_bytes(wd), 2 * JT),
                   _cb(crs, 4, ttnn.uint32, meta_bytes, 1),
                   _cb(crs, 16, ttnn.float32, _tile_bytes(ttnn.float32), 2)]
            wnamed = [("ncores", nc), ("count_word", 1), ("per", NT)]
        rnamed = named + [("meta_bytes", meta_bytes), ("meta_page", meta_page), ("woff", meta_bytes),
                          ("in_l1", int(_mem_key(a) == "L1"))]
        FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
        self.reader = ttnn.KernelDescriptor(kernel_source=str(KDIR / src[0]), source_type=FP, core_ranges=crs,
                                            compile_time_args=acc, named_compile_time_args=rnamed, runtime_args=rt,
                                            common_runtime_args=[0, 0, 0], config=ttnn.ReaderConfigDescriptor())
        self.writer = ttnn.KernelDescriptor(kernel_source=str(KDIR / "writer.cpp"), source_type=FP, core_ranges=crs,
                                            compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
                                            named_compile_time_args=wnamed, runtime_args=rt,
                                            common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
        self.compute = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / src[1]), source_type=FP, core_ranges=crs, named_compile_time_args=named,
            runtime_args=rt, config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, math_approx_mode=False,
                                                                 fp32_dest_acc_en=True))
        self.cbs = cbs
        # the program hash, computed once (generic_op would hash the whole descriptor on every call)
        self.hash = int.from_bytes(hashlib.sha1(repr(("fnx.tt.moe", key, nc)).encode()).digest()[:8], "little")

    def __call__(self, a, meta, w, out):
        self.reader.common_runtime_args = [a.buffer_address(), meta.buffer_address(), w.buffer_address()]
        self.writer.common_runtime_args = [out.buffer_address()]
        pd = ttnn.ProgramDescriptor(kernels=[self.reader, self.writer, self.compute], semaphores=[], cbs=self.cbs)
        pd.custom_program_hash = self.hash
        ttnn.generic_op([a, meta, w, out], pd)
        return out


_STAGES: dict[tuple, _Stage] = {}


def _mem_key(t) -> str:
    return "L1" if t.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"


class DeviceExperts:
    """A layer's resident routed experts on the device (module docstring).

    `resident`: sorted expert ids kept on the device (non-empty); `local[e]` = their index there, -1 for the CPU's.
    mesh None: host-only conversion of the weight caches."""

    def __init__(self, mesh, w, L: int, resident, fmt: str = "bfloat8_b", fidelity=ttnn.MathFidelity.HiFi4):
        """fmt (RESIDENT_FORMATS): bfloat8_b; bfloat4_b, the experts' values first snapped to their min-MSE BFP4 grid
        (snap_bfp4), which tt-metal's conversion then keeps (its own round-to-nearest BFP4 has the larger error);
        gptq4, bfloat4_b holding the layer's GPTQ values (fnx.quant.residents store, same experts), kept likewise."""
        self.mesh, self.L, self.fidelity = mesh, L, fidelity
        self.resident = np.asarray(sorted(int(e) for e in resident), np.int32)
        n = len(self.resident)
        if not n:
            raise ValueError("DeviceExperts needs at least one resident expert")
        self.local = np.full(N_EXPERT, -1, np.int32)
        self.local[self.resident] = np.arange(n, dtype=np.int32)
        tag = hashlib.sha1(self.resident.tobytes()).hexdigest()[:12]
        model = w.model

        def gate_up():
            a = np.empty((n * 2 * N_FF, N_EMBD), np.float32)
            for i, e in enumerate(self.resident):
                a[(2 * i) * N_FF:(2 * i + 1) * N_FF] = model.get_expert(f"blk.{L}.ffn_gate_exps.weight", int(e))
                a[(2 * i + 1) * N_FF:(2 * i + 2) * N_FF] = model.get_expert(f"blk.{L}.ffn_up_exps.weight", int(e))
            return a

        def down():
            a = np.empty((n * N_EMBD, N_FF), np.float32)
            for i, e in enumerate(self.resident):
                a[i * N_EMBD:(i + 1) * N_EMBD] = model.get_expert(f"blk.{L}.ffn_down_exps.weight", int(e))
            return a

        if fmt == "bfloat4_b":
            raw_gu, raw_dn = gate_up, down
            gate_up, down = (lambda: snap_bfp4(raw_gu())), (lambda: snap_bfp4(raw_dn()))
            tag += "-s4"
        elif fmt == "gptq4":
            from ..quant import residents
            tag += f"-g4{residents.layer_digest(L, self.resident)[:8]}"
            gate_up = lambda: residents.load_layer(L, self.resident, "gate_up")    # noqa: E731
            down = lambda: residents.load_layer(L, self.resident, "down")          # noqa: E731
        elif fmt != "bfloat8_b":
            raise ValueError(f"resident expert format {fmt!r}: one of {', '.join(RESIDENT_FORMATS)}")
        self.dtype = ttnn.bfloat8_b if fmt == "bfloat8_b" else ttnn.bfloat4_b
        self.gu = _weight(mesh, f"blk.{L}.ffn_gate_up_exps.n{n}-{tag}", gate_up, self.dtype)
        self.dn = _weight(mesh, f"blk.{L}.ffn_down_exps.n{n}-{tag}", down, self.dtype)

    def nbytes(self) -> int:
        """Device DRAM bytes of the resident routed experts."""
        return len(self.resident) * 3 * KT * JT * _tile_bytes(self.dtype)

    @staticmethod
    def entries(T: int, loc: np.ndarray, wt: np.ndarray):
        """Metadata entries for T rows: per 32-row group its resident experts in ascending local id (a group without
        one gets local expert 0 at route weight 0: exact zeros, so stage B writes every group).  loc [T,10] local ids
        (-1 = CPU slot), wt [T,10] route weights.  Returns (ent uint32 [S] = group << 16 | local id, route f32 [S, 32],
        group starts uint32 [G + 1])."""
        G = (T + 31) // 32
        tt, kk = np.nonzero(loc >= 0)
        empty = np.setdiff1d(np.arange(G), tt // 32).astype(np.int64)
        key = np.concatenate([(tt // 32).astype(np.int64) << 16 | loc[tt, kk], empty << 16])
        uk, inv = np.unique(key, return_inverse=True)            # (group, expert) pairs, group-major
        ent = uk.astype(np.uint32)
        route = np.zeros((len(uk), 32), np.float32)
        route[inv[:len(tt)], tt % 32] = wt[tt, kk]
        starts = np.searchsorted(uk >> 16, np.arange(G + 1)).astype(np.uint32)
        return ent, route, starts

    def _stage(self, name, a, meta, w, out, s_cap):
        key = (name, s_cap, _mem_key(a), _mem_key(out), w.dtype, self.fidelity)
        st = _STAGES.get(key)
        if st is None:
            st = _STAGES[key] = _Stage(self.mesh, name, a, meta, w, out, s_cap, self.fidelity, key)
        return st(a, meta, w, out)

    @staticmethod
    def prepare(x, rows: int):
        """x [1,1,rows,2560] f32 TILE -> the BF16 copy the kernels read (L1 for one 32-row group, else DRAM).
        Moe dispatches it before its blocking read-back, while the CPU is idle."""
        return ttnn.typecast(x, ttnn.bfloat16,
                             memory_config=ttnn.L1_MEMORY_CONFIG if rows <= 32 else ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def s_cap_for(S: int) -> int:
        """Metadata capacity (entries) for S entries: the next power of two >= max(32, S); one compiled program each."""
        return max(32, 1 << (S - 1).bit_length())

    @staticmethod
    def meta_words(T: int, ent, route, starts, g0: int = 0, g1: int | None = None, s_cap: int | None = None):
        """The uint32 metadata row of the round over groups [g0, g1) (default all; one round when T <= 32 * GMAX):
        [0] S, [1] groups, [2] g0, [3] T, [4 + g] round-relative group starts, [HDR + s] entries, then from
        HDR + s_cap the 32 route weights of each entry (f32 bits).  s_cap >= S (default s_cap_for(S)).
        Returns (words, S, s_cap)."""
        g1 = len(starts) - 1 if g1 is None else g1
        if g1 - g0 > GMAX:
            raise ValueError(f"{g1 - g0} groups in one round (at most {GMAX})")
        s0, s1 = int(starts[g0]), int(starts[g1])
        S = s1 - s0
        s_cap = DeviceExperts.s_cap_for(S) if s_cap is None else int(s_cap)
        if S > s_cap:
            raise ValueError(f"{S} entries exceed s_cap {s_cap}")
        words = np.zeros(HDR + s_cap + 32 * s_cap, np.uint32)
        words[0], words[1], words[2], words[3] = S, g1 - g0, g0, T
        words[4:4 + g1 - g0 + 1] = starts[g0:g1 + 1] - s0
        words[HDR:HDR + S] = ent[s0:s1]
        words[HDR + s_cap:HDR + s_cap + 32 * S] = route[s0:s1].reshape(-1).view(np.uint32)
        return words, S, s_cap

    def run_fixed(self, xb, meta, s_cap: int, rows: int, y=None, s_alloc: int | None = None):
        """Device only (capture-safe): both stages for one metadata round already on the device.  meta: int32
        ROW_MAJOR [1,1,1,HDR+33*s_cap] (meta_words); the kernels read S, groups, first group and T from it at run
        time, so a larger s_cap / s_alloc than needed gives bitwise the same y.  s_alloc: h capacity in entries
        (default s_cap).  y: f32 TILE [1,1,rows,2560] to write (allocated when None).  Frees neither xb nor meta.
        Returns y."""
        if y is None:
            y = ttnn.empty([1, 1, rows, N_EMBD], ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        S = s_cap if s_alloc is None else int(s_alloc)
        h = ttnn.empty([1, 1, 32 * S, N_FF], ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                       ttnn.L1_MEMORY_CONFIG if rows <= 32 else ttnn.DRAM_MEMORY_CONFIG)
        self._stage("A", xb, meta, self.gu, h, s_cap)
        self._stage("B", h, meta, self.dn, y, s_cap)
        ttnn.deallocate(h)
        return y

    def run(self, xb, rows: int, T: int, loc: np.ndarray, wt: np.ndarray):
        """Eager: xb = prepare(x) -> y [1,1,rows,2560] f32 TILE = the device slots' weighted routed experts (rows >= T:
        padding), in rounds of <= GMAX groups (one round unless T > 1888).  Frees xb.  Returns (y, entries)."""
        ent, route, starts = self.entries(T, loc, wt)
        G = len(starts) - 1
        y = ttnn.empty([1, 1, rows, N_EMBD], ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        for g0 in range(0, G, GMAX):
            words, S, s_cap = self.meta_words(T, ent, route, starts, g0, min(G, g0 + GMAX))
            meta = ttnn.from_torch(torch.from_numpy(words.view(np.int32)).reshape(1, 1, 1, -1), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.run_fixed(xb, meta, s_cap, rows, y, s_alloc=S)
            ttnn.deallocate(meta)
        ttnn.deallocate(xb)
        return y, len(ent)


@dataclass
class Plan:
    """Host side of one Moe call (Moe.plan): routing, the device/CPU split and the running CPU-expert call."""
    T: int
    x: np.ndarray            # [T, 2560] f32 read-back MoE input
    logits: np.ndarray       # [T, N_EXPERT] device router logits
    gate_logit: np.ndarray   # [T] shared-expert gate logit
    ids: np.ndarray          # [T, 10] routed experts (rank order)
    wt: np.ndarray           # [T, 10] route weights
    loc: np.ndarray          # [T, 10] device-local expert index, -1 = CPU slot
    refined: int
    parts: np.ndarray | None = None   # [T, 10, 2560] being filled by CpuExperts.submit (None: no CPU slot)
    dev: object = None       # the layer's DeviceExperts (None: none resident)
    words: np.ndarray | None = None   # with plan(s_cap=...): the device-expert metadata row (DeviceExperts.meta_words)
    cpu_ids: np.ndarray | None = None  # [T, 10] the CPU slots' expert ids (negative = not a CPU slot); None: no CPU slot

    @property
    def on_dev(self) -> np.ndarray:
        return self.loc >= 0


class Moe:
    def __init__(self, mesh, w, L: int, experts, shared_dtype=ttnn.bfloat8_b, resident=None,
                 resident_fmt=None, fidelity=ttnn.MathFidelity.HiFi4):
        """resident: expert ids of layer L to keep on the device (default: default_resident(L), i.e. the placement
        file FNX_EXPERT_PLACEMENT, else the variant's default); resident_fmt: theirs (default resident_format())."""
        self.mesh, self.L, self.experts = mesh, L, experts
        self.cfg = _hifi4()
        Wr = np.ascontiguousarray(w.layer(L, "ffn_gate_inp.weight"), np.float32)        # (N_EXPERT, 2560) BF16 values
        wsg = w.layer(L, "ffn_gate_inp_shexp.weight").reshape(-1)
        assert Wr.shape == (N_EXPERT, N_EMBD) and wsg.shape == (N_EMBD,)
        self.wr_host = Wr                                                                  # near-tie refinement
        self.wr_norm = np.linalg.norm(Wr.astype(np.float64), axis=1)
        R = np.zeros((N_EMBD, ROUTER_COLS), np.float32)
        R[:, :N_EXPERT] = Wr.T
        R[:, N_EXPERT] = wsg
        self.wr = _weight(mesh, f"blk.{L}.ffn_gate_inp+shexp.router", R, ttnn.bfloat16)   # exact (BF16 GGUF)
        sd = shared_dtype
        self.wg = _weight(mesh, f"blk.{L}.ffn_gate_shexp.weight", w.layer(L, "ffn_gate_shexp.weight").T, sd)
        self.wu = _weight(mesh, f"blk.{L}.ffn_up_shexp.weight", w.layer(L, "ffn_up_shexp.weight").T, sd)
        self.wd = _weight(mesh, f"blk.{L}.ffn_down_shexp.weight", w.layer(L, "ffn_down_shexp.weight").T, sd)
        res = default_resident(L) if resident is None else np.asarray(resident, np.int32)
        fmt = resident_format() if resident_fmt is None else resident_fmt
        self.dev = DeviceExperts(mesh, w, L, res, fmt=fmt, fidelity=fidelity) if len(res) else None
        self.local = self.dev.local if self.dev else np.full(N_EXPERT, -1, np.int32)
        self.last: dict = {}

    def reset(self) -> None:
        """The MoE block keeps no per-sequence state."""

    # ---- the call in parts: pre / shared / experts (device only, capture-safe), plan / finish (host), post (device)
    def pre(self, mixed, T: int | None = None) -> dict:
        """Device: f32 x, router logits, the row-major [x | logits] rows to read back (`both`, exactly T rows) and the
        BF16 copy of x for the device experts (`xb`, None without resident experts).  Does not free mixed."""
        rows = int(mixed.shape[-2])
        T = rows if T is None else int(T)
        x = mixed if mixed.dtype == ttnn.float32 else ttnn.typecast(mixed, ttnn.float32)
        logits = ttnn.linear(x, self.wr, dtype=ttnn.float32, compute_kernel_config=self.cfg)   # [1,1,rows,288]
        both = ttnn.untilize(ttnn.concat([x, logits], dim=-1))
        if T < rows:
            both = ttnn.slice(both, [0, 0, 0, 0], [1, 1, T, N_EMBD + ROUTER_COLS])
        xb = self.dev.prepare(x, rows) if self.dev else None
        return dict(x=x, logits=logits, both=both, xb=xb, rows=rows, T=T)

    def plan(self, host: np.ndarray, s_cap: int | None = None, submit: bool = True) -> Plan:
        """Host: read-back rows [T, 2560 + N_EXPERT + 32] -> routing (fnx.cpu_experts.route, libfnx_cpu's
        fnx_route.cpp: top-10 of the device logits with exact refinement of near ties, module docstring; f64 softmax
        weights over the 10 picks), the device/CPU split, and the CPU experts submitted to libfnx_cpu's leader thread
        (submit=False: not yet; start() does, for a call that must wait for another call's CPU experts).
        s_cap (T <= 32, resident experts): also the device experts' metadata row padded to s_cap entries (words)."""
        T = host.shape[0]
        xh = np.ascontiguousarray(host[:, :N_EMBD])
        lg = np.ascontiguousarray(host[:, N_EMBD:N_EMBD + N_EXPERT])
        ids, wt, cpu_ids, n_refined, words, _ = route(lg, xh, self.wr_host, self.wr_norm, self.local, PRODUCT_ERR,
                                                       s_cap if self.dev is not None else None)
        cpu = cpu_ids if (cpu_ids >= 0).any() else None
        parts = self.experts.submit(self.L, xh, cpu) if submit and cpu is not None else None
        return Plan(T=T, x=xh, logits=lg, gate_logit=host[:, N_EMBD + N_EXPERT].copy(), ids=ids, wt=wt,
                    loc=self.local[ids], refined=n_refined, parts=parts, dev=self.dev, words=words, cpu_ids=cpu)

    def shared(self, pre: dict):
        """Device: shared expert sigmoid(gate) * W_d (silu(W_g x) * W_u x), f32 TILE [1,1,rows,2560]."""
        x, rows = pre["x"], pre["rows"]
        g = ttnn.linear(x, self.wg, dtype=ttnn.float32, activation="silu", compute_kernel_config=self.cfg)
        u = ttnn.linear(x, self.wu, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        sh = ttnn.linear(ttnn.multiply(g, u), self.wd, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        gs = ttnn.sigmoid(ttnn.slice(pre["logits"], [0, 0, 0, N_EXPERT], [1, 1, rows, N_EXPERT + 1]))
        return ttnn.multiply(sh, gs)

    def finish(self, p: Plan) -> tuple[np.ndarray, float]:
        """Host: wait for the CPU experts -> (their weighted sum f32 [T, 2560] (zeros without CPU slots), their us on
        the leader thread)."""
        if p.parts is None:
            return np.zeros((p.T, N_EMBD), np.float32), 0.0
        cpu_us = self.experts.wait()
        return self.combine(p), cpu_us

    @staticmethod
    def combine(p: Plan) -> np.ndarray:
        """Host: the weighted sum f32 [T, 2560] of p's finished CPU expert parts."""
        wt = p.wt if p.dev is None else np.where(p.on_dev, 0, p.wt).astype(np.float32)
        return np.einsum("tkd,tk->td", p.parts, wt)

    @staticmethod
    def post(y, r):
        """Device: bo = y + r; r f32 [1,1,rows,2560], TILE or ROW_MAJOR (tilized here).  Frees neither."""
        if r.layout != ttnn.TILE_LAYOUT:
            r = ttnn.to_layout(r, ttnn.TILE_LAYOUT)
        return ttnn.add(y, r)

    # ---- the call in two phases, for pipelined prompt reads (TtModel.forward_prompt): begin() routes a row block
    # while the CPU experts of earlier blocks still run; start() runs the CPU experts of several begun blocks as ONE
    # library call (more tokens per expert: the weights are decoded once per group of up to 8 tokens) after the
    # earlier blocks' complete().  The device arithmetic is forward()'s, op for op; a token's CPU expert rows may
    # round differently than in a one-block call when its expert's token grouping changes (Strata #152).
    def begin(self, mixed) -> dict:
        """Device router + the read-back + the host routing of mixed [1,1,rows,2560]; starts nothing on the CPU.
        mixed must stay allocated until complete() (the f32 MoE input may be mixed itself)."""
        pr = self.pre(mixed)
        host = pr["both"].cpu(blocking=True).to_numpy().reshape(pr["T"], N_EMBD + ROUTER_COLS)
        return dict(pr=pr, p=self.plan(host, submit=False))

    def start(self, cs: list[dict]) -> None:
        """Start the CPU experts of the begun blocks cs (one library call over their rows; the library runs one call
        at a time: complete() any earlier blocks first), then dispatch each block's shared and resident experts."""
        cpu = [c["p"] for c in cs if c["p"].cpu_ids is not None]
        if cpu:
            parts = self.experts.submit(self.L, np.concatenate([p.x for p in cpu]),
                                        np.concatenate([p.cpu_ids for p in cpu]))
            a = 0
            for p in cpu:
                p.parts, a = parts[a:a + p.T], a + p.T
        for c in cs:
            pr, p = c["pr"], c["p"]
            y = self.shared(pr)
            if pr["xb"] is not None:
                if p.on_dev.any():
                    yd, _ = self.dev.run(pr["xb"], pr["rows"], pr["T"], p.loc, p.wt)
                    y = ttnn.add(y, yd)
                    ttnn.deallocate(yd)
                else:
                    ttnn.deallocate(pr["xb"])
            c["y"] = y

    def complete(self, cs: list[dict]) -> list:
        """Wait for start()'s CPU experts -> each block's bo [1,1,rows,2560] float32 TILE, as forward() returns it."""
        if any(c["p"].parts is not None for c in cs):
            self.experts.wait()
        out = []
        for c in cs:
            pr, p, y = c["pr"], c["p"], c["y"]
            if p.parts is None:
                out.append(y)
                continue
            routed = self.combine(p)
            rows, T = pr["rows"], pr["T"]
            padded = routed if T == rows else np.concatenate([routed, np.zeros((rows - T, N_EMBD), np.float32)])
            r = ttnn.from_torch(torch.from_numpy(padded).reshape(1, 1, rows, N_EMBD), dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(self.post(y, r))
            ttnn.deallocate(y)
            ttnn.deallocate(r)
        return out

    def forward(self, mixed, T: int | None = None):
        """mixed [1,1,rows,2560] TILE (float32 or bfloat16), tokens in rows [0, T) (default T = rows)
        -> bo [1,1,rows,2560] float32 TILE (rows >= T: padding).  Eager composition of the parts above."""
        t0 = time.perf_counter()
        pr = self.pre(mixed, T)
        rows, T = pr["rows"], pr["T"]
        host = pr["both"].cpu(blocking=True).to_numpy().reshape(T, N_EMBD + ROUTER_COLS)
        t1 = time.perf_counter()
        p = self.plan(host)
        t2 = time.perf_counter()
        y = self.shared(pr)                    # on the device while the CPU experts run
        n_entries = 0
        if pr["xb"] is not None:
            if p.on_dev.any():
                yd, n_entries = self.dev.run(pr["xb"], rows, T, p.loc, p.wt)
                y = ttnn.add(y, yd)
                ttnn.deallocate(yd)
            else:
                ttnn.deallocate(pr["xb"])
        t3 = time.perf_counter()
        routed, cpu_us = self.finish(p)
        t4 = time.perf_counter()
        if p.parts is not None:
            padded = routed if T == rows else np.concatenate([routed, np.zeros((rows - T, N_EMBD), np.float32)])
            r = ttnn.from_torch(torch.from_numpy(padded).reshape(1, 1, rows, N_EMBD), dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            bo = self.post(y, r)
            ttnn.deallocate(y)
            ttnn.deallocate(r)
        else:
            bo = y
        t5 = time.perf_counter()
        self.last = dict(ids=p.ids, wt=p.wt, logits=p.logits, x=p.x, gate_logit=p.gate_logit, refined=p.refined,
                         routed=routed, on_dev=p.on_dev, n_entries=n_entries,
                         us=dict(readback=(t1 - t0) * 1e6, route_submit=(t2 - t1) * 1e6, dev_dispatch=(t3 - t2) * 1e6,
                                 cpu=cpu_us, cpu_wait=(t4 - t3) * 1e6, combine_upload=(t5 - t4) * 1e6,
                                 total=(t5 - t0) * 1e6))
        return bo


MOCK_CLUSTER = ("/home/container_app_user/tt-metal/tt_metal/third_party/tt-cluster-descriptors/blackhole/"
                "p100_cluster_desc/p100_cluster_desc.yaml")      # in the image: block-float packing without a card


def main() -> int:
    """Host-only conversion of the device-expert weight caches for a placement (no device; ttnn's block-float
    packing needs a cluster descriptor, a mock one does, MOCK_CLUSTER):
        TT_METAL_MOCK_CLUSTER_DESC_PATH=<MOCK_CLUSTER> NODEV=1 tools/devrun.sh prep -m fnx.tt.moe \\
            --prepare PLACEMENT.json|default [--layers 0-47] [--gguf SHARD1]
    default: the variant's default placement (FNX_EXPERT_PLACEMENT unset: DEFAULT_RESIDENT), what the engine loads."""
    import argparse
    from ..weights import Weights
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", required=True, help="placement JSON (tools/expert_placement.py), or 'default'")
    ap.add_argument("--layers", default=f"0-{N_LAYERS - 1}")
    ap.add_argument("--gguf", default=None, help="the model's shard 1 (default fnx.paths.SHARD1)")
    a = ap.parse_args()
    lo, _, hi = a.layers.partition("-")
    if a.prepare == "default":
        os.environ.pop("FNX_EXPERT_PLACEMENT", None)
    else:
        os.environ["FNX_EXPERT_PLACEMENT"] = a.prepare
    placement, fmt = [default_resident(L) for L in range(N_LAYERS)], resident_format()
    w = Weights(a.gguf)
    for L in range(int(lo), int(hi or lo) + 1):
        t0 = time.time()
        if len(placement[L]):
            DeviceExperts(None, w, L, placement[L], fmt=fmt)
        print(f"layer {L}: {len(placement[L])} resident experts cached in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
