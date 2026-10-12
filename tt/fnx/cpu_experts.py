"""ctypes binding of libfnx_cpu.so: Flash-Next routed experts on the host CPU with Strata's kernels, the cpu MoE's
native routing step (`route`, cpu_experts/fnx_route.cpp) and the QSA selection's per-row assembly
(`assemble_selection`, cpu_experts/fnx_qsa.cpp)."""
from __future__ import annotations

import ctypes as C
import os

import numpy as np

from . import paths

_LIB = None


def lib(path: str | None = None):
    global _LIB
    if _LIB is None:
        # #152: multi-token kernels for every group, so a token's expert rows do not depend on how many tokens share
        # the expert in a call (the kernels read it once, at their first call)
        os.environ.setdefault("STRATA_IQ_MT_MIN", "1")
        # built in the TT image (glibc 2.35) by tt/cpu_experts/build.sh, so it loads there and on a newer host alike
        p = str(path or paths.CPU_LIB)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}: Strata's CPU expert kernels are not built (tt/cpu_experts/build.sh)")
        L = C.CDLL(p)
        L.fnx_init.argtypes = [C.c_int, C.c_int, C.c_char_p, C.c_int]
        L.fnx_set_layer.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p,
                                    C.c_char_p, C.c_int]
        L.fnx_moe.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_char_p, C.c_int]
        L.fnx_moe_submit.argtypes = L.fnx_moe.argtypes
        L.fnx_moe_wait.argtypes = [C.POINTER(C.c_double), C.c_char_p, C.c_int]
        L.fnx_route.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_double,
                                C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int,
                                C.POINTER(C.c_int)]
        L.fnx_qsa_assemble.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                       C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
        for f in (L.fnx_init, L.fnx_set_layer, L.fnx_moe, L.fnx_moe_submit, L.fnx_moe_wait, L.fnx_route,
                  L.fnx_qsa_assemble):
            f.restype = C.c_int
        _LIB = L
    return _LIB


def _check(rc: int, err) -> None:
    if rc != 0:
        raise RuntimeError(err.value.decode(errors="replace"))


META_HDR = 64          # the device-expert metadata header words (fnx.tt.moe.HDR; fnx_route.cpp HDR)


def route(lg: np.ndarray, x: np.ndarray, wr: np.ndarray, wr_norm: np.ndarray, local: np.ndarray, product_err: float,
          s_cap: int | None = None):
    """The cpu MoE's routing of T rows (fnx_route.cpp: top-10 with exact near-tie refinement, softmax weights).
    lg f32 [T, E] device router logits, x f32 [T, 2560] the rows, wr f32 [E, 2560] router weights, wr_norm f64 [E],
    local int32 [E] (device index of a resident expert, -1: CPU).  s_cap (T <= 32 only): also the resident slots'
    metadata row (fnx.tt.moe.DeviceExperts.meta_words layout).  Returns (ids int32 [T, 10], wt f32 [T, 10], cpu_ids
    int32 [T, 10] (-1 for resident slots), refined rows, words uint32 [HDR + 33 s_cap] or None, its entries)."""
    T, E = lg.shape
    ids = np.empty((T, 10), np.int32)
    wt = np.empty((T, 10), np.float32)
    cpu_ids = np.empty((T, 10), np.int32)
    words = None if s_cap is None else np.empty(META_HDR + 33 * s_cap, np.uint32)
    n_ent = C.c_int(0)
    r = lib().fnx_route(T, E, x.shape[1], lg.ctypes.data, x.ctypes.data, wr.ctypes.data, wr_norm.ctypes.data,
                        product_err, local.ctypes.data, ids.ctypes.data, wt.ctypes.data, cpu_ids.ctypes.data,
                        None if words is None else words.ctypes.data, s_cap or 0, C.byref(n_ent))
    if r < 0:
        raise ValueError(f"fnx_route: {'entries exceed s_cap ' + str(s_cap) if r == -1 else 'metadata needs T <= 32'}")
    return ids, wt, cpu_ids, r, words, n_ent.value


QSA_TOPK, QSA_WIDTH = 512, 2051     # fnx.tt.qsa.TOPK_BLOCKS, fnx.ref.qsa.SEL_WIDTH


def assemble_selection(sure: np.ndarray, cand: np.ndarray, ex: np.ndarray, e, nbt, n) -> list[np.ndarray]:
    """fnx.tt.qsa's sparse selection per row (fnx_qsa.cpp): sure / cand bool [2, R, nbm] (top K, top K + 1), ex f32
    [R, nbm] the exact scores of the candidates, e / nbt / n [R] -> each row's ascending cells, int32 [SEL_WIDTH]."""
    R, nbm = ex.shape
    s = np.ascontiguousarray(sure, np.uint8)
    c = np.ascontiguousarray(cand, np.uint8)
    x = np.ascontiguousarray(ex, np.float32)
    e, nbt, n = (np.ascontiguousarray(v, np.int32) for v in (e, nbt, n))
    cells = np.empty((R, QSA_WIDTH), np.int32)
    r = lib().fnx_qsa_assemble(R, nbm, QSA_TOPK, s.ctypes.data, c.ctypes.data, x.ctypes.data, e.ctypes.data,
                               nbt.ctypes.data, n.ctypes.data, QSA_WIDTH, cells.ctypes.data)
    if r:
        raise AssertionError(f"selection bound violated (row {-r - 1})")
    return list(cells)


TYPE_IDS = {"IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22,
            "IQ4_XS": 23, "IQ1_M": 29, "Q2_0": 42, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "Q8_0": 8}


class CpuExperts:
    """Every layer's routed experts, read in place from a gguf_load.Model's mappings.

    `capacity` (default n_layers) reserves extra layer slots for experts that are not in the GGUF, e.g. the MTP
    layer's: register them with `add_layer`.  One instance per process (the library's layer table is global).
    The library keeps n_threads - 1 workers and a leader thread pinned one per physical core, spinning between
    calls (cpu_experts/fnx_cpu.cpp); `moe` runs a call on the calling thread + the workers, `submit` / `wait` on the
    leader while the caller does other work (ctypes releases the GIL in both)."""

    def __init__(self, model, n_layers: int, n_threads: int, capacity: int | None = None):
        self.model = model
        self.L = lib()
        self._keep: dict[int, tuple] = {}
        self._err = C.create_string_buffer(512)
        self._us = C.c_double()
        self._pending: tuple | None = None
        _check(self.L.fnx_init(max(n_layers, capacity or 0), n_threads, self._err, 512), self._err)
        for l in range(n_layers):
            names = [f"blk.{l}.ffn_{r}_exps.weight" for r in ("gate", "up", "down")]
            ptrs = [model.raw(n).ctypes.data for n in names]
            gt, ut, dt = (model.type_of(n) for n in names)
            if gt != ut:
                raise ValueError(f"layer {l}: gate {gt} and up {ut} differ")
            gi = model.info(names[0])
            _check(self.L.fnx_set_layer(l, TYPE_IDS[gt], TYPE_IDS[dt], gi.shape[0], gi.shape[1], *ptrs, self._err, 512),
                   self._err)

    def add_layer(self, layer: int, gate: np.ndarray, up: np.ndarray, down: np.ndarray, gu_type: str, d_type: str,
                  n_embd: int = 2560, n_ff: int = 640) -> None:
        """Register expert stacks held in memory as layer slot `layer` (< capacity).

        gate/up/down: contiguous uint8 arrays in GGUF expert-stack layout (expert e's rows at e * rows * row_bytes,
        i.e. ne = [n_in, n_out, n_expert]); types are ggml names ("Q2_0", "IQ4_NL", ...).  The arrays are kept alive
        here for as long as this instance exists."""
        arrs = [np.ascontiguousarray(a, dtype=np.uint8) for a in (gate, up, down)]
        self._keep[layer] = tuple(arrs)
        _check(self.L.fnx_set_layer(layer, TYPE_IDS[gu_type], TYPE_IDS[d_type], n_embd, n_ff,
                                    *(a.ctypes.data for a in arrs), self._err, 512), self._err)

    @staticmethod
    def _args(x: np.ndarray, ids: np.ndarray, out: np.ndarray | None):
        x = np.ascontiguousarray(x, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        nt, k = ids.shape
        if out is None:
            out = np.empty((nt, k, x.shape[1]), dtype=np.float32)
        elif out.shape != (nt, k, x.shape[1]) or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError(f"out must be C-contiguous float32 {(nt, k, x.shape[1])}")
        return x, ids, out, nt, k

    def moe(self, layer: int, x: np.ndarray, ids: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """x (nt, 2560) f32, ids (nt, k) int32 (negative = skip) -> (nt, k, 2560) unweighted expert outputs (skipped
        slots zero)."""
        x, ids, out, nt, k = self._args(x, ids, out)
        _check(self.L.fnx_moe(layer, nt, k, x.ctypes.data, ids.ctypes.data, out.ctypes.data, self._err, 512), self._err)
        return out

    def submit(self, layer: int, x: np.ndarray, ids: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Start `moe` on the library's leader thread and return its `out` at once: filled when `wait` returns
        (x, ids and out are held until then)."""
        x, ids, out, nt, k = self._args(x, ids, out)
        _check(self.L.fnx_moe_submit(layer, nt, k, x.ctypes.data, ids.ctypes.data, out.ctypes.data, self._err, 512),
               self._err)
        self._pending = (x, ids, out)
        return out

    def wait(self) -> float:
        """Wait for the last `submit` (spinning); returns its time on the leader thread in microseconds."""
        rc = self.L.fnx_moe_wait(C.byref(self._us), self._err, 512)
        self._pending = None
        _check(rc, self._err)
        return self._us.value
