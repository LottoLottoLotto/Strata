"""Flash-Next on the P150 (TTNN; eager, or traced decode steps): host embedding + PLE row gathers, 48 HC-wrapped
GDN/QSA + MoE layers on device, output_hc read, LM head, greedy.

MoE (FNX_MOE, default DEFAULT_MOE): "device" = fnx.tt.moe_device.MoeDevice, the whole MoE block on the device (router,
top-10, routed experts re-quantized to block floats from the GPTQ plan FNX_MOE_PLAN of fnx.quant.store, default
DEFAULT_PLAN: its resident experts in device DRAM, the others in pinned host memory read over PCIe by the kernels;
no host round trip, so a dense decode step is one captured trace); "cpu" = fnx.tt.moe.Moe, the routed experts on the
host CPU through fnx.cpu_experts (Strata's own kernels and GGUF weights: Strata-exact experts).

`TtModel(mesh, w, experts).forward(tokens, positions)` consumes T consecutive tokens of one sequence (T rows travel
as tensors of LOGICAL shape [1, 1, T, ...], TILE-padded to 32; modules read T from shape[-2]) and returns
(logits f32 [T, 248320] on the host, greedy ids [T]).  `forward(..., dump=d)` reads the residual back after every
layer into d["R"] [n_layers+1, T, 4, 2560] (same ladder as fnx.ref.model.RefModel) plus d["R_ple"], d["head_in"].
`forward(..., head=False)` only advances the state (prompt reading: no output_hc / LM head, returns (None, None));
`forward(..., read_logits=False)` returns (None, ids): the device argmax without reading the logits back.
`forward(..., keep_R=True)` also returns the final residual (before output_hc) for the MTP drafter (fnx.tt.mtp).

Speculative decoding (fnx.serve_engine): `forward(window, verify=True)` runs a verify window of T rows at p..p+T-1,
then `commit(n_keep)` keeps its first n_keep rows: for T > 1 every GDN layer's kernel also saves its state after
row 0 and the raw rows 1..T-1 it read (GdnFused.forward(snap=True)), and a partial commit is one GdnFused.rollback per
layer (that snapshot + the kept rows replayed, no projections: bitwise what those rows alone leave); Ple.commit rolls
the PLE history and hash window back; QSA KV / indexer state is positional (rejected cells are rewritten by the next
window before any query reads them); pos = p + n_keep.  A row of a T-row window is bitwise the row of T=1 calls
(fnx/tests/test_spec.py invariance), so the committed text does not depend on the drafts.

`snapshot()` / `restore(snap)`: the per-sequence state that is NOT positional (GDN recurrent + conv state, the PLE
conv history and n-gram window, the position) copied to the host and back in place, for conversation checkpoints
(fnx.serve_engine).  The QSA KV caches are positional (cell = position, masked beyond the attending row), so a
snapshot of the first L tokens stays valid as long as cells [0, L) are not rewritten with other tokens.

`enable_trace(Ts=(1, 4))` captures the decode forwards of T in Ts (fnx.tt.trace: device segments between the host
round trips, replayed with execute_trace; bitwise the eager results); forward() then replays them for head=True calls
of T in Ts at the next positions and runs everything else eagerly.  A traced keep_R residual belongs to the trace:
release it with free_R() (which frees an eager one).

Residual R: float32 TILE [1, 1, T, 10240], column = stream * 2560 + d; every token starts with its embedding row
in all 4 streams (spec §A.2, verify.cpp:686-688).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import ttnn

from .. import paths
from ..config import HC, HC_DIM, N_EMBD, N_LAYERS, PLE_LAYER, is_qsa
from ..ref.embed import embed
from .gdn import Gdn
from .hc import HcRead, hc_write
from .head import Head
from .moe import Moe
from .moe_device import F_BFP4, MoeDevice, host_matrices, matrix_bytes, quant_layer
from .ple import Ple
from .qsa import Qsa

DEFAULT_MOE = "device"                 # FNX_MOE: "device" | "cpu" (= fnx.serve_engine.DEFAULT_MOE; quality gate 2026-10-07)
# FNX_MOE_PLAN default per variant (fnx.paths.MODEL): the fnx.quant.store plan of the device experts.
# coder: GPTQ-BFP4; 24.0e9 B, 8680 experts resident = 85.9% of routings.  Serving at 32K + MTP drafter + traces leaves
#   1.52 GB DRAM free (tools/dram_budget.py, 480 MiB trace region).  Same numerics as 2.2e+10 (placement only); T=1
#   41.2 -> 38.8 ms, MTP +6..9% (2026-10-07).
# iq3s: data-free BFP4 (fnx.quant.datafree: no calibration GPU), resident = the hottest of Strata's routing profile.
DEFAULT_PLANS = {"coder": "bfp4-tiered-2.4e+10", "iq3s": "search4-tiered-2.4e+10"}
DEFAULT_PLAN = DEFAULT_PLANS[paths.MODEL]


def moe_mode(v: str | None = None) -> str:
    """v, else FNX_MOE, else DEFAULT_MOE (checked)."""
    v = v or os.environ.get("FNX_MOE", DEFAULT_MOE)
    if v not in ("device", "cpu"):
        raise ValueError(f"FNX_MOE={v!r}: 'device' or 'cpu'")
    return v


# Trace region: reserved device DRAM outside the allocator, so every MB here is a MB less for resident experts
# (~1.1 ms/token per GB).  The serving traces (decode T=1 and T=4 + the MTP draft rounds) measured 0.200 GB at 32K
# context (tools/dram_budget.py, 2026-10-07); 480 MiB keeps > 2x headroom.  FNX_TRACE_MB overrides it.
TRACE_REGION_MB = int(os.environ.get("FNX_TRACE_MB", "480"))


def open_mesh():
    """The P150 as a 1x1 mesh, opened like the qwen36 runtime, with the trace region sized above."""
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, num_command_queues=1,
                                 trace_region_size=TRACE_REGION_MB << 20)


def to_host(t, rows: int) -> np.ndarray:
    """First `rows` rows of a [1, 1, T, n] device tensor as float32 numpy [rows, n]."""
    x = ttnn.to_torch(t)
    return x.reshape(-1, x.shape[-1])[:rows].float().numpy()


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _replace(old, new):
    """Free `old` once a module has returned its successor `new` (unless the module updated in place)."""
    if new is not old:
        ttnn.deallocate(old)
    return new


class TtModel:
    def __init__(self, mesh, w, experts, n_layers: int = N_LAYERS, max_ctx: int = 4096, verbose: bool = True,
                 moe: str | None = None, host_reserve: int = 0):
        """moe: "device" | "cpu" (default: FNX_MOE, else DEFAULT_MOE; module docstring).  experts (CpuExperts): the
        cpu MoE's routed experts (unused by the device MoE).  The device MoE's host tier is ONE pinned HostPool
        (`host_pool`, chunk mode FNX_HOST_CHUNK, default "auto": huge pages while the kernel gives them, then pinned
        4 KiB pages), allocated before anything else of the model (huge pages are easiest to get then) and kept for
        the model's life (every trace may read it).  host_reserve: extra
        items of that pool (each >= one BFP4 expert matrix, 921600 B) after the main host-tier items, for another
        reader of host memory (the MTP drafter, fnx.tt.mtp.host_items); the first is item `host_reserve_first` (None
        without a reserve)."""
        self.mesh, self.w, self.n_layers = mesh, w, n_layers
        self.moe = moe_mode(moe)
        t0 = time.time()
        lms, firsts, self.host_pool, self.host_reserve_first = None, [0] * n_layers, None, None
        n_host, item = 0, matrix_bytes(F_BFP4) if host_reserve else 0
        if self.moe == "device":
            from ..quant.store import Plan
            plan = Plan(os.environ.get("FNX_MOE_PLAN", DEFAULT_PLAN))
            lms = [quant_layer(w, L, plan) for L in range(n_layers)]
            hm = [host_matrices(lm) for lm in lms]
            firsts = list(np.cumsum([0] + [n for n, _ in hm])[:-1])
            n_host, item = sum(n for n, _ in hm), max([item] + [b for _, b in hm])
        if n_host + host_reserve:
            from .hostmem import HostPool, drop_page_cache
            # the expert store and the GGUF shards are streamed once anyway: their cached pages would only fragment
            # the free memory the pool's 2 MiB chunks need
            drop_page_cache(([plan.pool.root] if lms else []) + [f.path for f in w.model.files])
            self.host_pool = HostPool(n_host + host_reserve, item)
            self.host_reserve_first = n_host if host_reserve else None
            if verbose:
                print(f"TtModel: host pool of {n_host} expert matrices + {host_reserve} reserved items "
                      f"({self.host_pool.nbytes / 1e9:.2f} GB pinned, {self.host_pool.alloc_stats}) in "
                      f"{time.time() - t0:.0f} s", flush=True)
        self.ple = Ple(mesh, w)
        self.layers = []
        for L in range(n_layers):
            mix = Qsa(mesh, w, L, max_ctx) if is_qsa(L) else Gdn(mesh, w, L)
            moe_L = (MoeDevice(mesh, w, L, lms[L], pool=self.host_pool, pool_first=int(firsts[L])) if lms else
                     Moe(mesh, w, L, experts, shared_dtype=ttnn.bfloat16))
            self.layers.append((HcRead(mesh, w, f"blk.{L}.hc_attn"), mix, HcRead(mesh, w, f"blk.{L}.hc_ffn"), moe_L))
            if verbose:
                print(f"TtModel: layer {L} built ({time.time() - t0:.0f} s)", flush=True)
        self.hc_out = HcRead(mesh, w, "output_hc")
        self.head = Head(mesh, w)
        self.pos = 0
        self._win = None           # the uncommitted verify window: (p, T, GDN snapshot taken, Sequence | None)
        self._trace = None         # fnx.tt.trace.Tracer (enable_trace)
        if verbose:
            print(f"TtModel: {n_layers} layers + head on device in {time.time() - t0:.0f} s", flush=True)

    def reset(self) -> None:
        self._win = None
        if self._trace is not None:
            self._trace.sync()
        self.ple.reset()
        for _, mix, _, _ in self.layers:
            mix.reset()
        self.pos = 0

    def enable_trace(self, Ts=(1, 4), spans=None, verbose: bool = True, drafter=None) -> None:
        """Capture the traced forwards of T rows, T in Ts (fnx.tt.trace; spans: dense QSA span buckets, default
        FNX_TRACE_SPANS or fnx.tt.trace.DENSE_SPANS); drafter (fnx.tt.mtp.Mtp): also its draft() rounds after windows
        of up to max(Ts) rows (captured in the same pass).  Needs a committed state; the state is unchanged."""
        from .trace import Tracer
        self.disable_trace()
        self._trace = Tracer(self, tuple(Ts), spans, verbose, drafter)

    def disable_trace(self) -> None:
        if self._trace is not None:
            if self._win is not None and self._win[3] is not None:
                raise RuntimeError("TtModel.disable_trace: commit() the traced verify window first")
            self._trace.sync()
            self._trace.release()
            self._trace = None

    def free_R(self, R) -> None:
        """Release a forward(keep_R=True) residual (a traced forward's is the trace's own buffer: kept)."""
        if self._trace is None or not self._trace.free_R(R):
            ttnn.deallocate(R)

    def prepare_decode(self) -> None:
        """Between a prompt read and its decode (fnx.serve_engine): recapture the traces now if the prompt compiled
        new programs (fnx.tt.trace.Tracer.ensure), so the cost lands in the time to first token, not in decode."""
        if self._trace is not None and self._win is None:
            self._trace.ensure("after a prompt read")

    def embed(self, tokens: np.ndarray):
        """Host IQ4_XS row gather -> R [1, 1, T, 10240] f32 on device (embedding in every stream)."""
        e = embed(self.w, tokens)                                         # f32 [T, 2560]
        R = np.ascontiguousarray(np.tile(e, (1, HC)), dtype=np.float32)   # col = stream * 2560 + d
        return ttnn.from_torch(torch.from_numpy(R).reshape(1, 1, len(tokens), HC_DIM), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, tokens, positions=None, dump: dict | None = None, head: bool = True, read_logits: bool = True,
                keep_R: bool = False, verify: bool = False):
        """T tokens following the ones already consumed -> (logits f32 [T, 248320], greedy ids [T]) on the host
        (head=False: (None, None); read_logits=False: (None, ids)).  keep_R=True appends the final residual R
        (device f32 [1, 1, T, 10240], after the last layer's write, before output_hc; release it with free_R()).
        verify=True: a speculative verify window, to be kept in part by commit(n_keep) before the next call."""
        if self._win is not None:
            raise RuntimeError("TtModel.forward: commit() the last verify window first")
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        T = len(tokens)
        nxt = positions is None or np.array_equal(np.asarray(positions).reshape(-1), np.arange(self.pos, self.pos + T))
        pos = np.arange(self.pos, self.pos + T) if positions is None else np.asarray(positions, dtype=np.int64)
        tr = self._trace
        seq = tr.pick(T, pos) if tr is not None and nxt and head and dump is None else None
        if seq is not None:
            logits, ids = tr.forward(seq, tokens, pos, read_logits)
            self.pos = int(pos[-1]) + 1
            if verify:
                self._win = (int(pos[0]), T, seq.verify, seq)
            if keep_R:
                tr.lent = seq.E.R
                return logits, ids, seq.E.R
            return logits, ids
        if tr is not None:
            tr.sync()                                 # the host indexer rows a traced step deferred
        snap = verify and T > 1                   # a T=1 window has nothing to roll back
        if snap:
            self._window_buffers(T)
        e_ple = self.ple.upload_rows(tokens)          # host n-gram hash (Ple's own prev) + row gather -> device
        R = self.embed(tokens)
        lad = [to_host(R, T).reshape(T, HC, N_EMBD)] if dump is not None else None
        for L, (hc_attn, mix, hc_ffn, moe) in enumerate(self.layers):
            if L == PLE_LAYER:
                R = _replace(R, self.ple.forward(e_ple, R))
                _free(e_ple)
                if dump is not None:
                    dump["R_ple"] = to_host(R, T).reshape(T, HC, N_EMBD)
            mixed, inj = hc_attn.forward(R)
            bo = (mix.forward(mixed, pos) if is_qsa(L) else mix.forward(mixed, snap=True) if snap else
                  mix.forward(mixed))
            R = _replace(R, hc_write(R, bo, inj))
            _free(mixed, bo, inj)
            mixed, inj = hc_ffn.forward(R)
            bo = moe.forward(mixed)
            R = _replace(R, hc_write(R, bo, inj))
            _free(mixed, bo, inj)
            if lad is not None:
                lad.append(to_host(R, T).reshape(T, HC, N_EMBD))
        if dump is not None:
            dump["R"] = np.stack(lad)
        self.pos = int(pos[-1]) + 1
        if verify:
            self._win = (int(pos[0]), T, snap, None)
        if not head:
            if keep_R:
                return None, None, R
            _free(R)
            return None, None
        head_in, _ = self.hc_out.forward(R)
        if not keep_R:
            _free(R)
        logits_dev = self.head.forward(head_in)
        ids_dev = self.head.argmax(logits_dev)                  # device argmax, uint32 [1, 1, T, 1]
        logits = to_host(logits_dev, T) if read_logits else None
        ids = ttnn.to_torch(ids_dev).numpy().reshape(-1)[:T].astype(np.int64)
        _free(logits_dev, ids_dev)
        if dump is not None:
            dump["head_in"] = to_host(head_in, T)
        _free(head_in)
        return (logits, ids, R) if keep_R else (logits, ids)

    # ---- verify-window rollback
    def _gdn_layers(self) -> list:
        return [mix for L, (_, mix, _, _) in enumerate(self.layers) if not is_qsa(L)]

    def _window_buffers(self, T: int) -> None:
        """Every GDN layer's verify-window buffers for windows of up to T rows (GdnFused.window_buffers).  The
        captured forwards and commits record their addresses: when they grow, the traces are recaptured before the
        next replay (fnx.tt.trace)."""
        grown = [m.window_buffers(T) for m in self._gdn_layers()]
        if any(grown) and self._trace is not None:
            self._trace.n_prog = -1

    def _rollback(self, n_keep: int) -> None:
        """Every GDN layer's state <- what the first n_keep rows of the last verify window alone leave."""
        for m in self._gdn_layers():
            m.rollback(n_keep)

    def commit(self, n_keep: int) -> None:
        """Keep the first n_keep rows of the last verify window (forward(verify=True)) and roll the rest back:
        GDN state <- GdnFused.rollback (the state after row 0 + the kept rows replayed: bitwise the kept rows
        alone), PLE Ple.commit(n_keep), pos = p + n_keep.  QSA / MTP KV cells are positional: the next call
        rewrites them."""
        if self._win is None:
            raise RuntimeError("TtModel.commit: no verify window")
        p, T, snap, seq = self._win
        if not 1 <= n_keep <= T:
            raise ValueError(f"TtModel.commit({n_keep}) of a {T}-row window")
        self._win = None
        if snap and n_keep < T:
            if seq is None or self._trace.grown():    # eager window, or no replay after a post-capture compile
                self._rollback(n_keep)                # (fnx.tt.trace): the same programs, compiled before the capture
            else:
                ttnn.execute_trace(self.mesh, seq.commit_tids[n_keep], cq_id=0, blocking=False)
        if self.n_layers > PLE_LAYER:
            self.ple.commit(n_keep)
        self.pos = p + n_keep

    def _state_modules(self) -> list:
        """The modules with non-positional per-sequence state; each exposes state_tensors() (what reset() zeroes)."""
        return [self.ple] + self._gdn_layers()

    def snapshot(self) -> dict:
        """The non-positional per-sequence state on the host (see the module docstring); committed states only."""
        if self._win is not None:
            raise RuntimeError("TtModel.snapshot: commit() the verify window first")
        if self._trace is not None:
            self._trace.sync()
        return {"pos": self.pos, "ple_prev": self.ple.prev,
                "dev": [[ttnn.from_device(t) for t in m.state_tensors()] for m in self._state_modules()]}

    def restore(self, snap: dict) -> None:
        """Put a snapshot() back in place (same device addresses); the KV cells [0, snap pos) must be unchanged."""
        self._win = None
        if self._trace is not None:
            self._trace.sync()
        self.ple.drop_window()
        for m, saved in zip(self._state_modules(), snap["dev"], strict=True):
            for t, s in zip(m.state_tensors(), saved, strict=True):
                ttnn.copy_host_to_device_tensor(s, t)
        self.ple.prev = snap["ple_prev"]
        self.pos = snap["pos"]

    def generate(self, prompt, max_new: int, chunk: int = 32, eos=()) -> list[int]:
        """Greedy continuation of `prompt` from a fresh sequence: prompt chunks of <= `chunk` rows, then one token
        per step (the last prompt token is the first decode step, as in Strata)."""
        self.reset()
        prompt = list(prompt)
        for i in range(0, len(prompt) - 1, chunk):
            self.forward(prompt[i:min(i + chunk, len(prompt) - 1)], head=False)
        out, tok = [], prompt[-1]
        for _ in range(max_new):
            tok = int(self.forward([tok], read_logits=False)[1][0])
            out.append(tok)
            if tok in eos:
                break
        return out
