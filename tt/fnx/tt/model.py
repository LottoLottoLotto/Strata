"""Flash-Next on the P150 (TTNN; eager, or traced decode steps): host embedding + PLE row gathers, 48 HC-wrapped
GDN/QSA + MoE layers on device, output_hc read, LM head, greedy.

MoE (FNX_MOE, default DEFAULT_MOE of the variant): "device" = fnx.tt.moe_device.MoeDevice, the whole MoE block on the device (router,
top-10, routed experts re-quantized to block floats from the GPTQ plan FNX_MOE_PLAN of fnx.quant.store, default
DEFAULT_PLAN: its resident experts in device DRAM, the others in pinned host memory read over PCIe by the kernels;
no host round trip, so a dense decode step is one captured trace); "cpu" = fnx.tt.moe.Moe, the routed experts on the
host CPU through fnx.cpu_experts (Strata's own kernels and GGUF weights: Strata-exact experts), except a resident set
on the device as bfloat8_b (fnx.tt.moe.default_resident).

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
from ..config import HC, HC_DIM, N_EMBD, N_LAYERS, N_USED, PLE_LAYER, is_qsa
from ..ref.embed import embed
from .gdn import Gdn
from .hc import HcRead, hc_write
from .head import Head
from .moe import DeviceExperts, Moe
from .moe_device import F_BFP4, MoeDevice, host_matrices, matrix_bytes, quant_layer
from .ple import Ple
from .qsa import Qsa

# FNX_MOE default per variant (fnx.paths.DEFAULT_MOE; quality gates: coder 2026-10-07, iq3s 2026-10-10).  The Coder's
# GPTQ-BFP4 device MoE is its fastest path within the gate.  iq3s keeps 65% of its experts off the card: its device
# MoE reads them over PCIe and pins 50.6 GB of host RAM, and against the cpu MoE with 4671 bfloat8_b experts resident
# it is no faster on the golden prompts (traced MTP chat/code/doc 28.0/28.8/33.5 vs 33.8/29.7/33.5 tok/s; ~25% faster
# on general chat, served) at 9x the KL against the exact experts (.061 vs .0067; GSM8K-50 46 both, exact 47).
DEFAULT_MOE = paths.DEFAULT_MOE
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
            if lms:
                # the device MoE streams its store once and never reads the GGUF experts: their cached pages would
                # only fragment the free memory the pool's 2 MiB chunks need (the cpu MoE computes from those pages)
                drop_page_cache([plan.pool.root] + [f.path for f in w.model.files])
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

    def warm_prompt_shapes(self, sizes) -> None:
        """Compile the QSA programs of prompt calls of every row count in `sizes` at every attention bucket a prompt
        can reach: each dense span (every 256 positions up to 2304), the dense-to-sparse crossing at position 2048
        (where chunk-aligned prompts cross it), and each width of the sparse block scores (every 4096 positions up to
        the context).  One QSA layer runs them (the layers share every program) on zero rows at those positions, then
        the sequence restarts: the garbage it leaves in the positional KV / indexer state is overwritten before any
        later call reads it (every cell and block is written before it is attended).  Without this, a prompt of a
        new length past 2048 compiled programs, and so recaptured every decode trace (~2.7 s, IQ3_S)."""
        from ..ref.qsa import DENSE_MAX_POS
        from .qsa import BSPAN, SPAN
        qsa = next((mix for _, mix, _, _ in self.layers if isinstance(mix, Qsa)), None)
        if qsa is None:
            return
        cap = qsa.max_ctx
        starts = []
        for s in sizes:
            for b in range(SPAN, DENSE_MAX_POS + SPAN + 1, SPAN):   # dense: the call's last row in span bucket b
                end = min(b, DENSE_MAX_POS + 1, cap) - 1             # (within the model's context)
                if end - s + 1 >= 0:
                    starts.append((s, end - s + 1))
            starts = list(dict.fromkeys(starts))                     # the context cap can repeat a bucket
            if DENSE_MAX_POS - 2048 + 1 < s and 2048 + s <= cap:  # the chunk-aligned crossing: dense + sparse rows
                starts.append((s, 2048))
            seen, k = set(), 1
            while True:                                           # sparse: every block-score width
                p0 = max(DENSE_MAX_POS + 1, 4 * BSPAN * (k - 1) + 4)
                if p0 + s > cap:
                    break
                nbs = min(-(-max(p0 // 4, 1) // BSPAN) * BSPAN, qsa.blocks)
                if nbs not in seen:
                    seen.add(nbs)
                    starts.append((s, p0))
                k += 1
        for s, p0 in starts:
            x = ttnn.zeros([1, 1, s, N_EMBD], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _free(qsa.forward(x, list(range(p0, p0 + s))), x)
        qsa.reset()
        # the cpu MoE's resident experts: one program pair per metadata capacity (a power of two >= the call's
        # entries); a real prompt's rows spread over many more experts than warm_up's repeated token reaches
        devs = [m.dev for _, _, _, m in self.layers if isinstance(m, Moe) and m.dev is not None]
        n_caps = 0
        if devs:
            d, n_res = devs[0], max(len(x.resident) for x in devs)
            for s in sizes:
                G = -(-s // 32)
                ent, route, gs = DeviceExperts.entries(s, np.full((s, N_USED), -1, np.int32),
                                                       np.zeros((s, N_USED), np.float32))     # G dummy entries
                c, top = DeviceExperts.s_cap_for(len(ent)), DeviceExperts.s_cap_for(min(N_USED * s, G * n_res))
                while c <= top:
                    words, S, _ = DeviceExperts.meta_words(s, ent, route, gs, s_cap=c)
                    meta = ttnn.from_torch(torch.from_numpy(words.view(np.int32)).reshape(1, 1, 1, -1),
                                           dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    x = ttnn.zeros([1, 1, s, N_EMBD], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    xb = d.prepare(x, s)
                    _free(d.run_fixed(xb, meta, c, s, s_alloc=S), xb, x, meta)
                    c *= 2
                    n_caps += 1
        self.warmed_prompt_calls = len(starts) + n_caps

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

    def forward_prompt(self, tokens, groups: list[int], block: int, keep_R: bool = False):
        """Prompt rows without the head, as consecutive row groups (`groups`: rows per group, each a multiple of
        `block`) whose CPU experts overlap the device.  The device runs every layer in `block`-row calls (forward()'s
        ops and shapes); the MoE of a group is one CPU expert call over all its blocks' rows (Moe.start).

        Per layer, group g's blocks are mixed and routed while group g-1's CPU experts still run; group g-1's MoE then
        completes, g's CPU experts start, and the device moves on to group g+1 (the last group's CPU experts overlap
        the first group's next layer).  The device state and arithmetic are those of consecutive forward(head=False)
        calls of `block` rows; only the CPU experts' token grouping is the group's (Moe.begin).  The CPU MoE only (the
        device MoE has no host phase to hide).  -> None, or with keep_R each block's final residual, in token order
        (release each through free_R())."""
        if self._win is not None:
            raise RuntimeError("TtModel.forward_prompt: commit() the last verify window first")
        if self.moe != "cpu":
            raise RuntimeError("TtModel.forward_prompt: the CPU MoE only")
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if sum(groups) != len(tokens) or any(n <= 0 or n % block for n in groups):
            raise ValueError(f"groups {groups} of {block}-row blocks do not cover {len(tokens)} rows")
        if self._trace is not None:
            self._trace.sync()                         # the host indexer rows a traced step deferred
        gs, a = [], 0
        for n in groups:
            g = []
            for b in range(a, a + n, block):
                tok = tokens[b:b + block]
                e_ple = self.ple.upload_rows(tok)      # the PLE n-gram history: blocks in token order
                g.append(dict(pos=np.arange(self.pos + b, self.pos + b + block), e_ple=e_ple, R=self.embed(tok)))
            gs.append(g)
            a += n
        pending = None                                 # (blocks, their layer's Moe, the begun calls)

        def finish():
            g, moe, cs = pending
            for blk, c, bo in zip(g, cs, moe.complete(cs)):
                blk["R"] = _replace(blk["R"], hc_write(blk["R"], bo, c["inj"]))
                _free(c["mixed"], bo, c["inj"])

        for L, (hc_attn, mix, hc_ffn, moe) in enumerate(self.layers):
            for g in gs:
                cs = []
                for blk in g:
                    R = blk["R"]
                    if L == PLE_LAYER:
                        R = _replace(R, self.ple.forward(blk["e_ple"], R))
                        _free(blk["e_ple"])
                    mixed, inj = hc_attn.forward(R)
                    bo = mix.forward(mixed, blk["pos"]) if is_qsa(L) else mix.forward(mixed)
                    R = _replace(R, hc_write(R, bo, inj))
                    _free(mixed, bo, inj)
                    blk["R"] = R
                    mixed, inj = hc_ffn.forward(R)
                    c = moe.begin(mixed)               # waits for the device, not for the CPU experts
                    c.update(mixed=mixed, inj=inj)
                    cs.append(c)
                if pending is not None:
                    finish()
                moe.start(cs)
                pending = (g, moe, cs)
        finish()
        self.pos += len(tokens)
        Rs = [blk["R"] for g in gs for blk in g]
        if keep_R:
            return Rs
        _free(*Rs)
        return None

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
