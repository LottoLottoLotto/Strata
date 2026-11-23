"""Traced decode steps for fnx.tt.model.TtModel (TtModel.enable_trace): the device work between the host round trips
of one forward is captured once per shape and replayed with ttnn.execute_trace(blocking=False), so a step no longer
dispatches its ~2000 ttnn ops one by one from Python.

A forward of T rows (T in `Ts`, default 1 and 4) is a `Sequence` of captured device segments, each followed by its
host part (the round trips that cannot be captured):
    inputs   host: embedding rows, PLE rows (n-gram hash + gather), QSA step inputs -> persistent device buffers
    seg 0    embedding -> R (4 streams), layer 0 attention half, layer 0 hc_ffn + MoE pre
    host     MoE L: read back [x | router logits] (T rows), route, start the CPU experts, upload the device-expert
             metadata                                            (fnx.tt.moe Moe.plan / Plan.meta_words)
    seg      MoE L device part: shared expert + resident experts (runs while the CPU experts do)
    host     MoE L: wait for the CPU experts, upload their weighted sum (row-major, T rows)   (Moe.finish)
    seg      MoE L post (y + r) + hc_write, [PLE], layer L+1 attention half, hc_ffn + MoE pre ...
    ...      (sparse QSA positions: one more cut per QSA layer, Qsa.sparse_host: indexer + exact cell selection)
    seg 48   MoE 47 post + hc_write, output_hc, LM head, device argmax (+ the row-major logits it scans)
    host     read back the ids (and the logits when asked), the QSA layers' `mixed` rows (dense steps: their host
             indexer update is deferred to the next step, overlapping its first segment)
Shapes are position-independent: QSA K/V are appended at a device position (fnx.tt.qsa.kv_write), RoPE tables and the
dense mask are step inputs, the dense attention scores a fixed span of cells (one Sequence per span bucket, the
smallest covering the window; the masked padding cells change no bit), sparse block scores cover every block.  A
window that mixes dense and sparse rows, another T, positions out of order, dump / head=False calls run eagerly.

All-device MoE (TtModel FNX_MOE=device, fnx.tt.moe_device.MoeDevice: router, top-10, resident and host-pool experts
read by the kernels themselves): a layer's hc_ffn + MoE + hc_write stay inside the current segment, no cut.  A dense
forward is then ONE captured segment (inputs -> 48 layers -> head) followed by the ids / logits read-back; a sparse
one keeps the QSA indexer cut of each QSA layer (Qsa.sparse_host: the exact host selection of 2051 cells).

Verify windows (T > 1): every GDN layer runs GdnFused.forward(snap=True) (its kernel saves the state after row 0 and
the rows it read into the layer's window buffers, allocated before the captures), and `commit(n_keep)` replays a
captured trace per n_keep (GdnFused.rollback of every GDN layer, exactly TtModel's eager commit).

Drafter rounds (enable_trace(drafter=mtp), DraftTracer): a whole fnx.tt.mtp.Mtp.draft round (catch-up front, every
chain step with the all-device MoE, the draft head, the device argmax and the gather of the next step's embedding row)
is ONE captured segment, one program per (catch-up rows T = a + 1, attention span bucket of the round's last cell:
powers of two from 512, then the whole cache; the attention matmuls accumulate sequentially, so padding cells change
no bit); the host writes the round inputs before it and reads 128 B after it.

Memory discipline (a replay rewrites every buffer its capture allocated, freed or not):
  * every sequence is captured in order, its temporaries freed inside the captures (also the cross-segment ones:
    replays run the segments in captured order), and nothing is allocated on the device between its segments (host
    parts only read back and copy into buffers allocated before the captures);
  * the buffers a sequence keeps (final R, ids, logits, the PLE window) are consumed before any other trace replays
    (TtModel.commit / the drafter run right after the window);
  * nothing allocated after the captures may stay alive across a replay: module caches are filled before the captures
    (fnx.tt.qsa._Const.warm, the GDN window buffers) or cleared before every replay (fnx.tt.qsa._StepInputs.clear);
    a traced forward's keep_R residual is the trace's own buffer (TtModel.free_R leaves it alone).
  * no trace replays after a program was compiled since its capture (a prompt chunk of a new row count, a new
    attention span, a new device-expert capacity ...): that program's kernel binary may sit in DRAM a capture used
    for temporaries, the replay overwrites it and its next run hangs the card (qwen36 #48536).  Tracer.ensure()
    (before every forward / draft replay; commit runs its programs eagerly instead) compares the program cache size
    with the last capture's and recaptures everything first (~2-3 s; new shapes stop appearing once seen).
Every result is bitwise the eager forward's (fnx/tests/test_trace.py).
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import ttnn

from ..config import HC, N_EMBD, PLE_LAYER, is_qsa
from ..ref.embed import embed
from ..ref.qsa import DENSE_MAX_POS
from .hc import hc_write
from .moe import ROUTER_COLS, DeviceExperts, Moe
from .moe_device import MoeDevice
from .qsa import GMAX, SPAN, TraceInputs, _Const, _StepInputs

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
DENSE_SPANS = (512, 1024, 2304)       # dense QSA span buckets (cells); the last covers position 2050
TS = (1, 4)                           # traced row counts: plain decode and the default MTP verify window


def env_spans() -> tuple[int, ...]:
    v = os.environ.get("FNX_TRACE_SPANS")
    return tuple(int(s) for s in v.split(",")) if v else DENSE_SPANS


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _host_rm(a: np.ndarray, dtype=ttnn.float32):
    """Host ROW_MAJOR tensor of `a` (ttnn.Tensor straight from numpy: the same tensor as from_torch, ~4 us cheaper)."""
    return ttnn.Tensor(np.ascontiguousarray(a), data_type=dtype, layout=RM)


def _zeros_rm(mesh, shape, dtype=ttnn.float32):
    t = torch.zeros(shape, dtype=torch.int32 if dtype in (ttnn.int32, ttnn.uint32) else torch.float32)
    return ttnn.from_torch(t, dtype=dtype, layout=RM, device=mesh, memory_config=DRAM)


class Sequence:
    """One traced forward shape: T rows, dense QSA span `span` (cells) or sparse (span None).  `segs` lists
    (device functions, host function, kind) triples; a capture records each one's device functions as one trace."""

    def __init__(self, model, T: int, span: int | None):
        m = self.m = model
        self.T, self.span, self.sparse = T, span, span is None
        mesh = m.mesh
        self.qsa = [L for L in range(m.n_layers) if is_qsa(L)]
        self.gdn = [L for L in range(m.n_layers) if not is_qsa(L)]
        self.verify = T > 1 and bool(self.gdn)
        # persistent inputs (allocated before any capture)
        self.e_in = _zeros_rm(mesh, (1, 1, T, N_EMBD))
        self.ple_in = _zeros_rm(mesh, (1, 1, T, N_EMBD)) if m.n_layers > PLE_LAYER else None
        q0 = m.layers[self.qsa[0]][1] if self.qsa else None
        self.qin = TraceInputs(mesh, T, span or 0, q0.cells, self.sparse) if q0 is not None else None
        self.s_cap, self.s_alloc, self.meta, self.r = {}, {}, {}, {}
        self.dev = {L for L, lay in enumerate(m.layers) if isinstance(lay[3], MoeDevice)}   # all-device MoE: no cut
        for L, (_, _, _, moe) in enumerate(m.layers):
            if L in self.dev:
                continue
            if moe.dev is not None:
                n_res = len(moe.dev.resident)
                self.s_alloc[L] = max(1, min(10 * T, n_res))
                self.s_cap[L] = DeviceExperts.s_cap_for(self.s_alloc[L])
                self.meta[L] = _zeros_rm(mesh, (1, 1, 1, 64 + 33 * self.s_cap[L]), ttnn.int32)
            self.r[L] = _zeros_rm(mesh, (1, 1, T, N_EMBD))
        # host tensor the MoE rows are read into (copy_device_to_host_tensor: no host allocation per layer)
        self.h_both = (ttnn.allocate_tensor_on_host(ttnn.Shape([1, 1, T, N_EMBD + ROUTER_COLS]), ttnn.float32, RM, mesh)
                       if len(self.dev) < m.n_layers else None)
        self.h_q: dict = {}                           # per QSA layer: the host tensor its [x | s~] rows are read into
        self.E = SimpleNamespace(both={}, qboth={})   # the captured program's device handles; both / qboth: the
                                                      # read-back rows of each layer (kept, never freed)
        self.segs: list[tuple[list, object, str]] = []      # (device functions, host function, kind)
        self.tids: list = []
        self.commit_tids: dict[int, object] = {}
        self._build()

    # ---- the program ---------------------------------------------------------------------------------------------
    def _build(self) -> None:
        m, T, E = self.m, self.T, self.E
        pos_d = list(range(T))             # only len(pos) is used by the captured QSA parts
        cur: list = []

        def cut(host, kind: str):
            self.segs.append((list(cur), host, kind))
            cur.clear()

        def d_inputs():
            e = ttnn.to_layout(self.e_in, TILE)
            E.R = ttnn.repeat(e, ttnn.Shape([1, 1, 1, HC]))        # col = stream * 2560 + d (TtModel.embed)
            ttnn.deallocate(e)
            E.e_ple = ttnn.to_layout(self.ple_in, TILE) if self.ple_in is not None else None
            if self.qin is not None:
                self.qin.prepare()
            E.qx = []
        cur.append(d_inputs)

        for L, (hc_attn, mix, hc_ffn, moe) in enumerate(m.layers):
            if L > 0 and L - 1 not in self.dev:
                cur.append(lambda L=L: self._post(L - 1))
            if L == PLE_LAYER:
                cur.append(self._ple)
            if not is_qsa(L):
                cur.append(lambda hc_attn=hc_attn, mix=mix: self._gdn(hc_attn, mix))
            elif not self.sparse:
                cur.append(lambda hc_attn=hc_attn, mix=mix: self._qsa_dense(hc_attn, mix, pos_d))
            else:
                cur.append(lambda L=L, hc_attn=hc_attn, mix=mix: self._qsa_pre(L, hc_attn, mix, pos_d))
                cut(lambda ctx, L=L, mix=mix: self._qsa_host(ctx, L, mix), "layer")
                cur.append(lambda mix=mix: self._qsa_post(mix, pos_d))
            if L in self.dev:
                cur.append(lambda hc_ffn=hc_ffn, moe=moe: self._ffn_dev(hc_ffn, moe))
                continue
            cur.append(lambda L=L, hc_ffn=hc_ffn, moe=moe: self._ffn_pre(L, hc_ffn, moe))
            cut(lambda ctx, L=L, moe=moe: self._moe_host(ctx, L, moe), "layer")
            cur.append(lambda L=L, moe=moe: self._moe_dev(L, moe))
            cut(lambda ctx, L=L, moe=moe: self._moe_wait(ctx, L, moe), "moe_dev")
        if m.n_layers - 1 not in self.dev:
            cur.append(lambda: self._post(m.n_layers - 1))
        cur.append(self._head)
        cut(self._final, "head")

    # device parts (captured)
    def _post(self, L: int) -> None:
        E = self.E
        bo = Moe.post(E.y, self.r[L])
        R = hc_write(E.R, bo, E.inj)
        _free(E.y, bo, E.inj, E.R)
        E.R, E.y, E.inj = R, None, None

    def _ple(self) -> None:
        E = self.E
        R = self.m.ple.forward(E.e_ple, E.R)
        _free(E.e_ple, E.R)
        E.R, E.e_ple = R, None

    def _gdn(self, hc_attn, mix) -> None:
        E = self.E
        mixed, inj = hc_attn.forward(E.R)
        bo = mix.forward(mixed, snap=True) if self.verify else mix.forward(mixed)
        R = hc_write(E.R, bo, inj)
        _free(E.R, bo, inj, mixed)
        E.R = R

    def _qsa_dense(self, hc_attn, mix, pos_d) -> None:
        E = self.E
        mixed, inj = hc_attn.forward(E.R)
        dev = mix._project(mixed, pos_d, False, step=self.qin)
        bo = mix._attend(dev, pos_d, self.T, [], step=self.qin)
        R = hc_write(E.R, bo, inj)
        _free(E.R, bo, inj)
        E.R = R
        E.qx.append(mixed)                    # read back at the end of the step: the host indexer (index_rows)

    def _qsa_pre(self, L: int, hc_attn, mix, pos_d) -> None:
        E = self.E
        mixed, E.qinj = hc_attn.forward(E.R)
        E.qdev = mix._project(mixed, pos_d, True, step=self.qin)
        E.qboth[L] = mix._readback_dev(mixed, E.qdev)    # kept (never freed): the host part reads it back
        ttnn.deallocate(mixed)

    def _qsa_post(self, mix, pos_d) -> None:
        E = self.E
        bo = mix._attend(E.qdev, pos_d, 0, [self.qin.ids], step=self.qin)
        R = hc_write(E.R, bo, E.qinj)
        _free(E.R, bo, E.qinj)
        E.R, E.qinj, E.qdev = R, None, None

    def _ffn_dev(self, hc_ffn, moe) -> None:
        E = self.E
        mixed, inj = hc_ffn.forward(E.R)
        bo = moe.forward(mixed)
        R = hc_write(E.R, bo, inj)
        _free(E.R, mixed, bo, inj)
        E.R = R

    def _ffn_pre(self, L: int, hc_ffn, moe) -> None:
        E = self.E
        mixed, E.inj = hc_ffn.forward(E.R)
        E.pre = moe.pre(mixed, self.T)
        E.both[L] = E.pre["both"]             # kept (never freed): the host part reads it back

    def _moe_dev(self, L: int, moe) -> None:
        E, pr = self.E, self.E.pre
        y = moe.shared(pr)
        if moe.dev is not None:
            yd = moe.dev.run_fixed(pr["xb"], self.meta[L], self.s_cap[L], self.T, s_alloc=self.s_alloc[L])
            y2 = ttnn.add(y, yd)                  # yd = exact zeros without a device slot (dummy entry)
            _free(y, yd, pr["xb"])
            y = y2
        _free(pr["logits"], pr["x"])
        E.y, E.pre = y, None

    def _head(self) -> None:
        m, E = self.m, self.E
        head_in, _ = m.hc_out.forward(E.R)        # E.R stays: the keep_R residual
        logits = m.head.forward(head_in)
        E.ids, E.lrm = m.head.argmax(logits, keep_rm=True)
        _free(head_in, logits)
        E.qrows = None
        if E.qx:
            cat = ttnn.concat(E.qx, dim=-1) if len(E.qx) > 1 else E.qx[0]
            E.qrows = ttnn.untilize(cat)
            _free(*E.qx, *(() if cat is E.qx[0] else (cat,)))
        E.qx = []
        if self.qin is not None:
            self.qin.release()

    # host parts (ctx["mode"]: "run" a replay, "warm" the eager compile run; nothing runs during a capture)
    def _moe_host(self, ctx: dict, L: int, moe) -> None:
        t0 = time.perf_counter()
        ttnn.copy_device_to_host_tensor(self.E.both[L], self.h_both, blocking=True)
        host = self.h_both.to_numpy().reshape(self.T, N_EMBD + ROUTER_COLS)
        t1 = time.perf_counter()
        p = ctx["plan"] = moe.plan(host, self.s_cap[L] if moe.dev is not None else None)
        t2 = time.perf_counter()
        if moe.dev is not None:
            ttnn.copy_host_to_device_tensor(_host_rm(p.words.view(np.int32).reshape(1, 1, 1, -1), ttnn.int32),
                                            self.meta[L])
        moe.last = dict(ids=p.ids, wt=p.wt, on_dev=p.on_dev, refined=p.refined)
        prof = ctx.get("prof")
        if prof is not None:
            prof["readback"] += t1 - t0
            prof["route"] += t2 - t1
            prof["upload"] += time.perf_counter() - t2

    def _moe_wait(self, ctx: dict, L: int, moe) -> None:
        t0 = time.perf_counter()
        routed, cpu_us = moe.finish(ctx.pop("plan"))
        t1 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(_host_rm(routed.reshape(1, 1, self.T, N_EMBD)), self.r[L])
        prof = ctx.get("prof")
        if prof is not None:
            prof["cpu"] += cpu_us * 1e-6
            prof["cpu_wait"] += t1 - t0
            prof["upload"] += time.perf_counter() - t1

    def _qsa_host(self, ctx: dict, L: int, mix) -> None:
        t0 = time.perf_counter()
        if ctx["mode"] == "warm":            # compile run: any valid cells (no indexer update)
            self.qin.write_ids(np.zeros(int(self.qin.ids.shape[-1]), np.int32))
            return
        q = self.E.qboth[L]
        h = self.h_q.get(L)
        if h is None or list(h.shape) != list(q.shape) or h.dtype != q.dtype:
            h = self.h_q[L] = ttnn.allocate_tensor_on_host(ttnn.Shape(list(q.shape)), q.dtype, RM, self.m.mesh)
        ttnn.copy_device_to_host_tensor(q, h, blocking=True)
        host = h.to_numpy().astype(np.float32, copy=False).reshape(self.T, -1)
        self.qin.write_ids(mix.sparse_host(host, ctx["pos"]))
        prof = ctx.get("prof")
        if prof is not None:
            prof["qsa_host"] += time.perf_counter() - t0

    def _final(self, ctx: dict) -> None:
        t0 = time.perf_counter()
        E, T = self.E, self.T
        ctx["ids"] = ttnn.to_torch(E.ids).numpy().reshape(-1)[:T].astype(np.int64)
        ctx["logits"] = ttnn.to_torch(E.lrm).float().numpy().reshape(T, -1) if ctx["read_logits"] else None
        ctx["qrows"] = ttnn.to_torch(E.qrows).float().numpy().reshape(T, -1) if E.qrows is not None else None
        prof = ctx.get("prof")
        if prof is not None:
            prof["final"] += time.perf_counter() - t0

    # ---- inputs, warm-up, capture, replay ----------------------------------------------------------------------
    def write_inputs(self, tokens, pos) -> None:
        m, T = self.m, self.T
        ttnn.copy_host_to_device_tensor(_host_rm(embed(m.w, tokens).reshape(1, 1, T, N_EMBD)), self.e_in)
        if self.ple_in is not None:
            ttnn.copy_host_to_device_tensor(_host_rm(m.ple.rows_host(tokens).reshape(1, 1, T, N_EMBD)), self.ple_in)
        if self.qin is not None:
            self.qin.write(list(pos))

    def commit_fn(self, n: int):
        """Device: TtModel.commit(n) of this sequence's window (every GDN layer's rollback to its first n rows)."""
        return lambda: self.m._rollback(n)

    def warm(self, pos0: int) -> None:
        """Eager run of the program (compiles every program the capture records), then the partial commits."""
        m, T = self.m, self.T
        pos = list(range(pos0, pos0 + T))
        self.write_inputs([198] * T, pos)
        ctx = {"mode": "warm", "pos": pos, "read_logits": True}
        for devs, host, _ in self.segs:
            for f in devs:
                f()
            host(ctx)
        if self.verify:
            for n in range(1, T):
                self.commit_fn(n)()
        self._release_outputs()
        m.ple.drop_window()

    def _release_outputs(self) -> None:
        E = self.E
        _free(E.R, E.ids, E.lrm, E.qrows, *E.both.values(), *E.qboth.values())
        E.R, E.ids, E.lrm, E.qrows, E.both, E.qboth = None, None, None, None, {}, {}

    def capture(self) -> None:
        mesh, m = self.m.mesh, self.m
        m.ple.drop_window()
        for devs, _, _ in self.segs:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for f in devs:
                f()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            self.tids.append(tid)
        E = self.E
        E.ext = m.ple.ext if self.ple_in is not None else None       # the PLE window rows, this trace's buffer
        m.ple.ext, m.ple.window = None, None
        if self.verify:
            for n in range(1, self.T):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                self.commit_fn(n)()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                self.commit_tids[n] = tid

    def run(self, tokens, pos, read_logits: bool, after_first=None, prof: dict | None = None) -> dict:
        mesh = self.m.mesh
        t0 = time.perf_counter()
        self.write_inputs(tokens, pos)
        ctx = {"mode": "run", "pos": list(pos), "read_logits": read_logits, "prof": prof}
        if prof is not None:
            prof["inputs"] += time.perf_counter() - t0
        for i, ((_, host, _), tid) in enumerate(zip(self.segs, self.tids)):
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            if prof is not None:
                prof["launch"] += time.perf_counter() - t1
            if i == 0 and after_first is not None:
                after_first()
            host(ctx)
        if self.ple_in is not None:
            self.m.ple.ext, self.m.ple.ext_owned = self.E.ext, False
        return ctx

    def release(self, lent=None) -> None:
        """Release the traces and the buffers they own (except `lent`: a keep_R residual still held by a caller,
        which TtModel.free_R then frees)."""
        mesh = self.m.mesh
        for tid in self.tids + list(self.commit_tids.values()):
            ttnn.release_trace(mesh, tid)
        self.tids, self.commit_tids = [], {}
        E = self.E
        _free(E.ids, E.lrm, E.qrows, getattr(E, "ext", None), *E.both.values(), *E.qboth.values(),
              *(() if E.R is None or E.R is lent else (E.R,)))
        E.R, E.ids, E.lrm, E.qrows, E.ext, E.both, E.qboth = None, None, None, None, None, {}, {}


class Tracer:
    """TtModel's traced forwards (module docstring): sequences per T and QSA span bucket, deferred QSA indexer rows;
    with `drafter` (fnx.tt.mtp.Mtp) also its traced rounds (DraftTracer, installed as drafter._trace)."""

    def __init__(self, model, Ts=TS, spans=None, verbose: bool = True, drafter=None):
        m = self.m = model
        if m._win is not None:
            raise RuntimeError("enable_trace: commit() the verify window first")
        t0 = time.time()
        if not Ts or min(Ts) < 1 or max(Ts) > GMAX:
            raise ValueError(f"traced row counts {Ts}: 1..{GMAX} (one sparse attention group)")
        spans = tuple(sorted(env_spans() if spans is None else spans))
        self.qsa = [L for L in range(m.n_layers) if is_qsa(L)]
        q0 = m.layers[self.qsa[0]][1] if self.qsa else None
        self.max_ctx = q0.max_ctx if q0 is not None else None
        if q0 is not None:
            # dense rows attend cells <= min(position, 2050): spans past the cache would gather beyond its rows
            dense_max = min(-(-(DENSE_MAX_POS + 1) // SPAN) * SPAN, q0.cells)
            spans = tuple(sorted({min(s, q0.cells) for s in spans}))
            if spans[-1] < dense_max or any(s % SPAN for s in spans):
                raise ValueError(f"dense spans {spans}: multiples of {SPAN}, the last >= {dense_max}")
            _Const.warm(m.mesh, q0.cells, range(SPAN, dense_max + 1, SPAN))
        self.spans = spans if q0 is not None else (0,)
        if max(Ts) > 1:
            m._window_buffers(max(Ts))        # fixed addresses before any capture
        self.seqs: dict[tuple[int, int | None], Sequence] = {}
        for T in Ts:
            for s in self.spans:
                self.seqs[(T, s)] = Sequence(m, T, s)
            if q0 is not None and self.max_ctx - T > DENSE_MAX_POS:
                self.seqs[(T, None)] = Sequence(m, T, None)
        self.draft = DraftTracer(drafter, Ts=tuple(range(1, max(Ts) + 1))) if drafter is not None else None
        if self.draft is not None:
            self.draft.owner = self
        self.pending = None                   # (qrows [T, 12 * 2560], pos) of the last dense traced step
        self.prof: dict | None = None         # per-step host time breakdown (seconds), when a dict
        self.lent = None                      # the keep_R residual of the last traced forward, until TtModel.free_R
        self.recaptures, self.recapture_s = 0, 0.0
        self.events: list[tuple[str, int, float]] = []
        t1 = self._capture_all()
        if self.draft is not None:
            drafter._trace = self.draft
        if verbose:
            dr = (f"; drafter rounds: catch-up rows T={list(self.draft.Ts)} x spans {list(self.draft.spans)} "
                  f"({len(self.draft.rounds)} one-segment rounds)" if self.draft is not None else "")
            print(f"TtModel: traced T={list(Ts)}, {len(self.seqs)} sequences "
                  f"({sum(len(s.tids) for s in self.seqs.values())} segments + "
                  f"{sum(len(s.commit_tids) for s in self.seqs.values())} commits){dr}: compile run {t1 - t0:.1f} s, "
                  f"capture {time.time() - t1:.1f} s", flush=True)

    def _capture_all(self) -> float:
        """Compile runs + captures of every sequence that fits at the current position (and the drafter's rounds),
        the per-sequence state restored afterwards.  Returns the time the captures started."""
        m = self.m
        snap = m.snapshot()
        pos0 = m.pos
        for seq in self.seqs.values():
            if self.max_ctx is None or pos0 + seq.T <= self.max_ctx:     # else no free cells: stays eager
                seq.warm(pos0)
        if self.draft is not None:
            self.draft.warm()
        t1 = time.time()
        for seq in self.seqs.values():
            if self.max_ctx is None or pos0 + seq.T <= self.max_ctx:
                seq.capture()
        if self.draft is not None:
            self.draft.capture()
        m.restore(snap)
        self.n_prog = m.mesh.num_program_cache_entries()
        return t1

    def grown(self) -> bool:
        """Programs were compiled since the capture (eager calls with new shapes).  Their kernel binaries may sit
        where the traces keep their temporaries: a replay would overwrite them and the program's next run hang
        (qwen36 #48536), so no trace may replay before a recapture (ensure())."""
        return self.m.mesh.num_program_cache_entries() != self.n_prog

    def ensure(self, where: str = "") -> None:
        """Recapture every trace if programs were compiled since the last capture (the new temporaries then avoid
        every kernel binary allocated so far).  Needs a committed state; a lent keep_R residual stays valid.
        `events` records (where, programs compiled since the capture, seconds) per recapture."""
        if not self.grown():
            return
        t0 = time.time()
        new = self.m.mesh.num_program_cache_entries() - self.n_prog if self.n_prog >= 0 else 0
        self.sync()
        for s in self.seqs.values():
            s.release(self.lent)
        if self.draft is not None:
            self.draft.release(free=True)
        self._capture_all()
        self.recaptures += 1
        self.recapture_s += time.time() - t0
        self.events.append((where, new, time.time() - t0))

    def free_R(self, R) -> bool:
        """TtModel.free_R: True when R is a traced forward's own residual (kept), else the caller frees it."""
        if R is self.lent:
            self.lent = None
        return self.owns(R)

    def pick(self, T: int, pos: np.ndarray) -> Sequence | None:
        p0, p1 = int(pos[0]), int(pos[-1])
        if self.max_ctx is not None and p1 >= self.max_ctx:
            return None
        if self.max_ctx is None or p1 <= DENSE_MAX_POS:
            n = -(-(p1 + 1) // SPAN) * SPAN
            span = next((s for s in self.spans if s >= n), None) if self.max_ctx is not None else 0
            seq = self.seqs.get((T, span))
        elif p0 > DENSE_MAX_POS:
            seq = self.seqs.get((T, None))
        else:
            return None                       # a window across the dense limit
        if seq is not None and not seq.tids:  # not captured at the last (re)capture position: capture now
            self.n_prog = -1
        return seq

    def owns(self, t) -> bool:
        return any(t is s.E.R for s in self.seqs.values())

    def sync(self) -> None:
        """The deferred host indexer rows of the last dense traced step -> every QSA layer's host state."""
        if self.pending is None:
            return
        rows, pos = self.pending
        self.pending = None
        for i, L in enumerate(self.qsa):
            self.m.layers[L][1].index_rows(rows[:, i * N_EMBD:(i + 1) * N_EMBD], pos)

    def flush(self) -> None:
        """sync() + the deferred P2 page uploads (eager; before anything reads the device block scores)."""
        self.sync()
        for L in self.qsa:
            self.m.layers[L][1].flush_pooled()

    def forward(self, seq: Sequence, tokens, pos, read_logits: bool):
        _StepInputs.clear()                   # an eager QSA call's cached inputs must not live across a replay
        prof = self.prof
        t0 = time.perf_counter()
        if prof is not None:
            for k in ("inputs", "launch", "readback", "route", "upload", "cpu_wait", "cpu", "qsa_host", "final",
                      "indexer", "total"):
                prof.setdefault(k, 0.0)
            prof["steps"] = prof.get("steps", 0) + 1

        def after():                          # host work overlapping the first segment
            t = time.perf_counter()
            self.sync()
            if prof is not None:
                prof["indexer"] += time.perf_counter() - t
        if seq.sparse:
            after()
            self.flush()                      # the device scores read the P2 rows of the blocks completed so far
        self.ensure(f"forward T={len(tokens)} at {int(pos[0])}")   # after the eager uploads: they may compile too
        ctx = seq.run(tokens, pos, read_logits, None if seq.sparse else after, prof)
        if ctx["qrows"] is not None:
            self.pending = (ctx["qrows"], [int(p) for p in pos])
        if prof is not None:
            prof["total"] += time.perf_counter() - t0
        return ctx["logits"], ctx["ids"]

    def release(self) -> None:
        for s in self.seqs.values():
            s.release(self.lent)
        if self.draft is not None:
            self.draft.release(free=True)
            self.draft.mtp._trace = None
            self.draft = None


# ---------------------------------------------------------------------------------------------------- the drafter
def draft_spans(cells: int) -> tuple[int, ...]:
    """Attention span buckets of a traced drafter round: powers of two from 512 below the cache, then all its cells."""
    out, s = [], 512
    while s < cells:
        out.append(s)
        s *= 2
    return tuple(out) + (cells,)


class DraftRound:
    """One traced drafter round (fnx.tt.mtp.Mtp.round_dev): the catch-up front over T = a + 1 rows, every chain step
    (the drafts fed from step to step on the device) and the draft head, attending `span` cells, as ONE captured
    segment.  Host: the round inputs before it (Rin copy, RoundInputs.write: embedding rows, RoPE tables, K/V
    positions, the attending cell), one 128 B read-back after it (draft subset indices and probabilities)."""

    def __init__(self, mtp, T: int, span: int, Rin, io, cells_row):
        self.mtp, self.T, self.span = mtp, T, span
        self.Rin, self.io, self.cells_row = Rin, io, cells_row
        self.res = None                     # the captured round's result (kept: read back after each replay)
        self.tid = None

    def _dev(self, p: int) -> None:
        self.res, _ = self.mtp.round_dev(self.io, self.Rin, self.io.cells(p), self.span, self.cells_row)

    def _last_p(self) -> int:
        """The round whose last chain cell is the cache's last (positional K/V: rewritten before any query)."""
        return self.mtp.cells - self.T - self.mtp.steps + 1

    def warm(self) -> None:
        p = self._last_p()
        self.io.write(self.mtp, [198] * self.T, p)
        self._dev(p)
        ttnn.to_torch(self.res)
        _free(self.res)
        self.res = None

    def capture(self) -> None:
        mesh = self.mtp.mesh
        self.tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self._dev(self._last_p())
        ttnn.end_trace_capture(mesh, self.tid, cq_id=0)

    def run(self, tokens, p: int) -> np.ndarray:
        """Replay at p (Rin already holds the window rows) -> the result row (fnx.tt.mtp.Mtp.decode's input)."""
        self.io.write(self.mtp, tokens, p)
        ttnn.execute_trace(self.mtp.mesh, self.tid, cq_id=0, blocking=False)
        return ttnn.to_torch(self.res).float().numpy().reshape(-1)

    def release(self, free: bool = False) -> None:
        if self.tid is not None:
            ttnn.release_trace(self.mtp.mesh, self.tid)
        self.tid = None
        if free:                                            # the result buffer the capture allocated
            _free(self.res)
            self.res = None


class DraftTracer:
    """Mtp's traced rounds: one DraftRound per (first-step rows T, span bucket of the round's last cell).  Built by
    the model's Tracer (TtModel.enable_trace(drafter=mtp)): every buffer, the span constants and the per-T round
    inputs included, exists before the first capture, so no replay of one can write over a constant of another."""

    def __init__(self, mtp, Ts=(1, 2, 3, 4), spans=None):
        from .mtp import RoundInputs
        mesh = mtp.mesh
        self.mtp = mtp
        self.spans = tuple(sorted(draft_spans(mtp.cells) if spans is None else spans))
        if self.spans[-1] < mtp.cells or any(s % 256 for s in self.spans):
            raise ValueError(f"draft spans {self.spans}: multiples of 256, the last >= {mtp.cells} cells")
        self.Ts = tuple(sorted(set(Ts) | {1}))
        zeros = lambda T: ttnn.from_torch(torch.zeros(1, 1, T, HC * N_EMBD), dtype=ttnn.float32, layout=TILE,
                                          device=mesh, memory_config=DRAM)
        self.Rin = {T: zeros(T) for T in self.Ts}
        self.io = {T: RoundInputs(mesh, T, mtp.steps) for T in self.Ts}
        cells_rows = {s: ttnn.from_torch(torch.arange(s, dtype=torch.float32).reshape(1, 1, 1, s), dtype=ttnn.float32,
                                         layout=TILE, device=mesh, memory_config=DRAM) for s in self.spans}
        self.rounds = {(T, s): DraftRound(mtp, T, s, self.Rin[T], self.io[T], cells_rows[s])
                       for T in self.Ts for s in self.spans}

    def warm(self) -> None:
        """Compile runs at the cache's last cells, and the eager copies draft() makes of a window's residual rows
        into Rin (windows of 1..max(Ts) rows)."""
        mesh = self.mtp.mesh
        for rows in range(1, max(self.Ts) + 1):
            R = ttnn.from_torch(torch.zeros(1, 1, rows, HC * N_EMBD), dtype=ttnn.float32, layout=TILE, device=mesh,
                                memory_config=DRAM)
            for T in self.Ts:
                if T == rows:
                    ttnn.copy(R, self.Rin[T])
                elif T < rows:
                    ttnn.slice(R, [0, 0, 0, 0], [1, 1, T, HC * N_EMBD], output_tensor=self.Rin[T])
            ttnn.deallocate(R)
        for r in self.rounds.values():
            r.warm()

    def capture(self) -> None:
        for r in self.rounds.values():
            r.capture()

    def _span(self, cell: int) -> int | None:
        n = -(-(cell + 1) // 256) * 256
        return next((s for s in self.spans if s >= n), None)

    def draft(self, R_rows, outv, p: int, a: int, steps: int, min_p: float):
        """Mtp.draft's round, replayed; None when the round has no captured program (the caller runs it eagerly)."""
        T = a + 1
        m = self.mtp
        last = p + a + m.steps - 1
        if (T, self.spans[0]) not in self.rounds or last >= m.cells or R_rows.shape[-2] < T:
            return None
        _StepInputs.clear()
        Rin = self.Rin[T]
        for _ in range(3):                  # a recapture's compile runs reuse Rin: copy after it (the copy itself
            self.owner.ensure(f"draft a={a} at {p}")   # may compile once more: then recapture, copy again)
            if R_rows.shape[-2] == T:
                ttnn.copy(R_rows, Rin)
            else:
                ttnn.slice(R_rows, [0, 0, 0, 0], [1, 1, T, HC * N_EMBD], output_tensor=Rin)
            if not self.owner.grown():
                break
        res = self.rounds[(T, self._span(last))].run(list(outv)[:T], p)
        return m.decode(res, steps, min_p)

    def release(self, free: bool = False) -> None:
        for r in self.rounds.values():
            r.release(free)
