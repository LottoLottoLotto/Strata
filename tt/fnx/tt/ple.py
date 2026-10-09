"""Layer-1 PLE block on the P150 (TTNN), matching fnx.ref.ple.ple_block.

Host: n-gram hash (fnx.ref.ple.ngram_rows) + IQ4_NL row gather (fnx.ref.ple.PleTable) -> e_ple float32 [T, 2560],
uploaded as `[1,1,T,2560]` float32 TILE (`Ple.upload_rows(tokens)` does all three and advances the hash window).
Device: `Ple(mesh, w).forward(e_ple, R) -> R_new`, R `[1,1,T,10240]` float32 TILE after layer 0's FFN write.

State: the conv history lives ON THE DEVICE in one persistent ROW_MAJOR float32 buffer `[1,1,9,10240]` (row 8 newest;
fixed address, so a captured trace stays valid). A call with T rows concatenates it with this call's normalized rows
(row-major, exact data movement) into ext `[1,1,9+T,10240]`; row t's taps t-9, t-6, t-3 are ext rows t, t+3, t+6 (rows
of this call when t >= 3), and ext rows T..T+8 are written back into the buffer. So T tokens in one call equal T
sequential T=1 calls (spec-ple §6.4: normalized_t depends only on token t). The hash window (two_back, one_back) lives
on the host in `self.prev`. `reset()` zeroes both in place.
Verify-window rollback (Strata's hist_snap_[t] + commit, spec Part 2 §7): the last forward's ext stays on the device
(the history after row t is ext rows t+1..t+9, i.e. every row's snapshot) and upload_rows() records the hash window
before its tokens; `commit(n_keep)` keeps the first n_keep rows: history <- ext rows n_keep..n_keep+8 (copied into
the persistent buffer, address unchanged) and prev <- the window advanced by the kept tokens only.

Device weights: [Wk^T | Wv^T] BF16 [2560, 12800]; gammas * sqrt(2560) float32 [1,1,1,10240]; the 4 conv taps as
float32 [1,1,1,10240] rows (f16 values, exact).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

from ..ref.hc import D, EPS, HC, N
from ..ref.ple import HIST, SCALE, PleTable, ngram_rows, ple_weights
from .hc import BF16, F32, HIFI4, consts, gamma_dev, mm, rms_streams, stream_bcast, stream_sums, to_dev, upload

RM = ttnn.ROW_MAJOR_LAYOUT
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
SIGN = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGN)
SQRT = ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT)
SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
FAST = os.environ.get("FNX_PLE_FAST", "0") == "1"      # development flag (HeadQsa): row-stacked programs


def _rows(x, r0: int, r1: int):
    return ttnn.slice(x, [0, 0, r0, 0], [1, 1, r1, x.shape[-1]])


def _mm_post(x, w, post):
    """hc.mm(x, w) (precise hi/lo split) with the unary ops `post` fused into its final add (bitwise mm + ops)."""
    hi = ttnn.typecast(x, BF16)
    hi32 = ttnn.typecast(hi, F32)
    lo = ttnn.subtract(x, hi32)
    ttnn.deallocate(hi32)
    a = ttnn.matmul(hi, w, dtype=F32, compute_kernel_config=HIFI4)
    ttnn.deallocate(hi)
    b = ttnn.matmul(lo, w, dtype=F32, compute_kernel_config=HIFI4)
    ttnn.deallocate(lo)
    out = ttnn.add(a, b, activations=post)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return out


def _free(*ts) -> None:
    """Deallocate tensors, each buffer once (a reshape may return a view of its input)."""
    seen = set()
    for t in ts:
        a = t.buffer_address()
        if a not in seen:
            seen.add(a)
            ttnn.deallocate(t)


class Ple:
    def __init__(self, mesh, w, table: PleTable | None = None, precise: bool = True):
        Wk, Wv, gk, gq, gc, Wc = ple_weights(w)
        self.mesh, self.precise = mesh, precise
        self.table = table if table is not None else PleTable(w.model)
        self.wkv = upload(mesh, "blk.1.ple_key_value.T", np.concatenate([np.asarray(Wk).T, np.asarray(Wv).T], 1), BF16)
        self.gk = gamma_dev(mesh, "blk.1.ple_norm_key.weight", gk)
        self.gq = gamma_dev(mesh, "blk.1.ple_norm_query.weight", gq)
        self.gc = gamma_dev(mesh, "blk.1.ple_norm_conv.weight", gc)
        self.wc = [upload(mesh, f"blk.1.ple_conv1d.tap{k}", np.asarray(Wc)[:, k].reshape(1, D), F32) for k in range(4)]
        self._zeros = ttnn.from_torch(torch.zeros(1, 1, HIST, D), dtype=F32, layout=RM)   # host
        self.hist = to_dev(mesh, np.zeros((1, 1, HIST, D), np.float32), F32, RM)      # persistent device buffer
        self.prev = (-1, -1)
        self.window = None             # (prev before the last upload_rows, its tokens)
        self.ext = None                # the last forward's [1,1,9+T,10240] history rows (per-row snapshots)
        self.ext_owned = True          # False: ext belongs to a captured trace (fnx.tt.trace), never freed here

    def reset(self):
        """Zero the history in place (its device address never changes, so traces stay valid) and the hash window."""
        ttnn.copy_host_to_device_tensor(self._zeros, self.hist)
        self.prev = (-1, -1)
        self.drop_window()

    def drop_window(self):
        """Forget the last call's rollback data (TtModel.restore() must call it: a restored state has no window)."""
        if self.ext is not None and self.ext_owned:
            ttnn.deallocate(self.ext)
        self.window, self.ext, self.ext_owned = None, None, True

    def commit(self, n_keep: int) -> None:
        """Keep the first n_keep rows of the last call (upload_rows + forward) and roll the rest back."""
        if self.window is None or self.ext is None:
            raise RuntimeError("Ple.commit: no uploaded and computed window to commit")
        prev0, toks = self.window
        if not 1 <= n_keep <= len(toks) or self.ext.shape[-2] != HIST + len(toks):
            raise ValueError(f"commit({n_keep}) of a {len(toks)}-row window")
        if n_keep < len(toks):
            ttnn.slice(self.ext, [0, 0, n_keep, 0], [1, 1, n_keep + HIST, D], output_tensor=self.hist)
            _, self.prev = ngram_rows(toks[:n_keep], prev0)
        self.drop_window()

    def state_tensors(self) -> list:
        """The per-sequence device state that reset() zeroes (the hash window `prev` is host-side)."""
        return [self.hist]

    def upload_rows(self, tokens):
        """Hash `tokens` from the current window (advancing it), gather the rows, upload e_ple [1,1,T,2560] f32."""
        return to_dev(self.mesh, self.rows_host(tokens)[None, None])

    def rows_host(self, tokens) -> np.ndarray:
        """upload_rows without the upload: e_ple f32 [T, 2560] on the host (a traced step copies it into its input)."""
        self.drop_window()
        self.window = ((int(self.prev[0]), int(self.prev[1])), [int(t) for t in tokens])
        rows, self.prev = ngram_rows(tokens, self.prev)
        return self.table.gather(rows)

    def hist_host(self) -> np.ndarray:
        """The device history as float32 [10240, 9] (column 8 newest), the layout of fnx.ref.ple.PleState.hist."""
        return ttnn.to_torch(self.hist).float().numpy().reshape(HIST, D).T.copy()

    def _kv(self, e_ple, stack: bool):
        """e_ple @ [Wk^T | Wv^T] -> [T, 12800] f32 (hc.mm's exact hi/lo split). stack: hi and lo as the rows of ONE
        matmul (one read of the 65 MB weight instead of two; matmul rows are independent, so the bits are mm's)."""
        if not stack:
            return mm(e_ple, self.wkv, self.precise)
        T = e_ple.shape[-2]
        hi = ttnn.typecast(e_ple, BF16)
        hi32 = ttnn.typecast(hi, F32)
        ttnn.deallocate(hi)
        lo = ttnn.subtract(e_ple, hi32)
        x2 = ttnn.concat([hi32, lo], dim=2)                          # [2T, 2560]
        _free(hi32, lo)
        ab = ttnn.matmul(x2, self.wkv, dtype=F32, compute_kernel_config=HIFI4)
        ttnn.deallocate(x2)
        a, b = _rows(ab, 0, T), _rows(ab, T, 2 * T)
        ttnn.deallocate(ab)
        out = ttnn.add(a, b)
        _free(a, b)
        return out

    def _rms2(self, k_raw, R):
        """(rms_streams(k_raw, gk), rms_streams(R, gq)) with the stream sums and broadcasts of both on stacked rows."""
        T = R.shape[-2]
        x2 = ttnn.concat([k_raw, R], dim=2)                          # [2T, 10240]
        sq = ttnn.multiply(x2, x2)
        ttnn.deallocate(x2)
        s = stream_sums(sq, True)
        ttnn.deallocate(sq)
        s_e = ttnn.add(s, float(np.float32(N) * EPS))
        ttnn.deallocate(s)
        rs = ttnn.rsqrt(s_e)
        ttnn.deallocate(s_e)
        rsb = stream_bcast(rs, True)
        ttnn.deallocate(rs)
        out = []
        for x, g, r0 in ((k_raw, self.gk, 0), (R, self.gq, T)):
            xg = ttnn.multiply(x, g)
            rb = _rows(rsb, r0, r0 + T)
            out.append(ttnn.multiply(xg, rb))
            _free(xg, rb)
        ttnn.deallocate(rsb)
        return out

    def forward(self, e_ple, R):
        T = R.shape[-2]
        p = self.precise
        stack = FAST and p and 2 * T <= 32                            # stacked rows still fit one tile row
        kv = self._kv(e_ple, stack)                                   # [T, 12800]: key 10240 | value 2560
        k_raw = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, T, D])
        value = ttnn.slice(kv, [0, 0, 0, D], [1, 1, T, D + N])
        ttnn.deallocate(kv)
        if stack:
            key, query = self._rms2(k_raw, R)
        else:
            key = rms_streams(k_raw, self.gk, p)
            query = rms_streams(R, self.gq, p)
        ttnn.deallocate(k_raw)
        kq = ttnn.multiply(key, query)
        ttnn.deallocate(key)
        ttnn.deallocate(query)
        if FAST and p:
            # stream sums with the SCALE multiply fused into its final add, then one multiply computing
            # sigmoid(sign(s) * sqrt(max(|s|, 1e-6))) with the unary ops fused (bitwise the separate ops)
            s = _mm_post(kq, consts(kq.device())[0], [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU,
                                                                         float(SCALE))])
            ttnn.deallocate(kq)
            a = ttnn.abs(s)
            am = ttnn.maximum(a, 1e-6)
            ttnn.deallocate(a)
            gate = ttnn.multiply(s, am, input_tensor_a_activations=[SIGN], input_tensor_b_activations=[SQRT],
                                 activations=[SIGMOID])
            _free(s, am)
        else:
            s0 = stream_sums(kq, p)                                   # [T, 32], columns 0..3
            ttnn.deallocate(kq)
            s = ttnn.multiply(s0, float(SCALE))
            ttnn.deallocate(s0)
            a = ttnn.abs(s)
            am = ttnn.maximum(a, 1e-6)
            ttnn.deallocate(a)
            mag = ttnn.sqrt(am)
            ttnn.deallocate(am)
            sg = ttnn.sign(s)
            ttnn.deallocate(s)
            z = ttnn.multiply(sg, mag)
            ttnn.deallocate(sg)
            ttnn.deallocate(mag)
            gate = ttnn.sigmoid(z)
            ttnn.deallocate(z)
        gb = stream_bcast(gate, p)                                    # [T, 10240]
        ttnn.deallocate(gate)
        if FAST:                                  # value broadcast over the 4 stream blocks: no repeat, same products
            g4 = ttnn.reshape(gb, [1, HC, T, N])
            gated4 = ttnn.multiply(value, g4)
            _free(value, g4, gb)
            gated = ttnn.reshape(gated4, [1, 1, T, D])
            if gated.buffer_address() != gated4.buffer_address():
                ttnn.deallocate(gated4)
        else:
            v4 = ttnn.repeat(value, ttnn.Shape([1, 1, 1, HC]))
            ttnn.deallocate(value)
            gated = ttnn.multiply(v4, gb)
            ttnn.deallocate(v4)
            ttnn.deallocate(gb)
        normalized = rms_streams(gated, self.gc, p)
        n_rm = ttnn.to_layout(normalized, RM)
        ext = ttnn.concat([self.hist, n_rm], dim=2)                   # [1,1,9+T,10240] row-major
        ttnn.deallocate(n_rm)
        conv = None
        for k in range(4):
            if k < 3:
                tap_rm = _rows(ext, 3 * k, 3 * k + T)
                tap = ttnn.to_layout(tap_rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(tap_rm)
            else:
                tap = normalized
            term = ttnn.multiply(tap, self.wc[k])
            ttnn.deallocate(tap)
            if conv is None:
                conv = term
            else:
                c2 = ttnn.add(conv, term)
                ttnn.deallocate(conv)
                ttnn.deallocate(term)
                conv = c2
        if FAST:
            upd = ttnn.add(gated, conv, input_tensor_b_activations=[SILU])  # gated + silu(conv), the same bits
        else:
            act = ttnn.silu(conv)
            upd = ttnn.add(gated, act)
            ttnn.deallocate(act)
        ttnn.deallocate(gated)
        ttnn.deallocate(conv)
        out = ttnn.add(R, upd)
        ttnn.deallocate(upd)
        # new history = ext rows T..T+8, written into the persistent buffer (fixed address: trace-safe); ext itself is
        # kept until the next upload_rows/commit/reset: the per-row snapshots of this call
        ttnn.slice(ext, [0, 0, T, 0], [1, 1, T + HIST, D], output_tensor=self.hist)
        if self.ext is not None and self.ext_owned:
            ttnn.deallocate(self.ext)
        self.ext, self.ext_owned = ext, True
        return out
