"""Float32 reference of the eval set on the RTX 3090 (torch, CUDA): the published Coder GGUF computed with dequantized
weights everywhere, routed experts included, i.e. fnx.ref.model.RefModel with float experts (RefModel's math, module
by module: embed, PLE, HC read/write, GDN, QSA, MoE with ref.moe.expert_parts_float, output_hc, head).

    cd /var/tmp/strata-tt/flashnext_tt
    /var/tmp/strata-tt/venv/bin/python -m fnx.eval.torchref build [--topk 256] [--records id,..]
    /var/tmp/strata-tt/venv/bin/python -m fnx.eval.torchref check [--record ID] [--tokens 64]
    /var/tmp/strata-tt/venv/bin/python -m fnx.eval.torchref noise [--chunk 1000]

build: every record of fnx.eval.data's evalset, layer-major (all sequences through layer L, then L+1: one layer's 256
dequantized experts, 5 GB f32, are on the GPU at a time; worker processes dequantize the next layers into /dev/shm).
Writes FNX_EVAL/ref/<record id>.npz per record, position i = the logits after token i:
    lse f64 [n]          logsumexp of the float32 logits (computed in float64)
    top_ids i32 [n, K]   the K largest logits, descending;  top_lp f32 [n, K] their log-probabilities
    tgt_lp f32 [n]       log-probability of token i+1 (NaN at the last position)
    argmax i32 [n]       Strata's greedy pick (first maximal index)
and ref/meta.json (K, evalset sha256, timings, the check result when present).
check: the same code on one short sequence against RefModel (numpy, float experts) -> logits rel-L2, max |d|,
argmax agreement, KL; written to ref/check.json.
noise: the reference against itself with other row chunks (float32 rounding, amplified by routing flips and the
recurrences), scored like a TT run -> ref/noise.json: the floor below which configurations are indistinguishable.

Numerics: float32 everywhere RefModel is float32 (TF32 off), float64 where it is float64 (router logits, routing
softmax and weights, the MoE combine's fma steps, the RoPE angle table), fp16-rounded QSA K/V cells.  Only the dense
QSA path exists here: every sequence must stay within 2051 positions (fnx.eval.data keeps them <= 2048).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..config import (COMPRESS, EPS, HC, HEAD_DIM, IDX_TOP_K, N_EMBD, N_EXPERT, N_FF, N_HEAD, N_HEAD_KV, N_LAYERS,
                      N_USED, N_VOCAB, PLE_LAYER, ROPE_BASE, ROPE_DIM, SSM_K_HEADS, SSM_STATE, SSM_V_HEADS, is_qsa)
from . import data

DEV = torch.device("cuda")
F32, F64 = torch.float32, torch.float64
SHM = Path(os.environ.get("FNX_EVAL_SHM", "/dev/shm"))
REF = data.EVAL / "ref"
TOPK = 256
SEL_WIDTH = IDX_TOP_K + COMPRESS - 1       # 2051: positions < this attend densely
QK = SSM_K_HEADS * SSM_STATE                # 2048
W_CLAMP = 2.0 ** -14


def T(a, dtype=F32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(DEV, dtype)


# ------------------------------------------------------------------------------------------------ expert prefetch
_WM = {}


def _dequant(args) -> None:
    """Worker: experts e0..e1 of one expert stack into the /dev/shm .npy file at `path`."""
    shard1, name, path, e0, e1 = args
    if shard1 not in _WM:
        from fnx.gguf_load import Model
        _WM[shard1] = Model(shard1)
    m = _WM[shard1]
    mm = np.lib.format.open_memmap(path, mode="r+")
    for e in range(e0, e1):
        mm[e] = m.get_expert(name, e)
    mm.flush()
    del mm


class ExpertLoader:
    """Dequantizes layer L's routed experts (gate/up (256, 640, 2560), down (256, 2560, 640) f32) in worker processes
    ahead of use; get(L) waits and returns them on the GPU."""

    def __init__(self, shard1: str, workers: int = 12, ahead: int = 2):
        self.shard1, self.ahead = shard1, ahead
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        self.pending = {}

    def request(self, L: int) -> None:
        if L in self.pending or L >= N_LAYERS:
            return
        paths, futs = {}, []
        for r in ("gate", "up", "down"):
            shape = (N_EXPERT, N_FF, N_EMBD) if r != "down" else (N_EXPERT, N_EMBD, N_FF)
            p = SHM / f"fnx-eval-{os.getpid()}-{L}-{r}.npy"
            np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=shape)
            paths[r] = p
            futs += [self.pool.submit(_dequant, (self.shard1, f"blk.{L}.ffn_{r}_exps.weight", str(p), e0, e0 + 16))
                     for e0 in range(0, N_EXPERT, 16)]
        self.pending[L] = (paths, futs)

    def get(self, L: int) -> dict:
        for k in range(L, L + self.ahead + 1):
            self.request(k)
        paths, futs = self.pending.pop(L)
        for f in futs:
            f.result()
        out = {}
        for r, p in paths.items():
            out[r] = torch.from_numpy(np.load(p, mmap_mode="r")[...]).to(DEV)
            p.unlink()
        return out

    def close(self) -> None:
        for paths, futs in self.pending.values():
            for f in futs:
                f.cancel()
            for f in futs:
                if not f.cancelled():
                    f.result()
            for p in paths.values():
                p.unlink(missing_ok=True)
        self.pool.shutdown()


class FloatExperts:
    """RefModel's `experts` argument computing the routed experts from f32 dequantized weights (float experts)."""

    def __init__(self, model):
        self.model = model

    def moe(self, L, x, ids):
        from ..ref.moe import expert_parts_float
        return expert_parts_float(self.model, L, x, ids)


# ------------------------------------------------------------------------------------------------ modules (torch)
def rms_streams(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """ref.hc.rms_streams: x [n, 4, 2560], g [10240] -> (x*g)*rs per stream."""
    rs = 1.0 / torch.sqrt((x * x).mean(-1) + EPS)
    return (x * g.view(HC, N_EMBD)) * rs[..., None]


def hc_read(R: torch.Tensor, hw, chunk: int = 4096):
    g, Wd, Wu, Wi = hw
    n = R.shape[0]
    mixed = torch.empty(n, N_EMBD, device=DEV)
    inj = torch.empty(n, HC, device=DEV) if Wi is not None else None
    for a in range(0, n, chunk):
        xn = rms_streams(R[a:a + chunk], g)
        x = xn.reshape(-1, HC * N_EMBD)
        p = (x @ Wd.T) / HC
        lo = p * torch.sigmoid(p)
        gate = (lo @ Wu.T).view(-1, HC, N_EMBD)
        mixed[a:a + chunk] = (xn * torch.sigmoid(gate)).sum(1) / HC
        if Wi is not None:
            inj[a:a + chunk] = x @ Wi.T
    return mixed, inj


def hc_write(R: torch.Tensor, bo: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
    return R + (2.0 * torch.sigmoid(inj / HC))[:, :, None] * bo[:, None, :]


def silu(x):
    return x * torch.sigmoid(x)


class Ref:
    """Layer-major float32 forward of whole sequences (each starts at position 0 with fresh state).

    `loader`: the routed experts per layer: request(L) (a prefetch hint) and get(L) -> {"gate": [256, 640, 2560],
    "up": [256, 640, 2560], "down": [256, 2560, 640]} f32 on the GPU (ExpertLoader: the GGUF's dequantized experts;
    any object with these two methods swaps in other expert weights, e.g. quantized ones as their float values);
    None: dequantize in this process (slow; for short checks).  `moe_hook(L, mixed [n, 2560], ids [n, 10],
    wt [n, 10])`, if given, sees every layer's MoE input rows and routing (all sequences, packed in order).
    run(seqs, emit) calls emit(i, head_in [n_i, 2560]) per sequence; summarize() turns that into logit statistics."""

    def __init__(self, w, loader=None, moe_hook=None, chunk: int = 4096):
        self.w, self.loader, self.moe_hook, self.chunk = w, loader, moe_hook, chunk
        self.model = w.model

    def hcw(self, prefix: str, inject: bool = True):
        w = self.w
        return (T(w.get(f"{prefix}_norm.weight")), T(w.get(f"{prefix}_down.weight")), T(w.get(f"{prefix}_up.weight")),
                T(w.get(f"{prefix}_inject.weight")) if inject else None)

    # ---- embedding + PLE rows (host)
    def inputs(self, seqs):
        from ..ref.embed import embed
        from ..ref.ple import PleTable, ngram_rows
        tab = PleTable(self.model)
        e = np.concatenate([embed(self.w, s) for s in seqs])
        ep = np.concatenate([tab.gather(ngram_rows(s)[0]) for s in seqs])
        return T(e), T(ep)

    # ---- PLE block (layer 1), per sequence: conv taps t-9, t-6, t-3 over zero history
    def ple(self, R, e_ple, offs):
        w = self.w
        Wk, Wv = T(w.get("blk.1.ple_key.weight")), T(w.get("blk.1.ple_value.weight"))
        gk, gq, gc = (T(w.get(f"blk.1.ple_norm_{s}.weight")) for s in ("key", "query", "conv"))
        Wc = T(w.get("blk.1.ple_conv1d.weight"))                       # (10240, 4)
        scale = float(np.float32(1.0 / np.sqrt(np.float32(N_EMBD))))
        out = torch.empty_like(R)
        for o, n in offs:
            e, r = e_ple[o:o + n], R[o:o + n]
            key = rms_streams((e @ Wk.T).view(n, HC, N_EMBD), gk)
            query = rms_streams(r, gq)
            value = e @ Wv.T
            s = (key * query).sum(-1) * scale
            gate = torch.sigmoid(torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6)))
            gated = value[:, None, :] * gate[:, :, None]
            normalized = rms_streams(gated, gc).reshape(n, HC * N_EMBD)
            ext = torch.cat([torch.zeros(9, HC * N_EMBD, device=DEV), normalized])
            conv = ext[0:n] * Wc[:, 0]
            for k in (1, 2, 3):
                conv = conv + ext[3 * k:3 * k + n] * Wc[:, k]
            out[o:o + n] = r + (gated + silu(conv).view(n, HC, N_EMBD))
        return out

    # ---- GDN: projections and conv per token, the delta-rule recurrence over all sequences in lockstep
    def gdn(self, L, x, offs):
        g = lambda s: T(self.w.layer(L, s))
        Wqkv, Wz, Wa, Wb, Wconv, dt, A, norm, Wout = (g(s) for s in (
            "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
            "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ssm_out.weight"))
        N = x.shape[0]
        HV, DK = SSM_V_HEADS, SSM_STATE
        qkv = x @ Wqkv.T                                               # [N, 10240]
        z = x @ Wz.T
        a, b = x @ Wa.T, x @ Wb.T
        ap = a + dt
        gate = torch.where(ap > 20, ap, torch.log1p(torch.exp(torch.clamp(ap, max=20.0)))) * A
        decay, beta = torch.exp(gate), torch.sigmoid(b)
        h = torch.empty_like(qkv)
        for o, n in offs:
            q = torch.cat([torch.zeros(3, qkv.shape[1], device=DEV), qkv[o:o + n]])
            s = ((q[0:n] * Wconv[:, 0] + q[1:n + 1] * Wconv[:, 1]) + q[2:n + 2] * Wconv[:, 2]) + q[3:n + 3] * Wconv[:, 3]
            h[o:o + n] = silu(s)
        del qkv
        qk = h[:, :2 * QK].reshape(N, 2 * SSM_K_HEADS, DK)
        qk = qk / torch.sqrt((qk * qk).sum(-1, keepdim=True) + EPS)
        v = h[:, 2 * QK:].reshape(N, HV, DK)
        kmap = torch.arange(HV, device=DEV) % SSM_K_HEADS
        B, Tm = len(offs), max(n for _, n in offs)
        tix = torch.cat([torch.arange(n, device=DEV) for _, n in offs])
        bix = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(offs)])
        Qp = torch.zeros(Tm, B, SSM_K_HEADS, DK, device=DEV)
        Kp, Vp = torch.zeros_like(Qp), torch.zeros(Tm, B, HV, DK, device=DEV)
        Dp, Bp = torch.zeros(Tm, B, HV, device=DEV), torch.zeros(Tm, B, HV, device=DEV)
        Qp[tix, bix], Kp[tix, bix], Vp[tix, bix] = qk[:, :SSM_K_HEADS], qk[:, SSM_K_HEADS:], v
        Dp[tix, bix], Bp[tix, bix] = decay, beta
        del h, qk, v
        S = torch.zeros(B, HV, DK, DK, device=DEV)                     # [b, h, i=key, j=value]
        Op = torch.empty(Tm, B, HV, DK, device=DEV)
        inv = 1.0 / math.sqrt(DK)
        for t in range(Tm):
            qt, kt, gt = Qp[t][:, kmap], Kp[t][:, kmap], Dp[t]
            kv = (kt.unsqueeze(-2) @ S).squeeze(-2)                    # sum_i S[i, j] k[i]
            delta = (Vp[t] - gt[..., None] * kv) * Bp[t][..., None]
            S.mul_(gt[..., None, None]).add_(kt.unsqueeze(-1) * delta.unsqueeze(-2))
            Op[t] = (qt.unsqueeze(-2) @ S).squeeze(-2) * inv
        o = Op[tix, bix]                                                # [N, 48, 128]
        del Qp, Kp, Vp, Op, S
        y = o / torch.sqrt((o * o).mean(-1, keepdim=True) + EPS)
        ys = (y * norm * torch.sigmoid(z.view(N, HV, DK))).reshape(N, HV * DK)
        return ys @ Wout.T

    # ---- QSA, dense (every position < 2051 attends all cells 0..p)
    def qsa(self, L, x, offs):
        g = lambda s: T(self.w.layer(L, s))
        Wq, Wk, Wv, Wo, qn, kn = (g(s) for s in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                                                  "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"))
        group = N_HEAD // N_HEAD_KV
        out = torch.empty(x.shape[0], N_EMBD, device=DEV)
        for o, n in offs:
            assert n <= SEL_WIDTH, f"QSA reference is dense only: {n} positions > {SEL_WIDTH}"
            xs = x[o:o + n]
            qf = (xs @ Wq.T).view(n, N_HEAD, 2 * HEAD_DIM)
            k = (xs @ Wk.T).view(n, N_HEAD_KV, HEAD_DIM)
            v = (xs @ Wv.T).view(n, N_HEAD_KV, HEAD_DIM)
            i = torch.arange(ROPE_DIM // 2, dtype=F64, device=DEV)
            ang = torch.arange(n, dtype=F64, device=DEV)[:, None] * ROPE_BASE ** (-2.0 * i / ROPE_DIM)
            c, s = torch.cos(ang).float()[:, None], torch.sin(ang).float()[:, None]

            def rms(t, gam):
                ss = (t * t).sum(-1, keepdim=True)
                return ((1.0 / torch.sqrt(ss / t.shape[-1] + EPS)) * t) * gam

            def rope(t):
                y = t.clone()
                a, b = t[..., :32], t[..., 32:64]
                y[..., :32] = a * c - b * s
                y[..., 32:64] = a * s + b * c
                return y

            K = rope(rms(k, kn)).half().float()
            V = v.half().float()
            q = rope(rms(qf[..., :HEAD_DIM], qn))
            gate = qf[..., HEAD_DIM:]
            mask = torch.ones(n, n, dtype=torch.bool, device=DEV).tril()
            att = torch.empty(n, N_HEAD, HEAD_DIM, device=DEV)
            for j in range(N_HEAD_KV):
                qh = q[:, j * group:(j + 1) * group].transpose(0, 1)                 # [12, n, 256]
                sc = (qh @ K[:, j].T) * (1.0 / 16.0)                                # [12, n, n]
                sc = sc.masked_fill(~mask, float("-inf"))
                e = torch.exp(sc - sc.max(-1, keepdim=True).values)
                att[:, j * group:(j + 1) * group] = ((e @ V[:, j]) / e.sum(-1, keepdim=True)).transpose(0, 1)
            a = att * torch.sigmoid(gate)
            out[o:o + n] = a.reshape(n, N_HEAD * HEAD_DIM) @ Wo.T
        return out

    # ---- MoE (router + top-10 + shared + float routed experts + Strata's combine order)
    def route(self, L, x, chunk: int | None = None):
        """router_top10 on mixed [n, 2560] -> (ids [n, 10] int64 rank order, wt [n, 10] f32)."""
        Wr = T(self.w.layer(L, "ffn_gate_inp.weight"), F64)
        chunk = chunk or self.chunk
        ids, wt = [], []
        for a in range(0, x.shape[0], chunk):
            lg = (x[a:a + chunk].double() @ Wr.T).float()
            mx = lg.max(-1, keepdim=True).values
            e = torch.exp(lg.double() - mx.double())
            inv = (1.0 / e.sum(-1, keepdim=True)).float()
            p = (e * inv.double()).float()
            i = torch.sort(p, dim=-1, descending=True, stable=True).indices[:, :N_USED]
            ws = torch.gather(p, 1, i).double()
            ids.append(i)
            wt.append((ws / torch.clamp(ws.sum(-1, keepdim=True), min=W_CLAMP)).float())
        return torch.cat(ids), torch.cat(wt)

    def moe(self, L, x, ex, chunk: int | None = None):
        """bo = combine(routed experts `ex`, routing weights) + gated shared expert; self.moe_hook(L, x, ids, wt) sees
        every layer's MoE input rows and routing before the experts run."""
        w = self.w
        chunk = chunk or self.chunk
        ids_all, wt_all = self.route(L, x, chunk)
        if self.moe_hook is not None:
            self.moe_hook(L, x, ids_all, wt_all)
        Wsg = T(w.layer(L, "ffn_gate_inp_shexp.weight").reshape(-1), F64)
        Wg, Wu, Wd = (T(w.layer(L, f"ffn_{r}_shexp.weight")) for r in ("gate", "up", "down"))
        out = torch.empty_like(x)
        for a in range(0, x.shape[0], chunk):
            xs, ids, wt = x[a:a + chunk], ids_all[a:a + chunk], wt_all[a:a + chunk]
            n = xs.shape[0]
            sh = silu(xs @ Wg.T) * (xs @ Wu.T) @ Wd.T
            gs = torch.sigmoid((xs.double() @ Wsg).float())
            sh = sh * gs[:, None]
            parts = torch.zeros(n, N_USED, N_EMBD, device=DEV)
            flat = ids.reshape(-1)
            order = torch.argsort(flat, stable=True)
            counts = torch.bincount(flat, minlength=N_EXPERT).tolist()
            pos = 0
            for eid, cnt in enumerate(counts):
                if not cnt:
                    continue
                sel = order[pos:pos + cnt]
                pos += cnt
                t, j = sel // N_USED, sel % N_USED
                xe = xs[t]
                hh = silu(xe @ ex["gate"][eid].T) * (xe @ ex["up"][eid].T)
                parts[t, j] = hh @ ex["down"][eid].T
            r = parts[:, 0] * wt[:, 0:1]
            for j in range(1, N_USED):
                r = (parts[:, j].double() * wt[:, j:j + 1].double() + r.double()).float()
            out[a:a + n] = r + sh
        return out

    # ---- the whole model
    def run(self, seqs, emit) -> None:
        """seqs: token lists; emit(i, head_in [n, 2560]) per sequence after the last layer + output_hc."""
        offs, o = [], 0
        for s in seqs:
            offs.append((o, len(s)))
            o += len(s)
        e, e_ple = self.inputs(seqs)
        R = e[:, None, :].expand(-1, HC, -1).contiguous()
        del e
        t0 = time.time()
        for L in range(N_LAYERS):
            if self.loader is not None:
                self.loader.request(L)
                self.loader.request(L + 1)
            if L == PLE_LAYER:
                R = self.ple(R, e_ple, offs)
                del e_ple
            mixed, inj = hc_read(R, self.hcw(f"blk.{L}.hc_attn"), self.chunk)
            bo = self.qsa(L, mixed, offs) if is_qsa(L) else self.gdn(L, mixed, offs)
            R = hc_write(R, bo, inj)
            del mixed, inj, bo
            mixed, inj = hc_read(R, self.hcw(f"blk.{L}.hc_ffn"), self.chunk)
            t1 = time.time()
            ex = self.loader.get(L) if self.loader is not None else self.experts_direct(L)
            tw = time.time() - t1
            bo = self.moe(L, mixed, ex)
            del ex
            R = hc_write(R, bo, inj)
            del mixed, inj, bo
            torch.cuda.synchronize()
            print(f"torchref: layer {L} done ({time.time() - t0:.0f} s, waited {tw:.1f} s for experts)", flush=True)
        head_in, _ = hc_read(R, self.hcw("output_hc", inject=False), self.chunk)
        del R
        for i, (o, n) in enumerate(offs):
            emit(i, head_in[o:o + n])

    def experts_direct(self, L) -> dict:
        m = self.model
        return {r: T(m.get(f"blk.{L}.ffn_{r}_exps.weight")) for r in ("gate", "up", "down")}

    def head_weight(self) -> torch.Tensor:
        return T(self.w.get("output.weight"))


def summarize(Wh: torch.Tensor, head_in: torch.Tensor, tokens, topk: int, chunk: int = 512) -> dict:
    """head_in [n, 2560] -> the per-position reference record (lse, top-K, target log-prob, argmax)."""
    n = head_in.shape[0]
    tok = torch.as_tensor(np.asarray(tokens, np.int64), device=DEV)
    res = {"lse": np.empty(n, np.float64), "top_ids": np.empty((n, topk), np.int32),
           "top_lp": np.empty((n, topk), np.float32), "tgt_lp": np.full(n, np.nan, np.float32),
           "argmax": np.empty(n, np.int32)}
    for a in range(0, n, chunk):
        lg = head_in[a:a + chunk] @ Wh.T                                     # f32 [c, V]
        b = a + lg.shape[0]
        lse = torch.logsumexp(lg.double(), -1)
        v, ix = torch.topk(lg, topk, dim=-1)
        res["lse"][a:b] = lse.cpu().numpy()
        res["top_ids"][a:b] = ix.int().cpu().numpy()
        res["top_lp"][a:b] = (v.double() - lse[:, None]).float().cpu().numpy()
        res["argmax"][a:b] = torch.argmax(lg, -1).int().cpu().numpy()
        nt = min(b, n - 1) - a
        if nt > 0:
            t = torch.gather(lg[:nt], 1, tok[a + 1:a + 1 + nt, None])[:, 0]
            res["tgt_lp"][a:a + nt] = (t.double() - lse[:nt]).float().cpu().numpy()
    return res


def setup_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def evalset_sha() -> str:
    return hashlib.sha256((data.EVAL / "evalset.jsonl").read_bytes()).hexdigest()


def build(a) -> int:
    from ..weights import Weights
    setup_torch()
    recs = data.load()
    if a.records:
        want = set(a.records.split(","))
        recs = [r for r in recs if r["id"] in want]
    REF.mkdir(parents=True, exist_ok=True)
    w = Weights()
    loader = ExpertLoader(str(w.model.files[0].path), workers=a.workers)
    t0 = time.time()
    heads = {}
    try:
        Ref(w, loader).run([r["token_ids"] for r in recs], lambda i, h: heads.__setitem__(i, h))
    finally:
        loader.close()
    t1 = time.time()
    Wh = Ref(w, None).head_weight()
    for i, r in enumerate(recs):
        res = summarize(Wh, heads.pop(i), r["token_ids"], a.topk)
        np.savez(REF / f"{r['id']}.npz", **res)
    t2 = time.time()
    meta_p = REF / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update({"topk": a.topk, "evalset_sha256": evalset_sha(), "records": sorted(set(meta.get("records", []))
                 | {r["id"] for r in recs}), "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                 "layers_s": round(t1 - t0, 1), "head_s": round(t2 - t1, 1),
                 "tokens": sum(len(r["token_ids"]) for r in recs)})
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"torchref: {len(recs)} records, {meta['tokens']} tokens: layers {t1 - t0:.0f} s, head {t2 - t1:.0f} s; "
          f"wrote {REF}")
    return 0


def check(a) -> int:
    """torch (this module) vs RefModel(numpy, float experts) on the first a.tokens tokens of one record."""
    from ..ref.model import RefModel
    from ..weights import Weights
    setup_torch()
    rec = next(r for r in data.load() if r["id"] == a.record)
    toks = rec["token_ids"][:a.tokens]
    w = Weights()
    t0 = time.time()
    heads = {}
    loader = ExpertLoader(str(w.model.files[0].path), workers=a.workers)
    try:
        Ref(w, loader).run([toks], lambda i, h: heads.__setitem__(i, h))
    finally:
        loader.close()
    Wh = Ref(w, None).head_weight()
    lt = (heads[0] @ Wh.T).double().cpu().numpy()
    t1 = time.time()
    ref = RefModel(w, FloatExperts(w.model), max_ctx=max(64, a.tokens)).forward(toks).astype(np.float64)
    t2 = time.time()
    d = lt - ref
    lp_r = ref - np.logaddexp.reduce(ref, axis=-1, keepdims=True)
    lp_t = lt - np.logaddexp.reduce(lt, axis=-1, keepdims=True)
    kl = (np.exp(lp_r) * (lp_r - lp_t)).sum(-1)
    top_r, top_t = np.argsort(-ref, -1)[:, :TOPK], np.argsort(-lt, -1)[:, :TOPK]
    res = {"record": a.record, "tokens": len(toks), "torch_s": round(t1 - t0, 1), "numpy_s": round(t2 - t1, 1),
           "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ref)),
           "rel_l2_row_max": float((np.linalg.norm(d, axis=1) / np.linalg.norm(ref, axis=1)).max()),
           "max_abs": float(np.abs(d).max()), "logit_absmax": float(np.abs(ref).max()),
           "argmax_agree": float((ref.argmax(-1) == lt.argmax(-1)).mean()),
           "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
           "top256_overlap": float(np.mean([len(set(x) & set(y)) / TOPK for x, y in zip(top_r, top_t)]))}
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "check.json").write_text(json.dumps(res, indent=1))
    meta_p = REF / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta["check"] = res
    meta_p.write_text(json.dumps(meta, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def noise(a) -> int:
    """The float32 reference against itself: the same model with other row chunks (HC, router, shared expert and
    routed-expert batches of a.chunk rows instead of 4096), scored like a TT run against the cached reference ->
    ref/noise.json.  Rounding-level differences that the routing (top-10 flips) and the recurrences amplify: no
    configuration can be told apart from the reference below this floor."""
    from ..weights import Weights
    from . import score
    setup_torch()
    recs = data.load()
    w = Weights()
    loader = ExpertLoader(str(w.model.files[0].path), workers=a.workers)
    heads = {}
    t0 = time.time()
    try:
        Ref(w, loader, chunk=a.chunk).run([r["token_ids"] for r in recs], lambda i, h: heads.__setitem__(i, h))
    finally:
        loader.close()
    Wh = Ref(w, None).head_weight()
    per = {c: [] for c in score.CORPORA}
    (REF / "noise").mkdir(exist_ok=True)
    for i, r in enumerate(recs):
        toks = np.asarray(r["token_ids"], np.int64)
        tg = score.targets(r) - 1
        ref = score.load_ref(r["id"])
        h = heads.pop(i)
        parts = []
        for c0 in range(0, len(toks), 512):
            pos = tg[(tg >= c0) & (tg < c0 + 512)]
            if len(pos):
                lg = (h[torch.as_tensor(pos, device=DEV)] @ Wh.T).cpu().numpy()
                parts.append(score.score_rows(lg, pos, toks, ref))
        m = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
        np.savez(REF / "noise" / f"{r['id']}.npz", **m)
        per[r["corpus"]].append(m)
    tab = score.corpus_table(per)
    res = {"chunk": a.chunk, "seconds": round(time.time() - t0, 1), "table": tab}
    (REF / "noise.json").write_text(json.dumps(res, indent=1))
    print(score.fmt_table(tab))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--topk", type=int, default=TOPK)
    b.add_argument("--records", default="")
    b.add_argument("--workers", type=int, default=12)
    c = sub.add_parser("check")
    c.add_argument("--record", default="chat-ultrachat-test-2")
    c.add_argument("--tokens", type=int, default=64)
    c.add_argument("--workers", type=int, default=12)
    n = sub.add_parser("noise")
    n.add_argument("--chunk", type=int, default=1000)
    n.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    return {"build": build, "check": check, "noise": noise}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
