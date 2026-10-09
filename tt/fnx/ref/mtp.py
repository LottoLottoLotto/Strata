"""MTP draft layer (speculative decoding), numpy float32 reference of Strata's drafter (spec Part 7 §3, src/core/mtp.cpp).

Weights: the runtime form Strata's engine reads (`--mtp <data>/mtp/rt`, built by tools/mtp_fetch.py -> mtp_pack.py
--experts q2_0 -> mtp_rt.py, setup.py): rt/dense.txt + dense.bin (big projections Q8_0, hyper-connection and router
weights BF16, RMSNorm weights F32 already "1 + w") and rt/draft_vocab.bin (int32 ids of the draft head's rows, the
subset setup chose).  The 512 routed experts (Q2_0 gate/up/down) come from mtp-q2_0.gguf beside rt/ (experts.bin is a
lossless relayout of the same blocks): host CPU through Strata's kernels (fnx.cpu_experts.CpuExperts, layer slot
MTP_LAYER, the canonical Q2_0 pool arithmetic) or all-float dequantized (`float_experts`); the device's BFP4 copy
and the routing profile are this port's own files under FNX_DATA/mtp (fnx.paths.MTP).  The token embedding and the
head rows are the main GGUF's (token_embd IQ4_XS, output.weight Q6_K rows of the subset).

Cell c pairs the main model's final 4-stream residual at position c (after the last layer's write, BEFORE output_hc)
with the token at c+1; RoPE position c; its output predicts the token at c+2 (mtp.hpp:3-7).  One cell, full layer:
  e2  = fc_e @ rms(embed(tok), g_e)                                   [2560]
  hn  = rms(Rin.ravel(), g_h) over all 10240 ("pooled", Strata's default; --mtp-hnorm stream = per-stream rms)
  R   = (fc_h @ hn[c]) + e2  for each stream c                       [4, 2560]
  mixed, inj = hc_read(R, attn HC);  K/V of the cell into the layer's own cache (fp16, Strata's default --kv)
  attention: q/gate = per-head [q | gate] rows of q_proj, q/k per-head RMSNorm, NEOX RoPE (64 dims) at c, dense over
             cells 0..c (window 32768), * sigmoid(gate), o_proj -> bo;  R = hc_write(R, bo, inj)
  MoE: mixed2, inj2 = hc_read(R, mlp HC); router 512 experts (BF16) -> softmax -> top-10, renormalized; routed experts
       (unweighted parts, rank-order fmaf combine) + sigmoid(w_sg . x) * shared(x);  R = hc_write(R, y, inj2)  = R_out
  head: smix = hc_read(R_out, final mixer, no inject); lg = W_out[subset] @ smix; j = argmax (ties -> lowest subset
        index); prob = 1 / sum exp(lg - lg[j]); draft = subset[j]
Rounds (MtpDrafter::draft, mtp.cpp:1224-1352): after a verify window at p with outputs outv and a accepted drafts,
catch-up = the K/V part ("front") for rows t = 0..a (Rin = final_R[p+t], tok = outv[t], cell p+t), then the rest
of the layer for row a -> draft 0; chain step j = 1, 2: the full layer on (R_out of step j-1, draft j-1, cell p+a+j).
Speculative cells are overwritten by later catch-ups before any query reads them, so the cache is positional.
Prompt: K/V only over cells [0, n_prompt - 1) with the next prompt token (mtp.cpp:1116-1222).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .. import paths
from ..config import HC, HEAD_DIM, N_EMBD, N_HEAD, N_HEAD_KV, N_USED
from .embed import embed
from .hc import hc_read, hc_write
from .moe import combine, route, sigmoid, silu
from .qsa import SCALE, rms, rope64, rope_cos_sin

f32, f16, F64 = np.float32, np.float16, np.float64
MTP_DIR = paths.MTP                # this port's MTP files: experts-bfp4-<method>.u8, PROFILE
MTP_LAYER = 48                     # the drafter's slot in a shared CpuExperts (main layers are 0..47)
MTP_EXPERTS = 512                  # the canonical (unpruned) MTP MoE (spec §3.1)
N_FF = 640
BFP4_SHAPE = ((N_EMBD, N_FF), (N_EMBD, N_FF), (N_FF, N_EMBD))     # W^T [in, out] of gate, up, down
BFP4_ROW = N_EMBD * N_FF // 16 + N_EMBD * N_FF // 2               # bytes of one BFP4 store row (fnx.quant.bfp)
BFP4_METHOD = "search"
PROFILE = paths.MTP_PROFILE.name  # MTP_DIR file: routing counts [512] of the drafter's MoE (fnx.tests.test_mtp profile)
GROUP = N_HEAD // N_HEAD_KV
EXPERT_GU = "mtp.layers.0.mlp.experts.gate_up_proj"
EXPERT_DN = "mtp.layers.0.mlp.experts.down_proj"
# hyper-connection reads of the layer, under fnx.ref.hc.hc_weights / fnx.tt.hc.HcRead prefix names
HC_PREFIX = {"mtp.hc_attn": "attn_hyper_connection", "mtp.hc_ffn": "mlp_hyper_connection",
             "mtp.hc_out": "hyper_connection_mixer"}
HC_PART = {"_norm.weight": "hc_norm.weight", "_down.weight": "input_mix_weight_down.weight",
           "_up.weight": "input_mix_weight_up.weight", "_inject.weight": "block_inject_weight.weight"}


def dequant_q8_0(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    b = np.asarray(raw, np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(f32)
    return (b[:, 2:].view(np.int8).astype(f32) * d).reshape(rows, cols)


class MtpWeights:
    """The drafter's weights.  `get(short)` = a dense.txt tensor as float32 (torch [out, in] orientation; 1-D norms
    [n]); `get("mtp.hc_attn_down.weight")`-style names serve the HC reads (fnx.ref.hc.hc_weights / tt.hc.HcRead).
    rt: Strata's MTP runtime directory (the engine's --mtp; default fnx.paths.MTP_RT); `dir` = MTP_DIR."""

    def __init__(self, rt: str | os.PathLike | None = None, vocab_file: str | None = None):
        self.dir = MTP_DIR
        self.rt = rt = Path(rt) if rt else paths.MTP_RT
        self.index = {}
        for line in (rt / "dense.txt").read_text().split("\n"):
            if line.strip():
                name, kind, rows, cols, off, nbytes = line.split()
                self.index[name] = (kind, int(rows), int(cols), int(off), int(nbytes))
        self.dense = np.memmap(rt / "dense.bin", dtype=np.uint8, mode="r")
        self.vocab = np.fromfile(vocab_file or rt / "draft_vocab.bin", dtype=np.int32)
        self._cache: dict[str, np.ndarray] = {}
        self._gguf = None

    def _short(self, name: str) -> str:
        for pre, hf in HC_PREFIX.items():
            if name.startswith(pre):
                part = name[len(pre):]
                if part in HC_PART:
                    return f"{hf}.{HC_PART[part]}"
        return name

    def __contains__(self, name: str) -> bool:
        return self._short(name) in self.index

    def get(self, name: str) -> np.ndarray:
        s = self._short(name)
        if s not in self._cache:
            kind, rows, cols, off, nbytes = self.index[s]
            raw = self.dense[off:off + nbytes]
            if kind == "q8_0":
                a = dequant_q8_0(raw, rows, cols)
            elif kind == "bf16":
                a = (np.asarray(raw).view(np.uint16).astype(np.uint32) << 16).view(f32).reshape(rows, cols)
            elif kind == "f32":
                a = np.asarray(raw).view(f32).reshape(cols).copy()
            else:
                raise ValueError(f"{s}: unknown kind {kind}")
            self._cache[s] = a
        return self._cache[s]

    def hc(self, prefix: str):
        """(gamma [10240], down [320,10240], up [10240,320], inject [4,10240] | None) of an HC read."""
        g = (f"{prefix}{p}" for p in ("_norm.weight", "_down.weight", "_up.weight"))
        inj = f"{prefix}_inject.weight"
        return (*(self.get(n) for n in g), self.get(inj) if inj in self else None)

    # ---- routed experts (mtp-q2_0.gguf, Q2_0)
    @property
    def gguf(self):
        if self._gguf is None:
            from ..gguf_load import Model
            self._gguf = Model(self.rt.parent / "mtp-q2_0.gguf")
        return self._gguf

    def expert(self, e: int, fmt: str = "q2_0") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expert e as float32 (gate [640,2560], up [640,2560], down [2560,640]): fmt "q2_0" = its GGUF values
        dequantized, "bfp4" = the device's values (bfp4_rows, exact)."""
        if fmt == "bfp4":
            from ..quant import bfp
            rows = self.bfp4_rows()
            return tuple(np.ascontiguousarray(bfp.unpack(rows[e, m], "bfp4", *BFP4_SHAPE[m]).T) for m in range(3))
        gu = self.gguf.get_expert(EXPERT_GU, int(e))                 # (1280, 2560): gate rows 0..639, up 640..1279
        return gu[:N_FF], gu[N_FF:], self.gguf.get_expert(EXPERT_DN, int(e))

    def bfp4_rows(self, method: str = BFP4_METHOD) -> np.ndarray:
        """The routed experts for the device (fnx.tt.mtp): fnx.quant BFP4 store rows of W^T [in, out] quantized from
        their Q2_0 values, uint8 [512, 3 (gate, up, down), 921600], built once into fnx.paths.mtp_bfp4(method)
        (memory-mapped).  method "search": per-block min-MSE shared exponent (fnx.quant.methods.search; the Q2_0
        values are 4-level per 64-input block with a fp16 scale, so a BFP4 block of 16 output rows cannot hold them
        exactly: 7.7% rel weight error vs 8.0% for "rtn", tt-metal's own conversion)."""
        key = f"bfp4_{method}"
        if key not in self._cache:
            path = paths.mtp_bfp4(method)
            shape = (MTP_EXPERTS, 3, BFP4_ROW)
            if not path.exists():
                self._build_bfp4(path, shape, method)
            self._cache[key] = np.memmap(path, np.uint8, "r", shape=shape)
        return self._cache[key]

    def _build_bfp4(self, path: Path, shape: tuple, method: str) -> None:
        import torch

        from ..quant import bfp, methods
        quant = {"rtn": methods.rtn, "search": methods.search}[method]
        tmp = path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        out = np.memmap(tmp, np.uint8, "w+", shape=shape)
        for e in range(MTP_EXPERTS):
            for m, W in enumerate(self.expert(e)):
                q = quant(torch.from_numpy(W), "bfp4")                     # snapped [out, in]
                out[e, m] = bfp.pack(q.T.contiguous(), "bfp4")
        out.flush()
        del out
        os.replace(tmp, path)

    def expert_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GGUF expert-stack Q2_0 bytes for CpuExperts.add_layer: gate, up [512*640*720] and down [512*2560*180]
        (the fused gate_up stack split into its two halves)."""
        gu = self.gguf.raw(EXPERT_GU).reshape(MTP_EXPERTS, 2 * N_FF, -1)
        return (np.ascontiguousarray(gu[:, :N_FF]).reshape(-1), np.ascontiguousarray(gu[:, N_FF:]).reshape(-1),
                np.ascontiguousarray(self.gguf.raw(EXPERT_DN)))

    def head_rows(self, w) -> np.ndarray:
        """W_out rows of the draft vocabulary, float32 [n_dvocab, 2560] (from the main model's dequantized head)."""
        if "head" not in self._cache:
            W = w.get("output.weight")
            self._cache["head"] = np.ascontiguousarray(W[np.sort(self.vocab)])[np.argsort(np.argsort(self.vocab))]
        return self._cache["head"]


def register_experts(experts, mw: MtpWeights) -> None:
    """Add the drafter's routed experts as layer MTP_LAYER of a shared CpuExperts (built with capacity >= 49)."""
    experts.add_layer(MTP_LAYER, *mw.expert_rows(), "Q2_0", "Q2_0")


class MtpState:
    """The drafter's own K/V cache (positional: cell c, fp16, post-norm post-RoPE K)."""

    def __init__(self, max_cells: int = 8192):
        self.max_cells = max_cells
        self.K = np.zeros((max_cells, N_HEAD_KV, HEAD_DIM), f16)
        self.V = np.zeros((max_cells, N_HEAD_KV, HEAD_DIM), f16)

    def reset(self) -> None:
        self.K[:] = 0
        self.V[:] = 0


class MtpRef:
    """Numpy drafter.  w: fnx.weights.Weights (main model: embedding, head rows); experts: CpuExperts with the MTP
    layer registered (register_experts), or None with float_experts=True (all-float experts: expert_fmt "q2_0" = the
    GGUF values dequantized, "bfp4" = the device's BFP4 values, MtpWeights.expert)."""

    def __init__(self, w, mw: MtpWeights | None = None, experts=None, float_experts: bool = False,
                 max_cells: int = 8192, hnorm: str = "pooled", window: int = 32768, expert_fmt: str = "q2_0"):
        self.w, self.mw = w, mw or MtpWeights()
        self.experts, self.float_experts, self.expert_fmt = experts, float_experts, expert_fmt
        if experts is None and not float_experts:
            raise ValueError("MtpRef: pass a CpuExperts with the MTP layer, or float_experts=True")
        if hnorm not in ("pooled", "stream"):
            raise ValueError(hnorm)
        self.hnorm, self.window = hnorm, window
        g = self.mw.get
        self.g_e, self.g_h = g("pre_fc_norm_embedding.weight"), g("pre_fc_norm_hidden.weight")
        self.fc_e, self.fc_h = g("fc_embedding.weight"), g("fc_hidden.weight")
        self.hc_attn, self.hc_ffn, self.hc_out = (self.mw.hc(p) for p in HC_PREFIX)
        self.wq, self.wk, self.wv, self.wo = (g(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
        self.wr, self.wsg = g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight").reshape(-1)
        self.sh = tuple(g(f"mlp.shared_expert.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.state = MtpState(max_cells)
        self._experts_f32: dict[int, tuple] = {}
        self.last: dict = {}

    def reset(self) -> None:
        self.state.reset()

    # ---- the layer
    def front(self, Rin: np.ndarray, tokens, cells) -> dict:
        """K/V part for T rows: Rin [T,4,2560] f32, tokens [T] (token at cell+1), cells [T].  Appends K/V; returns
        the rows' R (after the input branches), inj and mixed (attention HC read)."""
        Rin = np.asarray(Rin, f32).reshape(-1, HC, N_EMBD)
        cells = np.asarray(cells, np.int64).reshape(-1)
        T = Rin.shape[0]
        e2 = (rms(embed(self.w, tokens), self.g_e) @ self.fc_e.T).astype(f32)               # [T, 2560]
        if self.hnorm == "pooled":
            hn = rms(Rin.reshape(T, HC * N_EMBD), self.g_h).reshape(T, HC, N_EMBD)
        else:
            hn = rms(Rin, self.g_h.reshape(HC, N_EMBD))
        R = ((hn @ self.fc_h.T) + e2[:, None, :]).astype(f32)
        mixed, inj = hc_read(R, *self.hc_attn)
        c, s = rope_cos_sin(cells)
        k = rope64(rms((mixed @ self.wk.T).reshape(T, N_HEAD_KV, HEAD_DIM), self.kn), c[:, None], s[:, None])
        v = (mixed @ self.wv.T).reshape(T, N_HEAD_KV, HEAD_DIM)
        self.state.K[cells] = k.astype(f16)
        self.state.V[cells] = v.astype(f16)
        return {"R": R, "inj": inj, "mixed": mixed}

    def attend(self, mixed: np.ndarray, cell: int) -> np.ndarray:
        """bo [2560] of one row (mixed [2560]) at `cell`: dense attention over cells [max(0, cell+1-window), cell]."""
        qf = (mixed @ self.wq.T).reshape(N_HEAD, 2 * HEAD_DIM)
        c, s = rope_cos_sin([cell])
        q = rope64(rms(qf[:, :HEAD_DIM], self.qn), c, s)
        lo = max(0, cell + 1 - self.window) if self.window > 0 else 0
        K = self.state.K[lo:cell + 1].astype(f32)
        V = self.state.V[lo:cell + 1].astype(f32)
        att = np.empty((N_HEAD, HEAD_DIM), f32)
        for j in range(N_HEAD_KV):
            sc = (q[j * GROUP:(j + 1) * GROUP] @ K[:, j].T) * SCALE
            e = np.exp(sc - sc.max(1, keepdims=True))
            att[j * GROUP:(j + 1) * GROUP] = (e @ V[:, j]) / e.sum(1, keepdims=True, dtype=f32)
        o = att * sigmoid(qf[:, HEAD_DIM:])
        return (o.reshape(-1) @ self.wo.T).astype(f32)

    def _expert_parts(self, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
        if not self.float_experts:
            return self.experts.moe(MTP_LAYER, x, ids)
        out = np.zeros((*ids.shape, N_EMBD), f32)
        for e in np.unique(ids):
            if int(e) not in self._experts_f32:
                if len(self._experts_f32) >= 160:                       # bound the host memory (~12 MB per expert)
                    self._experts_f32.pop(next(iter(self._experts_f32)))
                self._experts_f32[int(e)] = self.mw.expert(int(e), self.expert_fmt)
            Wg, Wu, Wd = self._experts_f32[int(e)]
            t, k = np.nonzero(ids == e)
            h = (silu(x[t] @ Wg.T) * (x[t] @ Wu.T)).astype(f32)
            out[t, k] = h @ Wd.T
        return out

    def moe(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """x [T,2560] -> y [T,2560] (spec §3.3 MoE; router f64-accumulated like fnx.ref.moe)."""
        x = np.asarray(x, f32).reshape(-1, N_EMBD)
        logits = (x.astype(F64) @ self.wr.astype(F64).T).astype(f32)                         # [T, 512]
        ids, wt, _ = route(logits, N_USED)
        Wg, Wu, Wd = self.sh
        sh = ((silu(x @ Wg.T) * (x @ Wu.T)).astype(f32) @ Wd.T).astype(f32)
        gs = sigmoid((x.astype(F64) @ self.wsg.astype(F64)).astype(f32))
        parts = self._expert_parts(x, ids)
        y = (combine(parts, wt) + (sh * gs[:, None]).astype(f32)).astype(f32)
        return y, {"ids": ids, "wt": wt, "router_logits": logits}

    def head(self, R: np.ndarray) -> tuple[int, float, np.ndarray]:
        """Draft head of one row R [4,2560] -> (draft token id, draft-layer probability, subset logits)."""
        smix, _ = hc_read(R, *self.hc_out)
        lg = (self.mw.head_rows(self.w) @ smix[0]).astype(f32)
        j = int(np.argmax(np.where(np.isnan(lg), -np.inf, lg)))                 # first maximal subset index
        prob = float(f32(1) / np.exp((lg - lg[j]).astype(f32)).sum(dtype=f32))
        return int(self.mw.vocab[j]), prob, lg

    def rest(self, fr: dict, t: int, cell: int) -> tuple[np.ndarray, int, float]:
        """Rest of the layer for row t of a front() result at `cell` -> (R_out [4,2560], draft, prob)."""
        bo = self.attend(fr["mixed"][t], cell)
        R = hc_write(fr["R"][t:t + 1], bo[None], fr["inj"][t:t + 1])
        mixed2, inj2 = hc_read(R, *self.hc_ffn)
        y, info = self.moe(mixed2)
        R = hc_write(R, y, inj2)
        d, prob, lg = self.head(R)
        self.last = {"bo": bo, "mixed2": mixed2[0], "y": y[0], "R_out": R[0], "logits": lg, **info}
        return R[0], d, prob

    # ---- rounds
    def prefill(self, R_rows: np.ndarray, next_tokens, cell0: int) -> None:
        """K/V of prompt cells cell0.. (R_rows [n,4,2560] = final residual at those positions, next_tokens = the
        tokens at cell+1)."""
        n = len(next_tokens)
        for c in range(0, n, 64):
            self.front(R_rows[c:c + 64], next_tokens[c:c + 64], np.arange(cell0 + c, cell0 + min(c + 64, n)))

    def draft(self, R_rows: np.ndarray, outv, p: int, a: int, steps: int = 3, min_p: float = 0.0,
              trace: list | None = None) -> tuple[list[int], list[float]]:
        """A round after a verify window at p: R_rows [>= a+1, 4, 2560] = final residuals of window rows 0..a,
        outv = the window's picks (outv[t] = token at p+t+1), a = accepted drafts.  Returns up to `steps` drafts
        and their probabilities (min_p > 0: the chain stops after the first draft below it, as Strata does)."""
        fr = self.front(np.asarray(R_rows)[:a + 1], list(outv)[:a + 1], np.arange(p, p + a + 1))
        R, d, prob = self.rest(fr, a, p + a)
        drafts, probs = [d], [prob]
        if trace is not None:
            trace.append(dict(self.last))
        for j in range(1, steps):
            if min_p > 0 and prob < min_p:
                break
            cell = p + a + j
            fr = self.front(R[None], [d], [cell])
            R, d, prob = self.rest(fr, 0, cell)
            drafts.append(d)
            probs.append(prob)
            if trace is not None:
                trace.append(dict(self.last))
        return drafts, probs


def read_strata_drafts(path) -> list[dict]:
    """mtp/gdb_drafts.py lines `p T a outv... | drafts... | probs...` -> [{p, T, a, outv, drafts, probs}]."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        head, d, pr = line.split("|")
        h = [int(x) for x in head.split()]
        out.append({"p": h[0], "T": h[1], "a": h[2], "outv": h[3:3 + h[1]], "drafts": [int(x) for x in d.split()],
                    "probs": [float(x) for x in pr.split()]})
    return out


def read_final_R(path) -> np.ndarray:
    """STRATA_PREFILL_DUMP_R_ALL file (BF16 [n, 10240]) -> float32 [n, 4, 2560]."""
    u = np.fromfile(path, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(f32).reshape(-1, HC, N_EMBD)
