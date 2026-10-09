"""Gated DeltaNet mixer on the P150 (TTNN), Strata decode semantics (spec Part 4 §2; numpy twin: fnx.ref.gdn).

`Gdn(mesh, w, L[, impl]).forward(mixed)`: mixed [1,1,T,2560] TILE (float32 or bfloat16, logical T rows, tokens in
sequence order, any T) -> bo [1,1,T,2560] float32 TILE; the layer's recurrent and conv state advance by T tokens
(same result as T single-token calls). `.reset()` zeroes the state in place; `.state_tensors()` lists the persistent
per-sequence device tensors (what reset() zeroes); `.state_numpy()` returns (S [128 key, 48 v-head, 128 value],
conv [10240, 3]) in fnx.ref.gdn.GdnState layout (GGUF head order). State lives at fixed device addresses, so a T=1
forward can be captured in a trace and replayed. `impl` (default env FNX_GDN_IMPL, else "fused"):

"fused" (GdnFused, default) - three device ops per <= 32 rows, FP32 recurrent state [1, 48, 128 key, 128 value] TILE
in DRAM (measured on L1/L4: 0.225 ms/token traced, 0.27 ms eager T=1, 0.4 ms eager T=8; bo vs refq 3e-3, S 1.1e-3
after 1024 tokens, both below "ops"):
  mega = x @ [qkv | z | a_hi | a_lo | b_hi | b_lo]^T (bfloat8_b [2560, 16640], HiFi4, fp32 out, L1; bank-packed weight
      and dual-NoC reader, fnx.tt.packed_matmul, MEGA_PC): the BF16 a/b weights as two bfloat8_b terms
      (hi = bfp8(w), lo = bfp8(w - hi): residual ~3e-5 of the weight scale)
  -> fused generic_op kernel (fnx/tt/kernels/gdn), one core per v-head h, GGUF grouping (key head h % 16), the
     T rows token by token: conv 4 taps + SiLU, gates g = exp(ssm_a softplus(a + dt)), beta = sigmoid(b), q/k L2
     norms, kv = k S, delta = (v - g kv) beta, S <- g S + k^T delta, o = q S, y = rms_norm(o) ssm_norm sigmoid(z).
     Elementwise math on the SFPU in exact FP32 (state decay included), contractions on the FPU (~10-bit inputs,
     FP32 accumulation); S stays in L1 across the tokens of a call; conv history as raw rows in DRAM.
  -> bo = y @ Wout^T (bfloat8_b [6144, 2560], 1D mcast program).
  Verify windows (TtModel.commit): `.forward(mixed, snap=True)` (T <= 32 rows) also has the kernel save the state
  after row 0 (S and conv history) and rows 1..T-1 as it reads them (raw q/k/v + a/b words per v-head: 1600 B per
  head and row) into buffers of `window_buffers(T)` (allocated once, fixed addresses; ~3.4 MB per layer at T = 4);
  `.rollback(n)` is then one kernel run: that snapshot plus rows 1..n-1 replayed by the same per-row arithmetic (no
  projection), bitwise the n-row forward's state.

"ops" (GdnOps) - plain ttnn ops, float32 state [1, 48, 128 key, 128 value] TILE in DRAM (0.484 ms traced):
  projections, batched over the rows of one pass (<= 32 rows; longer inputs run in 32-row passes):
      [qkv | z] = x @ [Wqkv | Wz]^T  (one bfloat8_b [2560, 16384] weight)
      [a | -b]  = x @ [Wa | -Wb]^T   (bfloat16 [2560, 128]: a in cols 0..47, -b in 64..111)
      decay = exp(ssm_a * log1p(exp(a + dt))), beta = exp(-log1p(exp(-b))) = sigmoid(b)   (one fp32 pipeline)
  conv in the dense head layout [1, T, 80, 128] over [hist (3 rows, oldest first) | qkv], SiLU; hist <- last 3
  heads [1,80,T,128] (q 0..15, k 16..31, v 32..79); q,k L2-normalized (rms_norm trick); q pre-scaled by 1/sqrt(128)
  GGUF v-heads are "tiled": v-head h uses key head h % 16, so the 48-head q/k are concat([q, q, q]) on the head axis
  recurrence on an L1 copy of S, token by token: kv = k S ; delta = (v - g kv) beta ; S <- g S + k^T delta ;
  o = q S  (48-batch matmuls; k^T delta as a K=1 matmul), then S written back to DRAM
  output: y = rms_norm(o) * ssm_norm * sigmoid(z) per v-head, bo = y @ Wout^T  (bfloat8_b [6144, 2560])

"native" (GdnNative) - the qwen36 package's 48-head decode kernels (validated on Qwen3.8-27B, same GDN geometry),
one token per step, BF16 state in L1: ~1.5x faster traced, but BF16 state/decay drift (~9% after 256 tokens):
  mega projection x @ [qkv | 0 pad64 | 0 pad64 | z] (bfloat8_b [2560, 16512], 1D mcast program, bf16 x)
  -> qwen36 native_frontend (conv 4 taps + SiLU + GQA scatter, BF16 3-row history in one core's L1)
  -> ttnn.transformer.chunk_gated_delta_rule(fuse_decode_frontend=True) = prim recurrent_gated_delta_rule
     (l2norm q/k, 1/sqrt(128), sigmoid(b), decay, delta rule, o; BF16 state [1,48,128,128] ROW_MAJOR
     height-sharded over the 110 cores, updated in place); alpha/beta come from a separate BF16 [a | b] projection
     and the gate is precomputed in fp32 (see GdnNative.__init__: the prim's BF16 softplus is 0 below -5)
  -> ttnn rms_norm(o) * ssm_norm * sigmoid(z) (the qwen36 native output kernels hard-code SiLU(z); Flash-Next
     gates with SIGMOID) -> ssm_out (bfloat8_b, 1D mcast program).
  The qwen36 kernels group v-heads as h // 3 (key head kh owns v-heads 3kh..3kh+2); GGUF order is h % 16. The
  weights are permuted at load: device v-head n = GGUF v-head (n % 3) * 16 + n // 3 (qkv v rows, conv v channels,
  z rows, alpha/beta rows, dt, ssm_a, ssm_out columns), so bo is unchanged and only the state order differs.
"""
from __future__ import annotations

import dataclasses
import math
import os
import struct
from pathlib import Path

import numpy as np
import torch
import ttnn

from .. import paths
from ..ref.gdn import CONV_CH, DK, HK, HV, N, QK, VALUE_DIM, gdn_weights
from .packed_matmul import PackedLayout, PackedMatmul, place_packed

EPS = 1e-6
CACHE = paths.TT_CACHE
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
F32, BF16, BFP8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)
AB = 64            # a and b segments of the projections, each padded to two tiles
NH = CONV_CH // DK  # 80 head blocks of the qkv / conv channels (q 0..15, k 16..31, v 32..79)
GDN_TENSORS = ("attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
               "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ssm_out.weight")

# device v-head n <- GGUF v-head PERM[n] (qwen36 grouping h // 3 over GGUF tiling h % 16)
PERM = np.array([(n % 3) * HK + n // 3 for n in range(HV)])
VPERM = (PERM[:, None] * DK + np.arange(DK)).reshape(-1)               # 6144 value channels
CHPERM = np.concatenate([np.arange(2 * QK), 2 * QK + VPERM])          # 10240 qkv / conv channels


def Gdn(mesh, w, L: int, impl: str | None = None):
    impl = impl or os.environ.get("FNX_GDN_IMPL", "fused")
    return {"ops": GdnOps, "native": GdnNative, "fused": GdnFused}[impl](mesh, w, L)


def _dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1]


def _cached(mesh, name: str, dtype, make):
    """Weight as a DRAM TILE tensor, converted once and cached under fnx.paths.TT_CACHE (`make()` on a miss only)."""
    base = CACHE / f"{name}_{_dtype_name(dtype)}"
    hit = any(base.parent.glob(base.name + "_dtype_*.tensorbin"))
    return ttnn.as_tensor(None if hit else make(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                          memory_config=DRAM, cache_file_name=str(base))


def _small(mesh, a: np.ndarray, dtype=F32):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a, np.float32)), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=mesh, memory_config=DRAM)


def _zero_(t) -> None:
    """Zero a device tensor in place (keeps its address, so captured traces stay valid)."""
    host = ttnn.from_torch(torch.zeros(tuple(t.shape), dtype=torch.float32), dtype=t.dtype, layout=t.layout)
    ttnn.copy_host_to_device_tensor(host, t)


def _layer_weights(w, L: int):
    missing = [s for s in GDN_TENSORS if not w.model.available(f"blk.{L}.{s}")]
    if missing:
        raise RuntimeError(f"blk.{L}: GDN tensors not downloaded yet: {missing}")
    return gdn_weights(w, L)


def _decode_progcfg(k: int, n: int, num_cores: int, grid_w: int = 11):
    """1D mcast_in0 program for one tile row of activations (copied from the image's
    models/demos/blackhole/qwen36/tt/tp_common.py::create_matmul_1d_decode_progcfg, m = 1, fp32_acc)."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    k_tiles, n_tiles = k // 32, n // 32
    per_core_k = next(d for d in range(8, 0, -1) if k_tiles % d == 0)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    sub_w = max(i for i in range(1, 5) if per_core_n % i == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows), in0_block_w=per_core_k, out_subblock_h=1, out_subblock_w=sub_w,
        per_core_M=1, per_core_N=per_core_n, fuse_batch=True, fused_activation=None, mcast_in0=True)


class GdnOps:
    def __init__(self, mesh, w, L: int):
        self.mesh, self.L = mesh, L
        W = _layer_weights(w, L)
        p = f"blk.{L}.gdn"
        self.w_qkvz = _cached(mesh, f"{p}.qkvz_T", BFP8, lambda: torch.from_numpy(np.concatenate([W.qkv, W.z]).T.copy()))

        def ab():   # b negated: the gate pipeline below yields sigmoid(b) = exp(-softplus(-b))
            m = np.zeros((N, 2 * AB), np.float32)
            m[:, :HV], m[:, AB:AB + HV] = W.a.T, -W.b.T
            return torch.from_numpy(m)

        self.w_ab = _cached(mesh, f"{p}.ab_negb_T", BF16, ab)
        self.w_out = _cached(mesh, f"{p}.out_T", BFP8, lambda: torch.from_numpy(W.out.T.copy()))
        self.ab_pc, self.out_pc = _decode_progcfg(N, 2 * AB, 4), _decode_progcfg(VALUE_DIM, N, 40)  # T <= 32 rows
        # conv operands in the dense head layout [1, T, 80, 128] (row = 128-channel head block)
        self.taps = [_small(mesh, W.conv[:, i].reshape(1, 1, NH, DK)) for i in range(4)]
        # one pipeline for both per-head scalars: exp(C * log1p(exp(ab + D))) gives
        #   cols 0..47  (a):  exp(ssm_a * softplus(a + dt)) = decay       (C = ssm_a, D = dt)
        #   cols 64..111 (-b): exp(-softplus(-b))           = sigmoid(b)  (C = -1,    D = 0); pad cols C = 0 -> 1
        D, C = np.zeros(2 * AB, np.float32), np.zeros(2 * AB, np.float32)
        D[:HV], C[:HV], C[AB:AB + HV] = W.dt, W.A, -1.0
        self.gate_d, self.gate_c = _small(mesh, D.reshape(1, 1, 1, -1)), _small(mesh, C.reshape(1, 1, 1, -1))
        # rms_norm(eps/128) of a head gives sqrt(128) x / sqrt(sum x^2 + eps): rescale k by 1/sqrt(128), q by 1/128
        # (L2 norm and the 1/sqrt(128) read-out scale)
        self.qk_scale = _small(mesh, np.repeat([1.0 / DK, DK ** -0.5], HK).reshape(1, 2 * HK, 1, 1))
        self.norm = _small(mesh, W.norm.reshape(1, 1, 1, DK))
        # persistent state, updated in place (fixed addresses: trace capture safe)
        self.S = _small(mesh, np.zeros((1, HV, DK, DK), np.float32))
        self.hist = [_small(mesh, np.zeros((1, 1, NH, DK), np.float32)) for _ in range(3)]  # oldest first

    def reset(self) -> None:
        for t in self.state_tensors():
            _zero_(t)

    def state_tensors(self) -> list:
        return [self.S, *self.hist]

    def state_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        S = ttnn.to_torch(self.S).float().numpy().reshape(HV, DK, DK).transpose(1, 0, 2)
        hist = np.stack([ttnn.to_torch(h).float().numpy().reshape(CONV_CH) for h in self.hist], axis=1)
        return np.ascontiguousarray(S), hist

    def uploaded_weights(self, W):
        """Parity hook: GdnWeights `W` with the projections replaced by the values as uploaded (bfloat8_b etc.)."""
        qkvz, ab, out = (ttnn.to_torch(t).float().numpy() for t in (self.w_qkvz, self.w_ab, self.w_out))
        return dataclasses.replace(W, qkv=qkvz[:, :CONV_CH].T.copy(), z=qkvz[:, CONV_CH:].T.copy(),
                                   a=ab[:, :HV].T.copy(), b=-ab[:, AB:AB + HV].T, out=out.T.copy())

    def forward(self, mixed):
        T = int(mixed.shape[-2])
        if T > 32:   # one tile row per call (1D decode programs; M > 32 rows broke the K=1 outer product matmul)
            outs = [self.forward(ttnn.slice(mixed, [0, 0, t0, 0], [1, 1, min(T, t0 + 32), N]))
                    for t0 in range(0, T, 32)]
            return ttnn.concat(outs, dim=2)
        lin = dict(dtype=F32, compute_kernel_config=HIFI4)
        qkvz = ttnn.linear(mixed, self.w_qkvz, memory_config=L1, **lin)                    # [1,1,T,16384]
        ab = ttnn.linear(mixed, self.w_ab, memory_config=L1, program_config=self.ab_pc, **lin)
        qkv = ttnn.reshape(ttnn.slice(qkvz, [0, 0, 0, 0], [1, 1, T, CONV_CH]), [1, T, NH, DK])  # [1,T,80,128]
        z = ttnn.slice(qkvz, [0, 0, 0, CONV_CH], [1, 1, T, CONV_CH + VALUE_DIM])
        ttnn.deallocate(qkvz)

        # per-head decay / beta for all T rows, as columns [1, 48, T, 1]. Not transpose(1, 3) / permute(0, 3, 2, 1):
        # those truncate float32 to 10 mantissa bits (measured 2^-11), and a decay of 0.99999 read as 0.99951
        # biases the long-memory heads (5% state drift after 1024 tokens); transpose(-2, -1) + reshape is exact.
        gb = ttnn.exp(ttnn.multiply(ttnn.log1p(ttnn.exp(ttnn.add(ab, self.gate_d))), self.gate_c))  # [1,1,T,128]
        gb = ttnn.reshape(ttnn.transpose(gb, -2, -1), [1, 2 * AB, T, 1])
        decay = ttnn.slice(gb, [0, 0, 0, 0], [1, HV, T, 1])
        ndecay = ttnn.neg(decay)
        beta = ttnn.slice(gb, [0, AB, 0, 0], [1, AB + HV, T, 1])

        # causal depthwise conv over [hist (oldest first) | qkv], spec order h0 w0 + h1 w1 + h2 w2 + x w3; SiLU
        if T == 1:
            rows = [*self.hist, qkv]
        else:
            X = ttnn.concat([*self.hist, qkv], dim=1)                                     # [1,T+3,80,128]
            rows = [ttnn.slice(X, [0, tap, 0, 0], [1, tap + T, NH, DK]) for tap in range(4)]
        s = ttnn.multiply(rows[0], self.taps[0])
        for tap in range(1, 4):
            s = ttnn.mac(rows[tap], self.taps[tap], s)
        for i, dst in enumerate(self.hist):                                               # h0<-h1, h1<-h2, h2<-x
            ttnn.copy(rows[i + 1] if T == 1 else ttnn.slice(X, [0, T + i, 0, 0], [1, T + i + 1, NH, DK]), dst)
        h = ttnn.silu(s)
        h = ttnn.reshape(h, [1, NH, 1, DK]) if T == 1 else ttnn.permute(h, (0, 2, 1, 3))   # [1,80,T,128]

        # zero tile padding on q/k/v rows keeps the K=1 outer-product matmul below exact (see recurrence)
        qk = ttnn.multiply(ttnn.rms_norm(ttnn.slice(h, [0, 0, 0, 0], [1, 2 * HK, T, DK]), epsilon=EPS / DK,
                                         compute_kernel_config=HIFI4), self.qk_scale)
        q = ttnn.slice(qk, [0, 0, 0, 0], [1, HK, T, DK], pad_value=0.0)
        k = ttnn.slice(qk, [0, HK, 0, 0], [1, 2 * HK, T, DK], pad_value=0.0)
        q48, k48 = ttnn.concat([q, q, q], dim=1), ttnn.concat([k, k, k], dim=1)           # v-head h <- h % 16
        k48t = ttnn.transpose(k48, -2, -1)                                                # [1,48,128,T], 0-padded
        v = ttnn.slice(h, [0, 2 * HK, 0, 0], [1, 2 * HK + HV, T, DK], pad_value=0.0)      # [1,48,T,128]

        # recurrence on an L1 copy of the state (batched matmuls on a DRAM state are ~5x slower), written back once.
        # The rank-1 update k^T delta is a matmul over K = 1: kcol's tile padding is 0 (transpose / pad_value),
        # delta's padding rows are 0 (from the zero-padded k, v rows), so the padded K adds nothing.
        S = ttnn.to_memory_config(self.S, L1)
        mm = dict(compute_kernel_config=HIFI4, memory_config=L1)
        outs = []
        for t in range(T):
            row = (lambda x: x) if T == 1 else \
                (lambda x: ttnn.slice(x, [0, 0, t, 0], [1, HV, t + 1, x.shape[-1]], pad_value=0.0))
            kcol = k48t if T == 1 else ttnn.slice(k48t, [0, 0, 0, t], [1, HV, DK, t + 1], pad_value=0.0)
            kv = ttnn.matmul(row(k48), S, **mm)                                           # [1,48,1,128]
            delta = ttnn.multiply(ttnn.mac(kv, row(ndecay), row(v)), row(beta))           # (v - g kv) beta
            ttnn.multiply(S, row(decay), output_tensor=S)                                 # S <- g S + k^T delta
            ttnn.add(S, ttnn.matmul(kcol, delta, **mm), output_tensor=S)
            outs.append(ttnn.matmul(row(q48), S, **mm))
        ttnn.copy(S, self.S)
        ttnn.deallocate(S)
        o = outs[0] if T == 1 else ttnn.concat(outs, dim=2)                               # [1,48,T,128]

        y = ttnn.rms_norm(o, epsilon=EPS, weight=self.norm, compute_kernel_config=HIFI4)
        y = ttnn.reshape(y if T == 1 else ttnn.permute(y, (0, 2, 1, 3)), [1, 1, T, VALUE_DIM])
        y = ttnn.multiply(y, ttnn.sigmoid(z))
        return ttnn.linear(y, self.w_out, memory_config=DRAM, program_config=self.out_pc, **lin)


class GdnNative:
    MEGA = CONV_CH + 2 * AB + VALUE_DIM      # 16512
    Z0 = CONV_CH + 2 * AB                    # gate (z) columns of the mega row

    def __init__(self, mesh, w, L: int):
        # qwen36 kernels (image package): conv frontend; the fused recurrence prim reads this flag per call
        from models.demos.blackhole.qwen36.tt.gdn.native_frontend import FrontendGeometry, allocate_native_frontend
        os.environ["QWEN_GDN_RECURRENT_FUSED"] = "1"
        grid = mesh.compute_with_storage_grid_size()
        if (grid.x, grid.y) != (11, 10):
            raise RuntimeError(f"GdnNative needs the P150 11x10 worker grid, got {grid.x}x{grid.y}")
        self.mesh, self.L = mesh, L
        W = _layer_weights(w, L)
        p = f"blk.{L}.gdn"

        def mega():   # [qkv | a | b | z]; the a/b columns stay zero: alpha/beta come from the BF16 w_ab below
            m = np.zeros((N, self.MEGA), np.float32)
            m[:, :CONV_CH] = W.qkv[CHPERM].T
            m[:, self.Z0:] = W.z[VPERM].T
            return torch.from_numpy(m)

        def ab():
            m = np.zeros((N, 2 * AB), np.float32)
            m[:, :HV], m[:, AB:AB + HV] = W.a[PERM].T, W.b[PERM].T
            return torch.from_numpy(m)

        self.w_mega = _cached(mesh, f"{p}.mega_grouped_noab_T", BFP8, mega)
        self.w_ab = _cached(mesh, f"{p}.ab_grouped_T", BF16, ab)
        self.w_out = _cached(mesh, f"{p}.out_grouped_T", BFP8, lambda: torch.from_numpy(W.out[:, VPERM].T.copy()))
        self.mega_pc = _decode_progcfg(N, self.MEGA, 44)
        self.out_pc = _decode_progcfg(VALUE_DIM, N, 40)
        self.taps = [_small(mesh, W.conv[CHPERM, i].reshape(1, 1, CONV_CH), BF16) for i in range(4)]
        # The prim computes decay = exp(softplus(g_in + decay_bias) * decay_scale) with a BF16 SFPU softplus that
        # returns 0 below -5 (ckernel_sfpu_softplus.h, BF16 tail clamp): with Flash-Next's |ssm_a| up to 158 that
        # made decay 1.0 instead of ~0.76 on heads with dt ~ -5..-8 (measured). So gate = softplus(a + dt) * ssm_a
        # is computed here in float32 and handed over as g_in = max(-gate * 2^20, 21) with decay_bias 0 and
        # decay_scale -2^-20: softplus passes inputs >= threshold 20 through unchanged and the power-of-two scale is
        # exact, so the prim sees gate rounded to BF16 (|gate| < 2e-5 reads as 2e-5).
        pad = lambda v: np.pad(v, (0, AB - HV)).reshape(1, 1, 1, AB)
        self.dt = _small(mesh, pad(W.dt[PERM]))
        self.A_scaled = _small(mesh, pad(-W.A[PERM] * 2.0 ** 20))
        self.dbias = _small(mesh, np.zeros((HV, 1, 1)), BF16)
        self.dscale = _small(mesh, np.full((HV, 1, 1), -(2.0 ** -20)), BF16)
        self.ones = _small(mesh, np.ones((1, 32, 32), np.float32), BF16)
        self.norm = _small(mesh, W.norm.reshape(1, 1, 1, DK))
        self.fe = allocate_native_frontend(mesh, FrontendGeometry(key_heads=HK, value_heads=HV))

        # persistent state, fixed addresses (qwen36 model.py layout): recurrent [1,48,128,128] BF16 row-major
        # height-sharded (56 rows/core) over all 110 cores; conv [1,3,10240] BF16 row-major on one core,
        # core picked by GDN ordinal (column-major), as qwen36 assigns its per-layer conv pool rows.
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        rec_mc = ttnn.create_sharded_memory_config(
            shape=(math.ceil(HV * DK / (grid.x * grid.y)), DK), core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        gi = L - L // 4
        c = ttnn.CoreCoord(gi // grid.y, gi % grid.y)
        conv_mc = ttnn.create_sharded_memory_config(
            shape=(3, CONV_CH), core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(c, c)}), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        self.state = ttnn.from_torch(torch.zeros(1, HV, DK, DK, dtype=torch.bfloat16), dtype=BF16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=rec_mc)
        self.conv = ttnn.from_torch(torch.zeros(1, 3, CONV_CH, dtype=torch.bfloat16), dtype=BF16,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=conv_mc)

    def reset(self) -> None:
        for t in self.state_tensors():
            _zero_(t)

    def state_tensors(self) -> list:
        return [self.state, self.conv]

    def state_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        Sg = ttnn.to_torch(self.state).float().numpy().reshape(HV, DK, DK)       # [device head n, i, j]
        S = np.empty((DK, HV, DK), np.float32)
        S[:, PERM, :] = Sg.transpose(1, 0, 2)
        hg = ttnn.to_torch(self.conv).float().numpy().reshape(3, CONV_CH)
        hist = np.empty((CONV_CH, 3), np.float32)
        hist[CHPERM] = hg.T
        return S, hist

    def uploaded_weights(self, W):
        """Parity hook: GdnWeights `W` (GGUF order) with the projections replaced by the uploaded values."""
        mega, ab, out = (ttnn.to_torch(t).float().numpy() for t in (self.w_mega, self.w_ab, self.w_out))
        qkv, z, a, b, wo = (np.empty_like(x) for x in (W.qkv, W.z, W.a, W.b, W.out))
        qkv[CHPERM] = mega[:, :CONV_CH].T
        z[VPERM] = mega[:, self.Z0:].T
        a[PERM], b[PERM] = ab[:, :HV].T, ab[:, AB:AB + HV].T
        wo[:, VPERM] = out.T
        return dataclasses.replace(W, qkv=qkv, z=z, a=a, b=b, out=wo)

    def forward(self, mixed):
        from models.demos.blackhole.qwen36.tt.gdn.native_frontend import native_frontend
        T = int(mixed.shape[-2])
        x = mixed if mixed.dtype == BF16 else ttnn.typecast(mixed, BF16)
        ab = ttnn.linear(mixed, self.w_ab, dtype=F32, compute_kernel_config=HIFI4, memory_config=DRAM)   # [1,1,T,128]
        sp = ttnn.log1p(ttnn.exp(ttnn.add(ttnn.slice(ab, [0, 0, 0, 0], [1, 1, T, AB]), self.dt)))
        g_in = ttnn.typecast(ttnn.maximum(ttnn.multiply(sp, self.A_scaled), 21.0), BF16)
        b_in = ttnn.typecast(ttnn.slice(ab, [0, 0, 0, AB], [1, 1, T, 2 * AB]), BF16)
        state_addr = self.state.buffer_address()
        outs = []
        for t in range(T):
            xt = x if T == 1 else ttnn.slice(x, [0, 0, t, 0], [1, 1, t + 1, N])
            gt, bt = (ttnn.slice(s, [0, 0, t, 0], [1, 1, t + 1, HV]) for s in (g_in, b_in))
            mega = ttnn.linear(xt, self.w_mega, memory_config=L1, compute_kernel_config=HIFI4,
                               program_config=self.mega_pc)                                      # [1,1,1,16512]
            q, k, v, _, _ = native_frontend(mega, self.taps, None, self.conv, self.fe)
            o, st = ttnn.transformer.chunk_gated_delta_rule(
                q, k, v, gt, bt, initial_state=self.state, output_final_state=True, chunk_size=32,
                compute_kernel_config=HIFI4, fuse_decode_frontend=True, decay_bias=self.dbias,
                decay_scale=self.dscale, ones=self.ones, output_head_major=True)                 # o [48,1,128]
            if st.buffer_address() != state_addr:
                raise RuntimeError("recurrent_gated_delta_rule did not update the L1 state in place")
            z = ttnn.reshape(ttnn.slice(mega, [0, 0, 0, self.Z0], [1, 1, 1, self.MEGA]), [1, HV, 1, DK])
            y = ttnn.rms_norm(ttnn.reshape(o, [1, HV, 1, DK]), epsilon=EPS, weight=self.norm,
                              compute_kernel_config=HIFI4)
            y = ttnn.reshape(ttnn.multiply(y, ttnn.sigmoid(z)), [1, 1, 1, VALUE_DIM])
            outs.append(ttnn.linear(y, self.w_out, dtype=F32, memory_config=DRAM, compute_kernel_config=HIFI4,
                                    program_config=self.out_pc))
            ttnn.deallocate(mega)
        return outs[0] if T == 1 else ttnn.concat(outs, dim=2)


KDIR = Path(__file__).resolve().parent / "kernels" / "gdn"
TB = 4096                                  # fp32 tile bytes
AB0 = CONV_CH + VALUE_DIM                  # 16384: mega columns [a_hi | a_lo | b_hi | b_lo], 64 wide each
ROWS = 12                                  # compact 32-float rows per conv element: q0..3, k0..3, v0..3
ROW_FLOATS = (ROWS * 128 + 64) // 4        # a saved window row per v-head: 12 compact rows + the a/b words
OUT, SNAP, REPLAY = 1, 2, 4                # kernel modes (gdn_reader / gdn_writer / gdn_compute)
# kernel CBs (gdn_compute.cpp): (indices sharing one buffer, tiles); a CB index has a single producer
FUSED_CBS = (((0,), 8), ((1,), 8), ((2,), 1), ((3,), 8), ((4,), 4), ((5,), 1), ((6, 7), 16), ((8, 9, 25, 26), 16),
             ((10,), 1), ((11,), 1), ((12,), 1), ((13,), 4), ((14,), 4), ((15,), 4), ((16,), 1), ((17,), 1),
             ((18,), 4), ((19,), 4), ((20,), 4), ((21,), 4), ((22,), 1), ((23,), 8), ((24,), 1), ((27,), 1),
             ((28,), 1))
FUSED_TO_DEST = (0, 1, 3, 4, 6, 8, 10, 11, 15, 16, 18, 21, 25)    # SFPU operands: exact FP32 unpack into DEST
# the mega projection's program: the default ttnn.linear choice's blocking (1D mcast, in0_block_w 2: bitwise it) on the
# full 11x10 grid (104 workers x 5 columns), run by fnx.tt.packed_matmul.PackedMatmul over the bank-packed weight
# (traced T=1: stock 11x8 x 6 columns 119.7 us -> packed 110.9 us, fnx/tests/test_packed.py)
MEGA_PC = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(11, 10), in0_block_w=2, out_subblock_h=1, out_subblock_w=1, per_core_M=1,
    per_core_N=5, fuse_batch=True, fused_activation=None, mcast_in0=True)


def _f32_bits(v) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _bfp8_values(a: np.ndarray) -> np.ndarray:
    """`a` [K, n * 32] as stored by a bfloat8_b TILE conversion (host only)."""
    t = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a, np.float32)), dtype=BFP8, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().numpy()


def _unique(io: dict) -> list:
    return list({id(t): t for t in io.values()}.values())


def _fused_program(io: dict, T: int, mode: int, dt_bits, a_bits):
    """The fused kernel over T rows.  `io`: tensors by role (mega, y: forward only; par, hist_in, hist_out, s_in,
    s_out; snap_s, snap_h, rows: verify-window buffers, absent otherwise)."""
    cores = [ttnn.CoreCoord(h % 8, h // 8) for h in range(HV)]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [ttnn.CBDescriptor(total_size=n * TB, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(i, F32, TB, tile) for i in idx])
           for idx, n in FUSED_CBS]
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True, dst_full_sync_en=True)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for i in FUSED_TO_DEST:
        unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode.extend(unpack)
    acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    addr = lambda k: int(io[k].buffer_address()) if k in io else 0
    fwd = mode & OUT
    dram = io["par"]                                  # stand-in accessor for the roles a mode does not use
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "gdn_reader.cpp"), core_ranges=grid,
        compile_time_args=acc(io["mega"] if fwd else dram, io["par"], io["hist_in"], io.get("rows", dram)),
        runtime_args=[(c, [h, addr("mega"), addr("par"), addr("hist_in"), addr("hist_out"), addr("rows"),
                           addr("snap_h"), T, mode]) for h, c in enumerate(cores)],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "gdn_writer.cpp"), core_ranges=grid,
        compile_time_args=acc(io["y"] if fwd else dram, io["s_in"]),
        runtime_args=[(c, [h, addr("y"), addr("s_in"), addr("s_out"), addr("snap_s"), T, mode])
                      for h, c in enumerate(cores)],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "gdn_compute.cpp"), core_ranges=grid,
        runtime_args=[(c, [T, dt_bits[h], a_bits[h], mode]) for h, c in enumerate(cores)], config=cc)
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs, semaphores=[])


class GdnFused:
    MEGA = AB0 + 4 * AB                    # 16640

    def __init__(self, mesh, w, L: int):
        grid = mesh.compute_with_storage_grid_size()
        if grid.x < 11 or grid.y < 10:
            raise RuntimeError(f"GdnFused needs at least the P150's 11x10 worker grid, got {grid.x}x{grid.y}")
        self.mesh, self.L = mesh, L
        W = _layer_weights(w, L)
        p = f"blk.{L}.gdn"

        def mega():   # [qkv | z | a_hi | a_lo | b_hi | b_lo]: a/b (BF16 GGUF) as two bfloat8_b terms each
            m = np.zeros((N, self.MEGA), np.float32)
            m[:, :CONV_CH], m[:, CONV_CH:AB0] = W.qkv.T, W.z.T
            for j, wt in enumerate((W.a, W.b)):
                blk = np.zeros((N, AB), np.float32)
                blk[:, :HV] = wt.T
                hi = _bfp8_values(blk)
                m[:, AB0 + 2 * j * AB:AB0 + (2 * j + 1) * AB] = hi
                m[:, AB0 + (2 * j + 1) * AB:AB0 + (2 * j + 2) * AB] = _bfp8_values(blk - hi)
            return torch.from_numpy(m)

        # packed in place: the packed weight is allocated first, so freeing the converted copy above it leaves no hole
        self.mega_layout = PackedLayout(N // 32, self.MEGA // 32, MEGA_PC.per_core_N)
        self.w_mega = ttnn.empty([N, self.MEGA], BFP8, ttnn.TILE_LAYOUT, mesh, DRAM)
        converted = _cached(mesh, f"{p}.mega_ab2_T", BFP8, mega)
        place_packed(converted, self.w_mega, 0, self.mega_layout)
        ttnn.deallocate(converted)
        self.mega = PackedMatmul(self.w_mega, MEGA_PC, HIFI4, F32, L1)
        self.w_out = _cached(mesh, f"{p}.out_T", BFP8, lambda: torch.from_numpy(W.out.T.copy()))
        self.out_pc = _decode_progcfg(VALUE_DIM, N, 40)
        # per v-head h: conv taps of its rows (q and k of key head h % 16, v of h), then ssm_norm
        par = np.zeros((HV, 4, ROWS, 32), np.float32)
        for h in range(HV):
            kh = h % HK
            ch = np.concatenate([kh * DK + np.arange(DK), QK + kh * DK + np.arange(DK),
                                 2 * QK + h * DK + np.arange(DK)])
            par[h] = W.conv[ch].T.reshape(4, ROWS, 32)
        par = np.concatenate([par.reshape(HV, -1), np.broadcast_to(W.norm, (HV, DK))], axis=1)
        self.params = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(par)), dtype=F32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)
        self.dt_bits = [_f32_bits(v) for v in W.dt]
        self.a_bits = [_f32_bits(v) for v in W.A]
        # persistent state, fixed addresses: S [1, 48, 128 key, 128 value] FP32 tiles; conv history per v-head:
        # 3 elements (oldest first) x the 12 compact rows of its q, k, v channels
        self.S = _small(mesh, np.zeros((1, HV, DK, DK), np.float32))
        self.hist = self._hist_buffer()
        self._win = None           # verify-window buffers (window_buffers): state after row 0, saved rows 1..
        self._progs: dict = {}

    def _hist_buffer(self):
        return ttnn.from_torch(torch.zeros(HV, 3 * ROWS * 32), dtype=F32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh, memory_config=DRAM)

    def window_buffers(self, T: int) -> bool:
        """Make sure verify windows of up to T rows have their buffers (fixed addresses: a trace that contains a
        forward(snap=True) or a rollback() records them): the state after row 0 (S and conv history) and rows
        1..T-1 as the kernel read them.  Grows only; True when it (re)allocated."""
        cap = self._win["cap"] if self._win is not None else 0
        if T - 1 <= cap:
            return False
        if self._win is None:
            self._win = {"snap_s": _small(self.mesh, np.zeros((1, HV, DK, DK), np.float32)),
                         "snap_h": self._hist_buffer()}
        else:
            ttnn.deallocate(self._win["rows"])
        self._win["cap"] = T - 1
        self._win["rows"] = ttnn.from_torch(torch.zeros((T - 1) * HV, ROW_FLOATS), dtype=F32,
                                            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM)
        return True

    def reset(self) -> None:
        for t in self.state_tensors():
            _zero_(t)

    def state_tensors(self) -> list:
        return [self.S, self.hist]

    def state_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        S = ttnn.to_torch(self.S).float().numpy().reshape(HV, DK, DK).transpose(1, 0, 2)
        rows = ttnn.to_torch(self.hist).float().numpy().reshape(HV, 3, ROWS, 32)
        hist = np.empty((CONV_CH, 3), np.float32)
        for h in range(HV):
            hist[2 * QK + h * DK:2 * QK + (h + 1) * DK] = rows[h, :, 8:].reshape(3, DK).T
            if h < HK:
                hist[h * DK:(h + 1) * DK] = rows[h, :, :4].reshape(3, DK).T
                hist[QK + h * DK:QK + (h + 1) * DK] = rows[h, :, 4:8].reshape(3, DK).T
        return np.ascontiguousarray(S), hist

    def uploaded_weights(self, W):
        """Parity hook: GdnWeights `W` with the projections replaced by the values as uploaded."""
        m = self.mega_layout.unpack(ttnn.to_torch(self.w_mega).float()).numpy()
        out = ttnn.to_torch(self.w_out).float().numpy()
        hi_lo = lambda j: m[:, AB0 + j * AB:AB0 + j * AB + HV]
        ab = [(hi_lo(2 * j) + hi_lo(2 * j + 1)).T for j in range(2)]
        return dataclasses.replace(W, qkv=m[:, :CONV_CH].T.copy(), z=m[:, CONV_CH:AB0].T.copy(),
                                   a=np.ascontiguousarray(ab[0]), b=np.ascontiguousarray(ab[1]), out=out.T.copy())

    def _program(self, io: dict, T: int, mode: int):
        key = (tuple((k, int(t.buffer_address())) for k, t in sorted(io.items())), T, mode)
        prog = self._progs.get(key)
        if prog is None:
            if len(self._progs) >= 16:
                self._progs.clear()
            prog = self._progs[key] = _fused_program(io, T, mode, self.dt_bits, self.a_bits)
        return prog

    def _recur(self, x, n: int, snap: bool = False):
        """Mega projection + fused kernel over the n rows of `x` (advances S and the conv history) -> y in L1.
        snap: also keep the state after row 0 and rows 1..n-1 for rollback()."""
        mega = self.mega(x)
        y = ttnn.empty([1, 1, n, VALUE_DIM], F32, ttnn.TILE_LAYOUT, self.mesh, L1)
        io = {"mega": mega, "par": self.params, "hist_in": self.hist, "hist_out": self.hist, "s_in": self.S,
              "s_out": self.S, "y": y}
        mode = OUT
        if snap:
            io.update(snap_s=self._win["snap_s"], snap_h=self._win["snap_h"], rows=self._win["rows"])
            mode |= SNAP
        ttnn.generic_op(_unique(io), self._program(io, n, mode))
        ttnn.deallocate(mega)
        return y

    def _chunks(self, mixed):
        T = int(mixed.shape[-2])
        for t0 in range(0, T, 32):
            n = min(32, T - t0)
            x = mixed if T <= 32 else ttnn.slice(mixed, [0, 0, t0, 0], [1, 1, t0 + n, N])
            yield x, n, x is not mixed

    def forward(self, mixed, snap: bool = False):
        """snap=True: a verify window (T <= 32 rows), to be kept in part by rollback(n_keep) before the next call."""
        T = int(mixed.shape[-2])
        snap = snap and T > 1
        if snap:
            if T > 32:
                raise ValueError(f"GdnFused verify window of {T} rows (at most 32)")
            self.window_buffers(T)
        outs = []
        for x, n, own in self._chunks(mixed):
            y = self._recur(x, n, snap)
            outs.append(ttnn.linear(y, self.w_out, dtype=F32, memory_config=DRAM, program_config=self.out_pc,
                                    compute_kernel_config=HIFI4))
            ttnn.deallocate(y)
            if own:
                ttnn.deallocate(x)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)

    def rollback(self, n_keep: int) -> None:
        """After forward(mixed, snap=True) of T > n_keep rows: S and the conv history <- what the first n_keep rows
        alone leave (TtModel.commit).  One kernel run: the state after row 0 from the snapshot, then rows
        1..n_keep-1 replayed from the saved rows by the forward's own per-row arithmetic (bitwise the n_keep-row
        forward; no projection weights read)."""
        w = self._win
        if w is None or not 1 <= n_keep <= w["cap"]:
            raise ValueError(f"GdnFused.rollback({n_keep}) without a verify window of more rows")
        io = {"par": self.params, "hist_in": w["snap_h"], "hist_out": self.hist, "s_in": w["snap_s"],
              "s_out": self.S, "rows": w["rows"]}
        ttnn.generic_op(_unique(io), self._program(io, n_keep - 1, REPLAY))
