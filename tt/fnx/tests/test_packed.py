"""A/B of the bank-packed dual-NoC decode matmul (fnx.tt.packed_matmul) against the stock matmuls it replaces, on the
real weights and realistic activations (the trunk's head_in / GDN mixer inputs, fnx/tests/test_hc.py): bitwise output
equality at rows 1..4 and 32 for every variant, then traced us/call.

    tools/devrun.sh packed -m fnx.tests.test_packed [head] [draft] [mega] [--mode gptq4] [--variants] [--time]
                                                    [integrated]

Uses (stock = the pre-packing call, rebuilt here from the same cached converted weights; `integrated` also checks the
modules' own paths: Head.forward, Mtp.head_logits, GdnFused.mega):
  head   LM head W^T [2560, 248320] (--mode, default bfp8): ttnn.linear with fnx.tt.head's 1D decode config
         (110 workers x 71 columns, in0_block_w 2), rows <= 32; prefill rows (33, 64, 128): bfp8 used ttnn's auto
         config, the store modes 32-row slices; packed = 32-row slices.
  draft  MTP draft head W^T [2560, 106304] bfloat8_b: four column groups through ttnn's auto config (1D, in0_block_w 2,
         per_core_N 8/7) + concat, vs one packed matmul over the joined weight (per_core_N 31, in0_block_w 2).
  mega   GDN mega projection of layers 0-2 [2560, 16640] bfloat8_b: stock 11x8 x 6 columns (in0_block_w 2, L1 out) vs
         packed fnx.tt.gdn.MEGA_PC (11x10 x 5 columns, in0_block_w 2).
Every output is float32 (HiFi4, fp32 accumulation, packer L1 accumulation), as in the model.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

import ttnn
from fnx.ref.hc import hc_read, hc_weights
from fnx.tests.test_hc import load_trunk
from fnx.tt.packed_matmul import PackedLayout, PackedMatmul, place_packed

TILE = 32
F32, BF16, BFP8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
ROWS = (1, 2, 3, 4, 32)
# (group, cb_slots, outstanding) explored with --variants; the first is the default
VARIANTS = ((2, 2, 1), (2, 3, 1), (2, 3, 2), (2, 4, 2), (2, 4, 3), (10, 2, 1), (10, 3, 2), (5, 3, 2))


def load() -> str:
    return f"load {os.getloadavg()[0]:.1f}"


def repack(weight, layout: PackedLayout):
    """A packed copy of a DRAM interleaved block-float weight (the source stays)."""
    out = ttnn.empty(list(weight.shape), weight.dtype, ttnn.TILE_LAYOUT, weight.device(), DRAM)
    place_packed(weight, out, 0, layout)
    return out


def bits(t) -> np.ndarray:
    return ttnn.to_torch(t).float().numpy().view(np.uint32)


def upload(mesh, x: np.ndarray, dtype=F32):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(1, 1, *x.shape), dtype=dtype,
                           layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)


def traced_us(mesh, fn, inner: int = 10, reps: int = 20) -> float:
    """Device time per fn() call: `inner` calls captured in one trace, replayed `reps` times (host-timed)."""
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(inner):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (reps * inner) * 1e6
    ttnn.release_trace(mesh, tid)
    return us


def eager_us(mesh, fn, n: int = 20) -> str:
    """Eager us/call of n back-to-back calls: host dispatch / total (max of host and device: the prefill regime)."""
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(fn())
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    return f"{(t1 - t0) / n * 1e6:.1f}/{(time.perf_counter() - t0) / n * 1e6:.1f}"


def ab(mesh, name: str, stock, packed: dict, xs: dict, timing: bool, time_rows=(1, 4)) -> bool:
    """stock(x), packed[variant](x) -> device tensors; bitwise equality per input; traced timings."""
    ok = True
    for tag, x in xs.items():
        ref = stock(x)
        want = bits(ref)
        ttnn.deallocate(ref)
        for v, fn in packed.items():
            out = fn(x)
            got = bits(out)
            ttnn.deallocate(out)
            eq = got.shape == want.shape and np.array_equal(got, want)
            ok &= eq
            nd = int((got != want).sum()) if got.shape == want.shape else -1
            print(f"{name} {tag} {v}: bitwise {'EQUAL' if eq else f'DIFFER ({nd} words)'} shape {tuple(got.shape)}",
                  flush=True)
    if timing:
        for tag, x in xs.items():
            if int(tag.split("=")[1].split()[0]) not in time_rows:
                continue
            line = [f"{name} traced us/call {tag} ({load()}): stock {traced_us(mesh, lambda: stock(x)):.1f}"]
            for v, fn in packed.items():
                line.append(f"{v} {traced_us(mesh, lambda: fn(x)):.1f}")
            print(" | ".join(line), flush=True)
        x = xs[next(iter(xs))]
        line = [f"{name} eager host/total us/call {next(iter(xs))}: stock {eager_us(mesh, lambda: stock(x))}"]
        for v, fn in packed.items():
            line.append(f"{v} {eager_us(mesh, lambda: fn(x))}")
        print(" | ".join(line), flush=True)
    return ok


def variants(args) -> tuple:
    return VARIANTS if "--variants" in args else VARIANTS[:1]


def vname(g, s, o) -> str:
    return f"g{g}s{s}o{o}d"


# ------------------------------------------------------------------------------------------------------------- head
def sliced(mm, N: int):
    """mm on each 32-row slice of x, joined (rows > 32)."""
    def fn(x):
        T = int(x.shape[-2])
        if T <= TILE:
            return mm(x)
        outs = []
        for r0 in range(0, T, TILE):
            xs = ttnn.slice(x, [0, 0, r0, 0], [1, 1, min(r0 + TILE, T), N])
            outs.append(mm(xs))
            ttnn.deallocate(xs)
        out = ttnn.concat(outs, dim=2, memory_config=DRAM)
        for o in outs:
            ttnn.deallocate(o)
        return out
    return fn


def test_head(mesh, w, x_rows: np.ndarray, args) -> bool:
    from fnx.tt.head import (HIFI4, N, N_VOCAB, Head, _host_parts, create_matmul_1d_decode_progcfg, mode_dtype,
                             place_cols)
    mode = args[args.index("--mode") + 1] if "--mode" in args else "bfp8"
    dtype = mode_dtype(mode)
    grid = mesh.compute_with_storage_grid_size()
    pc = create_matmul_1d_decode_progcfg(TILE, N, N_VOCAB, num_cores=min(110, grid.x * grid.y), grid_w=grid.x)
    pc.in0_block_w = 2
    lay = {g: PackedLayout(N // TILE, N_VOCAB // TILE, pc.per_core_N, g) for g in {v[0] for v in variants(args)}}
    t0 = time.time()
    stock_w = ttnn.empty([1, 1, N, N_VOCAB], dtype, ttnn.TILE_LAYOUT, mesh, DRAM)
    packed_w = {g: ttnn.empty([1, 1, N, N_VOCAB], dtype, ttnn.TILE_LAYOUT, mesh, DRAM) for g in lay}
    for v0, host in _host_parts(w, mode):
        part = ttnn.to_device(host, mesh, memory_config=DRAM)
        place_cols(part, stock_w, v0 // TILE)
        for g, pw in packed_w.items():
            place_packed(part, pw, v0 // TILE, lay[g])
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(part)
    print(f"head ({mode}, {dtype}): stock + packed weights built in {time.time() - t0:.1f} s; config 11x10 per_core_N "
          f"{pc.per_core_N} in0_block_w {pc.in0_block_w}; workers {lay[2].workers if 2 in lay else ''}", flush=True)

    def decode(x):
        return ttnn.linear(x, stock_w, program_config=pc, dtype=F32, compute_kernel_config=HIFI4, memory_config=DRAM)

    def stock(x):           # the pre-packing Head.forward: T > 32 bfp8 took ttnn's auto config, store modes slices
        if x.shape[-2] > TILE and mode == "bfp8":
            return ttnn.linear(x, stock_w, dtype=F32, compute_kernel_config=HIFI4, memory_config=DRAM)
        return sliced(decode, N)(x)

    packed = {vname(g, s, o): sliced(PackedMatmul(packed_w[g], pc, HIFI4, F32, DRAM, group=g, cb_slots=s,
                                                  outstanding=o), N) for g, s, o in variants(args)}
    xs = {f"T={T} f32": upload(mesh, x_rows[:T]) for T in ROWS + (33, 64, 128)}
    xs.update({f"T={T} bf16": upload(mesh, x_rows[:T], BF16) for T in (1, 4)})
    ok = ab(mesh, f"head {mode}", stock, packed, xs, "--time" in args)
    if "integrated" in args:          # the module's own path (fnx.tt.head.Head) against the stock call
        hd = Head(mesh, w, mode)
        ok &= ab(mesh, f"head {mode}", stock, {"Head.forward": hd.forward}, xs, "--time" in args)
        ttnn.deallocate(hd.w)
    for t in (stock_w, *packed_w.values(), *xs.values()):
        ttnn.deallocate(t)
    return ok


# ------------------------------------------------------------------------------------------------------------ draft
def test_draft(mesh, w, x_rows: np.ndarray, args) -> bool:
    from fnx.ref.mtp import MtpWeights
    from fnx.tt.head import place_cols
    from fnx.tt.mtp import HIFI4, Mtp, draft_head_parts
    mw = MtpWeights()
    vp, parts = draft_head_parts(mw, w)
    K, groups = 2560, 4                    # the stock path: HEAD_GROUPS = 4 joined column groups, auto config each
    per = -(-len(parts) // groups)
    t0 = time.time()
    stock_ws, joined = [], ttnn.empty([1, 1, K, vp], BFP8, ttnn.TILE_LAYOUT, mesh, DRAM)
    col = 0
    for g0 in range(0, len(parts), per):
        group = parts[g0:g0 + per]
        width = sum(int(t.shape[-1]) for t in group)
        gw = ttnn.empty([1, 1, K, width], BFP8, ttnn.TILE_LAYOUT, mesh, DRAM)
        gc = 0
        for t in group:
            part = ttnn.to_device(t, mesh, memory_config=DRAM)
            place_cols(part, gw, gc // TILE)
            place_cols(part, joined, col // TILE)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(part)
            gc += int(t.shape[-1])
            col += int(t.shape[-1])
        stock_ws.append(gw)
    grid = mesh.compute_with_storage_grid_size()
    pcn = -(-(-(-vp // (grid.x * grid.y))) // TILE)       # ttnn's auto 1D choice for the joined weight
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=2, out_subblock_h=1,
        out_subblock_w=max(i for i in range(1, 5) if pcn % i == 0), per_core_M=1, per_core_N=pcn, fuse_batch=True,
        fused_activation=None, mcast_in0=True)
    lay = {g: PackedLayout(K // TILE, vp // TILE, pcn, g) for g in {v[0] for v in variants(args)}}
    packed_w = {g: repack(joined, lay[g]) for g in lay}
    print(f"draft: Vp {vp}, stock groups {[int(t.shape[-1]) for t in stock_ws]}, joined per_core_N {pcn} -> "
          f"{lay[2].workers if 2 in lay else ''} workers; built in {time.time() - t0:.1f} s", flush=True)

    def stock(x):
        outs = [ttnn.linear(x, gw, dtype=F32, compute_kernel_config=HIFI4, memory_config=DRAM) for gw in stock_ws]
        out = ttnn.concat(outs, dim=-1, memory_config=DRAM)
        for o in outs:
            ttnn.deallocate(o)
        return out

    packed = {"joined-auto": lambda x: ttnn.linear(x, joined, dtype=F32, compute_kernel_config=HIFI4,
                                                   memory_config=DRAM)}
    packed.update({vname(g, s, o): PackedMatmul(packed_w[g], pc, HIFI4, F32, DRAM, group=g, cb_slots=s,
                                                outstanding=o) for g, s, o in variants(args)})
    xs = {f"T={T} f32": upload(mesh, x_rows[:T]) for T in ROWS}
    ok = ab(mesh, "draft", stock, packed, xs, "--time" in args, time_rows=(1,))
    if "integrated" in args:
        md = Mtp.__new__(Mtp)
        md.mesh, md.mw, md.w = mesh, mw, w
        md._head(mesh)
        ok &= ab(mesh, "draft", stock, {"Mtp.head_logits": md.head_logits}, xs, "--time" in args, time_rows=(1,))
        ttnn.deallocate(md.head_w)
    for t in (*stock_ws, joined, *packed_w.values(), *xs.values()):
        ttnn.deallocate(t)
    return ok


# ------------------------------------------------------------------------------------------------------------- mega
def test_mega(mesh, w, tr: dict, args) -> bool:
    from fnx.tt.gdn import HIFI4, MEGA_PC, GdnFused, _cached
    # the stock call: the default ttnn.linear choice's blocking on 88 cores (gdn.MEGA_PC before packing)
    stock_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 8), in0_block_w=2, out_subblock_h=1, out_subblock_w=1, per_core_M=1,
        per_core_N=6, fuse_batch=True, fused_activation=None, mcast_in0=True)
    ok = True
    for L in (0, 1, 2):
        def missing():
            raise RuntimeError(f"blk.{L}.gdn.mega_ab2_T not in the TT cache: run the model once")
        stock_w = _cached(mesh, f"blk.{L}.gdn.mega_ab2_T", BFP8, missing)
        K, Nn = int(stock_w.shape[-2]), int(stock_w.shape[-1])
        R = tr[f"R:blk.{L}.hc_attn"]
        mixed = hc_read(R, *hc_weights(w, f"blk.{L}.hc_attn"))[0]
        mixed = np.concatenate([mixed, mixed[::-1] * np.float32(0.9)])
        pcs = {f"pc11x10n{MEGA_PC.per_core_N}": MEGA_PC}
        if "--variants" in args:       # same in0_block_w / per-tile accumulation, the stock 88-core program
            pcs["pc11x8n6"] = stock_pc
        packed, pws = {}, []
        for pname, pc in pcs.items():
            for g, s, o in variants(args):
                pw = repack(stock_w, PackedLayout(K // TILE, Nn // TILE, pc.per_core_N, g))
                pws.append(pw)
                packed[f"{pname}-{vname(g, s, o)}"] = PackedMatmul(pw, pc, HIFI4, F32, L1, group=g, cb_slots=s,
                                                                   outstanding=o)
        if "--variants" in args:       # the stock program on the full grid (unpacked): the config change alone
            packed["stock-pc11x10n5"] = lambda x: ttnn.linear(x, stock_w, dtype=F32, memory_config=L1,
                                                              program_config=MEGA_PC, compute_kernel_config=HIFI4)

        def stock(x):
            return ttnn.linear(x, stock_w, dtype=F32, memory_config=L1, program_config=stock_pc,
                               compute_kernel_config=HIFI4)

        xs = {f"T={T} f32": upload(mesh, mixed[:T]) for T in ROWS}
        print(f"mega L{L}: {K}x{Nn}", flush=True)
        ok &= ab(mesh, f"mega L{L}", stock, packed, xs, "--time" in args and L == 0)
        if L == 0:        # harness sensitivity: the unpacked page order through the packed reader must differ
            ctl, ref = PackedMatmul(stock_w, stock_pc, HIFI4, F32, L1)(xs["T=1 f32"]), stock(xs["T=1 f32"])
            differ = not np.array_equal(bits(ctl), bits(ref))
            ok &= differ
            print(f"mega L0 control (unpacked weight through the packed reader): "
                  f"{'DIFFER (expected)' if differ else 'EQUAL: the harness cannot see the layout'}", flush=True)
            ttnn.deallocate(ctl)
            ttnn.deallocate(ref)
        if "integrated" in args:
            gd = GdnFused(mesh, w, L)
            ok &= ab(mesh, f"mega L{L}", stock, {"GdnFused.mega": gd.mega}, xs, "--time" in args and L == 0)
        for t in (stock_w, *pws, *xs.values()):
            ttnn.deallocate(t)
    return ok


if __name__ == "__main__":
    from fnx.weights import Weights
    args = sys.argv[1:]
    uses = [u for u in ("head", "draft", "mega") if u in args] or ["head", "draft", "mega"]
    w = Weights()
    tr = load_trunk(32)
    x = tr["head_in"]
    x_rows = np.concatenate([x, x[::-1] * np.float32(0.9), x * np.float32(1.1), x[::-1] * np.float32(0.8)])  # 128 rows
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    ok = True
    try:
        print(f"test_packed {uses} ({load()})", flush=True)
        if "mega" in uses:
            ok &= test_mega(mesh, w, tr, args)
        if "draft" in uses:
            ok &= test_draft(mesh, w, x_rows, args)
        if "head" in uses:
            ok &= test_head(mesh, w, x_rows, args)
    finally:
        ttnn.close_mesh_device(mesh)
    print("test_packed", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
