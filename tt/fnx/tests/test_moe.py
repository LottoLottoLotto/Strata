"""test_moe: TT Moe (device router, device shared expert + resident routed experts, host CPU experts for the rest)
against the numpy reference, real weights.

    tools/devrun.sh test_moe fnx/tests/test_moe.py [--layers 0-9] [--tokens 32] [--chunk 4] [--threads 16]
                    [--shared bfloat8_b,bfloat16] [--placement PATH] [--float-tokens 8] [--iters 30]

Inputs are realistic FFN-half mixed rows: real token embeddings (IQ4_XS rows of token_embd) as a 4-stream residual
(each stream = own embedding + a scaled embedding of another token, so the streams differ) read through each layer's
own hc_ffn weights (fnx.ref.hc.hc_read), i.e. RMS-normalized, gated, stream-averaged as in the model.

Resident experts (on the device): --placement (tools/expert_placement.py JSON; default $FNX_EXPERT_PLACEMENT, else the
tool's default placement computed here from Strata's profile), so a call mixes device and CPU slots.

Checks per layer and shared-expert dtype (T = --chunk tokens per call, logical [1,1,T,2560] f32 TILE as TtModel
passes them; FAIL on any):
  routing   TT ids (device logits + exact near-tie refinement) == numpy ids as sets; rank-order differences are
            listed (they only change the combine order).  Also reported: the ids a plain top-k of the raw device
            logits gives, with each mismatch's margin (10th - 11th exact logit)
  logits    device router-logit error / (||x|| ||W_e||) <= fnx.tt.moe.PRODUCT_ERR (the refinement's assumption)
  bo        ||bo_tt - bo_ref|| / ||bo_ref|| <= BO_TOL[dtype], where bo_ref is the numpy reference with the device
            slots' expert outputs emulated as the device computes them (bfloat8_b weights - bfp_round, a numpy copy
            of tt-metal's float->bfp8 conversion, checked bitwise against the device tensor - BF16 x and h, f64
            math) and the CPU slots' outputs from CpuExperts (the same rows TT uses: STRATA_IQ_MT_MIN=1).  So bo
            error = shared expert (+ gate) error + routing-weight error + device kernel error.  Also printed: bo
            against the all-CPU reference (the bfp8-vs-CPU-kernel difference of the device slots)
  experts   on the device slots of the first --float-tokens tokens: device-emulated parts vs all-float dequantized
            experts must not be worse than CpuExperts' parts vs all-float (the CPU kernels' Q8_K/Q8_0 error)
  split     over all layers both device and CPU slots occur
  bf16 in   the contract's bf16 mixed (informational): routing mismatches and bo error
  timing    T=1 and T=4 per call, with the placement and with no resident expert (every routed expert on the CPU):
            readback, host routing, device dispatch, CPU experts, combine+upload, total
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("STRATA_IQ_MT_MIN", "1")   # CPU expert rows independent of the window (golden runs use it)

import numpy as np

from fnx.cpu_experts import CpuExperts
from fnx.ref import moe as M
from fnx.ref.hc import hc_read, hc_weights
from fnx.weights import Weights

# bo tolerance per shared-expert weight dtype (the shared expert dominates the error: measured shared-expert matmul
# error ~8.5e-3 bfloat8_b, ~1.7e-3 bfloat16; the shared part is 0.4-1.3x the routed part of bo; the device routed
# experts match their emulation to ~1.3e-3)
BO_TOL = {"bfloat8_b": 1.5e-2, "bfloat16": 4e-3}
F64 = np.float64


def parse_layers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def embeddings(model, toks) -> np.ndarray:
    info = model.info("token_embd.weight")
    assert info.type_name == "IQ4_XS", info.type_name
    rb = 2560 // 256 * 136
    raw = model.raw("token_embd.weight")
    return np.stack([model._dequant("IQ4_XS", raw[t * rb:(t + 1) * rb], 2560) for t in toks])


def realistic_mixed(w, L: int, E: np.ndarray, rng) -> np.ndarray:
    T = E.shape[0]
    other = E[rng.permutation(T)]
    R = E[:, None, :] + 0.5 * rng.standard_normal((T, 4, 1)).astype(np.float32) * other[:, None, :]
    g, Wd, Wu, _ = hc_weights(w, f"blk.{L}.hc_ffn")
    with np.errstate(over="ignore"):
        mixed, _ = hc_read(R.astype(np.float32), g, Wd, Wu)
    return mixed


def rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, F64) - b) / np.linalg.norm(b))


def margins(lg_exact: np.ndarray, k: int = 10) -> np.ndarray:
    s = -np.sort(-lg_exact, axis=-1)
    return s[:, k - 1] - s[:, k]


def bfp_round(a: np.ndarray, bits: int = 7) -> np.ndarray:
    """tt-metal's float -> bfp (blockfloat_common.cpp convert_u32_to_bfp): per 16 consecutive values of a row (a
    TILE face row) the largest exponent is shared, mantissas (hidden bit included) shift down to it and round to
    nearest-even to `bits` bits (7: bfloat8_b, 3: bfloat4_b), saturating; zero/denormal -> 0."""
    a = np.ascontiguousarray(a, np.float32)
    u = a.reshape(-1, 16).view(np.uint32)
    exp = ((u >> 23) & 0xFF).astype(np.int64)
    man = ((u & 0x7FFFFF) | (1 << 23)).astype(np.int64)
    man[exp == 0] = 0
    se = exp.max(axis=1, keepdims=True)
    d = se - exp
    man = np.where(d > 31, 0, man >> np.minimum(d, 31))
    sh = 24 - bits
    rv, m = man & ((1 << sh) - 1), man >> sh
    tie = 1 << (sh - 1)
    m = np.minimum(m + ((rv > tie) | ((rv == tie) & ((m & 1) == 1))), (1 << bits) - 1)
    val = m * np.exp2((se - 127 - (bits - 1)).astype(F64))
    return np.where((u >> 31) == 1, -val, val).reshape(a.shape).astype(np.float32)


def bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16                        # round to nearest even
    return u.astype(np.uint32).view(np.float32)


class DevEmu:
    """Device routed experts as the kernels compute them (module docstring of fnx.tt.moe) and all-float experts."""

    def __init__(self, model, L: int):
        self.model, self.L, self.q, self.f = model, L, {}, {}

    def weights(self, e: int):
        if e not in self.f:
            self.f[e] = [self.model.get_expert(f"blk.{self.L}.ffn_{r}_exps.weight", e) for r in ("gate", "up", "down")]
            self.q[e] = [bfp_round(a).astype(F64) for a in self.f[e]]
        return self.q[e], self.f[e]

    def part(self, e: int, x: np.ndarray, wgt: float) -> np.ndarray:
        """Weighted device output of expert e for one row x (the route weight is applied to h, as the kernel does)."""
        (qg, qu, qd), _ = self.weights(e)
        xb = bf16(x).astype(F64)
        g, u = xb @ qg.T, xb @ qu.T
        return bf16((g / (1 + np.exp(-g))) * u * wgt).astype(F64) @ qd.T

    def part_float(self, e: int, x: np.ndarray) -> np.ndarray:
        _, (fg, fu, fd) = self.weights(e)
        xf = x.astype(F64)
        g, u = xf @ fg.T.astype(F64), xf @ fu.T.astype(F64)
        return ((g / (1 + np.exp(-g))) * u) @ fd.T.astype(F64)


def default_placement() -> list[np.ndarray]:
    p = os.environ.get("FNX_EXPERT_PLACEMENT")
    if p:
        from fnx.tt.moe import load_placement
        return load_placement(p)
    spec = importlib.util.spec_from_file_location("expert_placement",
                                                  Path(__file__).resolve().parents[2] / "tools" / "expert_placement.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    pairs = ep.read_profile(ep.STRATA / "data" / "expert-profile-coder.bin")
    n = int((27.29 - 3.0 - 0.5) * 1e9 // (ep.TILES * ep.TILE_BYTES["bfloat8_b"]))   # the tool's defaults
    mask = ep.resident_mask(pairs, n)
    return [np.nonzero(mask[l])[0].astype(np.int32) for l in range(mask.shape[0])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-9")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shared", default="bfloat8_b,bfloat16")
    ap.add_argument("--placement", default=None)
    ap.add_argument("--float-tokens", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    layers = parse_layers(a.layers)

    w = Weights()
    model = w.model
    need = ["token_embd.weight"] + [f"blk.{L}.{n}" for L in layers for n in (
        "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
        "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight")]
    missing = [n for n in need if not model.available(n)]
    if missing:
        print("not on disk yet:", missing[:8])
        return 2
    if a.placement:
        from fnx.tt.moe import load_placement
        placement = load_placement(a.placement)
    else:
        placement = default_placement()
    experts = CpuExperts(model, 48, a.threads)
    rng = np.random.default_rng(a.seed)
    toks = rng.integers(1000, 150000, size=a.tokens)
    E = embeddings(model, toks)
    print(f"tokens {a.tokens} (chunks of {a.chunk}), layers {layers}, STRATA_IQ_MT_MIN={os.environ['STRATA_IQ_MT_MIN']}, "
          f"cpu threads {a.threads}; resident experts per layer {[len(placement[L]) for L in layers]}")

    # ---- numpy reference (CPU experts) and the expert precision: CPU kernels and device emulation vs all-float
    X, REF, EMU = {}, {}, {}
    ok = True
    print(f"\n[ref] expert parts vs all-float dequantized experts, device slots of the first {a.float_tokens} tokens "
          f"(FAIL if the device's are worse than the CPU kernels')")
    for L in layers:
        X[L] = realistic_mixed(w, L, E, rng)
        tr = {}
        REF[L] = (M.moe(w, L, X[L], experts, trace=tr), tr)
        EMU[L] = DevEmu(model, L)
        res = set(placement[L].tolist())
        slots = [(t, k) for t in range(min(a.float_tokens, a.tokens)) for k in range(10) if int(tr["ids"][t, k]) in res]
        if slots:
            fl = np.stack([EMU[L].part_float(int(tr["ids"][t, k]), X[L][t]) for t, k in slots])
            dv = np.stack([EMU[L].part(int(tr["ids"][t, k]), X[L][t], 1.0) for t, k in slots])
            cp = np.stack([tr["parts"][t, k] for t, k in slots])
            e_dev, e_cpu = rel(dv, fl), rel(cp, fl)
            ok &= e_dev <= e_cpu
            print(f"  L{L}: {len(slots)} device slots: device (bfp8 emulation) {e_dev:.4f}, CPU kernels {e_cpu:.4f} "
                  f"{'OK' if e_dev <= e_cpu else 'FAIL'}  (mixed rms {np.sqrt((X[L] ** 2).mean()):.3f}, |sh|/|routed| "
                  f"{np.linalg.norm(tr['sh']) / np.linalg.norm(tr['routed']):.2f})")
        else:
            print(f"  L{L}: no device slot among the first {a.float_tokens} tokens")

    import torch
    import ttnn
    from fnx.tt.moe import PRODUCT_ERR, Moe

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        def up(x, dtype=ttnn.float32):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x)).reshape(1, 1, *x.shape), dtype=dtype,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def down(t, T):
            return ttnn.to_torch(t).float().numpy().reshape(-1, 2560)[:T]

        def run(mod, x, dtype=ttnn.float32):
            """All tokens of x through mod in chunks; returns per-token arrays."""
            acc = {k: [] for k in ("bo", "ids", "wt", "logits", "gate_logit", "routed", "on_dev", "raw_ids")}
            nref = 0
            for c in range(0, len(x), a.chunk):
                xc = x[c:c + a.chunk]
                acc["bo"].append(down(mod.forward(up(xc, dtype)), len(xc)))
                last = mod.last
                if dtype == ttnn.float32:
                    assert np.array_equal(last["x"], xc), "read-back rows differ from the uploaded f32 mixed"
                for k in ("ids", "wt", "logits", "gate_logit", "routed", "on_dev"):
                    acc[k].append(last[k])
                acc["raw_ids"].append(M.route(last["logits"])[0])
                nref += last["refined"]
            out = {k: np.concatenate(v) for k, v in acc.items()}
            out["refined"] = nref
            return out

        def ref_mixed(L, ids, on_dev):
            """bo_ref with TT's device slots (ids[on_dev]) emulated (docstring), reference route weights: the CPU
            slots combined as the reference does (rank-order fmaf, M.combine), the device slots' weighted parts
            added, + the reference shared expert."""
            _, tr = REF[L]
            on_ref = np.array([np.isin(tr["ids"][t], ids[t][on_dev[t]]) for t in range(len(ids))])
            dev = np.zeros((len(ids), 2560))
            for t, k in zip(*np.nonzero(on_ref)):
                dev[t] += EMU[L].part(int(tr["ids"][t, k]), X[L][t], float(tr["wt"][t, k]))
            return M.combine(tr["parts"], np.where(on_ref, 0, tr["wt"])) + tr["sh"] + dev

        mods = {}
        n_dev = n_cpu = 0
        for sd_name in a.shared.split(","):
            sd = getattr(ttnn, sd_name)
            print(f"\n[tt] shared expert {sd_name}, f32 mixed, T={a.chunk} per call.  thresholds: ids identical (sets), "
                  f"logit err/(|x||W_e|) <= {PRODUCT_ERR:.2e}, bo rel (device slots emulated) <= {BO_TOL[sd_name]:g}")
            tot = dict(set=0, order=0, raw=0, refined=0)
            for L in layers:
                t0 = time.time()
                mods[L] = Moe(mesh, w, L, experts, shared_dtype=sd, resident=placement[L])
                t_init = time.time() - t0
                bo_ref, tr = REF[L]
                x = X[L]
                o = run(mods[L], x)
                ids_ref, lg_ref = tr["ids"], tr["logits"]
                mg = margins(lg_ref)
                set_eq = np.array([set(p) == set(q) for p, q in zip(o["ids"], ids_ref)])
                raw_eq = np.array([set(p) == set(q) for p, q in zip(o["raw_ids"], ids_ref)])
                ord_eq = np.all(o["ids"] == ids_ref, axis=1)
                Wn = np.linalg.norm(w.layer(L, "ffn_gate_inp.weight").astype(F64), axis=1)
                lerr = np.abs(o["logits"] - lg_ref)
                kappa = float((lerr / (np.linalg.norm(x.astype(F64), axis=1)[:, None] * Wn[None, :])).max())
                gl_ref = M.shared_gate_logit(w, L, x)
                werr = max((float(np.abs(o["wt"][t][np.argsort(o["ids"][t])] - tr["wt"][t][np.argsort(ids_ref[t])]).max())
                            for t in np.nonzero(set_eq)[0]), default=0.0)
                bo_mix = ref_mixed(L, o["ids"], o["on_dev"]) if set_eq.all() else bo_ref
                r_bo = rel(o["bo"], bo_mix)
                r_cpu = rel(o["bo"], bo_ref)
                tok_rel = np.linalg.norm(o["bo"] - bo_mix, axis=1) / np.linalg.norm(bo_mix, axis=1)
                layer_ok = bool(set_eq.all()) and kappa <= PRODUCT_ERR and r_bo <= BO_TOL[sd_name]
                ok &= layer_ok
                nd = int(o["on_dev"].sum())
                n_dev += nd
                n_cpu += o["on_dev"].size - nd
                tot["set"] += int((~set_eq).sum()); tot["order"] += int((~ord_eq).sum())
                tot["raw"] += int((~raw_eq).sum()); tot["refined"] += o["refined"]
                print(f"  L{L}: device slots {nd}/{o['on_dev'].size} | ids set-mismatch {int((~set_eq).sum())}/{len(x)} "
                      f"order-diff {int((~ord_eq).sum())} | raw device top-k mismatch {int((~raw_eq).sum())} | refined "
                      f"{o['refined']} | logit err max {lerr.max():.2e} kappa {kappa:.1e} | gate-logit err "
                      f"{np.abs(o['gate_logit'] - gl_ref).max():.1e} | weight err {werr:.1e} | bo rel {r_bo:.3e} (tok max "
                      f"{tok_rel.max():.2e}; vs all-CPU ref {r_cpu:.2e}) | min margin {mg.min():.1e} | init "
                      f"{t_init:.1f}s  {'OK' if layer_ok else 'FAIL'}")
                for t in np.nonzero(~raw_eq)[0]:
                    print(f"     raw-device near tie tok {t}: margin {mg[t]:.3e}, ref has "
                          f"{sorted(set(ids_ref[t]) - set(o['raw_ids'][t]))}, raw device top-k has "
                          f"{sorted(set(o['raw_ids'][t]) - set(ids_ref[t]))} -> refined {'OK' if set_eq[t] else 'WRONG'}")
                for t in np.nonzero(~set_eq)[0]:
                    print(f"     MISMATCH tok {t}: margin {mg[t]:.3e} ref {ids_ref[t]} tt {o['ids'][t]}")
                for t in np.nonzero(set_eq & ~ord_eq)[0][:2]:
                    print(f"     order differs tok {t}: ref {ids_ref[t]} tt {o['ids'][t]} (combine order only)")
            print(f"  total ({sd_name}): set mismatches {tot['set']}, order diffs {tot['order']}, raw-device top-k "
                  f"mismatches {tot['raw']}, refined rows {tot['refined']} of {a.tokens * len(layers)}")
        split_ok = n_dev > 0 and n_cpu > 0
        ok &= split_ok
        print(f"\n[split] device slots {n_dev}, CPU slots {n_cpu} ({n_dev / max(1, n_dev + n_cpu):.1%} on the device) "
              f"{'OK' if split_ok else 'FAIL: not a mixed split'}")

        # ---- the device tensor holds exactly the emulated bfloat8_b weights
        L0 = layers[0]
        dev0 = mods[L0].dev
        if len(dev0.resident):
            e0 = int(dev0.resident[0])
            gu = ttnn.to_torch(ttnn.slice(dev0.gu, [0, 0], [1280, 2560])).float().numpy()
            dn = ttnn.to_torch(ttnn.slice(dev0.dn, [0, 0], [2560, 640])).float().numpy()
            (qg, qu, qd), _ = EMU[L0].weights(e0)
            same = (np.array_equal(gu, np.concatenate([qg, qu]).astype(np.float32))
                    and np.array_equal(dn, qd.astype(np.float32)))
            ok &= same
            print(f"[tt] L{L0} expert {e0}: device bfloat8_b gate/up/down == bfp_round emulation (bitwise): {same}")

        # ---- tokens in the first T rows of a 32-row tensor (forward(mixed, T)): same bo rows as logical-T calls
        xc = X[L0][:a.chunk]
        pad = np.zeros((32, 2560), np.float32)
        pad[:len(xc)] = xc
        bo_logical = down(mods[L0].forward(up(xc)), len(xc))
        bo_padded = down(mods[L0].forward(up(pad), len(xc)), len(xc))
        same = np.array_equal(bo_padded, bo_logical)
        ok &= same
        print(f"[tt] L{L0} padded 32-row input, T={len(xc)}: bo rows identical to the logical-T call: {same}")

        # ---- more than one 32-row group in a call (prefill chunks): same as one call per group
        if a.tokens > 32:
            xa = X[L0]
            bo_all = down(mods[L0].forward(up(xa)), len(xa))
            bo_grp = np.concatenate([down(mods[L0].forward(up(xa[i:i + 32])), len(xa[i:i + 32]))
                                     for i in range(0, len(xa), 32)])
            r = rel(bo_all, bo_grp)
            ok &= r <= 1e-6
            print(f"[tt] L{L0} T={len(xa)} in one call vs per 32-row group: bo rel {r:.2e} (<= 1e-6)")

        # ---- the contract's bf16 mixed (informational: x itself is rounded, so routing can legitimately differ)
        print(f"\n[tt] bf16 mixed (contract dtype), shared {a.shared.split(',')[-1]}, T={a.chunk}")
        for L in layers:
            bo_ref, tr = REF[L]
            o = run(mods[L], X[L], ttnn.bfloat16)
            mis = [t for t in range(len(X[L])) if set(o["ids"][t]) != set(tr["ids"][t])]
            mg = margins(tr["logits"])
            print(f"  L{L}: ids set-mismatch {len(mis)}/{len(X[L])} (margins {[f'{mg[t]:.1e}' for t in mis]}) "
                  f"bo rel {rel(o['bo'], bo_ref):.3e}")

        # ---- timing: the placement vs every routed expert on the CPU
        Lt = layers[0]
        cpu_only = Moe(mesh, w, Lt, experts, shared_dtype=getattr(ttnn, a.shared.split(",")[-1]), resident=[])
        print(f"\n[timing] median us per call, layer {Lt}, {a.iters} calls, CPU threads {a.threads}; readback = dispatch "
              f"router+concat+untilize+typecast + transfer; dev_dispatch = shared expert + metadata upload + device-expert kernels (overlaps "
              f"the CPU experts); cpu = CpuExperts' call on libfnx_cpu's leader thread")
        for name, mod in (("placement", mods[Lt]), ("cpu-only", cpu_only)):
            for T in (1, 4):
                xs = [up(X[Lt][i:i + T]) for i in range(0, a.tokens - T + 1, T)]
                for xx in xs:      # warm: program cache + first-touch page faults of these experts' mmap pages
                    ttnn.to_torch(mod.forward(xx))
                ttnn.synchronize_device(mesh)
                st, hits = [], []
                for i in range(a.iters):
                    mod.forward(xs[i % len(xs)])
                    t0 = time.perf_counter()
                    ttnn.synchronize_device(mesh)
                    us = dict(mod.last["us"])
                    us["final_sync"] = (time.perf_counter() - t0) * 1e6
                    st.append(us)
                    hits.append(mod.last["on_dev"].mean())
                med = {k: float(np.median([s[k] for s in st])) for k in st[0]}
                print(f"  {name:9s} T={T} (device slots {np.mean(hits):.0%}): " +
                      "  ".join(f"{k} {v:.0f}" for k, v in med.items()))
    finally:
        ttnn.close_mesh_device(mesh)
    print("\ntest_moe", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
