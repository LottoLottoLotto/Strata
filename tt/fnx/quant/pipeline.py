"""Streaming calibration + GPTQ-BFP quantization of every routed expert (RTX 3090, torch, float32).

    cd /var/tmp/strata-tt/flashnext_tt
    /var/tmp/strata-tt/venv/bin/python -m fnx.quant.pipeline capture --layers 2,11       # activation dumps only
    /var/tmp/strata-tt/venv/bin/python -m fnx.quant.pipeline quantize [--weighted 0|1] [--n-min 2560]

The forward is fnx.eval.torchref's float32 port of the published Coder GGUF (dequantized weights everywhere, float
routed experts), driven layer-major over the calibration corpus (fnx.quant.calib_data: 128 calibration + 16 held-out
sequences x 2048 tokens) in batches of sequences whose 4-stream residuals live in pinned host memory between layers.
At each layer the MoE input rows and their routing (Ref's moe_hook) feed, per expert e:
    Hgu[e] = sum_t s_t x_t x_t^T  (2560^2)      x = mixed row routed to e
    Hd[e]  = sum_t s_t h_t h_t^T  (640^2)       h = silu(Wg x) * (Wu x) with the float gate/up
    s_t = w_te^2 (routing weight squared; --weighted 1, the expert's share of the MoE output error) or 1.
Per-expert Hessians never leave the GPU (all 12288 in fp16 would be ~85 GB): each layer is quantized right after its
statistics are complete. Every expert gets a prior of pad = max(n_min - n, beta * n) pseudo-rows: Hgu += pad * Pgu
(Pgu = the layer's per-token mean of x x^T over all routed rows) and Hd += pad * diag(Hd[e]) / n (down inputs live in
each expert's own space, so only its own diagonal is a prior); fnx.quant.experiments B/B2 measured both.

Quantization per layer, batched over experts: RTN (tt-metal's own conversion) and GPTQ-BFP (act-order, 1% damping,
min-MSE block exponent with clips -1..2 (bfp4) / -1..3 (bfp2)) for bfp4 and bfp2; gate and up share Hgu.

Outputs under FNX_QUANT_DIR (/var/tmp/strata-tt/quant):
  stats/L{LL}.npz    n, w2, f2 (sum w^2 ||f||^2) per expert; Hgu/Hd diagonals; Pgu; residual/MoE scale statistics
  proxy/L{LL}.npz    per expert, method (rtn, gptq) x format (bfp4, bfp2):
                       rel_h_{gate,up,down}_{m}{f}: in-sample tr(dW H dW^T)/tr(W H W^T)
                       rel_x_{gate,up,down}_{m}{f}: held-out output error of the matrix alone (exact inputs)
                       abs_{combo}: held-out sum w^2 ||f_q(x) - f(x)||^2 of the whole expert (w = routing weight),
                         combo = gptq g/u/d formats "444", "442", ..., "222" and "rtn444", "rtn222"; fh2 = sum w^2 ||f||^2
                       pool_{combo}: the same on 512 generic held-out rows (any routing, s = 1), and pool_f2
  store/pool/L{LL}.b4.npy, .b4idx.npy, .b2.npy, .b2idx.npy   GPTQ rows (fnx.quant.store): bfp4 rows for the layer's
                      --pool4 most-routed experts, bfp2 rows for its --pool2 least-routed (calibration counts; Strata's
                      Coder profile ranking is uncorrelated with this model's routing: Spearman ~0 at layers 2 and 11)
  calib/acts/L{LL}.npz   (capture) the layer's calibration and held-out MoE input rows (fp16) + routing
  run.json            timings and settings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..config import HC, N_EMBD, N_EXPERT, N_FF, N_LAYERS, N_USED, PLE_LAYER, is_qsa
from ..eval.torchref import DEV, ExpertLoader, Ref, hc_read, hc_write, setup_torch, silu
from . import bfp, calib_data, methods

QDIR = Path(os.environ.get("FNX_QUANT_DIR", "/var/tmp/strata-tt/quant"))
FMTS = ("bfp4", "bfp2")
COMBOS = ("444", "442", "424", "422", "244", "242", "224", "222")
N_POOL_ROWS = 512


def pool_rows(L: int, tag: str) -> int:
    """Rows already in the pool file of layer L (0 if absent): a layer whose pool lacks rows the current --pool4
    asks for is re-quantized on resume."""
    p = QDIR / "store" / "pool" / f"L{L:02d}.{tag}idx.npy"
    return len(np.load(p)) if p.exists() else 0


def shm_hygiene() -> None:
    """torchref.ExpertLoader prefetches layers into /dev/shm/fnx-eval-<pid>-*.npy (~5 GB per layer) and removes them
    in close(). Make SIGTERM/SIGHUP run that close (raise SystemExit so `finally` blocks execute) and delete files
    left behind by dead processes (e.g. a SIGKILLed run)."""
    import signal

    def _exit(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGHUP, _exit)
    for f in Path("/dev/shm").glob("fnx-eval-*.npy"):
        try:
            pid = int(f.name.split("-")[2])
            os.kill(pid, 0)
        except ProcessLookupError:
            f.unlink(missing_ok=True)
        except (ValueError, PermissionError):
            pass


class Stream:
    """Layer-major driver of torchref.Ref over many sequences: residuals in pinned host memory between layers."""

    def __init__(self, w, splits: dict, batch: int):
        self.ref = Ref(w, loader=None, moe_hook=self._hook)
        self.batches, self.hook, self.cur = [], None, None
        for split, seqs in splits.items():
            for i in range(0, len(seqs), batch):
                chunk = seqs[i:i + batch]
                offs, o = [], 0
                for s in chunk:
                    offs.append((o, len(s)))
                    o += len(s)
                e, ep = self.ref.inputs(chunk)
                R = torch.empty((o, HC, N_EMBD), dtype=torch.float32, pin_memory=True)
                R.copy_(e[:, None, :].expand(-1, HC, -1))
                self.batches.append({"split": split, "offs": offs, "R": R, "eple": ep.cpu()})
                del e, ep

    def _hook(self, L, x, ids, wt):
        if self.hook is not None:
            self.hook(self.cur, L, x, ids, wt)

    def layer(self, L: int, ex: dict, scale: dict | None = None) -> None:
        ref = self.ref
        hcA, hcF = ref.hcw(f"blk.{L}.hc_attn"), ref.hcw(f"blk.{L}.hc_ffn")
        for b in self.batches:
            self.cur = b["split"]
            R = b["R"].to(DEV, non_blocking=True)
            if L == PLE_LAYER:
                R = ref.ple(R, b["eple"].to(DEV), b["offs"])
                b["eple"] = None
            mixed, inj = hc_read(R, hcA)
            bo = ref.qsa(L, mixed, b["offs"]) if is_qsa(L) else ref.gdn(L, mixed, b["offs"])
            R = hc_write(R, bo, inj)
            del mixed, inj, bo
            mixed, inj = hc_read(R, hcF)
            bo = ref.moe(L, mixed, ex)
            if scale is not None and b["split"] == "calib":
                gate = 2.0 * torch.sigmoid(inj / HC)
                scale["n"] += R.shape[0]
                scale["r2"] += float((R * R).sum(-1).mean(-1).sum())         # mean over streams of ||R_c||^2
                scale["bo2"] += float((bo * bo).sum(-1).sum())
                scale["g2bo2"] += float(((gate * gate).mean(-1) * (bo * bo).sum(-1)).sum())
            R = hc_write(R, bo, inj)
            b["R"].copy_(R, non_blocking=False)
            del R, mixed, inj, bo


class Acc:
    """Per-expert second moments of one layer (calibration rows)."""

    def __init__(self, ex: dict, weighted: bool):
        self.ex, self.weighted = ex, weighted
        self.Hgu = torch.zeros(N_EXPERT, N_EMBD, N_EMBD, device=DEV)
        self.Hd = torch.zeros(N_EXPERT, N_FF, N_FF, device=DEV)
        z = lambda: torch.zeros(N_EXPERT, dtype=torch.float64, device=DEV)
        self.n, self.w2, self.f2 = z(), z(), z()

    def add(self, x, ids, wt):
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=N_EXPERT).tolist()
        pos = 0
        for e, c in enumerate(counts):
            if not c:
                continue
            sel = order[pos:pos + c]
            pos += c
            t, j = sel // N_USED, sel % N_USED
            xe = x[t]
            we = wt[t, j]
            s = we * we if self.weighted else torch.ones_like(we)
            self.Hgu[e].addmm_((xe * s[:, None]).T, xe)
            h = silu(xe @ self.ex["gate"][e].T) * (xe @ self.ex["up"][e].T)
            self.Hd[e].addmm_((h * s[:, None]).T, h)
            f = h @ self.ex["down"][e].T
            self.n[e] += c
            self.w2[e] += float((we * we).sum())
            self.f2[e] += float(((we * we) * (f * f).sum(-1)).sum())      # MoE-output energy: always w^2

    _host = None      # one pinned 6.7 GB buffer reused by every layer (allocating it costs ~1 min)

    def offload(self) -> None:
        """After the layer's forward: the pooled per-token gate/up second moment (prior) stays on the GPU, the
        256 x 2560^2 Hessians (6.7 GB fp32) move to host memory; quantize_layer streams them back per chunk."""
        self.Pgu = self.Hgu.sum(0) / self.n.sum().float()
        if Acc._host is None:
            Acc._host = torch.empty(self.Hgu.shape, dtype=self.Hgu.dtype, pin_memory=True)
        Acc._host.copy_(self.Hgu)
        self.Hgu = Acc._host


def expert_rows(ids: torch.Tensor, wt: torch.Tensor):
    """Routing -> per expert (token indices, weights) lists."""
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=N_EXPERT).tolist()
    out, pos = [], 0
    for c in counts:
        sel = order[pos:pos + c]
        pos += c
        out.append((sel // N_USED, wt.reshape(-1)[sel]))
    return out


def ffn(Wg, Wu, Wd, x):
    return (silu(x @ Wg.transpose(-1, -2)) * (x @ Wu.transpose(-1, -2))) @ Wd.transpose(-1, -2)


def quantize_layer(L: int, ex: dict, acc: Acc, held: dict, a, sel4: np.ndarray, sel2: np.ndarray, tm: dict) -> dict:
    """GPTQ/RTN in both formats, proxies, packed pool rows. Returns the proxy arrays."""
    n = acc.n
    Pgu = acc.Pgu
    rows = expert_rows(held["ids"], held["wt"])
    xh = held["x"]
    gpool = torch.Generator(device="cpu").manual_seed(1000 + L)
    xp = xh[torch.randperm(xh.shape[0], generator=gpool)[:N_POOL_ROWS].to(DEV)]
    P = {k: np.zeros(N_EXPERT, np.float64) for k in ["n_held", "fh2", "pool_f2", "prior_tokens"]}
    for m in ("rtn", "gptq"):
        for f in FMTS:
            for mat in ("gate", "up", "down"):
                P[f"rel_h_{mat}_{m}{f[3]}"] = np.zeros(N_EXPERT)
                P[f"rel_x_{mat}_{m}{f[3]}"] = np.full(N_EXPERT, np.nan)
    for c in list(COMBOS) + ["rtn444", "rtn222"]:
        P[f"abs_{c}"] = np.zeros(N_EXPERT)
        P[f"pool_{c}"] = np.zeros(N_EXPERT)
    pool = {"bfp4": [], "bfp2": []}
    idx = {"bfp4": [], "bfp2": []}
    tr_perm, tr_k = [], []
    C = a.chunk
    for e0 in range(0, N_EXPERT, C):
        E = list(range(e0, min(e0 + C, N_EXPERT)))
        Wg, Wu, Wd = ex["gate"][E], ex["up"][E], ex["down"][E]
        Ho_gu = acc.Hgu[E].to(DEV, non_blocking=True)       # the layer's Hessians live on the host (Acc.offload)
        Hgu, Hd = Ho_gu.clone(), acc.Hd[E].clone()
        for i, e in enumerate(E):
            ne = float(n[e])
            pad = max(a.n_min - ne, a.beta * ne, 0.0)
            if pad > 0:
                P["prior_tokens"][e] = pad
                Hgu[i] += pad * Pgu
                dd = Hd[i].diagonal()
                # down inputs live in the expert's own space: its own per-token diagonal is the prior (identity if
                # the expert saw no calibration row at all)
                prior = dd / ne if ne > 0 else torch.ones_like(dd)
                Hd[i] += torch.diag(pad * prior)
        # references: RTN quantizes the published experts as they are; GPTQ quantizes the (exactly equivalent)
        # channel-transformed experts (--transform), whose rows are what the pool stores
        # (Ho_gu: the unpadded Hessians, for the in-sample proxies)
        REF = {"rtn": (Wg, Wu, Wd, acc.Hd[E])}
        if a.transform != "none":
            perm, kk = methods.channel_transform(Wg, Wu, a.transform)
            tg, tu, td, Hd = methods.apply_transform(Wg, Wu, Wd, Hd, perm, kk)
            REF["gptq"] = (tg, tu, td, methods.apply_transform(Wg, Wu, Wd, acc.Hd[E], perm, kk)[3])
            tr_perm.append(perm.to(torch.int16).cpu().numpy())
            tr_k.append(kk.to(torch.int8).cpu().numpy())
        else:
            REF["gptq"] = REF["rtn"]
        Q = {}
        t0 = time.time()
        for f in FMTS:
            Q[("rtn", f)] = (methods.rtn(Wg, f), methods.rtn(Wu, f), methods.rtn(Wd, f))
            tg, tu, td, _ = REF["gptq"]
            q = methods.gptq(torch.cat([tg, tu], 1), Hgu, f)
            Q[("gptq", f)] = (q[:, :N_FF], q[:, N_FF:], methods.gptq(td, Hd, f))
            del q
        torch.cuda.synchronize()
        tm["gptq"] += time.time() - t0
        t0 = time.time()
        for (m, f), (qg, qu, qd) in Q.items():
            k = f"{m}{f[3]}"
            rg, ru, rd, rHd = REF[m]
            P[f"rel_h_gate_{k}"][E] = methods.rel_err_h(rg, qg, Ho_gu).double().cpu().numpy()
            P[f"rel_h_up_{k}"][E] = methods.rel_err_h(ru, qu, Ho_gu).double().cpu().numpy()
            P[f"rel_h_down_{k}"][E] = methods.rel_err_h(rd, qd, rHd).double().cpu().numpy()
        # generic rows (pooled), batched over the chunk
        f0 = ffn(Wg, Wu, Wd, xp)                                                  # [c, 512, 2560]
        P["pool_f2"][E] = (f0 * f0).sum((-1, -2)).double().cpu().numpy()

        def combo_q(cmb):
            if cmb.startswith("rtn"):
                f = "bfp4" if cmb == "rtn444" else "bfp2"
                return Q[("rtn", f)]
            fm = {"4": "bfp4", "2": "bfp2"}
            return (Q[("gptq", fm[cmb[0]])][0], Q[("gptq", fm[cmb[1]])][1], Q[("gptq", fm[cmb[2]])][2])

        for cmb in list(COMBOS) + ["rtn444", "rtn222"]:
            qg, qu, qd = combo_q(cmb)
            d = ffn(qg, qu, qd, xp) - f0
            P[f"pool_{cmb}"][E] = (d * d).sum((-1, -2)).double().cpu().numpy()
        del f0
        # held-out rows routed to each expert
        for i, e in enumerate(E):
            t, we = rows[e]
            P["n_held"][e] = t.numel()
            if t.numel() == 0:
                continue
            x = xh[t]
            s = we * we                     # the expert's share of the MoE output error (independent of --weighted)
            fo = None
            for m in ("rtn", "gptq"):
                rg, ru, rd, _ = REF[m]
                g, u = x @ rg[i].T, x @ ru[i].T
                h = silu(g) * u
                fm_ = h @ rd[i].T
                if fo is None:
                    fo = fm_
                    P["fh2"][e] = float((s * (fo * fo).sum(-1)).sum())
                for f in FMTS:
                    qg, qu, qd = Q[(m, f)]
                    k = f"{m}{f[3]}"
                    dg, du, dd = x @ qg[i].T - g, x @ qu[i].T - u, h @ qd[i].T - fm_
                    P[f"rel_x_gate_{k}"][e] = float((dg * dg).sum() / (g * g).sum().clamp_min(1e-30))
                    P[f"rel_x_up_{k}"][e] = float((du * du).sum() / (u * u).sum().clamp_min(1e-30))
                    P[f"rel_x_down_{k}"][e] = float((dd * dd).sum() / (fm_ * fm_).sum().clamp_min(1e-30))
            for cmb in list(COMBOS) + ["rtn444", "rtn222"]:
                qg, qu, qd = combo_q(cmb)
                d = ffn(qg[i], qu[i], qd[i], x) - fo
                P[f"abs_{cmb}"][e] = float((s * (d * d).sum(-1)).sum())
        torch.cuda.synchronize()
        tm["proxy"] += time.time() - t0
        t0 = time.time()
        for f, sel in (("bfp4", sel4), ("bfp2", sel2)):
            qg, qu, qd = Q[("gptq", f)]
            keep = [i for i, e in enumerate(E) if sel[e]]
            if not keep:
                continue
            r_gu = bfp.pack_batch(torch.cat([qg[keep], qu[keep]]).transpose(1, 2).contiguous(), f)  # [in, out]
            r_d = bfp.pack_batch(qd[keep].transpose(1, 2).contiguous(), f)
            r_gu, r_d = r_gu.cpu().numpy(), r_d.cpu().numpy()
            k = len(keep)
            for j, i in enumerate(keep):
                e = E[i]
                pool[f] += [r_gu[j], r_gu[k + j], r_d[j]]
                idx[f] += [(e, 0), (e, 1), (e, 2)]
        tm["pack"] += time.time() - t0
        del Q, Hgu, Hd, REF, Ho_gu
    pdir = QDIR / "store" / "pool"
    pdir.mkdir(parents=True, exist_ok=True)
    for f, tag in (("bfp4", "b4"), ("bfp2", "b2")):
        arr = np.stack(pool[f]) if pool[f] else np.zeros((0, bfp.row_bytes(f, N_EMBD, N_FF)), np.uint8)
        np.save(pdir / f"L{L:02d}.{tag}.npy", arr)
        np.save(pdir / f"L{L:02d}.{tag}idx.npy", np.asarray(idx[f], np.int32).reshape(-1, 2))
    if tr_perm:   # stored rows are of the transformed experts: up row r scaled 2^k[r], channels permuted by perm
        np.save(pdir / f"L{L:02d}.perm.npy", np.concatenate(tr_perm))
        np.save(pdir / f"L{L:02d}.upk.npy", np.concatenate(tr_k))
    return P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["capture", "quantize"])
    ap.add_argument("--layers", default="", help="capture: layers to dump (comma list); quantize: subset to run")
    ap.add_argument("--batch", type=int, default=8, help="sequences per forward batch")
    ap.add_argument("--chunk", type=int, default=32, help="experts per GPTQ batch")
    ap.add_argument("--weighted", type=int, default=1)
    ap.add_argument("--n-min", type=int, default=2560)
    ap.add_argument("--beta", type=float, default=0.0, help="prior pseudo-rows per calibration row (all experts)")
    ap.add_argument("--transform", default="none", choices=["none", "scale", "perm", "scale+perm"],
                    help="exact intermediate-channel transform before GPTQ (methods.channel_transform)")
    ap.add_argument("--pool4", type=float, default=1.0, help="store bfp4 rows of the layer's most-routed fraction")
    ap.add_argument("--pool2", type=float, default=0.40, help="store bfp2 rows of the layer's least-routed fraction")
    ap.add_argument("--save-h", default="", help="layers whose per-expert Hessians are saved (fp16 upper triangles)")
    ap.add_argument("--n-calib", type=int, default=0, help="limit calibration sequences (0 = all)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    from ..weights import Weights
    setup_torch()
    shm_hygiene()
    want = sorted({int(x) for x in a.layers.split(",") if x.strip()})
    last = max(want) if (a.cmd == "capture" or want) and want else N_LAYERS - 1
    cal = [s["token_ids"] for s in calib_data.load("calib")]
    if a.n_calib:
        cal = cal[:a.n_calib]
    held = [s["token_ids"] for s in calib_data.load("heldout")]
    w = Weights()
    t_start = time.time()
    st = Stream(w, {"calib": cal, "held": held}, a.batch)
    print(f"pipeline: {len(cal)} calib + {len(held)} held-out sequences, inputs {time.time() - t_start:.0f} s",
          flush=True)
    n4, n2 = int(round(a.pool4 * N_EXPERT)), int(round(a.pool2 * N_EXPERT))
    loader = ExpertLoader(str(w.model.files[0].path), workers=a.workers)
    run_p = QDIR / f"run-{a.cmd}.json"
    log = {"cmd": a.cmd, "args": vars(a), "layers": {}}
    if a.cmd == "quantize" and run_p.exists():           # resume: keep the timings of layers already done
        log["layers"] = {k: v for k, v in json.loads(run_p.read_text()).get("layers", {}).items()
                         if (QDIR / "stats" / f"L{int(k):02d}.npz").exists()}
    save_h = {int(x) for x in a.save_h.split(",") if x.strip()}
    for d in ("stats", "proxy", "calib/acts", "calib/H"):
        (QDIR / d).mkdir(parents=True, exist_ok=True)
    try:
        for L in range(last + 1):
            tm = {"load": 0.0, "forward": 0.0, "gptq": 0.0, "proxy": 0.0, "pack": 0.0}
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ex = loader.get(L)
            tm["load"] = time.time() - t0
            cap = {"calib": [], "held": []}
            # a layer is done when its stats file exists (written last, atomically): resume only runs the forward
            done = (QDIR / "stats" / f"L{L:02d}.npz").exists() and pool_rows(L, "b4") >= 3 * n4
            todo = a.cmd == "quantize" and (not want or L in want) and not done
            acc = Acc(ex, bool(a.weighted)) if todo else None
            dump = a.cmd == "capture" and L in want

            def hook(split, L_, x, ids, wt):
                if dump or (split == "held" and acc is not None):
                    cap[split].append((x.half().cpu() if dump else x.clone(), ids.to(torch.int16).cpu() if dump
                                       else ids, wt.cpu() if dump else wt))
                if split == "calib" and acc is not None:
                    acc.add(x, ids, wt)

            st.hook = hook
            scale = {"n": 0, "r2": 0.0, "bo2": 0.0, "g2bo2": 0.0}
            t0 = time.time()
            st.layer(L, ex, scale)
            torch.cuda.synchronize()
            tm["forward"] = time.time() - t0
            tm["peak_forward_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.cuda.reset_peak_memory_stats()
            if dump:
                np.savez(QDIR / "calib" / "acts" / f"L{L:02d}.npz",
                         **{f"{s}_{k}": torch.cat([c[i] for c in cap[s]]).numpy()
                            for s in ("calib", "held") for i, k in enumerate(("x", "ids", "wt"))})
            if acc is not None:
                acc.offload()
                torch.cuda.empty_cache()
                heldd = {k: torch.cat([c[i] for c in cap["held"]]) for i, k in enumerate(("x", "ids", "wt"))}
                order = np.argsort(-acc.n.cpu().numpy(), kind="stable")          # most-routed first
                s4 = np.zeros(N_EXPERT, bool)
                s4[order[:n4]] = True
                s2 = np.zeros(N_EXPERT, bool)
                s2[order[N_EXPERT - n2:]] = True
                if L in save_h:
                    iu = torch.triu_indices(N_EMBD, N_EMBD, device=DEV)
                    ju = torch.triu_indices(N_FF, N_FF, device=DEV)
                    nn = acc.n.clamp_min(1).float()
                    np.savez(QDIR / "calib" / "H" / f"L{L:02d}.npz", n=acc.n.cpu().numpy(), weighted=a.weighted,
                             Hgu_mean_triu=np.concatenate([
                                 (acc.Hgu[i:i + 16].to(DEV) / nn[i:i + 16, None, None])[:, iu[0], iu[1]].half()
                                 .cpu().numpy() for i in range(0, N_EXPERT, 16)]),
                             Hd_mean_triu=(acc.Hd / nn[:, None, None])[:, ju[0], ju[1]].half().cpu().numpy())
                    del iu, ju
                P = quantize_layer(L, ex, acc, heldd, a, s4, s2, tm)
                np.savez(QDIR / "proxy" / f"L{L:02d}.npz", **P)
                tmp = QDIR / "stats" / f"L{L:02d}.tmp.npz"
                np.savez(tmp, n=acc.n.cpu().numpy(), w2=acc.w2.cpu().numpy(),
                         f2=acc.f2.cpu().numpy(), Hgu_diag=acc.Hgu.diagonal(dim1=1, dim2=2).numpy(),
                         Hd_diag=acc.Hd.diagonal(dim1=1, dim2=2).cpu().numpy(), Pgu=acc.Pgu.cpu().numpy(),
                         scale=np.array([scale["n"], scale["r2"], scale["bo2"], scale["g2bo2"]]))
                os.replace(tmp, QDIR / "stats" / f"L{L:02d}.npz")
                del heldd
                tm["peak_quant_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            del acc, ex, cap
            torch.cuda.empty_cache()
            if a.cmd == "capture" or todo:
                log["layers"][str(L)] = {k: round(v, 2) for k, v in tm.items()}
            print(f"pipeline: layer {L} {'quantized' if todo else 'forward only'} "
                  f"{ {k: round(v, 2) for k, v in tm.items()} } total {time.time() - t_start:.0f} s", flush=True)
            log["total_s"] = round(time.time() - t_start, 1)
            run_p.write_text(json.dumps(log, indent=1))
    finally:
        loader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
