"""GPTQ-BFP4 LM head (output.weight, Q6_K in the GGUF) on the RTX 3090 (torch, float32).

    cd /var/tmp/strata-tt/flashnext_tt
    /var/tmp/strata-tt/venv/bin/python -m fnx.quant.head capture [--every 2] [--batch 8]
    /var/tmp/strata-tt/venv/bin/python -m fnx.quant.head quantize [--methods bfp8,rtn4,search4,gptq4] [--damp 0.01]

capture: fnx.quant.pipeline.Stream (fnx.eval.torchref's float32 port of the published Coder GGUF, layer-major) over
every `--every`-th calibration sequence of fnx.quant.calib_data (disjoint from the quality gate's eval set; every 2nd
keeps the chat/code/instruct/docs mix) plus the 16 held-out ones, through all 48 layers and the output_hc read: the
head inputs. Writes FNX_QUANT_DIR/head/calib.npz: H f64 [2560, 2560] = sum x x^T over the calibration head inputs,
n; and head/held.npz: x f32 [n_held, 2560] (held-out head inputs), tokens int32 [n_held], seq_end bool (the last
position of each sequence: no next-token target).

quantize: per method, W = dequantized output.weight [248320, 2560] -> Q on the BFP grid of the device tensor W^T
([2560, 248320]: a block is 16 consecutive vocab rows at one input column, fnx.quant.methods orientation):
    bfp8     tt-metal's own bfloat8_b conversion (the current device head)
    rtn4     tt-metal's bfloat4_b conversion
    search4  per-block min-MSE exponent (fnx.quant.methods.search)
    gptq4    GPTQ-BFP4 (fnx.quant.methods.gptq: act-order, --damp, min-MSE block exponent with clips -1..2)
scored on the held-out head inputs against the float head: KL(float || method) mean / p99, top-1 agreement, mean
next-token NLL difference, and tr(dW H dW^T)/tr(W H W^T). Results -> head/quantize.json; the store rows of every bfp4
method -> FNX_QUANT_DIR/store/head/output.weight.T.<method>.npy (fnx.quant.bfp store row of W^T, K=2560, N=248320;
357 MB), read by fnx.tt.head (FNX_HEAD=<method>).
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

from ..config import N_EMBD, N_LAYERS, N_VOCAB
from . import bfp, methods

QDIR = Path(os.environ.get("FNX_QUANT_DIR", "/var/tmp/strata-tt/quant"))
HEAD_DIR = QDIR / "head"
STORE = QDIR / "store" / "head"
FMT = {"bfp8": "bfp8", "rtn4": "bfp4", "search4": "bfp4", "gptq4": "bfp4"}
K = N_EMBD            # W^T rows (input dim)
NV = N_VOCAB          # W^T columns (vocab)


def store_path(method: str) -> Path:
    return STORE / f"output.weight.T.{method}.npy"


_ROWS: dict = {}


def store_rows(method: str) -> np.ndarray:
    """The store row of a bfp4 method (memory-mapped, cached per process)."""
    if method not in _ROWS:
        p = store_path(method)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run python -m fnx.quant.head quantize --methods {method}")
        _ROWS[method] = np.load(p, mmap_mode="r")
    return _ROWS[method]


def unpack_cols(row: np.ndarray, fmt: str, v0: int, v1: int) -> np.ndarray:
    """Columns [v0, v1) of the store row of W^T [2560, 248320] (multiples of 16) -> float32 [2560, v1 - v0]."""
    assert v0 % bfp.BLOCK == 0 and v1 % bfp.BLOCK == 0
    exps, codes = bfp.split_row(row, fmt, K, NV)
    cb = bfp.CODE_BITS[fmt]
    sub = np.concatenate([np.ascontiguousarray(exps[:, v0 // bfp.BLOCK:v1 // bfp.BLOCK]).reshape(-1),
                          np.ascontiguousarray(codes[:, v0 * cb // 8:v1 * cb // 8]).reshape(-1)])
    return bfp.unpack(sub, fmt, K, v1 - v0)


# ------------------------------------------------------------------------------------------------ capture
def capture(a) -> int:
    from ..eval.torchref import DEV, ExpertLoader, hc_read, setup_torch
    from ..weights import Weights
    from .pipeline import Stream, shm_hygiene
    setup_torch()
    shm_hygiene()
    from . import calib_data
    cal = [s["token_ids"] for s in calib_data.load("calib")][::a.every]
    held = [s["token_ids"] for s in calib_data.load("heldout")]
    w = Weights()
    t0 = time.time()
    st = Stream(w, {"calib": cal, "held": held}, a.batch)
    print(f"head capture: {len(cal)} calib + {len(held)} held-out sequences, inputs {time.time() - t0:.0f} s",
          flush=True)
    loader = ExpertLoader(str(w.model.files[0].path), workers=a.workers)
    try:
        for L in range(N_LAYERS):
            t1 = time.time()
            ex = loader.get(L)
            st.layer(L, ex)
            del ex
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"head capture: layer {L} ({time.time() - t1:.1f} s, total {time.time() - t0:.0f} s)", flush=True)
    finally:
        loader.close()
    hw = st.ref.hcw("output_hc", inject=False)
    H = torch.zeros(K, K, dtype=torch.float64, device=DEV)
    n = 0
    hx, htok, hend = [], [], []
    held_iter = iter(held)
    for b in st.batches:
        x, _ = hc_read(b["R"].to(DEV), hw)
        if b["split"] == "calib":
            H += (x.T @ x).double()
            n += x.shape[0]
        else:
            hx.append(x.cpu().numpy())
            for _ in b["offs"]:
                s = next(held_iter)
                htok.append(np.asarray(s, np.int32))
                e = np.zeros(len(s), bool)
                e[-1] = True
                hend.append(e)
        b["R"] = None
    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(HEAD_DIR / "calib.npz", H=H.cpu().numpy(), n=n, every=a.every, n_seq=len(cal))
    np.savez(HEAD_DIR / "held.npz", x=np.concatenate(hx), tokens=np.concatenate(htok), seq_end=np.concatenate(hend))
    print(f"head capture: H over {n} calibration rows, {sum(len(t) for t in htok)} held-out rows, "
          f"{time.time() - t0:.0f} s -> {HEAD_DIR}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ quantize
def quant_rows(W: torch.Tensor, H: torch.Tensor | None, method: str, damp: float, rows: int) -> torch.Tensor:
    """W [V, 2560] f32 (host) -> snapped Q [V, 2560] f32 (host), row chunks of `rows` (a multiple of 32) on the GPU."""
    dev = torch.device("cuda")
    fmt = FMT[method]
    Q = torch.empty_like(W)
    for r0 in range(0, W.shape[0], rows):
        Wc = W[r0:r0 + rows].to(dev)
        if method in ("bfp8", "rtn4"):
            q = methods.rtn(Wc, fmt)
        elif method == "search4":
            q = methods.search(Wc, fmt)
        elif method == "gptq4":
            q = methods.gptq(Wc[None], H[None], fmt, damp=damp)[0]
        else:
            raise ValueError(method)
        Q[r0:r0 + rows] = q.cpu()
        del Wc, q
    return Q


def pack_rows(Q: torch.Tensor, fmt: str, cols: int = 31040) -> np.ndarray:
    """Snapped Q [V, 2560] -> the fnx.quant.bfp store row of W^T [2560, V] (packed on the GPU in vocab chunks)."""
    dev = torch.device("cuda")
    cb = bfp.CODE_BITS[fmt]
    W_ = bfp.MANT_BITS[fmt]
    exps = np.empty((K, NV // bfp.BLOCK), np.uint8)
    codes = np.empty((K, NV * cb // 8), np.uint8)
    for v0 in range(0, NV, cols):
        v1 = min(v0 + cols, NV)
        Wt = Q[v0:v1].to(dev).T.contiguous()                                       # [2560, c]
        q, E = bfp.quantize_codes(Wt, fmt)
        if not torch.equal(bfp.dequantize_codes(q, E, fmt, Wt.shape), Wt):
            raise ValueError(f"pack_rows: columns {v0}..{v1} are not {fmt}-snapped")
        code = (q.abs() | ((q < 0).to(q.dtype) << W_)).reshape(K, -1, 8 // cb)
        byte = torch.zeros(code.shape[:2], dtype=torch.int32, device=dev)
        for i in range(8 // cb):
            byte |= code[..., i] << (cb * i)
        exps[:, v0 // bfp.BLOCK:v1 // bfp.BLOCK] = E.to(torch.uint8).cpu().numpy()
        codes[:, v0 * cb // 8:v1 * cb // 8] = byte.to(torch.uint8).cpu().numpy()
    return np.concatenate([exps.reshape(-1), codes.reshape(-1)])


def score(Wf: torch.Tensor, Q: torch.Tensor, x: torch.Tensor, tokens: torch.Tensor, has_tgt: torch.Tensor,
          chunk: int = 256) -> dict:
    """Held-out logit metrics of the head Q against the float head Wf (both [V, 2560] on the GPU)."""
    kl, agree, dnll = [], [], []
    for a in range(0, x.shape[0], chunk):
        xc = x[a:a + chunk]
        lf = xc @ Wf.T
        lq = xc @ Q.T
        pf = torch.log_softmax(lf.double(), -1)
        pq = torch.log_softmax(lq.double(), -1)
        kl.append((pf.exp() * (pf - pq)).sum(-1).float().cpu())
        agree.append((lf.argmax(-1) == lq.argmax(-1)).cpu())
        n = min(a + chunk, x.shape[0] - 1) - a
        t = tokens[a + 1:a + 1 + n]
        m = has_tgt[a:a + n]
        d = -(pq[:n].gather(1, t[:, None])[:, 0] - pf[:n].gather(1, t[:, None])[:, 0])
        dnll.append(d[m].float().cpu())
        del lf, lq, pf, pq
    kl, agree, dnll = torch.cat(kl), torch.cat(agree), torch.cat(dnll)
    return {"kl_mean": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)), "top1": float(agree.float().mean()),
            "dnll": float(dnll.mean()), "n": int(kl.numel())}


def quantize(a) -> int:
    from ..eval.torchref import setup_torch
    from ..weights import Weights
    setup_torch()
    dev = torch.device("cuda")
    cz = np.load(HEAD_DIR / "calib.npz")
    hz = np.load(HEAD_DIR / "held.npz")
    H = torch.from_numpy(cz["H"]).float().to(dev)
    x = torch.from_numpy(hz["x"]).to(dev)
    tokens = torch.from_numpy(hz["tokens"].astype(np.int64)).to(dev)
    has_tgt = torch.from_numpy(~hz["seq_end"]).to(dev)
    W = torch.from_numpy(np.ascontiguousarray(Weights().get("output.weight"), np.float32))
    assert W.shape == (NV, K), W.shape
    Wf = W.to(dev)
    out_p = HEAD_DIR / "quantize.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    res["calib_rows"] = int(cz["n"])
    den = float(((Wf @ H) * Wf).sum())
    STORE.mkdir(parents=True, exist_ok=True)
    for method in a.methods.split(","):
        t0 = time.time()
        Q = quant_rows(W, H, method, a.damp, a.rows)
        tq = time.time() - t0
        Qd = Q.to(dev)
        dW = Wf - Qd
        rel_h = float(((dW @ H) * dW).sum()) / den
        del dW
        r = score(Wf, Qd, x, tokens, has_tgt)
        r.update({"rel_h": rel_h, "quant_s": round(tq, 1), "damp": a.damp if method == "gptq4" else None})
        del Qd
        torch.cuda.empty_cache()
        key = method if method != "gptq4" or a.damp == 0.01 else f"{method}-d{a.damp}"
        if FMT[method] == "bfp4":
            row = pack_rows(Q, "bfp4")
            tmp = store_path(key).with_suffix(".tmp.npy")
            np.save(tmp, row)
            os.replace(tmp, store_path(key))
            r["store"] = str(store_path(key))
        res[key] = r
        out_p.write_text(json.dumps(res, indent=1))
        print(f"head quantize: {key}: KL {r['kl_mean']:.5f} (p99 {r['kl_p99']:.4f}), top-1 {r['top1']:.4f}, "
              f"dNLL {r['dnll']:+.5f}, rel_h {rel_h:.3e} ({tq:.0f} s)", flush=True)
        del Q
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["capture", "quantize"])
    ap.add_argument("--every", type=int, default=2, help="capture: every N-th calibration sequence")
    ap.add_argument("--batch", type=int, default=8, help="capture: sequences per forward batch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--methods", default="bfp8,rtn4,search4,gptq4")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--rows", type=int, default=31040, help="quantize: vocab rows per GPU chunk (multiple of 32)")
    a = ap.parse_args(argv)
    return capture(a) if a.cmd == "capture" else quantize(a)


if __name__ == "__main__":
    sys.exit(main())
