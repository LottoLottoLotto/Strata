"""QSA: TTNN Qsa vs the numpy reference on real weights (layers 3 and 7), dense and sparse (indexer) positions.

  tools/devrun.sh qsa fnx/tests/test_qsa.py                        # TT vs numpy, 64 tokens, layers 3 and 7
  tools/devrun.sh qsalong fnx/tests/test_qsa.py long               # TT vs numpy, every position 0..2050, layer 3
  NODEV=1 tools/devrun.sh qsathr fnx/tests/test_qsa.py threshold   # numpy only: selection == all cells iff n_kv <= 2051
  NODEV=1 tools/devrun.sh qsaref fnx/tests/test_qsa.py sparse-ref  # numpy only: the sparse references (cached file)
  tools/devrun.sh qsasparse fnx/tests/test_qsa.py sparse           # TT vs numpy to 8256 positions, layers 3 and 7
  (sparse32k-ref / sparse32k: the same to 32800 positions, layer 3, f32 reference only)

Inputs are realistic attention-half inputs: real token embeddings (IQ4_XS rows of token_embd) in all 4 streams, read
through the layer's own hc_attn norm/down/up (spec Part 3 §3) -> `mixed` (f32, as tt HcRead returns it; the T=5 window
of the 64-token schedule is fed as bf16). Every call's bo is compared with
  ref : fnx.ref.qsa (dequantized GGUF weights in f32, fp16 KV) - the contract's reference, and
  bf8 : the same reference run with the device's bfloat8_b-rounded attn_q/k/v/output (read back from the device, or
        for the precomputed sparse references ttnn's identical host conversion), which isolates the TT
        implementation error from the weight-format error.
Pass per call: rel L2 vs ref <= REL_TOL and cos >= COS_TOL, and rel L2 vs bf8 <= REL_TOL_BF8 (measured: vs ref
1.3-1.8% = the bf8 weight floor; vs bf8 well below 1%).
Sparse modes additionally require every row's selected cells (Qsa.last_ids) to equal the reference's
(QsaState.last_ids); a mismatch is printed with the exact score margin of the blocks involved. The device block
scores of every row of the multi-row windows are checked against their exact values (max |s~ - s| / (sum_h |qi_h|
|pooled_b|) must stay below the module's SCORE_ERR bound; the single-token calls run as served, for timing). The
schedule mixes 128-row prompt windows, a window crossing the dense limit, wide windows and single-token calls.
Timing: eager wall ms per forward (host dispatch + device + the host round trip, synchronized), and the device time
of a T=1 step traced in its two device parts (Qsa._project, Qsa._attend; the host round trip between them cannot be
captured).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from fnx import paths
from fnx.config import HC, HEAD_DIM, N_EMBD, N_HEAD
from fnx.ref.qsa import DENSE_MAX_POS, SEL_WIDTH, QsaState, block_scores, qsa, rms, rope64, rope_cos_sin
from fnx.weights import Weights
import gguf.quants as gq  # noqa: E402  (fnx.gguf_load put gguf-py on sys.path)
from gguf.constants import GGMLQuantizationType  # noqa: E402

REL_TOL = 3e-2
COS_TOL = 0.9995
REL_TOL_BF8 = 1e-2
LAYERS = (3, 7)
N_TOK = 64
SCHEDULE = [(p, 1) for p in range(40)] + [(40, 8), (48, 5)] + [(p, 1) for p in range(53, 64)]
REF_DIR = os.environ.get("FNX_QSA_REF") or str(paths.CACHE / "qsa_ref")


def embed_rows(w: Weights, tokens) -> np.ndarray:
    m = w.model
    t = m.info("token_embd.weight")
    per = t.expected_bytes() // t.shape[1]
    raw = m.raw("token_embd.weight")
    return np.stack([np.asarray(gq.dequantize(raw[i * per:(i + 1) * per], GGMLQuantizationType[t.type_name]),
                                np.float32).reshape(N_EMBD) for i in tokens])


def hc_mixed(w: Weights, L: int, e: np.ndarray) -> np.ndarray:
    """hc_read (spec Part 3 §3) of R = 4 copies of each embedding row with blk.L.hc_attn_* -> mixed [T, 2560]."""
    g = np.asarray(w.layer(L, "hc_attn_norm.weight"), np.float32).reshape(HC, N_EMBD)
    Wd = np.asarray(w.layer(L, "hc_attn_down.weight"), np.float32)          # (320, 10240)
    Wu = np.asarray(w.layer(L, "hc_attn_up.weight"), np.float32)            # (10240, 320)
    sig = lambda x: 1 / (1 + np.exp(-x))
    R = np.repeat(e[:, None, :], HC, axis=1)                                 # [T, 4, 2560]
    rs = 1 / np.sqrt((R * R).sum(-1) / N_EMBD + np.float32(1e-6))
    xn = (R * g) * rs[..., None]
    p = (xn.reshape(len(e), -1) @ Wd.T) / HC
    gate = ((p * sig(p)) @ Wu.T).reshape(len(e), HC, N_EMBD)
    return ((xn * sig(gate)).sum(1) / HC).astype(np.float32)


def tokens(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 150000, n)


def errs(got: np.ndarray, ref: np.ndarray):
    d = got - ref
    rel = np.linalg.norm(d) / np.linalg.norm(ref)
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    return float(np.abs(d).max()), float(rel), cos


class RoundedWeights:
    """`Weights` view whose attn_q/k/v/output of layer L are the device's bfloat8_b values: qkvg [2560, 13312] =
    [q|gate|k|v]^T and o^T [6144, 2560] as the device holds them."""

    def __init__(self, w: Weights, L: int, qkvg: np.ndarray, oT: np.ndarray):
        QD = N_HEAD * HEAD_DIM
        q = qkvg[:, :QD].T.reshape(N_HEAD, 1, HEAD_DIM, N_EMBD)
        g = qkvg[:, QD:2 * QD].T.reshape(N_HEAD, 1, HEAD_DIM, N_EMBD)
        self.w, self.L = w, L
        self.over = {"attn_q.weight": np.concatenate([q, g], 1).reshape(2 * QD, N_EMBD),
                     "attn_k.weight": np.ascontiguousarray(qkvg[:, 2 * QD:2 * QD + 512].T),
                     "attn_v.weight": np.ascontiguousarray(qkvg[:, 2 * QD + 512:].T),
                     "attn_output.weight": np.ascontiguousarray(oT.T)}

    def layer(self, L: int, suffix: str):
        return self.over[suffix] if L == self.L and suffix in self.over else self.w.layer(L, suffix)


def device_rounded(w: Weights, L: int, tt) -> RoundedWeights:
    import ttnn
    return RoundedWeights(w, L, ttnn.to_torch(tt.w_qkvg).float().numpy(), ttnn.to_torch(tt.w_o).float().numpy())


def cache_rounded(w: Weights, L: int) -> RoundedWeights:
    """The same without a device: ttnn's host bfloat8_b conversion of the matrices fnx.tt.qsa.Qsa builds (the
    conversion ttnn.as_tensor caches)."""
    import torch
    import ttnn
    QD = N_HEAD * HEAD_DIM
    wq = np.asarray(w.layer(L, "attn_q.weight"), np.float32).reshape(N_HEAD, 2, HEAD_DIM, N_EMBD)
    qkvg = np.concatenate([wq[:, 0].reshape(QD, N_EMBD), wq[:, 1].reshape(QD, N_EMBD),
                           np.asarray(w.layer(L, "attn_k.weight"), np.float32),
                           np.asarray(w.layer(L, "attn_v.weight"), np.float32)]).T
    oT = np.asarray(w.layer(L, "attn_output.weight"), np.float32).T
    rnd = lambda a: ttnn.to_torch(ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.bfloat8_b,
                                                  layout=ttnn.TILE_LAYOUT)).float().numpy()
    return RoundedWeights(w, L, rnd(qkvg), rnd(oT))


def run_sequence(mesh, w, L, tt, wb, mixed, schedule, label, verbose, bf16_T=5):
    """Reset, feed `schedule` [(p0, T)] through TT and both references; returns (ok, rows, last ref)."""
    import torch
    import ttnn
    tt.reset()
    cells = schedule[-1][0] + schedule[-1][1]
    st, stb = QsaState(max_cells=cells), QsaState(max_cells=cells)
    rows, ok, ref = [], True, None
    for p0, T in schedule:
        pos = list(range(p0, p0 + T))
        x = mixed[p0:p0 + T]
        dt_in = ttnn.bfloat16 if T == bf16_T else ttnn.float32
        xr = x if dt_in == ttnn.float32 else torch.from_numpy(x).to(torch.bfloat16).float().numpy()
        ref = qsa(w, L, x, pos, st)
        refb = qsa(wb, L, xr, pos, stb)
        bo, dt = tt_call(mesh, tt, x, pos, dt_in)
        got = bo
        mx, rel, cos = errs(got, ref)
        _, relb, _ = errs(got, refb)
        good = rel <= REL_TOL and cos >= COS_TOL and relb <= REL_TOL_BF8 and bool(np.isfinite(got).all())
        ok &= good
        rows.append((T, rel, relb, dt))
        if verbose(p0) or not good:
            print(f"  L{L} {label} pos {p0:4d} T={T}{' bf16-in' if T == bf16_T else ''}: max|d| {mx:.4f} "
                  f"(max|ref| {np.abs(ref).max():.3f}) rel {rel:.4f} cos {cos:.6f} | vs bf8-weight ref "
                  f"{relb:.4f} | {dt:6.2f} ms {'ok' if good else 'FAIL'}")
    print(f"L{L} {label}: rel vs ref max {max(r[1] for r in rows):.4f} mean {np.mean([r[1] for r in rows]):.4f}; "
          f"vs bf8-weight ref max {max(r[2] for r in rows):.4f} mean {np.mean([r[2] for r in rows]):.4f}; eager ms "
          + ", ".join(f"T={T} median {np.median([r[3] for r in rows if r[0] == T]):.2f}"
                      for T in sorted({r[0] for r in rows})))
    return ok, rows, ref


def tt_call(mesh, tt, x: np.ndarray, pos, dt_in=None):
    """One synchronized TT forward of rows x at positions pos -> (bo [T, 2560] host, wall ms)."""
    import torch
    import ttnn
    T = len(pos)
    xt = ttnn.from_torch(torch.from_numpy(x.reshape(1, 1, T, N_EMBD)), dtype=dt_in or ttnn.float32,
                         layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    bo = tt.forward(xt, pos)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t1) * 1e3
    got = ttnn.to_torch(bo).float().numpy().reshape(-1, N_EMBD)[:T]
    ttnn.deallocate(xt)
    ttnn.deallocate(bo)
    return got, dt


def _replay_ms(mesh, fn, n: int = 30) -> float:
    import ttnn
    out = fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t1) * 1e3 / n
    ttnn.release_trace(mesh, tid)
    return ms, out


def traced_ms(mesh, tt, x_row: np.ndarray, p: int, ref_row: np.ndarray, L: int) -> tuple[bool, float]:
    """Device time of one T=1 step at position p (re-runs p with the same values): the two device parts traced
    separately (the host round trip between them cannot be captured). Returns (rel ok, device ms)."""
    import torch
    import ttnn
    from fnx.tt.qsa import _free
    got, wall = tt_call(mesh, tt, x_row.reshape(1, N_EMBD), [p])        # eager: step inputs, programs, ids
    _, rel, _ = errs(got, ref_row.reshape(1, N_EMBD))
    xt = ttnn.from_torch(torch.from_numpy(x_row.reshape(1, 1, 1, N_EMBD)), dtype=ttnn.float32,
                         layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sparse = p > DENSE_MAX_POS
    debug, tt.debug = tt.debug, False

    def part_a():
        d = tt._project(xt, [p], sparse)
        return d
    ms_a, d = _replay_ms(mesh, part_a)
    _free(d["qr"], d["gate"], d["scores"], d["qi"])
    ids = tt._sparse_ids(tt.last_ids[0:1]) if sparse else []
    dev = tt._project(xt, [p], False)
    ms_b, bo = _replay_ms(mesh, lambda: tt._attend(dev, [p], int(not sparse), ids, free_inputs=False))
    _free(dev["qr"], dev["gate"], bo, xt, *ids)
    tt.debug = debug
    print(f"L{L} T=1 step at pos {p}: device {ms_a + ms_b:.3f} ms traced (projections+KV+scores {ms_a:.3f}, "
          f"gather+attention+output {ms_b:.3f}); eager wall {wall:.2f} ms incl. host round trip; rel vs ref {rel:.4f}")
    return rel <= REL_TOL, ms_a + ms_b


def run_tt(long: bool) -> bool:
    import ttnn
    from fnx.tt.qsa import Qsa

    w = Weights()
    n_tok = SEL_WIDTH if long else N_TOK
    e = embed_rows(w, tokens(n_tok, seed=2 if long else 0))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    ok = True
    try:
        for L in ((3,) if long else LAYERS):
            mixed = hc_mixed(w, L, e)
            print(f"layer {L}: mixed rms {np.sqrt((mixed ** 2).mean()):.4f} max {np.abs(mixed).max():.3f}")
            t0 = time.time()
            tt = Qsa(mesh, w, L, max_ctx=SEL_WIDTH + 1 if long else 2048)
            print(f"layer {L}: Qsa init {time.time() - t0:.2f} s")
            wb = device_rounded(w, L, tt)
            if long:                      # every dense position, windows of 8 (last: 2048..2050)
                sched = [(p, min(8, SEL_WIDTH - p)) for p in range(0, SEL_WIDTH, 8)]
                good, _, ref = run_sequence(mesh, w, L, tt, wb, mixed, sched, "long", lambda p: p % 256 == 0 or
                                            p >= 2040, bf16_T=None)
                ok &= good
                ok &= traced_ms(mesh, tt, mixed[SEL_WIDTH - 1], SEL_WIDTH - 1, ref[-1], L)[0]
                continue
            for rep in range(2):          # pass 0 compiles; pass 1 (after reset) is the printed, timed one
                good, _, ref = run_sequence(mesh, w, L, tt, wb, mixed, SCHEDULE, f"pass{rep}", lambda p: rep == 1)
                ok &= good
            ok &= traced_ms(mesh, tt, mixed[N_TOK - 1], N_TOK - 1, ref[-1], L)[0]
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"test_qsa TT{' long' if long else ''} (pass: rel vs ref <= {REL_TOL}, cos >= {COS_TOL}, rel vs bf8-weight "
          f"ref <= {REL_TOL_BF8}):", "PASS" if ok else "FAIL")
    return ok


def run_threshold() -> bool:
    """Feed 2056 positions through the numpy reference (layer 3) in windows; record which rows select every cell."""
    w = Weights()
    n = SEL_WIDTH + 5
    e = embed_rows(w, tokens(n, seed=1))
    mixed = hc_mixed(w, 3, e)
    st = QsaState(max_cells=n + 4)
    dense_upto, first_sparse, ok = -1, None, True
    for p0 in range(0, n, 8):
        T = min(8, n - p0)
        qsa(w, 3, mixed[p0:p0 + T], list(range(p0, p0 + T)), st)
        for t, ids in enumerate(st.last_ids):
            p = p0 + t
            full = len(ids) == p + 1 and np.array_equal(ids, np.arange(p + 1))
            if full and first_sparse is None:
                dense_upto = p
            if not full:
                first_sparse = p if first_sparse is None else first_sparse
                tail = np.arange((p + 1) // 4 * 4, p + 1)
                print(f"  pos {p}: {len(ids)} of {p + 1} cells selected, tail cells {tail.tolist()} selected: "
                      f"{bool(np.isin(tail, ids).all())}, dropped cells {np.setdiff1d(np.arange(p + 1), ids)[:8].tolist()}")
                ok &= len(ids) == SEL_WIDTH and bool(np.isin(tail, ids).all())
    ok &= dense_upto == DENSE_MAX_POS and first_sparse == DENSE_MAX_POS + 1
    print(f"dense (all cells) up to pos {dense_upto}; first sparse pos {first_sparse}; expected {DENSE_MAX_POS}/"
          f"{DENSE_MAX_POS + 1}:", "PASS" if ok else "FAIL")
    return ok


# ---- sparse (indexer) positions -------------------------------------------------------------------------------
SPARSE_CFG = {"sparse": dict(n_end=8256, layers=LAYERS, win=64, bf8=True, seed=3),
              "sparse32k": dict(n_end=32800, layers=(3,), win=128, bf8=False, seed=4)}


def sparse_schedule(n_end: int, win: int) -> list[tuple[int, int]]:
    """128-row prompt windows to 1919, a 120-row window, T=1 at 2040..2047 (dense), a 32-row window crossing the
    dense limit (rows 2048..2050 dense, 2051..2079 sparse), T=1 at 2080..2111, `win`-row windows, T=1 for the last
    64 (8K) / 32 (32K) positions."""
    tail = 64 if n_end <= 10000 else 32
    s = [(p, 128) for p in range(0, 1920, 128)] + [(1920, 120)] + [(p, 1) for p in range(2040, 2048)]
    s += [(2048, 32)] + [(p, 1) for p in range(2080, 2112)]
    p = 2112
    while p + win <= n_end - tail:
        s.append((p, win))
        p += win
    if p < n_end - tail:
        s.append((p, n_end - tail - p))
    s += [(q, 1) for q in range(n_end - tail, n_end)]
    assert [a for a, _ in s] == list(np.cumsum([0] + [t for _, t in s[:-1]])) and sum(t for _, t in s) == n_end
    return s


def ref_path(mode: str, L: int) -> str:
    c = SPARSE_CFG[mode]
    return f"{REF_DIR}/{mode}_L{L}_n{c['n_end']}_w{c['win']}_s{c['seed']}.npz"


def sparse_inputs(w: Weights, mode: str, L: int) -> np.ndarray:
    c = SPARSE_CFG[mode]
    return hc_mixed(w, L, embed_rows(w, tokens(c["n_end"], seed=c["seed"])))


def run_sparse_ref(mode: str) -> bool:
    """The references of a sparse schedule, computed once on the host and cached: per call bo (ref, bf8 ref), per
    sparse row the selected cells (uint16 [rows, 2051])."""
    c = SPARSE_CFG[mode]
    w = Weights()
    os.makedirs(REF_DIR, exist_ok=True)
    sched = sparse_schedule(c["n_end"], c["win"])
    for L in c["layers"]:
        path = ref_path(mode, L)
        if os.path.isfile(path):
            print(f"{path}: exists")
            continue
        mixed = sparse_inputs(w, mode, L)
        wb = cache_rounded(w, L) if c["bf8"] else None
        st = QsaState(max_cells=c["n_end"])
        stb = QsaState(max_cells=c["n_end"]) if wb else None
        bo, bob, ids, t0 = [], [], [], time.time()
        for i, (p0, T) in enumerate(sched):
            pos = list(range(p0, p0 + T))
            bo.append(qsa(w, L, mixed[p0:p0 + T], pos, st))
            if wb is not None:
                bob.append(qsa(wb, L, mixed[p0:p0 + T], pos, stb))
            ids += [x.astype(np.uint16) for p, x in zip(pos, st.last_ids) if p > DENSE_MAX_POS]
            if i % 20 == 0:
                print(f"  L{L} ref pos {p0} ({time.time() - t0:.0f} s)", flush=True)
        np.savez(path, bo=np.concatenate(bo), bob=np.concatenate(bob) if bob else np.zeros(0),
                 ids=np.stack(ids), sched=np.asarray(sched))
        print(f"{path}: {len(sched)} calls, {len(ids)} sparse rows, {time.time() - t0:.0f} s")
    return True


def mismatch_margin(tt, p: int, got: np.ndarray, want: np.ndarray, x_row_qi: np.ndarray) -> str:
    """Exact scores (TT's host state = the reference's) of the blocks on which two selections differ."""
    nb = (p + 1) // 4
    bg, bw = set(np.unique(got // 4).tolist()), set(np.unique(want // 4).tolist())
    only_t, only_r = sorted((bg - bw) - {nb}), sorted((bw - bg) - {nb})
    s = block_scores(tt.pooled[:nb], x_row_qi)
    srt = np.sort(s)[::-1]
    return (f"TT-only blocks {only_t} scores {[float(s[b]) for b in only_t]}, ref-only blocks {only_r} scores "
            f"{[float(s[b]) for b in only_r]}; 512th/513th/514th scores {srt[511]:.7g}/{srt[512]:.7g}/{srt[513]:.7g}")


def run_sparse(mode: str) -> bool:
    import ttnn
    from fnx.tt.qsa import SCORE_ERR, Qsa

    c = SPARSE_CFG[mode]
    w = Weights()
    sched = sparse_schedule(c["n_end"], c["win"])
    refs = {}
    for L in c["layers"]:
        if not os.path.isfile(ref_path(mode, L)):
            print(f"computing the reference inline (precompute with NODEV=1 ... {mode}-ref to free the device sooner)")
            run_sparse_ref(mode)
        with np.load(ref_path(mode, L)) as z:                    # into memory once (NpzFile reads per access)
            refs[L] = {k: z[k] for k in z.files}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=256 << 20)
    ok = True
    try:
        for L in c["layers"]:
            R = refs[L]
            mixed = sparse_inputs(w, mode, L)
            tt = Qsa(mesh, w, L, max_ctx=c["n_end"])
            tt.reset()
            r0, row, mism, worst, rows, stats = 0, 0, [], dict(rel=0.0, relb=0.0, cos=1.0), [], []
            t_start = time.time()
            for p0, T in sched:
                pos = list(range(p0, p0 + T))
                tt.debug = T > 1                     # windows: check every device score; T=1 calls: timed as served
                got, dt = tt_call(mesh, tt, mixed[p0:p0 + T], pos)
                ref = R["bo"][r0:r0 + T]
                mx, rel, cos = errs(got, ref)
                relb = errs(got, R["bob"][r0:r0 + T])[1] if c["bf8"] else 0.0
                good = rel <= REL_TOL and cos >= COS_TOL and relb <= REL_TOL_BF8 and bool(np.isfinite(got).all())
                r0 += T
                for p, cells in zip(pos, tt.last_ids):
                    if p <= DENSE_MAX_POS:
                        if not np.array_equal(cells, np.arange(p + 1)):
                            mism.append((p, "dense row not all cells"))
                        continue
                    want = R["ids"][row].astype(np.int64)
                    row += 1
                    if not np.array_equal(cells, want):
                        cpos = rope_cos_sin([p])
                        xq = (mixed[p:p + 1] @ tt.h_iq.T).reshape(1, 4, 128)
                        qin = rope64(rms(xq, tt.h_iqn), cpos[0][:, None, :], cpos[1][:, None, :])[0]
                        mism.append((p, mismatch_margin(tt, p, cells, want, qin)))
                st = tt.last
                if "err_ratio" in st:
                    stats.append((st["err_ratio"], st.get("qi_rel", 0.0), st["cand"] / max(st["sparse_rows"], 1)))
                ok &= good
                worst = dict(rel=max(worst["rel"], rel), relb=max(worst["relb"], relb), cos=min(worst["cos"], cos))
                rows.append((p0, T, rel, dt, st.get("us", {})))
                if not good or p0 in (2040, 2048, 2080, 2111, c["n_end"] - 1) or (T > 1 and p0 % 2048 < T):
                    print(f"  L{L} pos {p0:5d} T={T:3d}: rel {rel:.4f} cos {cos:.6f} vs bf8 {relb:.4f} | {dt:7.2f} ms "
                          f"| err ratio {st.get('err_ratio', 0):.2e} qi rel {st.get('qi_rel', 0):.2e} rescored "
                          f"{st.get('cand', 0)} | {'ok' if good else 'FAIL'}", flush=True)
            n_sparse = row
            er = max(s[0] for s in stats)
            print(f"L{L} {mode}: {len(sched)} calls, {n_sparse} sparse rows in {time.time() - t_start:.0f} s; bo rel "
                  f"vs ref max {worst['rel']:.4f}, cos min {worst['cos']:.6f}" +
                  (f", vs bf8-weight ref max {worst['relb']:.4f}" if c["bf8"] else ""))
            print(f"L{L} device block scores: max |s~-s|/(sum|qi||k|) {er:.3e} (bound SCORE_ERR {SCORE_ERR:.3e}, "
                  f"{SCORE_ERR / er:.1f}x), device qi max rel err {max(s[1] for s in stats):.3e}; blocks rescored "
                  f"exactly per sparse window row: mean {np.mean([s[2] for s in stats]):.1f} "
                  f"max {max(s[2] for s in stats):.0f}")
            print(f"L{L} selection: {n_sparse - len(mism)}/{n_sparse} sparse rows identical to the reference" +
                  "".join(f"\n  MISMATCH pos {p}: {m}" for p, m in mism[:20]))
            ok &= not mism and er < SCORE_ERR
            for lab, lo_, hi_ in (("2K dense", 2040, 2047), ("2K sparse", 2080, 2111),
                                  (f"{c['n_end'] // 1024}K", c["n_end"] - 64, c["n_end"])):
                sel_rows = [r for r in rows if r[1] == 1 and lo_ <= r[0] <= hi_]
                if sel_rows:
                    us = {k: np.median([r[4][k] for r in sel_rows]) / 1e3 for k in sel_rows[0][4]}
                    print(f"L{L} T=1 eager at {lab} ({sel_rows[0][0]}..{sel_rows[-1][0]}): median "
                          f"{np.median([r[3] for r in sel_rows]):.2f} ms wall; parts (ms) "
                          + ", ".join(f"{k} {v:.2f}" for k, v in us.items()))
            for p in (2047, 2111, c["n_end"] - 1):                  # reference rows are positions (one sequence)
                ok &= traced_ms(mesh, tt, mixed[p], p, R["bo"][p], L)[0]
            del tt
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"test_qsa {mode} (pass: every sparse row's cells == reference, device score error < SCORE_ERR, rel vs ref "
          f"<= {REL_TOL}, cos >= {COS_TOL}" + (f", rel vs bf8-weight ref <= {REL_TOL_BF8}" if c["bf8"] else "") + "):",
          "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    args = sys.argv[1:] or ["tt"]
    res = []
    if "threshold" in args:
        res.append(run_threshold())
    if "tt" in args:
        res.append(run_tt(long=False))
    if "long" in args:
        res.append(run_tt(long=True))
    for m in SPARSE_CFG:
        if f"{m}-ref" in args:
            res.append(run_sparse_ref(m))
        if m in args:
            res.append(run_sparse(m))
    sys.exit(0 if all(res) else 1)
