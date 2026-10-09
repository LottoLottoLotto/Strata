"""Strata's `strata --serve` engine protocol on the P150: Strata's own server (serve/server.py, unchanged) drives
fnx.tt.model.TtModel through it as if it were the CUDA engine.

    python -m fnx.serve_engine --serve [--native SHARD1] [--ple-gguf SHARD2] [--mtp MTP_RT] [--pack PACK]
                               [--max-context 32768] [--prefill auto|N] [--prompt-cache N] [--threads N]
                               [--eos-ids a,b] [--no-warmup] [--spec N] [--spec-min-p P]

(tt/serve/tt-engine.sh runs it in the TT image holding the shared device lock.)  Model files come from the
arguments Strata's setup writes (--native shard 1, --ple-gguf the PLE shard, --mtp the drafter's mtp/rt; --pack is
the server's tokenizer pack, nothing the TT engine reads), this port's own data (GPTQ-BFP expert store, the drafter's
BFP4 experts, converted-tensor caches) from FNX_DATA (fnx.paths); Strata's other engine arguments are accepted and
ignored.  INFO reports engine=<the repository's CMakeLists.txt version>.  Protocol (spec Part 8, spec-serve-oracle.md
§1-§7; serve/server.py StrataEngine):
stdout carries ONLY protocol lines (fd 1 is re-pointed to stderr at start, so library output goes to the log):
`INFO key=val..`, `READY <ctx> stop`; per `GEN <max_new>[ key=val..] <id,id,..>`: `RESUME r`, `PP reached n ms
tok/s` after every prompt chunk, `REUSED r`, `T <id>` per token (EOS 248044/248046 included), then exactly one
`DONE generated n prompt_ms decode_ms finish 0 0 reused 0 0 0 0 0.0 read 0` (finish stop|length|cancel), or one
`ERR <reason>` instead.  `STOP` ends the running request (checked between prompt chunks and decode steps; a STOP
that arrives before the next GEN is stale and dropped when that GEN is queued), `QUIT` / EOF exit.  `GENI` (no
vision), `VRAM` (no elastic cache) answer ERR; `SAVE`/`RESTORE` answer `SERR invalid 0 ..` (no session files).
A device failure inside a request answers `ERR ..` and exits 1 (the server restarts the engine on the next request).

Positions: the context is up to 32768 tokens (READY min(--max-context, 32768)); the QSA layers attend exactly the
cells Strata selects (dense up to position 2050, the indexer's 2051 cells beyond: fnx.tt.qsa); requests that do not
fit are refused like Strata (`n + max_new + 8 > ctx`).

Conversation reuse (Strata's prompt cache, generate.cpp:8224-8236, 8900-8990): `live` is what the model state has
consumed (prompt + every fed decode token).  A GEN whose ids start with `live` (1 <= len <= n-1) continues from it;
otherwise from the longest checkpoint the ids start with (TtModel.snapshot(), taken where the prompt's last
<|im_start|> begins, i.e. before the new assistant turn whose header/thinking the next turn re-renders
differently); otherwise every module is reset.  Checkpoints keep only non-positional state (the QSA KV is
positional), so one that is not a prefix of `live` after a request is dropped.  `ckpt=0` takes no checkpoint and
leaves nothing to continue.  `--prompt-cache 0` turns reuse off (every request from token 0).

Prompt: positions [resume, n-1) in chunks of `--prefill` tokens through TtModel.forward(head=False) (split at the
checkpoint position); the last prompt token is the first decode step (T=1), as in Strata.

Sampling: greedy (no temperature key, or <= 0) uses the device argmax; penalty keys (penalty_last_n > 0 with a
non-neutral penalty) and temperature > 0 read the logits back and run Strata's sampled chain on the host, a
transcription of src/kernels/sampler_parity.cpp `sampled_reference` (spec Part 7 §1.4): penalties over the last
penalty_last_n consumed tokens (cap 4096) -> top_k (0 or >= 64 -> 64; ties to the lower id) -> top_p (double, over
the top_k list) -> min_p (on the unscaled logits) -> temperature -> inverse-CDF pick with Strata's Philox-like draw
`host_philox_uniform(seed, counter = position of the input token)`; seed 0/absent = steady clock (as Strata).  The
RNG stream is bit-identical to Strata's; the picked tokens can still differ where TT and CUDA logits differ.

Speculative decoding (--spec N, default $FNX_SPEC else DEFAULT_SPEC; 1 = plain decode): the MTP drafter
(fnx.tt.mtp) proposes N-1 tokens from the main model's final residual, the model verifies the window
[x, d1..d(N-1)] at positions p..p+N-1 in one forward (the first window is the last prompt token alone), row t's pick
is emitted while d1..dt matched the picks before it (greedy: the device argmax; sampled: Strata's chain with counter
p + t over the tokens consumed through row t), capped by max_new and EOS, and TtModel.commit(n_keep) rolls the
rejected rows back.  A window row computes bitwise what the serial loop computes (fnx/tests/test_spec.py), so the
tokens are those of --spec 1.  The drafter's K/V is filled from the prompt chunks' residuals and the committed rows;
conversation checkpoints are taken in the prompt only (committed states).  --spec-min-p stops a draft chain at a
draft whose drafter probability is below P.

Traced decode (default; FNX_TRACE=0 runs everything eagerly): after the warm-up the model captures its decode
forwards of 1 and --spec rows and the drafter's rounds (TtModel.enable_trace, fnx.tt.trace), replayed for every decode
step, verify window, partial commit and draft round; prompt chunks and other row counts stay eager.  Bitwise the eager
tokens.

MoE (FNX_MOE, default DEFAULT_MOE): "device" = every routed expert on the P150 (fnx.tt.moe_device: GPTQ-BFP4 experts of
the plan FNX_MOE_PLAN, the hottest in device DRAM, the rest in pinned host memory the kernels read over PCIe), so a
dense decode step or verify window is ONE captured trace; "cpu" = Strata's own CPU expert kernels on the GGUF experts
(Strata-exact logits; 2 host round trips per layer).  The MTP drafter's 512 routed experts run on the device either way
(fnx.tt.mtp: BFP4 from their Q2_0 values, fnx.tt.moe_device), so a traced draft round is one captured segment.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths
from .config import EOS_IDS, N_VOCAB


def source_version() -> str:
    """The repository's version (CMakeLists.txt `project(strata VERSION x.y.z.w)`, setup.py's source_version): the
    engine version INFO reports, numeric as Strata's server and setup parse it."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", paths.CMAKELISTS.read_text(encoding="utf-8"))
    if not m:
        raise RuntimeError(f"{paths.CMAKELISTS}: no `project(strata VERSION ...)`")
    return m.group(1)


CTX_MAX = 32768                # advertised context limit (READY): fnx.tt.qsa's selection verified to 32800 positions
TURN_TOKEN = 248045            # <|im_start|>: the checkpoint goes where the prompt's last one begins
PENALTY_WINDOW_CAP = 4096      # generate.cpp:5989
TOP_K_MAX = 64                 # the sampled path's widest list (sampler.cu:423-426)
DEFAULT_CHUNK = 128            # prompt rows per TtModel call: 16/32/64/128 rows gave 14.7/18.1/21.5/24.9 tok/s
                               # (256 rows of the code prompt, GdnOps, 2026-10-06)
ERR_EXPECTED = "ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>"
DEFAULT_SPEC = 4              # verify-window rows with the MTP drafter (1 + 3 drafts)
DEFAULT_MOE = "device"        # FNX_MOE: routed experts on the device (fnx.tt.moe_device) or the host CPU (fnx.tt.moe)


def log(msg: str) -> None:
    print(f"fnx serve: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ request parsing
class BadRequest(ValueError):
    """A request-level refusal: one `ERR <message>` line, the engine stays idle."""


def _f32(v: str) -> np.float32:
    """strtof: the longest numeric prefix, 0 when there is none."""
    m = re.match(r"\s*[+-]?(inf(inity)?|nan|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)", v, re.IGNORECASE)
    return np.float32(float(m.group(0))) if m else np.float32(0.0)


def _int(v: str) -> int:
    """atoi / strtoll: the leading integer, 0 when there is none."""
    m = re.match(r"\s*[+-]?\d+", v)
    return int(m.group(0)) if m else 0


def parse_ids(text: str) -> list[int]:
    """parse_i64_list (generate.cpp:877-895): commas are blanks; every token a non-negative int32."""
    ids = []
    for tok in text.replace(",", " ").split():
        if not re.fullmatch(r"-?\d+", tok) or int(tok) < 0 or int(tok) > 0x7FFFFFFF:
            raise BadRequest("bad request: invalid token id: expected an integer in [0, 2147483647]")
        ids.append(int(tok))
    if not ids:
        raise BadRequest("bad request: token list was empty")
    return ids


@dataclass
class Sampling:
    """A GEN line's sampling keys with Strata's per-request defaults (generate.cpp:8022-8063, 8872-8897)."""
    temperature: np.float32 = np.float32(0.0)
    top_p: np.float32 = np.float32(1.0)
    top_k: int = 20
    min_p: np.float32 = np.float32(0.0)
    penalty_last_n: int = 0
    penalty_repeat: np.float32 = np.float32(1.0)
    penalty_freq: np.float32 = np.float32(0.0)
    penalty_present: np.float32 = np.float32(0.0)
    seed: int = 0
    ckpt: bool = True

    @property
    def sampled(self) -> bool:
        return bool(self.temperature > 0)

    @property
    def penalized(self) -> bool:
        return self.penalty_last_n > 0 and (self.penalty_repeat != 1 or self.penalty_freq != 0
                                            or self.penalty_present != 0)


def parse_gen(line: str) -> tuple[int, Sampling, list[int]]:
    """`GEN <max_new>[ key=val...] <ids>` -> (max_new, sampling, ids); BadRequest with Strata's message."""
    rest = line[4:]
    m = re.match(r"\s*[+-]?\d+", rest)
    max_new = int(m.group(0)) if m else 0
    toks = rest[m.end() if m else 0:].split(" ")
    sp, i = Sampling(), 0
    while i < len(toks):
        tok = toks[i]
        if tok == "":
            i += 1
            continue
        if "=" not in tok:
            break
        k, v = tok.split("=", 1)
        if k == "temperature":
            sp.temperature = _f32(v)
        elif k == "top_p":
            sp.top_p = _f32(v)
        elif k == "top_k":
            sp.top_k = _int(v)
        elif k == "min_p":
            sp.min_p = np.float32(min(max(float(_f32(v)), 0.0), 1.0))
        elif k == "penalty_last_n":
            sp.penalty_last_n = min(max(_int(v), 0), PENALTY_WINDOW_CAP)
        elif k == "penalty_repeat":
            sp.penalty_repeat = _f32(v)
        elif k == "penalty_freq":
            sp.penalty_freq = _f32(v)
        elif k == "penalty_present":
            sp.penalty_present = _f32(v)
        elif k == "seed":
            sp.seed = _int(v) & 0xFFFFFFFFFFFFFFFF
        elif k == "ckpt":
            sp.ckpt = _int(v) != 0
        i += 1                                     # unknown keys (cvec, pcie_frac, spec_min_p, ...) are skipped
    if max_new < 1:
        raise BadRequest("bad request: max_new")
    return max_new, sp, parse_ids(" ".join(toks[i:]))


# ------------------------------------------------------------------------------------------------ host sampler
def philox_uniform(seed: int, counter: int) -> np.float32:
    """Strata's Philox-like 4x32-10 draw (sampler.cu:41-63 = sampler_parity.cpp host_philox_uniform)."""
    M = 0xFFFFFFFF
    c0, c1, c2, c3 = counter & M, (counter >> 32) & M, seed & M, (seed >> 32) & M
    for i in range(10):
        p0, p1 = 0x9E3779B9 * c0, 0xBB67AE85 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ i) & M, p1 & M, ((p0 >> 32) ^ c3) & M, p0 & M
    return np.float32(c0 >> 8) * np.float32(1.0 / 16777216.0)


def apply_penalties(logits: np.ndarray, hist: list[int], sp: Sampling) -> np.ndarray:
    """llama.cpp penalties over the window `hist` (sampler.cu:75-81), float32; returns a new array."""
    h = np.asarray(hist[-sp.penalty_last_n:] if sp.penalty_last_n else [], dtype=np.int64)
    h = h[(h >= 0) & (h < len(logits))]
    out = np.array(logits, dtype=np.float32, copy=True)
    if h.size == 0:
        return out
    ids, counts = np.unique(h, return_counts=True)
    x = out[ids]
    x = np.where(x <= 0, x * sp.penalty_repeat, x / sp.penalty_repeat).astype(np.float32)
    out[ids] = x - (counts.astype(np.float32) * sp.penalty_freq + np.float32(1.0) * sp.penalty_present)
    return out


def argmax_first(s: np.ndarray) -> int:
    """The lowest index of the maximum, NaN never picked (sampler.cu:143-151)."""
    return int(np.argmax(np.where(np.isnan(s), -np.inf, s)))


def sample_chain(s: np.ndarray, sp: Sampling, counter: int) -> int:
    """top_k -> top_p -> min_p -> temperature -> Philox pick over (penalized) logits `s` (sampled_reference)."""
    s = np.where(np.isnan(s), np.float32(-np.inf), s).astype(np.float32)
    k = min(sp.top_k if 0 < sp.top_k < TOP_K_MAX else TOP_K_MAX, len(s))
    thr = s[np.argpartition(-s, k - 1)[:k]].min()
    cand = np.nonzero(s >= thr)[0]                               # every tie at the threshold, ascending ids
    sel_ids = cand[np.lexsort((cand, -s[cand]))][:k]             # value descending, ties -> lower id
    sel = s[sel_ids]
    n_keep = len(sel)
    if sp.top_p < 1.0:
        m0 = float(sel[0])
        total = 0.0
        for x in sel:
            total += math.exp(float(x) - m0)
        cum, cut = 0.0, n_keep
        for i, x in enumerate(sel):
            cum += math.exp(float(x) - m0) / total
            if cum >= float(sp.top_p):
                cut = i + 1
                break
        n_keep = max(cut, min(1, n_keep))                        # min_keep = 1
    if sp.min_p > 0:
        thresh = np.float32(sel[0] + np.float32(np.log(sp.min_p)))
        for i in range(n_keep):
            if sel[i] < thresh:
                n_keep = i
                break
    inv_t = np.float32(1.0) / np.float32(sp.temperature)
    sc = [float(np.float32(x * inv_t)) for x in sel[:n_keep]]
    smx = max(sc)
    total = 0.0
    for x in sc:
        total += math.exp(x - smx)
    u = float(philox_uniform(sp.seed, counter))
    cum = 0.0
    for i, x in enumerate(sc):
        cum += math.exp(x - smx) / total
        if u < cum:
            return int(sel_ids[i])
    return int(sel_ids[n_keep - 1])


def pick(logits: np.ndarray | None, device_argmax: int, hist: list[int], sp: Sampling, counter: int) -> int:
    """The next token: device argmax for plain greedy, else Strata's chain on the read-back logits."""
    if not sp.sampled and not sp.penalized:
        return device_argmax
    s = apply_penalties(logits, hist, sp) if sp.penalized else np.asarray(logits, dtype=np.float32)
    return sample_chain(s, sp, counter) if sp.sampled else argmax_first(s)


# ------------------------------------------------------------------------------------------------ engine
@dataclass
class Checkpoint:
    ids: tuple
    snap: dict
    used: int
    tail: tuple | None = None          # Drafter.tail at the checkpoint (MTP on)


class Drafter:
    """The MTP drafter's side of the decode loop: fnx.tt.mtp.Mtp fed with the final residuals that
    TtModel.forward(keep_R=True) hands over.  Its K/V cell c holds (R of position c, token at c + 1), so the cells
    [0, len(live) - 1) are written as the prompt chunks and the committed window rows go by; `tail` = (cell, R row on
    the host) of the last consumed position, whose next token is the next request's (rewritten when it resumes)."""

    def __init__(self, mtp, k: int = 3, min_p: float = 0.0):
        import ttnn
        from .tt.model import to_host
        from .tt.mtp import _rows
        self.ttnn, self.to_host, self.rows = ttnn, to_host, _rows
        self.mtp, self.k, self.min_p = mtp, k, min_p
        self.tail: tuple | None = None

    def reset(self) -> None:
        self.mtp.reset()
        self.tail = None

    def _row(self, R, i: int):
        """Row i of R on the host [1, 10240] (a full-range slice would alias R: never deallocate that one)."""
        if R.shape[-2] == 1:
            return self.to_host(R, 1)
        r = self.rows(R, i, i + 1)
        h = self.to_host(r, 1)
        self.ttnn.deallocate(r)
        return h

    def cells(self, R, cell0: int, next_tokens: list[int]) -> None:
        """K/V of cells cell0.. from R rows 0.. (one per next token); the last row becomes the tail.  R stays the
        caller's to free (Mtp.prefill frees only its own sub-slices)."""
        n = len(next_tokens)
        self.tail = (cell0 + n - 1, self._row(R, n - 1))
        Rn = R if R.shape[-2] == n else self.rows(R, 0, n)
        self.mtp.prefill(Rn, next_tokens, cell0)
        if Rn is not R:
            self.ttnn.deallocate(Rn)

    def resume(self, cell: int, next_token: int) -> None:
        """A request continues after `cell`: its K/V with the token that actually follows."""
        if self.tail is not None and self.tail[0] == cell:
            self.mtp.prefill(self.tail[1], [next_token], cell)

    def draft(self, R, picks: list[int], p: int) -> list[int]:
        """Drafts after a window at p whose first len(picks) rows were committed (catch-up K/V of those cells)."""
        a = len(picks) - 1
        self.tail = (p + a, self._row(R, a))
        drafts, _ = self.mtp.draft(R, picks, p, a, steps=self.k, min_p=self.min_p)
        return drafts


class Engine:
    """One sequence on a TtModel-like `model` (forward(tokens, head=, read_logits=, keep_R=, verify=), commit,
    free_R, reset, snapshot, restore, pos); `drafter` (Drafter | None): speculative decoding with the MTP drafts."""

    def __init__(self, model, emit, ctx: int = CTX_MAX, chunk: int = DEFAULT_CHUNK, prompt_cache: int = 6,
                 eos=EOS_IDS, drafter: Drafter | None = None):
        self.model, self.emit, self.drafter = model, emit, drafter
        self.ctx, self.chunk, self.prompt_cache, self.eos = ctx, chunk, prompt_cache, set(eos)
        self.live: list[int] = []          # tokens the model state has consumed (model.pos == len(live))
        self.live_ok = False               # a later request may continue from `live` (not after ckpt=0)
        self.checks: list[Checkpoint] = []
        self.clock = 0
        self.stats: dict = {}              # the last request's decode: tokens, ms, rounds, drafted, accepted
        self.model.reset()
        if drafter is not None:
            drafter.reset()

    # ---- commands
    def handle(self, line: str, stop: threading.Event) -> None:
        if line.startswith("GEN "):
            try:
                max_new, sp, ids = parse_gen(line)
                n = len(ids)
                if n + max_new + 8 > self.ctx:
                    raise BadRequest(f"prompt ({n} tokens) + max_new ({max_new}) exceeds the context ({self.ctx})")
                if any(t >= N_VOCAB for t in ids):
                    raise BadRequest("a token id is outside the vocabulary")
            except BadRequest as e:
                self.emit(f"ERR {e}")
                return
            self.generate(ids, max_new, sp, stop)
        elif line.startswith("GENI "):
            self.emit("ERR this engine was started without --vision")
        elif line == "VRAM" or line.startswith("VRAM "):
            self.emit("ERR VRAM needs an engine started with --vram-elastic (the fnx TT engine has no expert cache)")
        elif line.startswith(("SAVE ", "RESTORE ")):
            self.emit("SERR invalid 0 session files are not supported by the fnx TT engine")
        else:
            self.emit(ERR_EXPECTED)

    # ---- conversation cache
    def _resume(self, ids: list[int]) -> tuple[int, str]:
        """Restore the longest reusable prefix of `ids` -> (tokens reused, source)."""
        n = len(ids)
        best, src = 0, None
        if self.prompt_cache > 0:
            L = len(self.live)
            if self.live_ok and 1 <= L <= n - 1 and ids[:L] == self.live:
                best, src = L, "live"
            for c in self.checks:
                L = len(c.ids)
                if best < L <= n - 1 and tuple(ids[:L]) == c.ids:
                    best, src = L, c
        if src is None:
            self.model.reset()
            if self.drafter is not None:
                self.drafter.reset()
            self.live = []
            return 0, "none"
        if src == "live":
            return best, "live"
        t0 = time.perf_counter()
        self.model.restore(src.snap)
        if self.drafter is not None:
            self.drafter.tail = src.tail
        self.clock += 1
        src.used = self.clock
        self.live = list(src.ids)
        log(f"conversation cache: restored {best} tokens (checkpoint) in {1e3 * (time.perf_counter() - t0):.1f} ms")
        return best, "checkpoint"

    def _checkpoint(self) -> None:
        L = len(self.live)
        self.clock += 1
        for c in self.checks:
            if len(c.ids) == L:                         # every checkpoint is a prefix of live: the same one
                c.used = self.clock
                return
        t0 = time.perf_counter()
        self.checks.append(Checkpoint(tuple(self.live), self.model.snapshot(), self.clock,
                                      self.drafter.tail if self.drafter is not None else None))
        while len(self.checks) > self.prompt_cache:
            self.checks.remove(min(self.checks, key=lambda c: c.used))
        log(f"conversation cache: checkpoint at {L} tokens in {1e3 * (time.perf_counter() - t0):.1f} ms; "
            f"{len(self.checks)} held")

    def _drop_stale(self) -> None:
        """A checkpoint is valid only while the positional KV cells [0, L) hold its tokens."""
        live = tuple(self.live)
        self.checks = [c for c in self.checks if live[:len(c.ids)] == c.ids]

    # ---- one request
    def generate(self, ids: list[int], max_new: int, sp: Sampling, stop: threading.Event) -> None:
        t0 = time.perf_counter()
        n = len(ids)
        m = self.model
        resume, src = self._resume(ids)
        self.live_ok = False                            # until this request ends cleanly
        self.emit(f"RESUME {resume}")
        turn_at = -1
        if self.prompt_cache > 0 and sp.ckpt:
            turn_at = next((i for i in range(n - 1, resume, -1) if ids[i] == TURN_TOKEN), -1)
        cancelled = False
        dr = self.drafter
        if dr is not None and resume > 0:
            dr.resume(resume - 1, ids[resume])
        at = resume
        for to in (turn_at, n - 1):
            if to <= at:
                continue
            for c0 in range(at, to, self.chunk):
                c1 = min(c0 + self.chunk, to)
                if dr is None:
                    m.forward(ids[c0:c1], head=False)
                else:
                    _, _, R = m.forward(ids[c0:c1], head=False, keep_R=True)
                    dr.cells(R, c0, ids[c0 + 1:c1 + 1])
                    m.free_R(R)
                self.live.extend(ids[c0:c1])
                ms = 1e3 * (time.perf_counter() - t0)
                self.emit(f"PP {c1} {n} {ms:.0f} {1e3 * (c1 - resume) / max(ms, 1e-3):.1f}")
                if stop.is_set():
                    cancelled = True
                    break
            if cancelled:
                break
            at = to
            if to == turn_at:
                self._checkpoint()
        read = len(self.live) - resume
        if not cancelled and hasattr(m, "prepare_decode"):
            m.prepare_decode()                          # traced decode: recapture after new prompt shapes (TTFT)
        t1 = time.perf_counter()
        prompt_ms = 1e3 * (t1 - t0)
        self.emit(f"REUSED {resume}")
        out, finish = [], "cancel" if cancelled else "length"
        need = sp.sampled or sp.penalized
        st = {"rounds": 0, "drafted": 0, "accepted": 0}
        x, p = ids[-1], n - 1
        if dr is None:
            while not cancelled and len(out) < max_new:
                if stop.is_set():
                    finish = "cancel"
                    break
                logits, am = m.forward([x], read_logits=need)
                self.live.append(x)
                y = pick(logits[0] if need else None, int(am[0]), self.live, sp, p)
                self.emit(f"T {y}")
                out.append(y)
                if y in self.eos:
                    finish = "stop"
                    break
                x, p = y, p + 1
        else:
            # verify windows [x, drafts...] at p..: row t's pick (device argmax, or Strata's chain with counter p + t
            # over the tokens consumed through row t) is emitted while the drafts before it matched; the window rows
            # are bitwise the serial loop's (TtModel docstring), so the emitted tokens are the serial loop's
            drafts: list[int] = []
            while not cancelled and len(out) < max_new:
                if stop.is_set():
                    finish = "cancel"
                    break
                win = [x] + drafts
                logits, am, R = m.forward(win, read_logits=need, keep_R=True, verify=True)
                picks = []
                for t in range(len(win)):
                    self.live.append(win[t])
                    y = pick(logits[t] if need else None, int(am[t]), self.live, sp, p + t)
                    picks.append(y)
                    self.emit(f"T {y}")
                    out.append(y)
                    if y in self.eos:
                        finish = "stop"
                        break
                    if len(out) >= max_new or t + 1 == len(win) or win[t + 1] != y:
                        break
                n_keep = len(picks)
                m.commit(n_keep)
                st["rounds"] += 1
                st["drafted"] += len(drafts)
                st["accepted"] += n_keep - 1
                if finish == "stop" or len(out) >= max_new or stop.is_set():
                    dr.cells(R, p, picks)                   # the committed cells; no more drafts needed
                    m.free_R(R)
                    break
                drafts = dr.draft(R, picks, p)
                m.free_R(R)
                x, p = picks[-1], p + n_keep
        if finish == "length" and stop.is_set():
            finish = "cancel"
        decode_ms = 1e3 * (time.perf_counter() - t1)
        assert m.pos == len(self.live), (m.pos, len(self.live))
        self.live_ok = sp.ckpt and not cancelled
        self._drop_stale()
        self.stats = dict(st, tokens=len(out), decode_ms=decode_ms, prompt_ms=prompt_ms, finish=finish)
        self.emit(f"DONE {len(out)} {n} {prompt_ms:.1f} {decode_ms:.1f} {finish} 0 0 {resume} 0 0 0 0 0.0 {read} 0")
        spec = (f"; MTP {st['rounds']} windows, {st['accepted']}/{st['drafted']} drafts accepted "
                f"({st['accepted'] / max(st['drafted'], 1):.2f}), {len(out) / max(st['rounds'], 1):.2f} tokens/window"
                if dr is not None else "")
        tr = getattr(m, "_trace", None)
        if tr is not None:
            spec += f"; traces recaptured {tr.recaptures}x in total ({tr.recapture_s:.1f} s)"
        log(f"prompt {n} tokens = {resume} reused ({src}) + {read} read in {prompt_ms:.0f} ms "
            f"({1e3 * read / max(prompt_ms, 1e-3):.1f} tok/s), {len(out)} generated in {decode_ms:.0f} ms "
            f"({1e3 * len(out) / max(decode_ms, 1e-3):.2f} tok/s), finish {finish}; "
            f"{'sampled' if sp.sampled else 'greedy'}{' + penalties' if sp.penalized else ''}{spec}")


def make_drafter(mesh, w, max_ctx: int, k: int, min_p: float = 0.0, model=None, mw=None) -> Drafter:
    """The MTP drafter (fnx.tt.mtp.Mtp: its 512 routed experts on the device, fnx.tt.moe_device; FNX_MTP_RESIDENT
    (default fnx.tt.mtp.DEFAULT_RESIDENT) of them, the hottest, in device DRAM, the others and its embedding rows in
    pinned host memory: `model`'s pool when it was built with host_reserve=fnx.tt.mtp.reserve_items(), else its own).
    mw: its fnx.ref.mtp.MtpWeights (default: MtpWeights(), fnx.paths.MTP_RT)."""
    from .ref.mtp import MtpWeights
    from .tt.mtp import Mtp, env_resident
    t0 = time.time()
    resident = env_resident()
    first = getattr(model, "host_reserve_first", None)
    pool = model.host_pool if first is not None else None
    # Strata's drafter attends a 32768-cell window; the served positions stay below the context (<= 32768), so a
    # window over every cell the drafter holds (context + draft steps, Mtp's 256-cell rounding) is the same attention
    mtp = Mtp(mesh, w, mw or MtpWeights(), max_ctx=max_ctx, steps=k,
              window=max(32768, -(-(max_ctx + k + 1) // 256) * 256),
              resident=resident, pool=pool, pool_first=first or 0)
    mtp.probs = min_p > 0               # the draft probabilities only serve the min_p chain stop
    log(f"MTP drafter built in {time.time() - t0:.0f} s ({k} drafts per window; {min(resident, 512)} of its 512 "
        f"experts in device DRAM ({mtp.moe.store.nbytes / 1e9:.2f} GB), {mtp.moe.store.host_bytes / 1e9:.2f} GB in its host "
        f"pool; {mtp.pinned / 1e9:.2f} GB pinned with its embedding rows, {'the model' if pool else 'its own'} pool)")
    return Drafter(mtp, k, min_p)


def warm_up(model, drafter: Drafter | None, chunk: int) -> None:
    """Program builds for the prompt chunk and decode shapes, then a fresh sequence.  With the drafter: the first
    window (1 row), verify windows of 2..1+k rows with every partial commit (1..T rows kept: the PLE window slice,
    the drafter's catch-up over the kept rows, the residual row it keeps as the tail) and the committed-cells
    prefill of a request's last window.  Every eager op of a decode step runs here once, so a traced decode
    (TtModel.enable_trace, captured after this) compiles nothing and never has to recapture.  Starts from a fresh
    sequence whatever the model held before (a caller may have used it, e.g. the quality gate's teacher forcing)."""
    model.reset()
    if drafter is not None:
        drafter.reset()
    prompt = [TURN_TOKEN] + [198] * (chunk - 1)
    if drafter is None:
        model.forward(prompt, head=False)
    else:
        _, _, R = model.forward(prompt, head=False, keep_R=True)
        drafter.cells(R, 0, prompt[1:] + [198])
        model.free_R(R)
    if drafter is None:
        model.forward([198], read_logits=True)
    else:
        for T in range(1, drafter.k + 2):               # windows of 1..1+k rows, every partial commit
            for n_keep in range(1, T + 1):
                p = model.pos
                _, ids, R = model.forward([198] * T, read_logits=True, keep_R=True, verify=True)
                picks = [int(i) for i in ids[:n_keep]]
                model.commit(n_keep)
                drafter.draft(R, picks, p)
                drafter.cells(R, p, picks)              # a request's last window: its committed cells
                drafter.resume(drafter.tail[0], 198)    # a continued request: the tail cell from the host row
                model.free_R(R)
        drafter.reset()
    model.reset()


def serve(engine: Engine, fd: int, emit) -> int:
    """The command loop (generate.cpp:7076-7130): a reader thread splits file descriptor `fd` into lines with
    read(2) (no buffered stdin object: a daemon thread blocked in one aborts the interpreter's shutdown), acts on
    STOP at once and queues everything else; requests run one at a time.  Returns the exit code (0 QUIT/EOF, 1
    device failure)."""
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def put(raw: bytes) -> None:
        line = raw.decode("utf-8", "replace")
        line = line[:-1] if line.endswith("\r") else line
        if line == "STOP":
            stop.set()
            return
        if line.startswith(("GEN ", "GENI ")):         # a STOP before this GEN belonged to the previous request
            stop.clear()
        q.put(line)

    def reader():
        buf = b""
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError:
                chunk = b""
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                put(raw)
        if buf:                                         # a last line without its newline
            put(buf)
        q.put(None)

    threading.Thread(target=reader, daemon=True, name="stdin").start()
    while True:
        line = q.get()
        if line is None or line == "QUIT":
            log("QUIT" if line else "stdin closed")
            return 0
        try:
            engine.handle(line, stop)
        except Exception as e:                        # noqa: BLE001 - a device failure: report, then exit
            traceback.print_exc(file=sys.stderr)
            emit("ERR fnx engine: " + f"{type(e).__name__}: {e}".replace("\n", " ")[:500])
            return 1


def protocol_stdout():
    """The real stdout for protocol lines only; fd 1 (C++ and Python prints) goes to stderr from here on."""
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(fd, "w", encoding="utf-8", buffering=1)


def main(argv=None) -> int:
    proto = protocol_stdout()
    lock = threading.Lock()

    def emit(s: str) -> None:
        with lock:
            proto.write(s + "\n")
            proto.flush()

    ap = argparse.ArgumentParser(prog="fnx.serve_engine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="Strata's engine protocol on stdin/stdout (required)")
    ap.add_argument("--max-context", type=int, default=CTX_MAX, help=f"context, at most {CTX_MAX}")
    ap.add_argument("--native", default=None, help="GGUF shard 1 (default: fnx.paths.SHARD1, $FNX_GGUF)")
    ap.add_argument("--ple-gguf", default=None,
                    help="the GGUF shard with the PLE table (default: the other shards named like --native's)")
    ap.add_argument("--mtp", default=None, help="Strata's MTP runtime directory, <data>/mtp/rt (default: "
                                                "fnx.paths.MTP_RT)")
    ap.add_argument("--pack", default=None, help="Strata's pack of the model: the server's tokenizer; the TT engine "
                                                 "reads its tensors from the GGUF shards and FNX_DATA")
    ap.add_argument("--prefill", default="auto", help=f"prompt rows per device call (auto = {DEFAULT_CHUNK})")
    ap.add_argument("--prompt-cache", type=int, default=6, help="conversation checkpoints kept (0 = no reuse)")
    ap.add_argument("--threads", type=int, default=16, help="CPU expert threads")
    ap.add_argument("--eos-ids", default=None, help="comma list (default 248044,248046)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the warm-up forward passes before READY")
    ap.add_argument("--spec", type=int, default=int(os.environ.get("FNX_SPEC", DEFAULT_SPEC)),
                    help=f"verify-window rows with the MTP drafter: 1 + drafts per window (default $FNX_SPEC, else "
                         f"{DEFAULT_SPEC}; 1 = plain decode, no drafter)")
    ap.add_argument("--spec-min-p", type=float, default=float(os.environ.get("FNX_SPEC_MIN_P", 0.0)),
                    help="stop a window's draft chain at a draft whose drafter probability is below this")
    a, unknown = ap.parse_known_args(argv)
    if not a.serve:
        ap.error("only --serve is implemented")
    if unknown:
        log(f"ignoring engine arguments the TT engine does not use: {' '.join(unknown)}")
    ctx = min(a.max_context, CTX_MAX)
    if a.max_context > CTX_MAX:
        log(f"--max-context {a.max_context}: fnx.tt.qsa's sparse selection is verified to {CTX_MAX} positions; "
            f"serving {CTX_MAX}")
    chunk = DEFAULT_CHUNK if a.prefill == "auto" else max(1, int(a.prefill))
    eos = tuple(int(x) for x in a.eos_ids.split(",")) if a.eos_ids else EOS_IDS

    def on_term(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)

    t0 = time.time()
    import ttnn

    from .cpu_experts import CpuExperts
    from .config import N_LAYERS
    from .tt.model import DEFAULT_PLAN, TtModel, open_mesh
    from .weights import Weights

    spec = max(1, a.spec)
    w = Weights(a.native, ple=a.ple_gguf)
    mw = None
    if spec > 1:
        from .ref.mtp import MtpWeights
        mw = MtpWeights(a.mtp)
    log(f"model: {', '.join(str(f.path) for f in w.model.files)}; MTP {mw.rt if mw else '(off: --spec 1)'}; "
        f"pack {a.pack or '-'} (the server's tokenizer); data {paths.DATA}; engine {source_version()}")
    experts = CpuExperts(w.model, N_LAYERS, a.threads)
    mesh = open_mesh()
    try:
        moe = os.environ.get("FNX_MOE", DEFAULT_MOE)
        from .tt.mtp import reserve_items
        model = TtModel(mesh, w, experts, max_ctx=max(ctx, chunk), moe=moe,   # KV + indexer state (warm-up: chunk rows)
                        host_reserve=reserve_items(mw=mw) if spec > 1 else 0)   # + the drafter's host items
        log(f"TtModel built in {time.time() - t0:.0f} s: MoE {moe}" + (
            f" (plan {os.environ.get('FNX_MOE_PLAN', DEFAULT_PLAN)}: experts "
            f"{sum(lay[3].store.nbytes for lay in model.layers) / 1e9:.2f} GB device DRAM + "
            f"{sum(lay[3].store.host_bytes for lay in model.layers) / 1e9:.2f} GB host pool, "
            f"{(model.host_pool.nbytes if model.host_pool else 0) / 1e9:.2f} GB pinned)" if moe == "device" else ""))
        drafter = make_drafter(mesh, w, max(ctx, chunk), spec - 1, a.spec_min_p, model, mw) if spec > 1 else None
        if not a.no_warmup:                              # program builds for the prompt and decode shapes
            t1 = time.time()
            warm_up(model, drafter, chunk)
            log(f"warm-up in {time.time() - t1:.1f} s")
        if os.environ.get("FNX_TRACE", "1") != "0":     # traced decode steps (fnx.tt.trace); FNX_TRACE=0: eager
            t1 = time.time()
            model.enable_trace(Ts=(1, spec) if spec > 1 else (1,), verbose=False,
                               drafter=drafter.mtp if drafter is not None else None)
            segs = {k: len(s.tids) for k, s in model._trace.seqs.items()}
            log(f"decode traced (T = 1{f', {spec}' if spec > 1 else ''}{', drafter rounds' if drafter else ''}) in "
                f"{time.time() - t1:.1f} s; captured segments per forward: dense "
                f"{sorted({n for (T, sp), n in segs.items() if sp is not None})}, sparse "
                f"{sorted({n for (T, sp), n in segs.items() if sp is None})}")
        engine = Engine(model, emit, ctx=ctx, chunk=chunk, prompt_cache=max(0, a.prompt_cache), eos=eos,
                        drafter=drafter)
        log(f"ready in {time.time() - t0:.0f} s: context {ctx}, prompt chunk {chunk}, "
            f"{engine.prompt_cache} conversation checkpoints, eos {sorted(engine.eos)}, "
            + (f"MTP verify windows of {spec} rows (draft min_p {a.spec_min_p})" if drafter else "plain decode"))
        emit(f"INFO context={ctx} kv=bf16 kv_resident=0 spec={spec} mtp_max={spec if drafter else 0} lookup=0 "
             f"prompt_chunk={chunk} conversation_cache_slots={engine.prompt_cache} tail_role_token=-1 "
             f"vram_elastic=0 cvec=0 device=p150 engine={source_version()}")
        ready = os.environ.get("FNX_READY_FILE")          # tt/serve/tt-engine.sh's start bound: READY reached
        if ready:
            Path(ready).touch()
        emit(f"READY {ctx} stop")
        return serve(engine, sys.stdin.fileno(), emit)
    finally:
        t1 = time.time()
        ttnn.close_mesh_device(mesh)
        log(f"device closed in {time.time() - t1:.1f} s")


if __name__ == "__main__":
    raise SystemExit(main())
