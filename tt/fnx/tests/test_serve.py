"""Strata's serving path on the P150: fnx.serve_engine over Strata's engine protocol, then Strata's own server
(serve/server.py of this repository) over HTTP with tt/serve/tt-engine.sh as its engine.

Runs on the HOST (Strata's venv), not through tools/devrun.sh: tt-engine.sh takes the device lock itself (a devrun
wrapper would hold it and deadlock).  From the repository's tt/:

    $FNX_DATA/venv/bin/python -m fnx.tests.test_serve [--skip proto,http] [--port 8090] [--config CONFIG]

The engine configuration is a Strata run config (--config, e.g. one `setup.py --backend tt` wrote; default: the one
this test writes to FNX_DATA/logs/test-serve-config.json from fnx.paths' dev data: exe tt/serve/tt-engine.sh, args
--pack --native --ple-gguf --mtp --prefill 128 --max-context 32768 --spec 4 --spec-min-p 0.5, env FNX_DATA).  The
protocol part runs the config's exe with its model-file arguments (--pack, --native, --ple-gguf, --mtp) and its env.

Reference answer of the golden chat prompt (greedy, thinking off): with the Strata-exact CPU experts (FNX_MOE=cpu, the
Coder) Strata's own 50 serve tokens incl. EOS; otherwise (the device MoE: re-quantized experts, other logits than
Strata's; iq3s, which has no Strata golden) the same engine configuration's plain eager greedy decode (one more engine
process with --spec 1 and FNX_TRACE=0), so the checks below prove the served path (traced, MTP verify windows,
checkpoints) exact against it.
The MoE is FNX_MOE from the environment, else the variant's fnx.paths.DEFAULT_MOE.

P. Protocol, one engine process (<exe> --serve <model files> --max-context 2048 --prefill auto):
   READY <2048> stop, INFO engine= the repository's version; request-level refusals (ERR / SERR, engine stays idle);
   the golden chat prompt greedy -> the reference tokens, exactly; a follow-up that extends the conversation ->
   RESUME = the 80 held tokens and only the suffix read; the chat prompt again -> resumed from the turn checkpoint
   (RESUME 24) and the same 50 tokens (snapshot/restore exactness); seeded sampling reproducible; a large presence
   penalty never repeats a windowed token; STOP mid-decode -> DONE cancel; a stale STOP is ignored; QUIT -> exit 0,
   device lock released.  TTFT and decode tok/s of the golden request.  The engine runs its default speculative
   decoding (MTP verify windows, INFO spec=4; FNX_SPEC=1 in the environment runs plain decode): P8 allows a window's
   tokens of lag.
H. HTTP on --port (tt/tools/fnx-serve.sh: this repository's serve/server.py --engine strata --config CONFIG):
   /v1/chat/completions non-stream (thinking off) = the reference answer's text; a follow-up turn reusing the
   conversation (timings.cache_n = everything the engine held, prompt_n = the new suffix only); the same chat
   streamed (content equal, from the turn checkpoint); /v1/messages; a streamed request whose client disconnects
   -> the engine ends it (DONE cancel in its log) and the next request is served; server stop -> engine QUIT, lock
   released.  Prints the chat transcript, TTFT and decode tok/s.
"""
from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from fnx import paths

LOCK = paths.DEVICE_LOCK
GOLDEN = paths.GOLDEN
LOGS = paths.LOGS
MODEL_ARGS = ("--pack", "--native", "--ple-gguf", "--mtp")     # the config's model files, given to every engine
EOS = (248044, 248046)
TURN = 248045
CHAT_Q = "What is the capital of France, and what is it famous for? Answer in two sentences."
FOLLOW_Q = "And which river flows through it? One sentence."
START_TIMEOUT = 3600            # the device lock may be held by another team's run
MAX_NEW = 64                    # tokens of the golden chat requests

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
    return bool(ok)


def read_ids(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().replace(",", " ").split()]


def dev_config(port: int) -> Path:
    """A Strata run config for this repository's engine wrapper on fnx.paths' dev data (what setup's to_tt writes)."""
    cfg = {"exe": str(paths.TT_DIR / "serve" / "tt-engine.sh"),
           "args": ["--pack", str(paths.PACK), "--native", str(paths.SHARD1), "--ple-gguf", str(paths.SHARD2),
                    "--mtp", str(paths.MTP_RT), "--prefill", "128", "--max-context", "32768", "--spec", "4",
                    "--spec-min-p", "0.5"],
           "cwd": str(paths.REPO), "tokenizer": str(paths.PACK / "tokenizer"),
           "model_name": f"qwen3.8-flash-next-{paths.PACK.name}", "log": str(LOGS / "test-serve-strata.log"),
           "lib_dirs": [], "port": port, "idle_unload_s": 900, "backend": "tt", "env": {"FNX_DATA": str(paths.DATA)}}
    p = LOGS / "test-serve-config.json"
    p.write_text(json.dumps(cfg, indent=2) + "\n")
    return p


def model_args(cfg: dict) -> list[str]:
    a, out = cfg["args"], []
    for k in MODEL_ARGS:
        if k in a:
            out += [k, a[a.index(k) + 1]]
    return out


def tokenizer(cfg: dict):
    sys.path.insert(0, str(paths.STRATA_TOOLS))
    import strata_tokenizer as ST
    t = Path(cfg["tokenizer"])
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    return ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((t / "token_type.json").read_text()))


def lock_free() -> bool:
    with open(LOCK) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def engine_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "--filter", "name=tt-engine-", "--format", "{{.Names}}"],
                         capture_output=True, text=True).stdout
    return out.split()


def released(wait_s: float = 20.0) -> bool:
    """The engine's container is gone (its docker client, the only holder of the device lock, has exited).  The
    lock itself may already be taken by the next waiting job, so its state is only reported."""
    t0 = time.time()
    while engine_containers() and time.time() - t0 < wait_s:
        time.sleep(1)
    print(f"   device lock now {'free' if lock_free() else 'held (the next device job)'}", flush=True)
    return not engine_containers()


# ------------------------------------------------------------------------------------------------ protocol
class EngineProc:
    """The config's engine exe (tt/serve/tt-engine.sh) as serve/server.py runs it: line pipes, stderr appended to a
    log, the config's cwd and env."""

    def __init__(self, cfg: dict, args: list[str], log: Path, env: dict | None = None):
        self.log = open(log, "a", encoding="utf-8")
        self.p = subprocess.Popen([cfg["exe"], "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, encoding="utf-8", bufsize=1, cwd=cfg["cwd"],
                                  env={**os.environ, **cfg.get("env", {}), **(env or {})})
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put((time.perf_counter(), line.rstrip("\n")))
        self.q.put((time.perf_counter(), None))

    def send(self, line: str) -> float:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return time.perf_counter()

    def line(self, timeout: float = 600.0) -> tuple[float, str | None]:
        return self.q.get(timeout=timeout)

    def ready(self) -> tuple[dict, list[str]]:
        info = {}
        while True:
            _, line = self.line(START_TIMEOUT)
            if line is None:
                raise RuntimeError("engine exited before READY")
            if line.startswith("INFO "):
                info = dict(kv.split("=", 1) for kv in line.split()[1:])
            if line.startswith("READY"):
                return info, line.split()

    def gen(self, ids: list[int], max_new: int, keys: str = "", stop_after: int | None = None) -> dict:
        """One GEN; STOP after `stop_after` T lines.  -> lines by kind, tokens, DONE fields, timings."""
        r = {"T": [], "PP": [], "t_T": [], "other": []}
        r["t0"] = self.send(f"GEN {max_new}{keys} {','.join(map(str, ids))}")
        while True:
            t, line = self.line()
            if line is None:
                raise RuntimeError("engine exited during a request")
            kind = line.split(" ", 1)[0]
            if kind == "T":
                r["T"].append(int(line[2:]))
                r["t_T"].append(t)
                if stop_after is not None and len(r["T"]) == stop_after:
                    r["t_stop"] = self.send("STOP")
            elif kind == "PP":
                r["PP"].append([float(x) for x in line.split()[1:]])
            elif kind in ("RESUME", "REUSED"):
                r[kind] = int(line.split()[1])
            elif kind == "DONE":
                f = line.split()
                r["done"], r["t_done"] = f, t
                r["finish"] = f[5]
                return r
            elif kind == "ERR":
                r["err"] = line
                return r
            else:
                r["other"].append(line)

    def answer(self, line: str) -> str:
        self.send(line)
        return self.line()[1]

    @property
    def container(self) -> str:
        """tt-engine.sh names its container after its own pid (the Popen pid)."""
        return f"tt-engine-{self.p.pid}"

    def close(self) -> None:
        """Make sure the engine and its container are gone, whatever state the test left them in: QUIT (if the
        engine still runs), then SIGTERM (tt-engine.sh forwards it, removes a container that does not exit within
        20 s and resets the board), then SIGKILL and `docker rm -f` of the container (the shared card)."""
        if self.p.poll() is None:
            try:
                self.send("QUIT")
                self.p.wait(timeout=60)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(timeout=30)
        if self.container in engine_containers():
            print(f"   engine container {self.container} still running after its client exited: docker rm -f",
                  flush=True)
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)


def rates(r: dict) -> tuple[float, float]:
    """(TTFT ms, decode tok/s after the first token) of a gen() result."""
    ttft = 1e3 * (r["t_T"][0] - r["t0"])
    n = len(r["t_T"]) - 1
    return ttft, (n / (r["t_T"][-1] - r["t_T"][0]) if n > 0 else float("nan"))


def moe() -> str:
    from fnx.paths import DEFAULT_MOE
    return os.environ.get("FNX_MOE", DEFAULT_MOE)


def reference(cfg: dict) -> list[int]:
    """The greedy answer to the golden chat prompt the engine must reproduce (module docstring): Strata's own for the
    cpu MoE when the variant has one (golden/chat/serve_gen.ids: the Coder's), else this configuration's plain eager
    greedy decode (the device MoE; iq3s, whose golden directory has only the prompts)."""
    strata = GOLDEN / "chat" / "serve_gen.ids"
    if moe() == "cpu" and strata.exists():
        return read_ids(strata)
    print(f"\n== reference: the {moe()} MoE's plain eager greedy decode (--spec 1, FNX_TRACE=0)", flush=True)
    chat = read_ids(GOLDEN / "prompts" / "chat.ids")
    t0 = time.perf_counter()
    e = EngineProc(cfg, model_args(cfg) + ["--max-context", "2048", "--spec", "1"], LOGS / "test-serve-engine.log",
                   {"FNX_TRACE": "0"})
    try:
        e.ready()
        r = e.gen(chat, MAX_NEW)
        e.send("QUIT")
        e.p.wait(timeout=120)
    finally:
        e.close()
    print(f"   {len(r['T'])} tokens, finish {r['finish']} ({time.perf_counter() - t0:.0f} s incl. load)", flush=True)
    return r["T"]


def finish_of(ref: list[int]) -> str:
    return "stop" if ref and ref[-1] in EOS else "length"


def part_proto(tok, cfg: dict, golden: list[int]) -> None:
    print(f"\n== P. protocol: {cfg['exe']} --serve (MoE {moe()})", flush=True)
    chat = read_ids(GOLDEN / "prompts" / "chat.ids")
    t0 = time.perf_counter()
    from fnx.serve_engine import source_version
    e = EngineProc(cfg, model_args(cfg) + ["--max-context", "2048", "--prefill", "auto"],
                   LOGS / "test-serve-engine.log")
    try:
        info, ready = e.ready()
        check("P1 READY", ready[1:] == ["2048", "stop"] and info.get("engine") == source_version(),
              f"{' '.join(ready)}, INFO {info} after {time.perf_counter() - t0:.0f} s (incl. lock wait)")
        refusals = [("GEN 0 1,2", "ERR bad request: max_new"), ("GEN 5 1,-2", "ERR bad request: invalid token id"),
                    ("GEN 5 1,248320", "ERR a token id is outside the vocabulary"),
                    (f"GEN 2017 {','.join(map(str, chat))}", "ERR prompt (31 tokens) + max_new (2017) exceeds the "
                                                             "context (2048)"),
                    ("HELLO", "ERR expected: GEN"), ("GENI 4 /tmp/x.emb 1,2", "ERR this engine was started without"),
                    ("VRAM 512", "ERR VRAM needs"), ("SAVE /tmp/fnx.session", "SERR invalid 0 ")]
        got = [(cmd, e.answer(cmd)) for cmd, _ in refusals]
        bad = [(c, g) for (c, w), (_, g) in zip(refusals, got) if not (g or "").startswith(w)]
        check("P2 request-level refusals", not bad, f"{len(refusals)} commands" + (f"; wrong: {bad}" if bad else ""))
        e.send("STOP")                                              # stale: no request runs
        r1 = e.gen(chat, MAX_NEW)
        ttft, tps = rates(r1)
        check("P3 golden chat greedy == the reference tokens", r1["T"] == golden and r1["finish"] == finish_of(golden)
              and r1["done"][1:3] == [str(len(golden)), "31"] and r1["RESUME"] == 0,
              f"{len(r1['T'])} tokens, first divergence "
              f"{next((i for i, (a, b) in enumerate(zip(r1['T'], golden)) if a != b), min(len(r1['T']), len(golden)))}"
              f"; DONE {' '.join(r1['done'][1:])}; TTFT {ttft:.0f} ms (31-token prompt), decode {tps:.2f} tok/s")
        follow = tok.encode(f"\n<|im_start|>user\n{FOLLOW_Q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                            parse_special=True)
        ids2 = chat + r1["T"] + follow
        held = len(chat) + len(r1["T"]) - 1                         # the final EOS was never fed
        r2 = e.gen(ids2, 48)
        ttft2, tps2 = rates(r2)
        read2 = int(r2["done"][14])
        check("P4 follow-up continues the held conversation", r2["RESUME"] == held and read2 == len(ids2) - 1 - held
              and r2["PP"][0][0] > held, f"RESUME {r2['RESUME']} (held {held}), read {read2} of {len(ids2)}; "
              f"TTFT {ttft2:.0f} ms; answer {tok.decode(r2['T'])!r}")
        r3 = e.gen(chat, MAX_NEW)
        turn = max(i for i, t in enumerate(chat) if t == TURN)
        check("P5 same chat again: from the turn checkpoint, same tokens", r3["RESUME"] == turn and r3["T"] == golden,
              f"RESUME {r3['RESUME']} (last <|im_start|> at {turn}), tokens equal: {r3['T'] == golden}")
        keys = " temperature=0.7 top_p=0.95 top_k=20 seed=42"
        s1, s2 = e.gen(chat, 16, keys), e.gen(chat, 16, keys)
        check("P6 seeded sampling reproducible", s1["T"] == s2["T"] and len(s1["T"]) > 0,
              f"{tok.decode(s1['T'])!r} (greedy: {tok.decode(golden[:len(s1['T'])])!r})")
        pen = e.gen(chat, 24, " penalty_present=1000.0 penalty_last_n=16")
        hist = chat + pen["T"]
        rep = [j for j, y in enumerate(pen["T"]) if y in hist[len(chat) + j - 16:len(chat) + j]]
        check("P7 presence penalty: no token repeats within its 16-token window", not rep,
              f"{tok.decode(pen['T'])!r}" + (f"; repeats at {rep}" if rep else ""))
        code = read_ids(GOLDEN / "prompts" / "code.ids")
        r4 = e.gen(code, 400, stop_after=3)
        lag = len(r4["T"]) - 3
        burst = max(1, int(info.get("spec", 1))) - 1               # a verify window's tokens arrive together
        check("P8 STOP mid-decode -> DONE cancel", r4["finish"] == "cancel" and 0 <= lag <= 2 + burst,
              f"{len(r4['T'])} tokens (STOP after 3; allowed lag {2 + burst}), DONE {' '.join(r4['done'][1:6])}, "
              f"stop->DONE {1e3 * (r4['t_done'] - r4['t_stop']):.0f} ms")
        e.send("STOP")                                              # stale again
        r5 = e.gen(chat, 2)
        check("P9 stale STOP ignored", r5["finish"] == "length" and len(r5["T"]) == 2,
              f"DONE {' '.join(r5['done'][1:6])}")
        t1 = time.perf_counter()
        e.send("QUIT")
        rc = e.p.wait(timeout=120)
        quit_s = time.perf_counter() - t1
        check("P10 QUIT -> exit 0, device lock released", rc == 0 and released(),
              f"rc {rc} after {quit_s:.1f} s")
    finally:
        e.close()


# ------------------------------------------------------------------------------------------------ HTTP
def post(port: int, path: str, body: dict, timeout: float = 900.0) -> tuple[dict, float]:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = c.getresponse()
    data = json.loads(resp.read())
    if resp.status != 200:
        raise RuntimeError(f"{path}: HTTP {resp.status}: {data}")
    return data, time.perf_counter() - t0


def stream(port: int, body: dict, close_after: int | None = None) -> dict:
    """POST a streaming chat completion; content deltas with arrival times.  close_after: drop the connection
    after that many content deltas (a client that went away)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}),
              {"Content-Type": "application/json"})
    sock = c.sock                                   # the response takes it over (the connection drops its handle)
    resp = c.getresponse()
    out = {"t0": t0, "deltas": [], "t": [], "final": None}
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        for ch in ev.get("choices") or []:
            d = (ch.get("delta") or {}).get("content")
            if d:
                out["deltas"].append(d)
                out["t"].append(time.perf_counter())
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
        if ev.get("timings"):
            out["final"] = ev
        if close_after is not None and len(out["deltas"]) >= close_after:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            resp.close()
            return out
    return out


def chat_body(messages: list[dict], max_tokens: int = 64) -> dict:
    return {"model": "qwen3.8-flash-next-coder-iq1_m", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def engine_log_since(log: Path, offset: int) -> str:
    with open(log, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace")


def part_http(tok, config: Path, port: int, golden: list[int]) -> None:
    serve = paths.TT_DIR / "tools" / "fnx-serve.sh"
    print(f"\n== H. HTTP: {serve} (serve/server.py --engine strata --config {config} --port {port})", flush=True)
    cfg = json.loads(config.read_text())
    log = Path(cfg["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    want = tok.decode([t for t in golden if t not in EOS]).strip()
    chat = read_ids(GOLDEN / "prompts" / "chat.ids")
    t0 = time.perf_counter()
    srv = subprocess.Popen([str(serve)], env={**os.environ, "PORT": str(port), "CONFIG": str(config)},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: queue.Queue = queue.Queue()

    def pump():
        for line in srv.stdout:
            print(f"   server| {line.rstrip()}", flush=True)
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        while True:
            line = lines.get(timeout=START_TIMEOUT)
            if line is None:
                raise RuntimeError("the server exited before it was ready")
            if line.startswith("ready: "):
                break
        ctx = cfg["args"][cfg["args"].index("--max-context") + 1]
        check("H0 server ready", f"context {ctx} tokens" in line, f"after {time.perf_counter() - t0:.0f} s")
        transcript = []
        q1 = [{"role": "user", "content": CHAT_Q}]
        d1, s1 = post(port, "/v1/chat/completions", chat_body(q1))
        a1 = d1["choices"][0]["message"]["content"]
        tm1 = d1.get("timings") or {}
        transcript += [("user", CHAT_Q), ("assistant", a1)]
        check("H1 /v1/chat/completions = the reference answer", a1.strip() == want and
              d1["choices"][0]["finish_reason"] == finish_of(golden),
              f"{s1:.1f} s; prompt {tm1.get('prompt_n')} read in {tm1.get('prompt_ms')} ms, "
              f"{tm1.get('predicted_n')} tokens at {tm1.get('predicted_per_second')} tok/s; usage {d1.get('usage')}")
        q2 = q1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOW_Q}]
        d2, s2 = post(port, "/v1/chat/completions", chat_body(q2))
        a2 = d2["choices"][0]["message"]["content"]
        tm2 = d2.get("timings") or {}
        held = len(chat) + len(golden) - 1                      # H1's prompt + answer; its last token was never fed
        transcript += [("user", FOLLOW_Q), ("assistant", a2)]
        check("H2 follow-up turn reuses the conversation (only the suffix read)",
              tm2.get("cache_n") == held and tm2.get("prompt_n", 0) < 40,
              f"{s2:.1f} s; cache_n {tm2.get('cache_n')} (held {held}), prompt_n {tm2.get('prompt_n')} read in "
              f"{tm2.get('prompt_ms')} ms, {tm2.get('predicted_n')} tokens at {tm2.get('predicted_per_second')} tok/s")
        st = stream(port, chat_body(q1))
        text = "".join(st["deltas"])
        ttft = 1e3 * (st["t"][0] - st["t0"]) if st["t"] else float("nan")
        n_d = len(st["t"]) - 1
        tps = n_d / (st["t"][-1] - st["t"][0]) if n_d > 0 else float("nan")
        tm3 = (st["final"] or {}).get("timings") or {}
        turn = max(i for i, t in enumerate(chat) if t == TURN)
        check("H3 streamed chat = the same answer (from the turn checkpoint)",
              text.strip() == want and st.get("finish") == finish_of(golden) and tm3.get("cache_n") == turn,
              f"client TTFT {ttft:.0f} ms, {len(st['deltas'])} deltas at {tps:.2f}/s; cache_n {tm3.get('cache_n')}, "
              f"prompt_n {tm3.get('prompt_n')}")
        d4, s4 = post(port, "/v1/messages", {"model": "x", "max_tokens": 64, "thinking": {"type": "disabled"},
                                             "messages": [{"role": "user", "content": CHAT_Q}]})
        a4 = "".join(b.get("text", "") for b in d4.get("content", []) if b.get("type") == "text")
        check("H4 /v1/messages (Anthropic)", a4.strip() == want and
              d4.get("stop_reason") == ("end_turn" if finish_of(golden) == "stop" else "max_tokens"),
              f"{s4:.1f} s; stop_reason {d4.get('stop_reason')}, usage {d4.get('usage')}")
        off = log.stat().st_size
        long_q = [{"role": "user", "content": "Write a Python function that prints the numbers from 1 to 300, one "
                                              "per line, without using loops. Explain it in detail."}]
        dc = stream(port, chat_body(long_q, 400), close_after=4)
        t_close = time.perf_counter()
        cancel = None
        while time.perf_counter() - t_close < 120 and cancel is None:
            time.sleep(1)
            cancel = re.search(r"(\d+) generated in \d+ ms .*finish cancel", engine_log_since(log, off))
        d5, s5 = post(port, "/v1/chat/completions", chat_body(q1, 8))
        check("H5 client disconnect -> STOP -> engine ends the request", cancel is not None and
              int(cancel.group(1)) < 60 and d5["choices"][0]["message"]["content"],
              f"{len(dc['deltas'])} deltas read, then closed; engine: "
              f"{cancel.group(0) if cancel else 'no cancel line'}; next request {s5:.1f} s")
        print("\n   transcript (HTTP, thinking off):", flush=True)
        for role, text in transcript:
            print(f"   {role:>9}: {text}", flush=True)
    finally:
        if srv.poll() is None:
            t1 = time.perf_counter()
            srv.send_signal(signal.SIGINT)                       # the server's Ctrl+C path: QUIT to the engine
            try:
                rc = srv.wait(timeout=120)
            except subprocess.TimeoutExpired:
                srv.kill()
                rc = srv.wait()
            check("H6 server stop -> engine QUIT, device lock released", rc == 0 and released(),
                  f"server rc {rc} in {time.perf_counter() - t1:.1f} s")
        remove_engines()


def remove_engines(grace_s: float = 40.0) -> None:
    """Engine containers left behind (a killed server, a hung engine): SIGTERM to their tt-engine.sh wrapper (pid =
    the container name's suffix; it forwards the signal, removes a container that does not exit and resets the
    board), then `docker rm -f` whatever is still there.  The device lock serializes engines, so any running
    tt-engine-* container here is this test's."""
    left = engine_containers()
    if not left:
        return
    print(f"   leftover engine containers {left}: terminating", flush=True)
    for name in left:
        try:
            os.kill(int(name.rsplit("-", 1)[1]), signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    t0 = time.time()
    while engine_containers() and time.time() - t0 < grace_s:
        time.sleep(1)
    for name in engine_containers():
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma list of parts to skip (proto,http)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--config", type=Path, default=None,
                    help="a Strata run config with the TT engine (default: written from fnx.paths' dev data)")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    LOGS.mkdir(parents=True, exist_ok=True)
    config = a.config or dev_config(a.port)
    cfg = json.loads(config.read_text())
    print(f"config {config}: exe {cfg['exe']}, args {' '.join(cfg['args'])}, env {cfg.get('env', {})}", flush=True)
    tok = tokenizer(cfg)
    try:
        golden = reference(cfg)
        if "proto" not in skip:
            part_proto(tok, cfg, golden)
        if "http" not in skip:
            part_http(tok, config, a.port, golden)
    finally:
        remove_engines()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\ntest_serve {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)} ({len(RESULTS)} checks)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
