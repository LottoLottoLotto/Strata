#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""tt/bundle/serve.py   Strata's server and the P150 engine in one container: the tt-model bundle's launch command.

    python tt/bundle/serve.py --host 0.0.0.0 --port 8000 [--mtp | --no-mtp] [--idle-unload S] [ENGINE ARGS...]

The tt-model package (kind http-server) ships this repository's tt/, serve/ and tools/ inside its tt-metal tree,
points at the weights (the GGUF shards and the P150 expert set, both pinned Hugging Face snapshots) and gives this
container the card.  What `./setup.sh --backend tt` does on a PC is done here at the start, into FNX_DATA (a persistent
cache directory; the steps are skipped once done):

  1. the P150 expert set: FNX_DATA/quant/store and the drafter's BFP4 experts + routing profile in FNX_DATA/mtp are
     links into the expert-set snapshot (FNX_P150_DIR), where setup's tt_artifacts.py fetch would have put them;
  2. the server's tokenizer: tools/strata_tokenizer.py from GGUF shard 1 into FNX_DATA/packs/coder-iq1_m/tokenizer
     (setup's pack step; the TT engine reads nothing else of the pack);
  3. with --mtp (speculative decoding, the default of a setup install): setup's MTP draft layer, built the same way -
     tools/mtp_fetch.py fetch (its 31 tensors read by HTTP range requests from Qwen/Qwen3.8-Flash-Next at the pinned
     revision, SHA-256 checked; HF_ENDPOINT selects a mirror), mtp_pack.py --experts q2_0, mtp_rt.py, and the
     draft-vocabulary subset setup installs by default (data/draft_vocab.bin);
  4. the run config setup writes for the Coder (strata-coder-iq1_m.json), with tt/bundle/engine.sh as the engine (the
     engine runs in this container: no docker, no device lock, no board reset), then exec tt/serve/server_tt.py.

Environment: FNX_P150_DIR (the expert-set snapshot), STRATA_GGUF (GGUF shard 1; the PLE shard is found beside it as
setup finds it), FNX_DATA (default $TT_CACHE_PATH/fnx), FNX_CPU_LIB / STRATA_CMAKELISTS / STRATA_DATA (default: the
fnx_cpu package, tt/cpu_experts/pyproject.toml, which also carries data/*.bin: tt-model's code staging drops *.bin),
STRATA_GGUF_PY (llama.cpp's gguf-py for the MTP tools; default <repo>/gguf-py).
The server's log (the engine's stderr included) is FNX_DATA/logs/strata-coder-iq1_m.log, also copied to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TT = Path(__file__).resolve().parents[1]           # <repo>/tt
ROOT = TT.parent
TOOLS = ROOT / "tools"
MODEL_NAME = "qwen3.8-flash-next-coder-iq1_m"       # setup's model_name for the Coder IQ1_M
PACK = "coder-iq1_m"                                # setup's pack name (fnx.paths.PACK)
PREFILL = "128"                                     # fnx.paths.VARIANTS["coder"]'s prompt chunk (setup_tt.to_tt)
CTX = "32768"                                       # setup_tt.CTX_MAX
IDLE_UNLOAD_S = 900                                 # setup_tt's idle_unload_s


def say(msg: str) -> None:
    print(f"[bundle] {msg}", flush=True)


def need(key: str) -> Path:
    v = os.environ.get(key)
    if not v:
        sys.exit(f"[bundle] {key} is not set (the tt-model package sets it in serve.env)")
    p = Path(v)
    if not p.exists():
        sys.exit(f"[bundle] {key}={v} does not exist (are the weights fetched? `tt-model pull <bundle> --with-weights`)")
    return p


def run(cmd: list[str], env: dict | None = None) -> None:
    say("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def p150_files(data: Path, release: Path) -> None:
    """Step 1, checked against tt/artifacts.json (the expert set's pinned files and sizes) for what the Coder reads."""
    manifest = json.loads((TT / "artifacts.json").read_text(encoding="utf-8"))
    if release.name != manifest["revision"]:
        sys.exit(f"[bundle] FNX_P150_DIR is revision {release.name}, tt/artifacts.json pins {manifest['revision']}")
    bad = [f["path"] for f in manifest["files"] if f["path"].startswith(("quant/", "mtp/"))
           and (not (release / f["path"]).exists() or (release / f["path"]).stat().st_size != f["size"])]
    if bad:
        sys.exit(f"[bundle] {len(bad)} file(s) of the P150 expert set are missing or incomplete in {release}, "
                 f"e.g. {bad[0]}")
    link(release / "quant" / "store", data / "quant" / "store")      # quant/ itself stays writable
    for name in ("experts-bfp4-search.u8", "expert-profile.npy"):
        link(release / "mtp" / name, data / "mtp" / name)


def ple_shard(shard1: Path) -> Path:
    """The shard with the PLE table (setup: the shard holding per_layer_token_embd.weight)."""
    sys.path.insert(0, str(TOOLS))
    from gguf_reader import GGUFFile
    shards = sorted(shard1.parent.glob(shard1.name.replace("-00001-of-", "-0000?-of-")))
    for s in shards:
        if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors):
            return s
    sys.exit(f"[bundle] no GGUF shard beside {shard1} has per_layer_token_embd.weight")


def tokenizer(data: Path, shard1: Path) -> Path:
    """Step 2: <pack>/tokenizer, written by tools/strata_tokenizer.py (vocab.json is written last)."""
    pack = data / "packs" / PACK
    tok = pack / "tokenizer"
    if not (tok / "vocab.json").exists() or not (tok / "chat_template.jinja").exists():
        run([sys.executable, str(TOOLS / "strata_tokenizer.py"), "--gguf", str(shard1), "--out", str(pack)])
    return tok


def mtp_runtime(data: Path) -> Path:
    """Step 3: <data>/mtp/rt as setup builds it (setup.py step 6)."""
    mtp = data / "mtp"
    rt = mtp / "rt"
    env = dict(os.environ, STRATA_GGUF_PY=os.environ.get("STRATA_GGUF_PY") or str(ROOT / "gguf-py"))
    if not (rt / "experts.bin").exists():
        say("the MTP draft layer comes from the original Qwen checkpoint: only its ~5 GB of MTP tensors are read")
        run([sys.executable, str(TOOLS / "mtp_fetch.py"), "fetch", "--out", str(mtp)], env=env)
        run([sys.executable, str(TOOLS / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(TOOLS / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    vocab = rt / "draft_vocab.bin"
    if not vocab.exists():
        shutil.copyfile(Path(os.environ["STRATA_DATA"]) / "draft_vocab.bin", vocab)
    return rt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True,
                    help="speculative decoding with the MTP draft layer (--spec 4 --spec-min-p 0.5 unless given)")
    ap.add_argument("--idle-unload", type=float, default=IDLE_UNLOAD_S,
                    help="unload the model after this many idle seconds (0: never; the next request loads it)")
    ap.add_argument("--protocol-log", metavar="NAME",
                    help="keep a copy of the engine protocol (prompt and output token ids) in "
                         "FNX_DATA/logs/NAME.in / NAME.out (FNX_PROTOCOL_LOG, tt/bundle/engine.sh)")
    a, engine_args = ap.parse_known_args()

    release, shard1 = need("FNX_P150_DIR"), need("STRATA_GGUF")
    data = Path(os.environ.get("FNX_DATA") or Path(os.environ.get("TT_CACHE_PATH") or "/tensor-cache") / "fnx")
    data.mkdir(parents=True, exist_ok=True)
    os.environ["FNX_DATA"] = str(data)
    os.environ.setdefault("FNX_MODEL", "coder")
    if not all(os.environ.get(k) for k in ("FNX_CPU_LIB", "STRATA_CMAKELISTS", "STRATA_DATA")):
        import fnx_cpu
        os.environ.setdefault("FNX_CPU_LIB", str(fnx_cpu.LIB))
        os.environ.setdefault("STRATA_CMAKELISTS", str(fnx_cpu.CMAKELISTS))
        os.environ.setdefault("STRATA_DATA", str(fnx_cpu.DATA))
    os.environ["STRATA_ENGINE_READY_S"] = "0"       # setup's run script: the first start builds ~40 GB of caches

    p150_files(data, release)
    ple = ple_shard(shard1)
    tok = tokenizer(data, shard1)
    args = ["--pack", str(tok.parent), "--native", str(shard1), "--ple-gguf", str(ple), "--prefill", PREFILL]
    if a.mtp:
        rt = mtp_runtime(data)
        if "--spec" not in engine_args:
            args += ["--spec", "4", "--spec-min-p", "0.5"]
        args += ["--mtp", str(rt)]
    elif "--spec" not in engine_args:
        args += ["--spec", "1"]
    if "--max-context" not in engine_args:
        args += ["--max-context", CTX]
    args += engine_args

    logs = data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"strata-{PACK}.log"
    log.touch()
    if a.protocol_log:
        os.environ["FNX_PROTOCOL_LOG"] = str(logs / Path(a.protocol_log).name)
    cfg = {"exe": str(TT / "bundle" / "engine.sh"), "args": args, "cwd": str(ROOT), "tokenizer": str(tok),
           "model_name": MODEL_NAME, "log": str(log), "port": a.port, "backend": "tt", "open_browser": False,
           "idle_unload_s": a.idle_unload, "lib_dirs": [], "env": {"FNX_DATA": str(data)}}
    cfg_path = data / f"strata-{PACK}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    say(f"run config {cfg_path}: engine {' '.join(args)}")

    # the engine log on stdout too (the container log), from here on
    subprocess.Popen(["tail", "-n", "0", "-F", str(log)], stdin=subprocess.DEVNULL)
    server = [sys.executable, str(TT / "serve" / "server_tt.py"), "--engine", "strata", "--config", str(cfg_path),
              "--host", a.host, "--port", str(a.port)]
    say("$ " + " ".join(server))
    os.execv(sys.executable, server)                 # the server is this process: docker stop's SIGTERM reaches it


if __name__ == "__main__":
    main()
