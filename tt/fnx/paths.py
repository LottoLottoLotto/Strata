"""Every data location of the Tenstorrent port, in one place.  Standard library only: the engine, the host tools,
tt/serve/tt-engine.sh and tt/tools/devrun.sh (through `--shell`) and setup's tt code all read it.

FNX_DATA (an absolute directory; default DEV_DATA, this port's development machine) holds what the TT engine reads
and writes besides the model files its arguments name (--native, --ple-gguf, --pack, --mtp):
    quant/store/{plans/<plan>/.., pool/L{LL}.*}   the GPTQ-BFP expert store (fnx.quant.store; FNX_QUANT_DIR replaces quant/)
    mtp/experts-bfp4-search.u8                    the MTP drafter's routed experts as BFP4 (fnx.ref.mtp.MtpWeights.bfp4_rows)
    mtp/expert-profile.npy                        the drafter's routing profile (fnx.tt.mtp.host_experts)
    cache/tt, cache/dequant                       converted-tensor caches (FNX_TT_CACHE replaces cache/tt, FNX_CACHE cache/)
    kcache/                                       tt-metal's kernel cache (TT_METAL_CACHE in the container)
    logs/                                         tt-engine.sh's READY marks, tools/devrun.sh's logs
Development and test defaults (models/coder/*.gguf, packs/coder-iq1_m, mtp/rt, golden/, eval/, ...) resolve under it
too: on the development machine the data needs no move.

Repository resources come from the checkout that contains tt/ (REPO): tools/ (gguf_reader, strata_tokenizer),
data/expert-profile-coder.bin, src/kernels/ple_oracle_vectors.inc, CMakeLists.txt (the engine version).

    python3 tt/fnx/paths.py --shell [ENGINE ARGS...]
prints bash assignments (DATA, KCACHE, LOGS, IMAGE, LOCK, TT_METAL, CPU_LIB, MOUNTS=(...)) for the docker wrappers;
MOUNTS are the host directories the container mounts at the same path (`mounts`).
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


def _env_path(key: str, default: Path | str) -> Path:
    return Path(os.environ.get(key) or default)


# ------------------------------------------------------------------------------------------------ repository
REPO = Path(__file__).resolve().parents[2]            # <repo>/tt/fnx/paths.py
TT_DIR = REPO / "tt"
STRATA_TOOLS = _env_path("STRATA_TOOLS", REPO / "tools")           # gguf_reader.py, gguf_writer.py, strata_tokenizer.py
EXPERT_PROFILE = REPO / "data" / "expert-profile-coder.bin"        # Strata's Coder routing profile
PLE_ORACLE = REPO / "src" / "kernels" / "ple_oracle_vectors.inc"   # Strata's PLE oracle vectors (fnx.tests.test_ple)
CMAKELISTS = REPO / "CMakeLists.txt"                               # project(strata VERSION x.y.z.w): the engine version
CPU_LIB = _env_path("FNX_CPU_LIB", TT_DIR / "cpu_experts" / "build-tt" / "libfnx_cpu.so")   # tt/cpu_experts/build.sh
# A llama.cpp gguf-py ahead of the installed `gguf` package (the dev host's; absent elsewhere: the image's / venv's gguf)
GGUF_PY = _env_path("GGUF_PY", "/home/lotto/llama.cpp/gguf-py")

# ------------------------------------------------------------------------------------------------ FNX_DATA
DEV_DATA = "/var/tmp/strata-tt"
DATA = _env_path("FNX_DATA", DEV_DATA)
QUANT = _env_path("FNX_QUANT_DIR", DATA / "quant")
STORE = QUANT / "store"
MTP = DATA / "mtp"
MTP_PROFILE = MTP / "expert-profile.npy"


def mtp_bfp4(method: str = "search") -> Path:
    """The drafter's BFP4 expert file of a quantization method (fnx.ref.mtp.MtpWeights.bfp4_rows)."""
    return MTP / f"experts-bfp4-{method}.u8"


MTP_BFP4 = mtp_bfp4()
CACHE = _env_path("FNX_CACHE", DATA / "cache")
DEQUANT = CACHE / "dequant"
TT_CACHE = _env_path("FNX_TT_CACHE", DATA / "cache" / "tt")
KCACHE = DATA / "kcache"
LOGS = DATA / "logs"

# ------------------------------------------------------------------------------------------------ dev / test defaults
MODELS = DATA / "models" / "coder"
SHARD1 = _env_path("FNX_GGUF", MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")
SHARD2 = MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
PACK = DATA / "packs" / "coder-iq1_m"
MTP_RT = MTP / "rt"
GOLDEN = _env_path("FNX_GOLDEN", DATA / "golden")
EVAL = _env_path("FNX_EVAL", DATA / "eval")
VENV_PY = DATA / "venv" / "bin" / "python"          # the dev host's venv (Strata's requirements)
STRATA_REF = DATA / "Strata"                         # the dev host's Strata checkout pinned for the CUDA reference
LLAMA_CPP = DATA / "third_party" / "llama.cpp"       # the pinned llama.cpp source (third_party/ggml/VERSION.txt)

# ------------------------------------------------------------------------------------------------ engine container
DEFAULT_IMAGE = ("lottolabs/qwen36-35b-tt-p150@sha256:"
                 "118e0d90fe6aa9d626b557aff2aab856b115b62edfebf8d876c379eecc2f9457")
IMAGE = os.environ.get("FNX_IMAGE") or DEFAULT_IMAGE
DEVICE_LOCK = os.environ.get("FNX_DEVICE_LOCK") or "/var/tmp/gemma4-12b/device.lock"   # shared by every team on the card
TT_METAL = "/home/container_app_user/tt-metal"       # tt-metal (and its python_env) inside the image


def mounts(args=()) -> list[str]:
    """Host directories a container mounts at the same path: the repository, FNX_DATA, the FNX_QUANT_DIR / FNX_CACHE /
    FNX_TT_CACHE / FNX_CPU_LIB directories, and the parent directory of every absolute path argument (`--k=/v` too),
    as given and resolved; existing directories only, none inside another, never /."""
    want = [REPO, DATA, QUANT, CACHE, TT_CACHE, CPU_LIB.parent]
    for a in args:
        v = a.split("=", 1)[1] if a.startswith("--") and "=" in a else a
        if v.startswith("/"):
            want += [Path(v).parent, Path(v).resolve().parent]
    out: list[Path] = []
    for d in sorted({Path(os.path.abspath(d)) for d in want}, key=lambda p: (len(p.parts), str(p))):
        if d.parent != d and d.is_dir() and not any(d == m or m in d.parents for m in out):
            out.append(d)
    return [str(d) for d in out]


def shell(args=()) -> str:
    """Bash assignments for the docker wrappers; creates FNX_DATA's logs/ and kcache/ first (mounted, so they must
    exist, owned by the caller)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    KCACHE.mkdir(parents=True, exist_ok=True)
    q = shlex.quote
    return "\n".join([f"DATA={q(str(DATA))}", f"KCACHE={q(str(KCACHE))}", f"LOGS={q(str(LOGS))}",
                      f"IMAGE={q(IMAGE)}", f"LOCK={q(DEVICE_LOCK)}", f"TT_METAL={q(TT_METAL)}",
                      f"CPU_LIB={q(str(CPU_LIB))}", "MOUNTS=(" + " ".join(q(m) for m in mounts(args)) + ")"]) + "\n"


if __name__ == "__main__":
    if sys.argv[1:2] != ["--shell"]:
        sys.exit("usage: paths.py --shell [ENGINE ARGS...]")
    sys.stdout.write(shell(sys.argv[2:]))
