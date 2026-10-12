"""setup.py for a Tenstorrent Blackhole P150: upstream's installer, steered onto the TT engine (tt/) from outside.

    ./setup.sh --backend tt [setup.py's options]       (this file, run through setup.sh's virtual environment)

The TT engine keeps out of the shared files (upstream merges stay clean), so this does not edit setup.py: it imports
it and replaces the few steps that are NVIDIA/AMD-specific, then runs setup's own main().  Everything else - the
download (or --gguf-dir), the pack, the tokenizer, the MTP draft layer - is setup.py's, unchanged.  What is replaced:

  - the GPU check: the P150 (/dev/tenstorrent/N, named by the tt-kmd driver's sysfs or `tt-smi -s`) is offered
    through setup's AMD path (the one with no ready-made engine);
  - the choices: the Coder (IQ1_M) and the full model at IQ3_S, a context of at most 32768 tokens, no images (what
    the engine was built and measured for: tt/CONTRACT.md);
  - the engine step: Docker and the engine image (pulled when missing) instead of a compiled engine, Strata's CPU
    expert kernels compiled in that image (tt/cpu_experts/build.sh, libfnx_cpu.so), and tt/serve/BUILD.json beside
    the wrapper, which setup and the server read the engine's version from;
  - the RAM rule: the CUDA engine keeps every expert in RAM; on the P150 the Coder keeps most of its experts in the
    card's 32 GB and pins a ~12 GB host pool for the rest, IQ3_S runs the experts the card does not hold on the CPU
    from the GGUF (warm in the page cache), so setup's rule does not apply (TT_MODELS' own minimum instead);
  - after the pack and the MTP draft layer: the P150's own files (tt/artifacts.json: for the Coder the GPTQ-BFP4
    expert store of the default plan and the drafter's BFP4 experts, for IQ3_S the drafter's only) are fetched into
    FNX_DATA by tt/tools/tt_artifacts.py; for IQ3_S the card's 4-bit expert caches are then built on the CPU (no card,
    `fnx.tt.moe --prepare default`), so the first start does not spend ~15 minutes on them;
  - the config: the engine is tt/serve/tt-engine.sh (the engine in Docker, holding the shared device lock), backend
    "tt"; the run script starts tt/serve/server_tt.py (serve/server.py plus the P150's Monitor readings and start
    narration).

FNX_DATA (the P150's files): the environment's FNX_DATA, else the one the model's config already names, else
<data folder>/tt.  FNX_IMAGE: the engine image (default fnx.paths.DEFAULT_IMAGE); FNX_DEVICE_LOCK, FNX_CPU_LIB as
tt-engine.sh reads them.

When setup.py changes the steps this relies on, this stops with a message rather than writing a wrong config.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tt"))
sys.path.insert(0, str(ROOT / "tt" / "serve"))
import setup as S  # noqa: E402
from fnx import paths as P  # noqa: E402  the TT port's locations (standard library only)
from tt_telemetry import CARDS as TT_CARDS  # noqa: E402  tt_card_type -> (name, GB): every P150 has 32 GB

TT = ROOT / "tt"
WRAPPER = TT / "serve" / "tt-engine.sh"
ARTIFACTS_TOOL = TT / "tools" / "tt_artifacts.py"
CPU_BUILD = TT / "cpu_experts" / "build.sh"
DEVRUN = TT / "tools" / "devrun.sh"
# in the engine image: a P150's cluster descriptor, for ttnn's block-float packing without a card (fnx.tt.moe)
MOCK_CLUSTER = ("/home/container_app_user/tt-metal/tt_metal/third_party/tt-cluster-descriptors/blackhole/"
                "p100_cluster_desc/p100_cluster_desc.yaml")
MANIFEST = TT / "artifacts.json"
DOCS = "docs/TT_P150.md"
DEFAULT_IMAGE = P.DEFAULT_IMAGE    # tt-engine.sh's image unless FNX_IMAGE names another
IMAGE_GB = 5.6                      # the default image, unpacked
# the models the engine was built and measured for: setup's size -> its family, the fnx variant (fnx.paths.VARIANTS),
# the RAM it needs, the tt/artifacts.json files it reads (None: all), what --check says of it
TT_MODELS = {
    "IQ1_M": {"family": "coder", "variant": "coder", "ram_gb": 32, "artifacts": None,
              "menu": "its experts on the card and in a ~12 GB host pool",
              "fit": "the Coder: fits the P150 (most of its experts in the card's 32 GB, the rest in a ~12 GB pinned "
                     "host pool)"},
    "IQ3_S": {"family": "qwen", "variant": "iq3s", "ram_gb": 64, "artifacts": ("mtp/*",), "cache_gb": 30,
              "menu": "8,823 of its experts on the card, the rest on the CPU (64 GB of RAM)",
              "fit": "the full model: 8,823 experts on the card, the rest on the CPU from the GGUF (64 GB of RAM keep "
                     "it warm)"},
}
VARIANT_ARTIFACTS = {d["variant"]: d["artifacts"] for d in TT_MODELS.values()}
CTX_MAX = 32768                     # fnx.serve_engine CTX_MAX: fnx.tt.qsa's sparse selection is verified to 32,800
PREFILL = 128                       # fnx.serve_engine DEFAULT_CHUNK: prompt rows per device call
IDLE_UNLOAD_S = 900                 # the shared card is given back after 15 idle minutes
MIN_RAM_GB = min(d["ram_gb"] for d in TT_MODELS.values())   # the card note; each model's own after its choice
PREP_WORKERS = 4                    # IQ3_S's card caches: layer ranges converted at a time (~8 GB of RAM each)
# the engine's flags (fnx.serve_engine); every other flag setup writes is the CUDA engine's and is left out
FNX_ARGS = frozenset({"--pack", "--native", "--ple-gguf", "--mtp", "--max-context", "--prefill", "--spec",
                      "--spec-min-p", "--threads", "--prompt-cache", "--eos-ids", "--no-warmup"})
DROP_KEYS = ("lib_dirs", "env", "vision", "gpu", "gpus_asked", "layer_split", "cuda", "parallel")
TT_CHIPS = {"b140": "Blackhole", "401e": "Wormhole", "faca": "Grayskull"}   # PCI device ids (vendor 0x1e52)
STATE: dict = {}                    # what main() found out that the replaced steps need: the data folder


def _read(path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def tt_smi_snapshot() -> dict:
    """`tt-smi -s` (its JSON snapshot), or {} without tt-smi.  Only asked when the driver's sysfs does not name a card
    (tt-kmd before tt_card_type): it talks to the card, the sysfs files do not."""
    exe = shutil.which("tt-smi") or os.path.expanduser("~/.local/bin/tt-smi")
    if not os.access(exe, os.X_OK):
        return {}
    try:
        r = subprocess.run([exe, "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout[r.stdout.find("{"):]) if r.returncode == 0 and "{" in r.stdout else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def tt_devices(dev="/dev", sysfs="/sys", snapshot=tt_smi_snapshot) -> list[dict]:
    """Tenstorrent cards: /dev/tenstorrent/N (the tt-kmd driver), named by the driver's sysfs (tt_card_type), else by
    `tt-smi -s`'s board_type, numbered as the driver numbers them.  dev/sysfs/snapshot: tests pass a mocked tree."""
    nodes = sorted(int(p.name) for p in Path(dev, "tenstorrent").glob("*") if p.name.isdigit())
    kmd = _read(Path(sysfs, "module", "tenstorrent", "version"))
    smi, found = None, []
    for n in nodes:
        cls = Path(sysfs, "class", "tenstorrent", f"tenstorrent!{n}")
        card = (_read(cls / "tt_card_type") or "").lower()
        devid = (_read(cls / "device" / "device") or "").lower().replace("0x", "")
        if not card:
            if smi is None:
                smi = snapshot() or {}
            infos = smi.get("device_info") or []
            if n < len(infos):
                card = str((infos[n].get("board_info") or {}).get("board_type") or "").lower().split(" ")[0]
        name, vram = TT_CARDS.get(card, (f"Tenstorrent {card or TT_CHIPS.get(devid, 'card')}", 0))
        found.append({"index": n, "name": name, "vram_gb": float(vram), "arch": card or devid or "unknown", "card": card,
                      "driver": "tenstorrent", "kmd": kmd, "node": str(Path(dev, "tenstorrent", str(n))),
                      "pci": os.path.realpath(cls / "device") if (cls / "device").exists() else None})
    return found


def tt_problem(g) -> str | None:
    """Why setup cannot use this card, or None."""
    if g.get("card") not in TT_CARDS:
        return f"not supported - the TT engine runs on a Blackhole P150 (32 GB) only, this is {g['name']}"
    if g["index"] != 0:
        return "the TT engine opens card 0 (/dev/tenstorrent/0) only"
    return None


def pcie_text(pci) -> str | None:
    """The card's PCIe link as trained: the host pool's experts are read over it every token."""
    gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
    if not pci:
        return None
    cur = gen.get((_read(Path(pci, "current_link_speed")) or "").split(" ")[0])
    top = gen.get((_read(Path(pci, "max_link_speed")) or "").split(" ")[0])
    width = _read(Path(pci, "current_link_width"))
    if not cur or not width:
        return None
    return f"PCIe Gen{cur} x{width}" + (f" (the card can do Gen{top})" if top and top > cur else "")


def docker_problem() -> tuple | None:
    """(message, hint) when this user cannot run containers, else None."""
    if not shutil.which("docker"):
        return ("Docker is not installed: the TT engine runs in a Docker image (tt-metal)",
                f"install Docker Engine (https://docs.docker.com/engine/install/), then run this again - {DOCS}")
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"docker does not answer ({e})", "start the Docker service: sudo systemctl start docker"
    if r.returncode == 0:
        return None
    err = (r.stderr or r.stdout or "").strip().splitlines()
    err = err[-1] if err else f"exit {r.returncode}"
    if "permission denied" in err.lower():
        return ("this user may not use Docker (" + err + ")",
                "sudo usermod -aG docker $USER, then log out and in again (or reboot)")
    return f"docker does not answer: {err}", "start the Docker service: sudo systemctl start docker"


def image_present(image) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True,
                              timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker_root() -> Path:
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.DockerRootDir}}"], capture_output=True, text=True,
                           timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path("/var/lib/docker")


def free_gb_at(path: Path) -> float | None:
    """Free space of the file system `path` is (or will be) on."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return None


def hugepages_1g(sysfs="/sys") -> int:
    v = _read(Path(sysfs, "kernel", "mm", "hugepages", "hugepages-1048576kB", "nr_hugepages"))
    return int(v) if v and v.isdigit() else 0


def lock_path() -> Path:
    """The shared device lock tt-engine.sh takes (FNX_DEVICE_LOCK, else every team's default on this card)."""
    return Path(P.DEVICE_LOCK)


def lock_problem(lock: Path) -> str | None:
    """tt-engine.sh opens (creates) the shared device lock: this user must be able to."""
    if lock.exists():
        return None if os.access(lock, os.R_OK | os.W_OK) else f"{lock} is not writable by this user"
    d = lock.parent
    while not d.exists() and d != d.parent:
        d = d.parent
    return None if os.access(d, os.W_OK) else f"{lock} cannot be created by this user ({d} is not writable)"


def card_users(n=0) -> list[str]:
    """Processes that have the card open right now (the driver's list: one line per open file, so a pid repeats)."""
    return list(dict.fromkeys((_read(f"/proc/driver/tenstorrent/{n}/pids") or "").split()))


def flag(args, name):
    return args[args.index(name) + 1] if name in args[:-1] else None


def take(argv: list, name: str, value=True) -> tuple[list, object]:
    """argv without `name` (`--name V`, `--name=V`, or a bare switch), and its last value (None when absent)."""
    out, val, i = [], None, 0
    while i < len(argv):
        x = argv[i]
        if x == name:
            if value:
                val = argv[i + 1] if i + 1 < len(argv) else ""
                i += 2
            else:
                val = True
                i += 1
            continue
        if value and x.startswith(name + "="):
            val = x.split("=", 1)[1]
        else:
            out.append(x)
        i += 1
    return out, val


def tt_argv(argv) -> list:
    """setup.py's arguments for the P150: its HIP path (the one with no ready-made engine), no images, no low-RAM mode,
    the KV answer given (the engine keeps its KV cache in bf16: the question would be about another engine), the
    speed projection off.  What the TT engine cannot do is refused (a context past 32768, several cards, the tuning)
    or left out with a word (--parallel, --vram-reserve-mib, --kv, images, the low-RAM mode, the projection)."""
    argv = list(argv)
    _, ctx = take(argv, "--context")
    if ctx is not None and str(ctx).strip().isdigit() and int(ctx) > CTX_MAX:
        S.fail(f"--context {ctx}: the TT engine serves up to {CTX_MAX} tokens (its sparse attention is verified to "
               "32,800 positions)", f"use --context {CTX_MAX} or less")
    _, gpus = take(argv, "--gpus")
    if gpus is not None:
        S.fail("several cards sharing one model (--gpus): the TT engine runs on one P150", "leave --gpus out")
    _, gpu = take(argv, "--gpu")
    if gpu is not None and str(gpu).strip() != "0":
        S.fail(f"--gpu {gpu}: the TT engine opens card 0 (/dev/tenstorrent/0) only", "leave --gpu out")
    _, cal = take(argv, "--calibrate", value=False)
    if cal:
        S.fail("--calibrate tunes the CUDA engine's settings; the TT engine has none of those",
               "run ./setup.sh --backend tt without --calibrate")
    argv, par = take(argv, "--parallel")
    if par is not None and str(par).strip() not in ("", "1"):
        S.warn("--parallel: the TT engine answers one request at a time (the others wait): left out")
    argv, reserve = take(argv, "--vram-reserve-mib")
    if reserve is not None:
        S.warn("--vram-reserve-mib: the P150's memory is planned by the engine itself (the card is not shared while "
               "the model is loaded): left out")
    argv, kv = take(argv, "--kv")
    if kv not in (None, "int8"):
        S.warn(f"--kv {kv}: the TT engine keeps its KV cache in bf16: left out")
    argv, vision = take(argv, "--vision")
    if vision not in (None, "no", "none"):
        S.warn("images: the TT engine has no image encoder yet: off")
    argv, low = take(argv, "--low-ram")
    if low not in (None, "auto", "off"):
        S.warn(f"--low-ram {low}: the CUDA engine's mode; the P150 keeps its experts on the card and in a host pool: off")
    argv, esp = take(argv, "--experimental-speed-projection")
    if esp is not None and str(esp).strip().lower() not in ("off", "no", "n", "0"):
        S.warn("the experimental speed projection (a control vector) is not implemented by the TT engine: off")
    argv, _ = take(argv, "--backend")
    return [*argv, "--backend", "hip", "--vision", "none", "--low-ram", "off", "--kv", "int8",
            "--experimental-speed-projection", "off"]


def restrict() -> None:
    """The choices setup offers: TT_MODELS (the Coder first: the default), contexts up to 32768.  setup.main() builds
    --family/--model's argparse choices and its menus from these, so another model is refused before anything is
    done."""
    fams = {}
    for m, d in TT_MODELS.items():
        if d["family"] not in S.FAMILIES or m not in S.MODELS:
            S.fail(f"setup.py has no {d['family']} / {m} model any more: tt/setup_tt.py needs updating for this setup.py")
        fams[d["family"]] = S.FAMILIES[d["family"]]
    S.FAMILIES = fams
    S.MODELS = {m: S.MODELS[m] for m in TT_MODELS}
    S.CONTEXTS = [c for c in S.CONTEXTS if c <= CTX_MAX]


def model_tag(model: str) -> str:
    return (S.FAMILIES[TT_MODELS[model]["family"]]["tag"] + model).lower()


def chosen_model() -> str | None:
    """The size chosen in setup's step 2 (None before it)."""
    return STATE.get("model")


def config_variant(cfg: dict) -> str:
    """The fnx variant of a TT config: from its --native shard's name, as tt-engine.sh decides it."""
    return P.detect_model([str(x) for x in cfg.get("args") or []])


def configured_env(key: str, cfg: dict | None = None) -> str | None:
    """`key` as the environment gives it, else as the model's config names it (a value set there before is kept)."""
    if os.environ.get(key):
        return os.environ[key]
    if cfg is None:
        cfg = {}
        models = [chosen_model()] if chosen_model() else list(TT_MODELS)
        for m in models:
            try:
                c = json.loads((S.ROOT / f"strata-{model_tag(m)}.json").read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if c.get("backend") == "tt" and (c.get("env") or {}).get(key):
                cfg = c
                break
    return (cfg.get("env") or {}).get(key) or None


def fnx_data_dir(cfg: dict | None = None) -> Path:
    """FNX_DATA: the environment's, else the config's, else <setup's data folder>/tt."""
    v = configured_env("FNX_DATA", cfg)
    if v:
        if not os.path.isabs(v):
            S.fail(f"FNX_DATA={v} is not an absolute path", "set FNX_DATA to a full path, or leave it unset")
        return Path(v)
    data = STATE.get("data")
    if data is None:
        S.fail("setup.py's data folder is unknown: tt/setup_tt.py needs updating for this setup.py")
    return Path(data) / "tt"


def engine_image(cfg: dict | None = None) -> str:
    return configured_env("FNX_IMAGE", cfg) or DEFAULT_IMAGE


def manifest_files(manifest: Path = MANIFEST) -> list[dict]:
    try:
        return list(json.loads(manifest.read_text(encoding="utf-8")).get("files") or [])
    except (OSError, ValueError, AttributeError):
        return []


def artifact_path(fnx_data: Path, rel: str, env=None) -> Path:
    """Where a manifest file lives: under FNX_DATA, and quant/... under FNX_QUANT_DIR when that is set (fnx.paths)."""
    env = os.environ if env is None else env
    if env.get("FNX_QUANT_DIR") and rel.startswith("quant/"):
        return Path(env["FNX_QUANT_DIR"]) / rel[len("quant/"):]
    return Path(fnx_data) / rel


def missing_artifacts(fnx_data: Path, manifest: Path = MANIFEST, env=None, variant: str = "coder") -> list[dict]:
    """The manifest's files the variant reads (VARIANT_ARTIFACTS) that are absent or of the wrong size (presence
    only: tt_artifacts.py fetch checks the SHA-256)."""
    pats = VARIANT_ARTIFACTS[variant]
    out = []
    for f in manifest_files(manifest):
        if pats is not None and not any(fnmatch.fnmatch(f["path"], p) for p in pats):
            continue
        p = artifact_path(fnx_data, f["path"], env)
        try:
            if p.stat().st_size == int(f["size"]):
                continue
        except (OSError, KeyError, ValueError):
            pass
        out.append(f)
    return out


def cpu_lib() -> Path:
    """libfnx_cpu.so (Strata's CPU expert kernels behind a C ABI), which every engine start loads: FNX_CPU_LIB, else
    what tt/cpu_experts/build.sh builds."""
    return Path(P.CPU_LIB)


def cpu_src() -> str:
    """A fingerprint of the CPU expert library's sources (tt/cpu_experts and Strata's engine sources): when a `git pull`
    changes them, it is compiled again."""
    h = hashlib.sha256(S.source_hash(S.ENGINE_SOURCES).encode())
    for f in ("CMakeLists.txt", "fnx_cpu.cpp", "fnx_route.cpp", "fnx_qsa.cpp"):
        h.update((CPU_BUILD.parent / f).read_bytes())
    return h.hexdigest()[:16]


def build_json(folder: Path, image: str, cpu: str | None = None) -> Path:
    """BUILD.json beside the wrapper: setup's engine_version() and the server's pre-READY version read it.  The
    version is the source tree's (the engine is this checkout's tt/ code, run in the image); cpu_src: the sources the
    CPU expert library was compiled from."""
    path = Path(folder) / "BUILD.json"
    path.write_text(json.dumps({"backend": "tt", "source": "docker", "image": image, "version": S.source_version(),
                                "lib_dirs": [], **({"cpu_src": cpu} if cpu else {})}, indent=1) + "\n", encoding="utf-8")
    return path


def engine_step(image: str, llama: Path) -> Path:
    """setup's step 4 on the P150: Docker and the engine image (pulled when missing) instead of a compiled engine, the
    CPU expert library compiled in that image (again only when its sources changed), the room for the P150's files
    checked before the model download, and BUILD.json beside the wrapper.  llama: setup's llama.cpp (ggml)."""
    problem = docker_problem()
    if problem:
        S.fail(*problem)
    if not WRAPPER.exists():
        S.fail(f"{WRAPPER} is missing", f"use a full Strata checkout (git clone) - {DOCS}")
    if image_present(image):
        S.ok(f"engine image: {image}")
    else:
        root = docker_root()
        free = free_gb_at(root)
        if free is not None and free < IMAGE_GB + 2:
            S.fail(f"not enough free disk space for the engine image in {root}: need ~{IMAGE_GB + 2:.0f} GB, "
                   f"{free:.0f} GB free", "free some space (docker system prune removes unused images)")
        S.say(f"  Downloading the engine image (tt-metal and the TT kernels' runtime, ~{IMAGE_GB:.0f} GB once) ...")
        S.run(["docker", "pull", image])
        S.ok(f"engine image: {image}")
    lib, src = cpu_lib(), cpu_src()
    try:
        built = json.loads((WRAPPER.parent / "BUILD.json").read_text(encoding="utf-8")).get("cpu_src")
    except (OSError, ValueError):
        built = None
    if os.environ.get("FNX_CPU_LIB"):                   # the user's own build: used as it is
        if not lib.exists():
            S.fail(f"FNX_CPU_LIB={lib} does not exist", "leave FNX_CPU_LIB unset: setup builds the library itself")
        src = built
    elif not lib.exists() or built != src:
        S.say("  Compiling Strata's CPU expert kernels for the TT engine (in the engine image; a few minutes, once) ...")
        S.run(["bash", str(CPU_BUILD)], env={**os.environ, "STRATA_GGML_DIR": str(llama), "FNX_IMAGE": image})
        if not lib.exists():
            S.fail(f"{CPU_BUILD} did not write {lib}", f"the build's output is above - {DOCS}")
    S.ok(f"CPU expert kernels: {lib}")
    data = fnx_data_dir()
    model = chosen_model() or "IQ1_M"
    todo = missing_artifacts(data, variant=TT_MODELS[model]["variant"])
    need = sum(int(f.get("size", 0)) for f in todo) / 1e9 + TT_MODELS[model].get("cache_gb", 0)
    if need:
        free = free_gb_at(data)
        if free is not None and free < need + 1:
            S.fail(f"not enough free disk space in {data} for the P150's expert files: need ~{need + 1:.0f} GB, "
                   f"{free:.0f} GB free", "free some space, or set FNX_DATA to a folder on a bigger drive")
        S.ok(f"the P150's expert files: {len(todo)} to fetch after the model"
             + (f" + the card's expert caches to build" if TT_MODELS[model].get("cache_gb") else "")
             + f" (~{need:.1f} GB, into {data})")
    else:
        S.ok(f"the P150's expert files: {data}")
    build_json(WRAPPER.parent, image, src)
    return WRAPPER.parent


def prepare(fnx_data: Path, variant: str = "coder", gguf: str | None = None) -> None:
    """The P150's own files the variant reads (tt/artifacts.json, VARIANT_ARTIFACTS), fetched into FNX_DATA and
    checked against their SHA-256 (a file already there and checked is not read again); for iq3s then its card
    caches from the model's shard 1 `gguf` (prepare_caches)."""
    if not ARTIFACTS_TOOL.exists():
        S.fail(f"{ARTIFACTS_TOOL} is missing", f"use a full Strata checkout (git clone) - {DOCS}")
    pats = VARIANT_ARTIFACTS[variant]
    size = sum(int(f.get("size", 0)) for f in manifest_files()
               if pats is None or any(fnmatch.fnmatch(f["path"], p) for p in pats)) / 1e9
    S.say(f"  The P150's {'expert' if pats is None else 'draft-layer'} files (~{size:.0f} GB the first time) into "
          f"{fnx_data}:")
    Path(fnx_data).mkdir(parents=True, exist_ok=True)
    S.run([sys.executable, str(ARTIFACTS_TOOL), "fetch", "--data", str(fnx_data),
           *(["--only", ",".join(pats)] if pats else [])])
    S.ok(f"the P150's files: {fnx_data}")
    if variant == "iq3s":
        if not gguf:
            S.fail("the model's file is unknown to the cache step: tt/setup_tt.py needs updating for this setup.py")
        prepare_caches(fnx_data, variant, gguf)


def prepare_caches(fnx_data: Path, variant: str, gguf: str) -> None:
    """The card's resident-expert caches of the variant's default placement (fnx.tt.moe --prepare default) from the
    model's shard 1 `gguf`, built in the engine image on the CPU with a mock card (no device, no lock), PREP_WORKERS
    layer ranges at a time.  Layers already cached take a second."""
    S.say("  Building the card's 4-bit expert caches from the model file (on the CPU, ~15 minutes the first time) ...")
    env = {**os.environ, "FNX_DATA": str(fnx_data), "FNX_MODEL": variant, "NODEV": "1",
           "TT_METAL_MOCK_CLUSTER_DESC_PATH": MOCK_CLUSTER, "FNX_IMAGE": engine_image()}
    env.pop("FNX_EXPERT_PLACEMENT", None)
    n = 48
    step = -(-n // PREP_WORKERS)
    procs = []
    for lo in range(0, n, step):
        rng = f"{lo}-{min(n, lo + step) - 1}"
        procs.append((rng, subprocess.Popen(["bash", str(DEVRUN), f"setup-prep-{variant}-{rng}", "-m", "fnx.tt.moe",
                                             "--prepare", "default", "--layers", rng, "--gguf", gguf], env=env,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)))
    bad = [rng for rng, p in procs if p.wait() != 0]
    if bad:
        S.fail(f"building the card's expert caches failed for layers {', '.join(bad)}",
               f"see {Path(fnx_data) / 'logs' / 'dev'}/setup-prep-{variant}-*.log - {DOCS}")
    S.ok("the card's expert caches: built")


def to_tt(cfg: dict, fnx_data, image: str = DEFAULT_IMAGE) -> dict:
    """setup's config (written for its HIP path) -> the TT engine's: the wrapper as the engine, the engine's own flags
    only (the CUDA engine's - expert cache and profile, KV precision and streaming, VRAM reserve, pool workers, rope,
    PLE I/O, the control vector - left out), an explicit prompt chunk (setup's 'auto' is the CUDA engine's sizing, and
    a script engine would read as 0.1.12 to upgrade_config without BUILD.json), the context capped at 32768, FNX_DATA
    (and a FNX_IMAGE other than the default) for the wrapper, the shared card given back after 15 idle minutes.  The
    paths stay the host's: tt-engine.sh mounts each one at the same place."""
    args = [str(x) for x in cfg.get("args") or []]
    out_args, i = [], 0
    while i < len(args):
        f, vals = args[i], []
        i += 1
        while i < len(args) and not args[i].startswith("--"):
            vals.append(args[i])
            i += 1
        if f == "--prefill":
            out_args += [f, str(PREFILL)]
        elif f == "--max-context" and vals and vals[0].isdigit():
            out_args += [f, str(min(int(vals[0]), CTX_MAX))]
        elif f in FNX_ARGS:
            out_args += [f, *vals]
    if "--max-context" not in out_args:
        out_args += ["--max-context", str(CTX_MAX)]
    if "--prefill" not in out_args:
        out_args += ["--prefill", str(PREFILL)]
    env = {k: v for k, v in (cfg.get("env") or {}).items() if k not in S.SETUP_ENV and k != "FNX_IMAGE"}
    env["FNX_DATA"] = str(fnx_data)
    if image != DEFAULT_IMAGE:
        env["FNX_IMAGE"] = image
    out = {k: v for k, v in cfg.items() if k not in DROP_KEYS}
    out.update({"backend": "tt", "exe": str(WRAPPER), "cwd": str(ROOT), "args": out_args, "lib_dirs": [], "env": env})
    out.setdefault("idle_unload_s", IDLE_UNLOAD_S)
    return out


READY_LINE = ("# the engine waits for the shared device lock, which has no bound; tt-engine.sh bounds the start once it "
              "holds it (FNX_READY_S)\nSTRATA_ENGINE_READY_S=0\nexport STRATA_ENGINE_READY_S\n")


def run_script(write, model, cfg_path, port, open_browser=True) -> Path:
    """setup's run-<model>.sh, starting tt/serve/server_tt.py with the server's READY timeout off (it would SIGKILL a
    start that waits for the device lock, skipping the wrapper's cleanup) and passing on its own arguments."""
    script = write(model, cfg_path, port, open_browser)
    text = script.read_text(encoding="utf-8")
    server = str(S.ROOT / "serve" / "server.py")
    if server not in text or "\nexec " not in text:
        S.fail("setup.py's run script changed: tt/setup_tt.py needs updating for this setup.py")
    text = text.replace(server, str(S.ROOT / "tt" / "serve" / "server_tt.py"))
    text = text.replace("\nexec ", "\n" + READY_LINE + "exec ", 1).rstrip("\n") + ' "$@"\n'
    script.write_text(text, encoding="utf-8")
    return script


def start_banner(cfg: dict, lock: Path, browser: bool) -> None:
    S.say()
    S.say("  " + "-" * 100)
    S.say(f"  Starting {cfg.get('model_name', 'the model')} on the Tenstorrent P150: the engine takes the card's shared "
          f"lock ({lock}),")
    S.say("  waiting while another job holds it, then builds the model on the card (1-2 minutes), then "
          + ("the browser opens." if browser else "the server says it is ready."))
    S.say(f"  The card is given back after {int(cfg.get('idle_unload_s') or IDLE_UNLOAD_S) // 60} idle minutes and "
          "when this window is closed (Ctrl+C).")
    S.say("  " + "-" * 100)


def say_rewriter(real_ram: float, fake_ram: float, image: str):
    """setup's words for its AMD path, its RAM rule and the CUDA engine, said for the P150."""
    say = S.say
    stub_exe = str(WRAPPER.parent / S.EXE)

    def say_tt(msg=""):
        msg = str(msg)
        msg = msg.replace("(AMD: docs/AMD_HIP.md)", f"(Tenstorrent: {DOCS})")
        msg = msg.replace("Your AMD GPUs:", "Your Tenstorrent cards:")
        msg = msg.replace("just run ./setup.sh", "just run ./setup.sh --backend tt")
        msg = msg.replace("Run it again without --check to install.", "Run ./setup.sh --backend tt to install.")
        msg = msg.replace(f"RAM: {fake_ram:.0f} GB",
                          f"RAM: {real_ram:.0f} GB (the Coder pins ~12 GB of it for the experts the card does not hold; "
                          "IQ3_S runs those on the CPU from the model file in RAM)")
        msg = msg.replace(f"engine: {stub_exe}", f"engine: {WRAPPER} (Docker image {image})")
        m = re.match(r"\s+\d+\) (\S+)\s", msg)                # a size in step 2's menu: the P150's use of RAM
        if m and m.group(1) in TT_MODELS:
            msg = re.sub(r", uses ~\d+ GB of RAM", ", " + TT_MODELS[m.group(1)]["menu"], msg)
        if re.match(r"\s+\[ok\] KV cache: ", msg):
            msg = "  [ok] KV cache: bf16 (the TT engine's)"
        m = re.match(r"\s+(\S+)\s+needs ~\d+ GB RAM:", msg)    # --check's CUDA verdict: replaced by the P150's
        if m and m.group(1) in TT_MODELS:
            msg = f"  {m.group(1):8s} {TT_MODELS[m.group(1)]['fit']}"
        say(msg)
    return say_tt


# setup.py's functions this replaces or calls: when one is gone, setup.py changed under it
NEEDED = ("gpus", "amd_gpus", "amd_problem", "amd_device_access_problem", "hip_vision", "build_engine_hip",
          "hipblaslt_table", "ram_gb", "bench_tips", "parallel_note", "desktop_reserve_note", "previous_config",
          "installed_configs", "data_folder", "write_run_script", "start", "upgrade_config", "write_config",
          "source_version", "low_ram_wanted", "run", "step", "say", "main")


def install(argv) -> None:
    if S.WIN:
        S.fail("the Tenstorrent engine runs on Linux only (the tt-kmd driver, tt-metal in Docker)", f"use Linux - {DOCS}")
    for name in NEEDED:
        if not callable(getattr(S, name, None)):
            S.fail(f"setup.py has no {name}() any more: tt/setup_tt.py needs updating for this setup.py")
    argv = tt_argv(argv)                                # refuses what the engine cannot do, before anything is done
    restrict()

    cards = tt_devices()
    if not cards:
        S.fail("no Tenstorrent card found (/dev/tenstorrent is empty or missing)",
               f"install the tt-kmd driver (Tenstorrent's tt-installer), reboot, and run this again - {DOCS}")
    usable = [g for g in cards if tt_problem(g) is None]
    if not usable:
        for g in cards:
            S.say(f"    card {g['index']}: {g['name']} - {tt_problem(g)}")
        S.fail("no Tenstorrent card the TT engine runs on", f"it runs on a Blackhole P150 as /dev/tenstorrent/0 - {DOCS}")
    card = usable[0]
    problem = docker_problem()
    if problem:
        S.fail(*problem)
    real_ram = S.ram_gb()
    fake_ram = max(real_ram, 1024.0)
    image = engine_image()
    lock = lock_path()

    def card_notes():
        """Said under step 1's header: the card, its driver and link, what the engine needs around it."""
        link = pcie_text(card["pci"])
        S.ok(f"Tenstorrent card: {card['name']}, {card['vram_gb']:.0f} GB" + (f", tt-kmd {card['kmd']}" if card["kmd"]
                                                                              else "") + (f", {link}" if link else ""))
        if not os.access(card["node"], os.R_OK | os.W_OK):
            S.warn(f"{card['node']} is not readable and writable by this user: the engine cannot open the card "
                   f"(tt-kmd's udev rule makes it so; {DOCS})")
        if real_ram < MIN_RAM_GB:
            S.warn(f"RAM: {real_ram:.0f} GB - the Coder pins a ~12 GB host pool for the experts the card does not "
                   f"hold, beside its own working memory: {MIN_RAM_GB} GB or more is recommended")
        if hugepages_1g() < 1:
            S.warn("no 1 GiB huge pages are reserved: tt-metal maps the card's host memory through them when the "
                   f"IOMMU is off (Tenstorrent's tt-installer sets them up; {DOCS})")
        why = lock_problem(lock)
        if why:
            S.warn(f"the shared device lock: {why} (FNX_DEVICE_LOCK names another one)")
        users = card_users(card["index"])
        S.ok(f"device lock: {lock} (every job on this card takes it; the model holds it while it is loaded)"
             + (f"; the card is in use right now (pid {', '.join(users)}): the model starts when it is free"
                if users else ""))
        S.ok(f"engine image: {image}" + ("" if image_present(image) else " (not downloaded yet: step 4 pulls it)"))

    data_folder = S.data_folder

    def data_folder_tt(requested):
        dest, elsewhere = data_folder(requested)
        STATE["data"] = dest
        return dest, elsewhere
    S.data_folder = data_folder_tt

    S.say = say_rewriter(real_ram, fake_ram, image)
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: cards
    S.amd_problem = tt_problem
    S.amd_device_access_problem = lambda *a, **k: None
    S.hip_vision = lambda asked: "none"                 # no image encoder on the TT engine
    S.build_engine_hip = lambda gpu, llama, vision="none": engine_step(image, llama)
    S.hipblaslt_table = lambda *a, **k: None
    S.ram_gb = lambda: fake_ram                         # the experts are on the card and in its host pool
    S.bench_tips = lambda *a, **k: []                   # the community bench numbers are the CUDA engine's
    S.parallel_note = lambda *a, **k: []                # one request at a time
    S.desktop_reserve_note = lambda *a, **k: []         # an AMD card's desktop VRAM
    low_ram_wanted = S.low_ram_wanted

    def low_ram_wanted_tt(model, ram, choice="auto"):
        """Called once right after step 2's size choice: the model this install is for, and its own RAM need."""
        STATE["model"] = model
        need = TT_MODELS[model]["ram_gb"]
        if real_ram < need:
            S.warn(f"RAM: {real_ram:.0f} GB - {model} on the P150 wants {need} GB or more ({TT_MODELS[model]['fit']})")
        return low_ram_wanted(model, ram, choice)
    S.low_ram_wanted = low_ram_wanted_tt
    step = S.step

    def step_tt(n, title):
        step(n, title)
        if n == 1:
            card_notes()
    S.step = step_tt

    previous = S.previous_config

    def previous_config(*a, **k):
        """An earlier install's choices are taken over only from a P150 install (another one's model is not this)."""
        prev = previous(*a, **k)
        try:
            return prev if prev is not None and json.loads(prev.read_text(encoding="utf-8-sig")).get("backend") == "tt" \
                else None
        except (OSError, ValueError):
            return None
    S.previous_config = previous_config

    installed = S.installed_configs

    def installed_configs():
        """This folder's P150 models (a CUDA or AMD model in the same folder is started by ./setup.sh)."""
        out = []
        for p in installed():
            try:
                if json.loads(p.read_text(encoding="utf-8-sig")).get("backend") == "tt":
                    out.append(p)
            except (OSError, ValueError):
                pass
        return out
    S.installed_configs = installed_configs

    write = S.write_run_script

    def write_run_script(model, cfg_path, port, open_browser=True):   # setup.write_run_script's signature (#870)
        cfg_path = Path(cfg_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        data = fnx_data_dir(cfg)
        # the P150's files, after the pack and the MTP draft layer (iq3s: and its card caches from the GGUF)
        prepare(data, config_variant(cfg), flag([str(x) for x in cfg.get("args") or []], "--native"))
        S.write_config(cfg_path, to_tt(cfg, data, image))
        return run_script(write, model, cfg_path, port, open_browser)
    S.write_run_script = write_run_script

    start = S.start

    def start_tt(cfg_path, port=None, gpu=None, open_browser=True, yes=False, layer_split=None, keep=None):
        cfg_path = Path(cfg_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        if cfg.get("backend") != "tt":
            return start(cfg_path, port, gpu, open_browser, yes, layer_split, keep)
        cfg = S.upgrade_config(cfg_path, cfg)
        a = cfg["args"]
        paths = [cfg["exe"], *[x for x in a if x.endswith(".gguf")], *[flag(a, f) for f in ("--pack", "--mtp") if flag(a, f)]]
        missing = [p for p in paths if not Path(p).exists()]
        data = Path((cfg.get("env") or {}).get("FNX_DATA") or fnx_data_dir(cfg))
        env = {**os.environ, **(cfg.get("env") or {})}
        missing += [str(artifact_path(data, f["path"], env))
                    for f in missing_artifacts(data, env=env, variant=config_variant(cfg))]
        if missing:
            S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}"
                   + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""),
                   "run ./setup.sh --backend tt --setup to repair")
        keep = {k: v for k, v in (keep or {}).items() if v is not None and k in ("host", "api_key", "open_browser")}
        if keep and any(cfg.get(k) != v for k, v in keep.items()):   # as setup.start: kept for this model from now on
            cfg.update(keep)
            S.write_config(cfg_path, cfg)
            S.ok("saved for this model: " + ", ".join(
                "api key" if k == "api_key" else ("the browser opens" if v else "no browser") if k == "open_browser"
                else f"{k} {v}" for k, v in keep.items()))
        cfg_path.touch()                                # the most recently used model
        browser = open_browser and cfg.get("open_browser") is not False
        # the script as the config says (its port, the browser); a --port given for this start only goes on its line
        script = run_script(write, cfg_path.stem[len("strata-"):], cfg_path, cfg.get("port", 8080), browser)
        start_banner(cfg, lock, browser)
        once = ["--port", str(port)] if port and str(port) != str(cfg.get("port", 8080)) else []
        S.say(f"\nstarting {' '.join([script.name, *once])} ...")
        sys.stdout.flush()
        os.execv("/bin/sh", ["/bin/sh", str(script), *once])
    S.start = start_tt

    sys.argv = [str(ROOT / "setup.py"), *argv]
    sys.exit(S.main())


if __name__ == "__main__":
    try:
        install(sys.argv[1:])
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
