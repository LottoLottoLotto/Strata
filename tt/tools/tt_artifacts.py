#!/usr/bin/env python3
"""The tt backend's downloadable artifacts: the files under FNX_DATA that the P150 engine (fnx) reads and a fresh
install cannot derive from setup's own downloads (GGUF shards, pack, mtp/rt).  Published on Hugging Face
(HF_REPO), pinned to one commit in tt/artifacts.json {repo, revision, plan, files: [{path, size, sha256}]}, paths
relative to FNX_DATA:

  quant/store/plans/<plan>/        plan.json + L{LL}.fmt/row/res.npy: the default precision plan (fnx/tt/model.py
                                   DEFAULT_PLAN, read through fnx.quant.store.Plan)
  quant/store/pool/L{LL}.b4.npy    GPTQ-BFP4 rows of every routed expert, + L{LL}.b4idx.npy (expert, matrix) per row;
                                   only the formats the plan uses (the default plan is all-BFP4: no b2 rows)
  mtp/experts-bfp4-search.u8       the MTP drafter's 512 routed experts in BFP4 (fnx.ref.mtp.MtpWeights.bfp4_rows)
  mtp/expert-profile.npy           the drafter's routing counts (fnx.tt.mtp.host_experts; made by a device run)

Not in the set: caches rebuilt by the first engine start (cache/dequant ~16 GB, cache/tt ~14 GB, kcache ~9 GB on the
dev machine), the opt-in plans (h2 BFP2 rows, the gptq4 LM head), the calibration corpus.

    python3 tt/tools/tt_artifacts.py fetch  --data DIR [--only GLOB,..] [--jobs 4]   resumable, sha256-checked
    python3 tt/tools/tt_artifacts.py verify --data DIR [--only GLOB,..]               local files vs the manifest
    python3 tt/tools/tt_artifacts.py verify --remote                                  published files vs the manifest
    venv/bin/python tt/tools/tt_artifacts.py manifest [--data DIR]                    sha256 + size -> artifacts.json
    venv/bin/python tt/tools/tt_artifacts.py publish  [--data DIR]                    upload + card, pins the revision
    venv/bin/python tt/tools/tt_artifacts.py regenerate --data DIR --gguf SHARD1 --mtp RT   rebuild (CUDA GPU; see
                                                                                           cmd_regenerate)

fetch / verify use only the standard library (setup's interpreter); manifest needs numpy, publish huggingface_hub,
regenerate the fnx venv (torch with CUDA).  HF_ENDPOINT selects a mirror as in setup.py's hf_endpoint().  Verified
files are remembered in DIR/tt-artifacts.verified.json (size + mtime + sha256), so a later fetch skips them without
reading them again; verify always reads every file.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TT = Path(__file__).resolve().parents[1]                 # <repo>/tt
MANIFEST = TT / "artifacts.json"
HF_REPO = "Lottolabs/Qwen3.8-Flash-Next-Coder-P150"
HF_DEFAULT = "https://huggingface.co"
STAMPS = "tt-artifacts.verified.json"
UA = {"User-Agent": "strata-tt-artifacts"}
BLOCK = 8 << 20
ATTEMPTS = 30
POOL_TAG = {4: "b4", 2: "b2"}
COMMIT_BYTES = 4 << 30                                   # publish: data per commit (a resumable unit)
# the published GPTQ run (quant/run-quantize.json of the dev store, 2026-10-06): unweighted Hessians, prior to 8192
# rows, bfp4 rows of every expert; regenerate's defaults
QUANTIZE_ARGS = ["--weighted", "0", "--n-min", "8192", "--beta", "0", "--transform", "none", "--pool4", "1.0",
                 "--batch", "8", "--chunk", "32"]
QWEN_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE"
QWEN_LICENSE_SHA256 = "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fail(msg: str, code: int = 1):
    say(f"tt_artifacts: {msg}")
    raise SystemExit(code)


def hf_endpoint() -> str:
    """HF_ENDPOINT as huggingface_hub (and setup.py) read it, else huggingface.co."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def default_data() -> Path:
    sys.path.insert(0, str(TT))
    from fnx import paths
    return Path(paths.DATA)


def local(data: Path, rel: str) -> Path:
    """Where `rel` (a manifest path) lives: under DATA, except quant/ which FNX_QUANT_DIR relocates (fnx.paths)."""
    q = os.environ.get("FNX_QUANT_DIR")
    if q and rel.startswith("quant/"):
        return Path(q) / rel[len("quant/"):]
    return data / rel


def default_plan() -> str:
    """The Coder's DEFAULT_PLANS entry of fnx/tt/model.py (read from the source: the module needs ttnn)."""
    for node in ast.parse((TT / "fnx" / "tt" / "model.py").read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "DEFAULT_PLANS" for t in targets):
            return ast.literal_eval(node.value)["coder"]
    fail("no DEFAULT_PLANS in fnx/tt/model.py")


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        fail(f"no manifest {path} (tt_artifacts.py manifest writes it)")
    return json.loads(path.read_text())


def selected(files: list, only: str | None) -> list:
    if not only:
        return files
    pats = [p.strip() for p in only.split(",") if p.strip()]
    out = [f for f in files if any(fnmatch.fnmatchcase(f["path"], p) for p in pats)]
    if not out:
        fail(f"--only {only}: no manifest file matches")
    return out


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def sha256_file(p: Path, h=None, progress=None) -> str:
    h = h or hashlib.sha256()
    buf = bytearray(BLOCK)
    mv = memoryview(buf)
    with open(p, "rb", buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
            if progress:
                progress(n)
    return h.hexdigest()


class Progress:
    """One aggregate progress line on stderr (in place on a terminal, every 30 s otherwise)."""

    def __init__(self, what: str, total: int, done: int = 0):
        self.what, self.total, self.done = what, total, done
        self.lock = threading.Lock()
        self.t0 = self.last = time.time()
        self.start = done
        self.tty = sys.stderr.isatty()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last >= (2 if self.tty else 30):
                self.last = now
                self._line(now)

    def _line(self, now: float, end: str = "") -> None:
        rate = (self.done - self.start) / max(now - self.t0, 1e-9)
        pct = 100 * self.done / self.total if self.total else 100
        msg = f"  {self.what}: {gb(self.done)} / {gb(self.total)} ({pct:.0f}%), {rate / 1e6:.1f} MB/s"
        if self.tty:
            print("\r" + msg + "   ", end=end, file=sys.stderr, flush=True)
        else:
            print(msg, file=sys.stderr, flush=True)

    def close(self) -> None:
        self._line(time.time(), "\n" if self.tty else "")


class Stamps:
    """DATA/tt-artifacts.verified.json: rel path -> [size, mtime_ns, sha256] of files whose sha256 was checked."""

    def __init__(self, data: Path):
        self.path = data / STAMPS
        self.lock = threading.Lock()
        try:
            self.d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.d = {}

    def ok(self, rel: str, p: Path, size: int, sha: str) -> bool:
        st = p.stat()
        return self.d.get(rel) == [size, st.st_mtime_ns, sha] and st.st_size == size

    def set(self, rel: str, p: Path, sha: str) -> None:
        st = p.stat()
        with self.lock:
            self.d[rel] = [st.st_size, st.st_mtime_ns, sha]
            self._save()

    def drop(self, rel: str) -> None:
        with self.lock:
            if self.d.pop(rel, None) is not None:
                self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.d, indent=0, sort_keys=True))
        os.replace(tmp, self.path)


# ------------------------------------------------------------------------------------------------------- manifest
def artifact_paths(data: Path, plan: str) -> list[str]:
    import numpy as np
    pdir = local(data, f"quant/store/plans/{plan}")
    if not (pdir / "plan.json").is_file():
        fail(f"plan {plan}: no {pdir / 'plan.json'}")
    layers = sorted(int(p.name[1:3]) for p in pdir.glob("L[0-9][0-9].fmt.npy"))
    if not layers:
        fail(f"plan {plan}: no L??.fmt.npy in {pdir}")
    rels = [f"quant/store/plans/{plan}/plan.json"]
    bits = set()
    for L in layers:
        bits |= {int(b) for b in np.unique(np.load(pdir / f"L{L:02d}.fmt.npy"))}
        for kind in ("fmt", "row", "res"):
            if (pdir / f"L{L:02d}.{kind}.npy").is_file():
                rels.append(f"quant/store/plans/{plan}/L{L:02d}.{kind}.npy")
    for L in layers:
        for b in sorted(bits, reverse=True):
            rels += [f"quant/store/pool/L{L:02d}.{POOL_TAG[b]}.npy", f"quant/store/pool/L{L:02d}.{POOL_TAG[b]}idx.npy"]
    return rels + ["mtp/experts-bfp4-search.u8", "mtp/expert-profile.npy"]


def hash_all(data: Path, files: list, jobs: int, what: str) -> dict:
    """{rel: sha256 | None (missing)} reading every file once, `jobs` files at a time."""
    present = [f for f in files if local(data, f["path"]).is_file()]
    prog = Progress(what, sum(local(data, f["path"]).stat().st_size for f in present))

    def one(f):
        return f["path"], sha256_file(local(data, f["path"]), progress=prog.add)
    with ThreadPoolExecutor(jobs) as ex:
        got = dict(ex.map(one, present))
    prog.close()
    return {f["path"]: got.get(f["path"]) for f in files}


def cmd_manifest(a) -> int:
    data = a.data or default_data()
    plan = default_plan()
    rels = artifact_paths(data, plan)
    missing = [r for r in rels if not local(data, r).is_file()]
    if missing:
        fail(f"missing under {data}: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    files = [{"path": r, "size": local(data, r).stat().st_size} for r in rels]
    sums = hash_all(data, files, a.jobs, "hashing")
    for f in files:
        f["sha256"] = sums[f["path"]]
    old = json.loads(a.manifest.read_text()) if a.manifest.is_file() else {}
    same = old.get("files") == files
    out = {"repo": old.get("repo", HF_REPO), "revision": old.get("revision") if same else None, "plan": plan,
           "files": files}
    a.manifest.write_text(json.dumps(out, indent=1) + "\n")
    stamps = Stamps(data)
    for f in files:
        stamps.set(f["path"], local(data, f["path"]), f["sha256"])
    say(f"manifest: {len(files)} files, {gb(sum(f['size'] for f in files))}, plan {plan} -> {a.manifest}"
        + ("" if same or not old.get("revision") else " (files changed: revision cleared, publish again)"))
    return 0


# ---------------------------------------------------------------------------------------------------------- verify
def verify_local(data: Path, files: list, jobs: int) -> list[str]:
    sums = hash_all(data, files, jobs, "verifying")
    stamps = Stamps(data)
    bad = []
    for f in files:
        p = local(data, f["path"])
        got = sums[f["path"]]
        if got is None:
            bad.append(f"missing  {f['path']}")
        elif p.stat().st_size != f["size"] or got != f["sha256"]:
            stamps.drop(f["path"])
            bad.append(f"differs  {f['path']} ({p.stat().st_size} B, sha256 {got[:12]}..)")
        else:
            stamps.set(f["path"], p, got)
    return bad


def http_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        nxt = None
        for part in (r.headers.get("Link") or "").split(","):
            if 'rel="next"' in part:
                nxt = part[part.index("<") + 1:part.index(">")]
        return json.loads(r.read()), nxt


def remote_tree(repo: str, revision: str) -> dict:
    """path -> tree entry of every file at `revision` (Hub API, paginated)."""
    url = f"{hf_endpoint()}/api/models/{repo}/tree/{revision}?recursive=true&expand=false"
    out = {}
    while url:
        items, url = http_json(url)
        out.update({x["path"]: x for x in items if x.get("type") == "file"})
    return out


def resolve_url(man: dict, rel: str) -> str:
    return f"{hf_endpoint()}/{man['repo']}/resolve/{man['revision']}/{urllib.parse.quote(rel, safe='/')}"


def verify_remote(man: dict, files: list) -> list[str]:
    """Every file at the pinned revision has the manifest's size and sha256: LFS files by their LFS oid (= sha256)
    from the tree API, regular (small) files by downloading and hashing them."""
    tree = remote_tree(man["repo"], man["revision"])
    bad = []
    for f in files:
        x = tree.get(f["path"])
        if x is None:
            bad.append(f"absent   {f['path']}")
        elif x.get("lfs"):
            if x["lfs"].get("oid") != f["sha256"] or int(x["lfs"].get("size", -1)) != f["size"]:
                bad.append(f"differs  {f['path']} (remote {x['lfs'].get('size')} B, sha256 {x['lfs'].get('oid')})")
        else:
            with urllib.request.urlopen(urllib.request.Request(resolve_url(man, f["path"]), headers=UA),
                                        timeout=60) as r:
                body = r.read()
            if len(body) != f["size"] or hashlib.sha256(body).hexdigest() != f["sha256"]:
                bad.append(f"differs  {f['path']} (remote {len(body)} B)")
    return bad


def cmd_verify(a) -> int:
    man = load_manifest(a.manifest)
    files = selected(man["files"], a.only)
    total = gb(sum(f["size"] for f in files))
    if a.remote:
        if not man.get("revision"):
            fail("the manifest has no revision (not published)")
        bad = verify_remote(man, files)
        where = f"{man['repo']}@{man['revision']}"
    else:
        data = a.data or default_data()
        bad = verify_local(data, files, a.jobs)
        where = str(data)
    for b in bad:
        say("  " + b)
    say(f"verify {where}: {len(files)} files, {total}: " + (f"{len(bad)} problem(s)" if bad else "all match"))
    return 1 if bad else 0


# ----------------------------------------------------------------------------------------------------------- fetch
def download(url: str, dst: Path, f: dict, prog: Progress, stop: threading.Event) -> bool:
    """Resumable download into dst.part (HTTP range), sha256 over the whole file, then an atomic rename.  True when
    dst is in place; False when `stop` was set (another file failed): the .part stays for the next run."""
    part = dst.with_name(dst.name + ".part")
    size, sha = f["size"], f["sha256"]
    bad_sums = 0
    for attempt in range(1, ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            prog.add(-have)
            have = 0
        h = hashlib.sha256()
        if have:
            sha256_file(part, h)                         # the bytes already there, then the rest of the stream
        try:
            if have < size:
                req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as out:
                    if have and r.status != 206:         # the server ignored the range: start over
                        say(f"  {f['path']}: the server ignored the range request, starting it over")
                        out.seek(0)
                        out.truncate()
                        prog.add(-have)
                        have = 0
                        h = hashlib.sha256()
                    elif have:
                        say(f"  {f['path']}: resuming at {have / 1e6:.0f} MB")
                    while have < size and not stop.is_set():
                        b = r.read(BLOCK)
                        if not b:
                            break
                        out.write(b)
                        h.update(b)
                        have += len(b)
                        prog.add(len(b))
                    out.flush()
                    os.fsync(out.fileno())
            if stop.is_set():
                return False
            if have != size:
                raise OSError(f"connection ended at {have:,} of {size:,} bytes")
            if h.hexdigest() != sha:
                part.unlink()
                prog.add(-have)
                bad_sums += 1
                if bad_sums >= 2:
                    fail(f"{f['path']}: sha256 {h.hexdigest()} twice, the manifest says {sha}")
                say(f"  {f['path']}: sha256 mismatch, downloading it again")
                continue
            os.replace(part, dst)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                fail(f"{f['path']}: {e.code} {e.reason} from {url.split('/')[2]} ({url})")
            say(f"  {f['path']}: {e} (attempt {attempt}/{ATTEMPTS}), retrying in 10 s")
        except OSError as e:
            say(f"  {f['path']}: {e} (attempt {attempt}/{ATTEMPTS}), retrying in 10 s")
        if stop.wait(10):
            return False
    fail(f"{f['path']}: could not finish the download after {ATTEMPTS} attempts (run it again: it resumes)")


def cmd_fetch(a) -> int:
    man = load_manifest(a.manifest)
    data = a.data or default_data()
    files = selected(man["files"], a.only)
    stamps = Stamps(data)
    todo, done_bytes = [], 0
    for f in files:
        p = local(data, f["path"])
        if p.is_file() and p.stat().st_size == f["size"]:
            good = stamps.ok(f["path"], p, f["size"], f["sha256"])
            if not good and sha256_file(p) == f["sha256"]:
                stamps.set(f["path"], p, f["sha256"])
                good = True
            if good:
                done_bytes += f["size"]
                continue
            say(f"  {f['path']}: content differs from the manifest, downloading it again")
        todo.append(f)
    total = sum(f["size"] for f in files)
    if not todo:
        say(f"fetch: {len(files)} files, {gb(total)} already in {data}")
        return 0
    if not man.get("revision"):
        fail(f"{a.manifest} has no pinned revision (the artifacts are not published yet): {len(todo)} files missing "
             f"or different in {data}")
    parts = {f["path"]: local(data, f["path"]).with_name(Path(f["path"]).name + ".part") for f in todo}
    have = sum(min(p.stat().st_size, f["size"]) for f in todo for p in [parts[f["path"]]] if p.is_file())
    need = sum(f["size"] for f in todo) - have
    probe = data
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if need + (1 << 30) > free:
        fail(f"{gb(need)} to download into {data}, {gb(free)} free there (keep 1 GB spare)")
    say(f"fetch: {len(todo)} of {len(files)} files, {gb(need)} from {hf_endpoint()}/{man['repo']} "
        f"@ {man['revision'][:12]} into {data}")
    prog = Progress("tt artifacts", total, done_bytes + have)
    stop = threading.Event()

    def one(f):
        dst = local(data, f["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if download(resolve_url(man, f["path"]), dst, f, prog, stop):
            stamps.set(f["path"], dst, f["sha256"])
    ex = ThreadPoolExecutor(max(1, a.jobs))
    try:
        for fut in as_completed([ex.submit(one, f) for f in todo]):
            fut.result()                                 # the first failure ends the run (the others stop too)
    except BaseException:
        stop.set()
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    prog.close()
    say(f"fetch: {len(files)} files, {gb(total)} in {data}, sha256 verified")
    return 0


# --------------------------------------------------------------------------------------------------------- publish
def card(man: dict) -> str:
    plan = json.loads((Path(man["_data"]) / f"quant/store/plans/{man['plan']}/plan.json").read_text()) \
        if man.get("_data") else {}
    total = sum(f["size"] for f in man["files"])
    n_pool = sum(1 for f in man["files"] if f["path"].startswith("quant/store/pool/") and f["path"].endswith(".b4.npy"))
    res = plan.get("n_resident", 8680)
    share = plan.get("resident_routing_share", 0.859)
    return f"""---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - tenstorrent
  - blackhole
  - p150
  - strata
  - moe
  - bfp4
  - gptq
---

# Qwen3.8-Flash-Next Coder for the Tenstorrent P150 (Strata `tt` backend)

Block-float weights that let [Strata](https://github.com/Niko1221/Strata)'s `tt` backend run Qwen3.8-Flash-Next
Coder on one Tenstorrent Blackhole P150 card:

- the routed experts of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (IQ1_M), 48 layers x 256 experts x gate/up/down, re-quantized with GPTQ to Tenstorrent's BFP4 (`bfp4_b`) format,
  in the exact byte layout the P150 kernels read;
- the precision/placement plan `{man['plan']}`: which experts live in the card's DRAM and which in pinned host
  memory;
- the MTP draft layer's 512 routed experts in BFP4 and their routing profile (speculative decoding).

**This is not a standalone model.** It holds no tokenizer, attention, embedding or LM-head weights: those come from
the GGUF that Strata's setup downloads. The files are only useful to the Strata `tt` backend (the `fnx` engine in
Strata's `tt/` folder) and are fetched by it at a pinned commit (`tt/artifacts.json`, `tt/tools/tt_artifacts.py fetch`).

## Files ({len(man['files'])} files, {total / 1e9:.1f} GB)

Paths are relative to the backend's data folder (`FNX_DATA`).

| path | what |
|---|---|
| `quant/store/pool/L{{00..47}}.b4.npy` | GPTQ-BFP4 rows of every routed expert of the layer: uint8 [768, 921600], one row per (expert, matrix), W^T [in, out] with one shared exponent per 16 output channels ({n_pool} files, ~708 MB each) |
| `quant/store/pool/L{{00..47}}.b4idx.npy` | int32 [768, 2]: (expert, matrix) of each row; matrix 0 gate, 1 up, 2 down |
| `quant/store/plans/{man['plan']}/` | the plan: per layer `fmt` (all 4 = BFP4), `row` (pool row of each matrix), `res` (device-resident or host), `plan.json` |
| `mtp/experts-bfp4-search.u8` | the MTP draft layer's 512 routed experts: uint8 [512, 3, 921600] BFP4 rows |
| `mtp/expert-profile.npy` | int64 [512]: how often the drafter routes to each expert (the 254 hottest stay in device DRAM) |

## How it was made

- **Source**: the GGUF shards of ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF at revision
  `5348543e0147355ac9cbcb031184a3546350988e`, IQ1_M (the revision Strata pins), a quantization of
  [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) that keeps 256 of its 512 routed experts.
- **Calibration** (`fnx.quant.calib_data`): 128 sequences x 2048 tokens (+16 held out), chat 48 (UltraChat 200k
  train_sft), code 48 (Python, C/C++/CUDA, TypeScript/JavaScript, Rust, Go source trees), instruct 16
  (databricks-dolly-15k), docs 16 (Markdown documentation); disjoint from the quality-gate evaluation set.
- **GPTQ-BFP4** (`fnx.quant.pipeline quantize --weighted 0 --n-min 8192`): a float32 PyTorch forward of the GGUF
  (dequantized weights) over the calibration set, layer by layer on one RTX 3090 (62 minutes for 48 layers); per
  expert input Hessians of its routed rows (gate/up share one; a layer-mean prior up to 8192 rows for rarely routed
  experts), then GPTQ with act-order, 1% damping and a min-MSE shared exponent per 16-value block (clips -1..2),
  emitted as bit-exact tt-metal `bfp4_b` blocks.
- **Plan** (`fnx.quant.alloc --tiered 2.4e10`): every expert BFP4; the {res} most-routed experts (calibration
  routing counts, {100 * share:.1f}% of all routings, 24.0 GB) are resident in the P150's DRAM, the other
  {48 * 256 - res} are read from pinned host memory over PCIe by the MoE kernels.
- **MTP drafter**: the MTP layer of Qwen/Qwen3.8-Flash-Next as Strata packs it (Q2_0 experts), converted to BFP4 with
  a per-block min-MSE exponent (`fnx.quant.methods.search`); the routing profile comes from draft rounds on
  calibration text on the P150 (`fnx/tests/test_mtp.py profile`).

`tt/tools/tt_artifacts.py regenerate` in Strata reruns these steps (CUDA GPU with >= 16 GB).

## Quality and speed (P150, Strata `tt` backend, default configuration)

Quality gate against the published GGUF evaluated in float32 (teacher-forced chat/code/doc records + GSM8K), with
these BFP4 experts on the device (gated 2026-10-07 on the 22.0 GB placement of the same rows; the placement does not
change the numerics):

| metric | P150 |
|---|---|
| perplexity change, all records | +0.09% (chat +0.37%, code -0.18%) |
| top-1 agreement | 0.937 |
| KL divergence | 0.038 |
| GSM8K, first 50 | 45/50 (Strata's CUDA engine on an RTX 3090: 44/50) |

Decode speed on one P150, traced, chat / code / doc prompts: 30.9 / 29.2 / 26.6 tok/s plain, 46.9 / 50.9 / 50.0
tok/s with MTP speculative decoding (2026-10-08). Context up to 32,768 tokens; text only.

## License

These weights are a derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and are
distributed under the **Qwen Community License 1.0** (the [LICENSE](LICENSE) file here, copied unchanged from the
base model repository). Its condition 1 requires that copyright and permission notice in all copies. Its condition 2
applies to every downstream user: a licensee (or affiliate) that runs a *Model as a Service* or *AI Work Assistant*
business must obtain a separate license from Qwen before using the weights or derivatives for any commercial purpose
(internal use that exposes nothing to third parties excepted).

The routed experts were re-quantized from ISTA-DASLab's
[Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
(GSQ-RCO expert pruning and quantization by IST Austria's DASLab, published under Apache-2.0); credit for the
expert selection and the IQ1_M quantization goes to them. The MTP drafter weights come from the Qwen base checkpoint.
"""


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def cmd_publish(a) -> int:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        fail("publish needs huggingface_hub (pip install huggingface_hub hf_xet)")
    man = load_manifest(a.manifest)
    data = a.data or default_data()
    repo = man["repo"]
    files = man["files"]
    say(f"publish: checking {len(files)} local files against {a.manifest}")
    bad = verify_local(data, files, a.jobs) if not a.no_verify else []
    if bad:
        for b in bad:
            say("  " + b)
        fail("local files differ from the manifest (run manifest first)")
    lic = urllib.request.urlopen(urllib.request.Request(QWEN_LICENSE_URL, headers=UA), timeout=60).read()
    if hashlib.sha256(lic).hexdigest() != QWEN_LICENSE_SHA256:
        fail(f"{QWEN_LICENSE_URL}: not the Qwen Community License 1.0 text this card was written for")
    docs = {"README.md": card({**man, "_data": str(data)}).encode(), "LICENSE": lic}
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    tree = {x.path: x for x in api.list_repo_tree(repo, recursive=True) if hasattr(x, "blob_id")}

    def remote_ok(f) -> bool:
        x = tree.get(f["path"])
        if x is None:
            return False
        if x.lfs is not None:
            return x.lfs.sha256 == f["sha256"] and x.lfs.size == f["size"]
        return x.size == f["size"] and x.blob_id == git_blob_sha1(local(data, f["path"]).read_bytes())
    doc_ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in docs.items()
               if k not in tree or tree[k].blob_id != git_blob_sha1(v)]
    if doc_ops:
        api.create_commit(repo, operations=doc_ops, commit_message="model card + Qwen Community License 1.0")
        say(f"publish: {', '.join(o.path_in_repo for o in doc_ops)} committed")
    todo = [f for f in files if not remote_ok(f)]
    say(f"publish: {len(files) - len(todo)} files already in {repo}, {len(todo)} to upload "
        f"({gb(sum(f['size'] for f in todo))})")
    batch, batches = [], []
    for f in todo:
        if batch and (sum(x["size"] for x in batch) + f["size"] > COMMIT_BYTES or len(batch) >= 200):
            batches.append(batch)
            batch = []
        batch.append(f)
    if batch:
        batches.append(batch)
    t_all, sent = time.time(), 0
    for i, b in enumerate(batches, 1):
        t0 = time.time()
        n = sum(f["size"] for f in b)
        ops = [CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=str(local(data, f["path"]))) for f in b]
        info = api.create_commit(repo, operations=ops,
                                 commit_message=f"tt artifacts {i}/{len(batches)}: {len(b)} files, {gb(n)}")
        sent += n
        say(f"publish: commit {i}/{len(batches)} {str(info.oid)[:12]}: {len(b)} files {gb(n)} in "
            f"{time.time() - t0:.0f} s ({n / max(time.time() - t0, 1e-9) / 1e6:.0f} MB/s); "
            f"{gb(sent)} of {gb(sum(f['size'] for f in todo))} after {(time.time() - t_all) / 60:.0f} min")
    revision = api.repo_info(repo, repo_type="model").sha
    out = json.loads(a.manifest.read_text())
    out["revision"] = revision
    a.manifest.write_text(json.dumps(out, indent=1) + "\n")
    say(f"publish: {repo} @ {revision} pinned in {a.manifest}; checking the published files")
    bad = verify_remote(out, files)
    for b in bad:
        say("  " + b)
    say(f"verify {repo}@{revision}: {len(files)} files: " + (f"{len(bad)} problem(s)" if bad else "all match"))
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------------ regenerate
def cmd_regenerate(a) -> int:
    """The GPTQ-BFP4 store, the default plan and the MTP BFP4 experts from the GGUF, into DATA in the manifest layout.

    Steps (--steps, run in this order; each resumes or skips what exists):
      calib     DATA/quant/calib/seqs.json: --calib FILE is copied, else fnx.quant.calib_data builds it (it reads the
                dev machine's corpora: UltraChat/dolly in the HF cache and local source trees; elsewhere use --calib)
      quantize  fnx.quant.pipeline quantize with the published settings (QUANTIZE_ARGS): an NVIDIA GPU with >= 16 GB
                (peak 15 GB) and ~55 GB of host RAM (pinned residuals of the 144 sequences + the layer prefetch in
                /dev/shm: 53 GB RSS seen); 62 min on an RTX 3090 for 48 layers (one layer: 2.5 min with the inputs);
                writes DATA/quant/store/pool (34 GB of BFP4 rows, + BFP2 rows with --pool2), DATA/quant/{stats,proxy}
                and the dequantized non-expert weights DATA/cache/dequant (FNX_CACHE; ~16 GB, the cache the engine
                itself uses); layers already done are skipped
      alloc     fnx.quant.alloc --tiered <DEFAULT_PLAN's budget> --write: the plan (all 48 layers needed; seconds)
      mtp       fnx.ref.mtp.MtpWeights(--mtp RT).bfp4_rows(): DATA/mtp/experts-bfp4-search.u8 from mtp-q2_0.gguf
                beside RT (setup's MTP runtime folder, the engine's --mtp); CPU only, 1.5 min with 8 threads
    mtp/expert-profile.npy is a P150 measurement (fnx/tests/test_mtp.py profile through tools/devrun.sh); regenerate
    does not rebuild it.  On the dev machine (RTX 3090, torch of the venv) layer 0's pool files and the MTP file came
    out bitwise equal to the published ones (2026-10-08); another GPU/driver/torch may round GPTQ differently: the
    summary line compares the result with the manifest."""
    data = a.data.absolute()
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    unknown = set(steps) - {"calib", "quantize", "alloc", "mtp"}
    if unknown:
        fail(f"unknown steps {sorted(unknown)} (calib, quantize, alloc, mtp)")
    if "quantize" in steps and not a.gguf:
        fail("the quantize step needs --gguf: shard 1 of the Coder IQ1_M GGUF (shard 2 beside it)")
    if "mtp" in steps and not a.mtp:
        fail("the mtp step needs --mtp RT: setup's MTP runtime folder (rt/, with mtp-q2_0.gguf beside it)")
    env = {**os.environ, "FNX_DATA": str(data), "PYTHONPATH": os.pathsep.join(
        [str(TT)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("FNX_QUANT_DIR", None)                   # everything under DATA/quant (the manifest layout)
    quant = data / "quant"
    py = sys.executable

    def run(*cmd) -> None:
        say("regenerate: " + " ".join(str(c) for c in cmd))
        t0 = time.time()
        rc = subprocess.call([str(c) for c in cmd], env=env, cwd=str(TT))
        if rc:
            fail(f"step failed (rc {rc}): {' '.join(str(c) for c in cmd)}", rc)
        say(f"regenerate: done in {(time.time() - t0) / 60:.1f} min")
    if "quantize" in steps:
        gguf = Path(a.gguf).absolute()                  # as given: shard 2 is looked up beside this name
        if not gguf.is_file():
            fail(f"--gguf {gguf}: no such file")
        env["FNX_GGUF"] = str(gguf)
        probe = subprocess.run([py, "-c", "import torch; p = torch.cuda.get_device_properties(0); "
                                "print(torch.cuda.is_available() and p.total_memory >> 20, p.name)"],
                               capture_output=True, text=True, env=env)
        mib = probe.stdout.split(" ", 1)[0] if probe.returncode == 0 else ""
        if probe.returncode or not mib.isdigit() or int(mib) < 16 * 1024 - 512:
            why = probe.stdout.strip() or (probe.stderr.strip().splitlines() or [""])[-1]
            fail("the quantize step needs an NVIDIA GPU with >= 16 GB visible to torch (CUDA) in this interpreter "
                 f"({py}); the GPTQ pass peaks at 15 GB ({why})", 2)
        say(f"regenerate: CUDA {probe.stdout.strip().split(' ', 1)[1]}, {int(mib) // 1024} GB")
    plan = default_plan()
    if "calib" in steps:
        seqs = quant / "calib" / "seqs.json"
        if a.calib:
            seqs.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a.calib, seqs)
            say(f"regenerate: calibration sequences {a.calib} -> {seqs}")
        elif not seqs.is_file():
            run(py, "-m", "fnx.quant.calib_data")
    if "quantize" in steps:
        run(py, "-m", "fnx.quant.pipeline", "quantize", *QUANTIZE_ARGS, "--pool2", str(a.pool2),
            *(["--layers", a.layers] if a.layers else []))
    if "alloc" in steps:
        done = sorted(quant.glob("stats/L[0-9][0-9].npz"))
        if len(done) < 48:
            say(f"regenerate: alloc skipped ({len(done)}/48 layers quantized; the plan needs all of them)")
        else:
            prefix = "bfp4-tiered-"
            if not plan.startswith(prefix):
                fail(f"DEFAULT_PLAN {plan} is not a {prefix}<budget> plan: make it with fnx.quant.alloc by hand")
            run(py, "-m", "fnx.quant.alloc", "--tiered", plan[len(prefix):], "--write")
            if not (quant / "store" / "plans" / plan / "plan.json").is_file():
                fail(f"fnx.quant.alloc did not write plan {plan}")
    if "mtp" in steps:
        rt = Path(a.mtp).absolute()
        if not (rt / "dense.txt").is_file() or not (rt.parent / "mtp-q2_0.gguf").is_file():
            fail(f"--mtp {rt}: needs rt/dense.txt and ../mtp-q2_0.gguf (setup's MTP step makes them)")
        out = data / "mtp" / "experts-bfp4-search.u8"
        if out.exists():
            say(f"regenerate: {out} exists (delete it to rebuild)")
        else:
            run(py, "-c", "import sys; from fnx.ref.mtp import MtpWeights; MtpWeights(sys.argv[1]).bfp4_rows()", rt)
            if not out.is_file():
                fail(f"MtpWeights.bfp4_rows did not write {out}")
    man = load_manifest(a.manifest) if a.manifest.is_file() else None
    if man:
        files = [f for f in man["files"] if local(data, f["path"]).is_file()]
        bad = verify_local(data, files, 8) if files else []
        for b in bad:
            say("  " + b)
        say(f"regenerate: {len(files)} of {len(man['files'])} manifest files present in {data}; "
            + (f"{len(bad)} differ from the published bytes" if bad else "all equal the published bytes"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data_required=False):
        p.add_argument("--data", type=Path, required=data_required,
                       help="the data folder (FNX_DATA); default: fnx.paths.DATA")
        p.add_argument("--manifest", type=Path, default=MANIFEST, help=f"default {MANIFEST}")
        p.add_argument("--jobs", type=int, default=4, help="files hashed / downloaded at a time")
        return p
    common(sub.add_parser("manifest", help="sha256 + size of the artifact set -> artifacts.json"))
    p = common(sub.add_parser("publish", help="upload to the HF repo (huggingface_hub), card + LICENSE, pin revision"))
    p.add_argument("--no-verify", action="store_true", help="skip the local sha256 check before uploading")
    p = common(sub.add_parser("fetch", help="download the pinned files into --data (resumable, sha256-checked)"), True)
    p.add_argument("--only", help="comma list of glob patterns over manifest paths")
    p = common(sub.add_parser("verify", help="--data: local files vs the manifest; --remote: the published files"))
    p.add_argument("--only", help="comma list of glob patterns over manifest paths")
    p.add_argument("--remote", action="store_true")
    p = common(sub.add_parser("regenerate", help="rebuild the artifacts from the GGUF (CUDA GPU)"), True)
    p.add_argument("--gguf", help="shard 1 of the Coder IQ1_M GGUF, shard 2 beside it (the quantize step)")
    p.add_argument("--mtp", help="setup's MTP runtime folder rt/ with mtp-q2_0.gguf beside it (the mtp step)")
    p.add_argument("--calib", help="calibration sequences (fnx.quant.calib_data's seqs.json) to use instead of "
                                   "building them (calib_data reads the dev machine's corpora)")
    p.add_argument("--steps", default="calib,quantize,alloc,mtp", help="comma list of calib, quantize, alloc, mtp")
    p.add_argument("--layers", default="", help="quantize only these layers (comma list; a smoke test: no plan)")
    p.add_argument("--pool2", type=float, default=0.0,
                   help="also store BFP2 rows for this coldest fraction (0.4 = the dev store, for the opt-in h2 plans)")
    a = ap.parse_args(argv)
    return {"manifest": cmd_manifest, "publish": cmd_publish, "fetch": cmd_fetch, "verify": cmd_verify,
            "regenerate": cmd_regenerate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
