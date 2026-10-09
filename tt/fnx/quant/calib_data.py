"""Calibration corpus for the routed-expert GPTQ: 128 calibration + 16 held-out sequences of 2048 Coder tokens.

Disjoint from the quality gate's eval set (fnx.eval.data, FNX_DATA/eval/provenance.json): no ultrachat
test split, no CPython stdlib, no llama.cpp / Strata sources, no alpaca, no Opticks, no GSM8K. Same framing as the eval
(chat template with enable_thinking=False; code and docs as raw text). Balanced mix (ttquant lesson: an all-chat
calibration made code worse than RTN), weighted toward code for the Coder:

    corpus    calib  held-out  source
    chat         48      6     ultrachat_200k train_sft shard 0, multi-turn conversations packed back to back
    code         48      6     python 20/2 (sglang, dist-packages), c/c++/cuda 12/2 (sglang, vllm, localmaxxing),
                               ts/js 8/1 (hermes-agent, /usr/share/nodejs), rust 5/1 (sglang), go 3/0 (~/go)
    instruct     16      2     databricks-dolly-15k single-turn chats (instruction [+ context] -> response) packed
    docs         16      2     markdown documentation (sglang, vllm, hermes-agent, localmaxxing) packed raw

Output: FNX_QUANT_DIR/calib/seqs.json {"calib": [{"id", "corpus", "source", "token_ids"}], "heldout": [...],
"provenance": {...}}.  Deterministic (seed 0).  Run with the venv python (pyarrow, jinja2).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fnx import paths  # noqa: E402
from fnx.eval.data import Builder, ULTRACHAT, sha256  # noqa: E402

QDIR = paths.QUANT
SEQ = 2048
HOME = Path("/home/lotto")
DOLLY = HOME / ".cache/huggingface/hub/datasets--databricks--databricks-dolly-15k/snapshots/" \
    "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl"
SKIP_PARTS = ("node_modules", "llama.cpp", "Strata", "third_party", "/test", "/.git", "dist/", "build/", ".min.")
CODE = {  # language: (roots, suffixes, n_calib, n_heldout)
    "python": ([HOME / "sglang-dflash-sm86/python", Path("/usr/lib/python3/dist-packages")], (".py",), 20, 2),
    "cpp": ([HOME / "sglang-dflash-sm86", HOME / "vllm", HOME / "localmaxxing"], (".cu", ".cuh", ".cpp", ".cc", ".h"), 12, 2),
    "tsjs": ([HOME / "hermes-agent", Path("/usr/share/nodejs")], (".ts", ".tsx", ".js"), 8, 1),
    "rust": ([HOME / "sglang-dflash-sm86"], (".rs",), 5, 1),
    "go": ([HOME / "go"], (".go",), 3, 0),
}
DOCS = [HOME / "sglang-dflash-sm86", HOME / "vllm", HOME / "hermes-agent", HOME / "localmaxxing"]


def files(roots, suffixes, lo=1500, hi=120_000) -> list[Path]:
    out = []
    for r in roots:
        for p in r.rglob("*"):
            s = str(p)
            if p.suffix in suffixes and not any(k in s for k in SKIP_PARTS) and p.is_file() and not p.is_symlink():
                try:
                    n = p.stat().st_size
                except OSError:
                    continue
                if lo <= n <= hi:
                    out.append(p)
    return sorted(out)


def pack_raw(B: Builder, paths: list[Path], n_seq: int, rng: random.Random, used: set) -> list[dict]:
    """Whole files (random order) concatenated with blank lines until SEQ tokens; each sequence starts at a file."""
    seqs, order = [], paths[:]
    rng.shuffle(order)
    it = iter(order)
    while len(seqs) < n_seq:
        ids, src = [], []
        while len(ids) < SEQ:
            p = next(it)
            if p in used:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if max((len(l) for l in text.splitlines()), default=0) > 400:   # minified / generated
                continue
            used.add(p)
            ids += B.raw(text.strip() + "\n\n")
            src.append(str(p))
        seqs.append({"token_ids": ids[:SEQ], "source": src})
    return seqs


def pack_chats(B: Builder, convs, n_seq: int) -> list[dict]:
    seqs, it = [], iter(convs)
    while len(seqs) < n_seq:
        ids, src = [], []
        while len(ids) < SEQ:
            key, msgs = next(it)
            one, _ = B.chat(msgs)
            ids += one
            src.append(key)
        seqs.append({"token_ids": ids[:SEQ], "source": src})
    return seqs


def build() -> None:
    import pyarrow.parquet as pq

    B = Builder()
    rng = random.Random(0)
    calib, held = [], []

    def add(corpus, seqs, n_cal):
        for i, s in enumerate(seqs):
            rec = {"id": f"{corpus}-{i}", "corpus": corpus, **s}
            (calib if i < n_cal else held).append(rec)

    # chat: ultrachat train_sft shard 0, random rows (>= 2 user turns, user/assistant only)
    path = next((ULTRACHAT / "data").glob("train_sft-00000-*.parquet"))
    rows = pq.read_table(path, columns=["messages"]).to_pylist()
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    convs = []
    for r_i in idx:
        msgs = [{"role": m["role"], "content": m["content"]} for m in rows[r_i]["messages"]]
        if sum(m["role"] == "user" for m in msgs) >= 2 and all(m["role"] in ("user", "assistant") for m in msgs):
            convs.append((f"ultrachat train_sft-00000 row {r_i}", msgs))
        if len(convs) >= 400:
            break
    add("chat", pack_chats(B, convs, 54), 48)
    prov = {"chat": {"file": str(path), "sha256": sha256(path)}}

    # code
    used: set = set()
    prov["code"] = {}
    for lang, (roots, suf, n_cal, n_held) in CODE.items():
        seqs = pack_raw(B, files(roots, suf), n_cal + n_held, rng, used)
        for s in seqs:
            s["language"] = lang
        cal, hld = seqs[:n_cal], seqs[n_cal:]
        calib.extend({"id": f"code-{lang}-{i}", "corpus": "code", **s} for i, s in enumerate(cal))
        held.extend({"id": f"code-{lang}-h{i}", "corpus": "code", **s} for i, s in enumerate(hld))
        prov["code"][lang] = {"roots": [str(r) for r in roots], "suffixes": suf}

    # instruct: dolly single-turn chats
    rows = [json.loads(l) for l in DOLLY.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    convs = []
    for r_i in idx[:2000]:
        r = rows[r_i]
        q = r["instruction"].strip() + (("\n\n" + r["context"].strip()) if r["context"].strip() else "")
        convs.append((f"dolly row {r_i}", [{"role": "user", "content": q},
                                           {"role": "assistant", "content": r["response"].strip()}]))
    add("instruct", pack_chats(B, convs, 18), 16)
    prov["instruct"] = {"file": str(DOLLY), "sha256": sha256(DOLLY)}

    # docs
    add("docs", pack_raw(B, files(DOCS, (".md",), lo=2000), 18, rng, used), 16)
    prov["docs"] = {"roots": [str(r) for r in DOCS]}

    for s in calib + held:
        assert len(s["token_ids"]) == SEQ, (s["id"], len(s["token_ids"]))
    out = QDIR / "calib"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seqs.json").write_text(json.dumps({"seq_len": SEQ, "seed": 0, "calib": calib, "heldout": held,
                                               "provenance": prov}))
    by = {}
    for s in calib:
        by[s["corpus"]] = by.get(s["corpus"], 0) + 1
    print(f"calib {len(calib)} {by}, heldout {len(held)} -> {out / 'seqs.json'}")


def load(split: str = "calib") -> list[dict]:
    return json.loads((QDIR / "calib" / "seqs.json").read_text())[split]


if __name__ == "__main__":
    build()
