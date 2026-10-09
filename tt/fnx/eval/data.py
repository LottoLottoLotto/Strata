"""The quality gate's eval set: chat / code / instruct / book token sequences plus GSM8K prompts, Coder tokenizer.

    NODEV=1 tools/devrun.sh evaldata -m fnx.eval.data        (the TT image: pyarrow for the ultrachat parquet)

Writes FNX_EVAL (default /var/tmp/strata-tt/eval)/:
  evalset.jsonl    one record per sequence: id, corpus, source, token_ids, score = [[a, b), ..] token ranges whose
                   tokens are scored as TARGETS (position i-1's logits predict token i; a >= 1)
  gsm8k.jsonl      idx, question, gold, prompt_ids (user turn + the template's generation prompt, thinking off)
  provenance.json  every source: path, sha256, dataset revision, rows / offsets taken

Disjoint from the expert-quantization calibration (ExpertQuant: ultrachat train_sft, site-packages modules, dolly-15k,
wikitext-103 train): chat = ultrachat_200k TEST_sft, code = CPython 3.14 stdlib + llama.cpp ggml C/C++ + llama.cpp
web UI TypeScript + Strata CUDA, instruct = alpaca-cleaned, book = Newton's Opticks (Project Gutenberg), GSM8K test.

Framing: chats and instructions are rendered with the Coder's own chat template (the pack's chat_template.jinja,
rendered like Strata's serve/frontend.py) with enable_thinking=False, so every assistant turn reads
"<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n" + content + "<|im_end|>"; the content tokens and the closing
<|im_end|> are scored, the template tokens are not.  Code and book passages are raw text, every token scored.
Every sequence holds a multiple of 128 tokens (fnx.serve_engine's prompt chunk: the TT run feeds whole 128-row
chunks, one program shape) and at most 2048 (QSA stays dense: positions <= 2050 attend every cell).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

EVAL = Path(os.environ.get("FNX_EVAL", "/var/tmp/strata-tt/eval"))
PACK_TOK = Path("/var/tmp/strata-tt/packs/coder-iq1_m/tokenizer")
STRATA_TOOLS = Path(os.environ.get("STRATA_TOOLS", "/var/tmp/strata-tt/Strata/tools"))
HF = Path("/home/lotto/.cache/huggingface/hub")
ULTRACHAT = HF / "datasets--HuggingFaceH4--ultrachat_200k/snapshots/8049631c405ae6576f93f445c6b8166f76f5505a"
ALPACA = HF / "datasets--yahma--alpaca-cleaned/snapshots/12567cabf869d7c92e573c7c783905fc160e9639"
GSM8K = Path("/var/tmp/gemma4-12b/quality/data/gsm8k-test.jsonl")   # openai/gsm8k "main" test split, file order
OPTICKS = Path("/var/tmp/qwen27b-ctx/quality/data/opticks.txt")     # Gutenberg #33504, Newton, Opticks (1730)
CHUNK = 128
MAX_LEN = 2048
CODE = [  # (path, language); a 1024-token window from the first line break after 25% of the file
    ("/usr/lib/python3.14/fractions.py", "python"),
    ("/usr/lib/python3.14/difflib.py", "python"),
    ("/usr/lib/python3.14/statistics.py", "python"),
    ("/usr/lib/python3.14/calendar.py", "python"),
    ("/var/tmp/strata-tt/third_party/llama.cpp/ggml/src/ggml-alloc.c", "c"),
    ("/var/tmp/strata-tt/third_party/llama.cpp/ggml/src/ggml-backend.cpp", "c++"),
    ("/home/lotto/llama.cpp/tools/ui/src/lib/utils/contenteditable-tokenizer.ts", "typescript"),
    ("/var/tmp/strata-tt/Strata/src/kernels/cuda/qsa_select.cu", "cuda"),
]
CODE_LEN = 1024
N_CHAT, CHAT_MAX = 6, 1536
N_INSTRUCT, INSTRUCT_LEN = 3, 1024
BOOK_STARTS, BOOK_LEN = (0.15, 0.45, 0.75), 2048
GSM_N = 100
GSM_SUFFIX = "\n\nSolve the problem step by step, then give the final answer on the last line in the form \"#### <number>\"."


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer():
    """Strata's tokenizer (tools/strata_tokenizer.py) over the pack's exported files (vocab, merges, token types)."""
    sys.path.insert(0, str(STRATA_TOOLS))
    from strata_tokenizer import Tokenizer
    vocab = json.loads((PACK_TOK / "vocab.json").read_text(encoding="utf-8"))
    tokens = [""] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (PACK_TOK / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((PACK_TOK / "token_type.json").read_text())
    cfg = json.loads((PACK_TOK / "tokenizer.json").read_text())
    assert len(tokens) == cfg["vocab_size"] and len(merges) == cfg["n_merges"], (len(tokens), len(merges))
    return Tokenizer(tokens, merges, types, cfg["pre"], cfg["special_ids"])


def chat_template():
    """The pack's chat template, rendered as Strata's serve/frontend.py ChatTemplate does."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string((PACK_TOK / "chat_template.jinja").read_text(encoding="utf-8"))


class Builder:
    def __init__(self):
        self.tok = tokenizer()
        self.tpl = chat_template()

    def render(self, messages, gen: bool = False) -> str:
        return self.tpl.render(messages=messages, add_generation_prompt=gen, enable_thinking=False)

    def chat(self, messages) -> tuple[list[int], list[list[int]]]:
        """Messages (user/assistant) -> (ids, scored target ranges): every assistant turn's content + <|im_end|>."""
        text = self.render(messages)
        spans = []
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            head = self.render(messages[:i], gen=True)
            body = m["content"].strip() + "<|im_end|>"
            if not text.startswith(head) or text[len(head):len(head) + len(body)] != body:
                raise ValueError("assistant turn does not render as head + content + <|im_end|>")
            spans.append((len(head.encode()), len((head + body).encode())))
        ids = self.tok.encode(text, parse_special=True)
        assert self.tok.decode(ids) == text
        starts, b = [], 0
        for t in ids:
            starts.append(b)
            b += len(self.tok.token_bytes(t))
        scored = [any(a <= s < e for a, e in spans) for s in starts]
        for a, e in spans:                                   # every span starts on a token boundary
            assert a in set(starts), "an assistant turn starts inside a token"
        return ids, ranges(scored)

    def raw(self, text: str) -> list[int]:
        ids = self.tok.encode(text, parse_special=False)
        assert self.tok.decode(ids) == text
        return ids


def ranges(mask) -> list[list[int]]:
    """Boolean mask -> [[a, b), ..] of True runs, never including position 0 (it has no predecessor)."""
    out, a = [], None
    for i, m in enumerate(list(mask) + [False]):
        m = m and i > 0
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append([a, i])
            a = None
    return out


def clip(ids: list[int], score: list[list[int]], n: int) -> tuple[list[int], list[list[int]]]:
    return ids[:n], [[a, min(b, n)] for a, b in score if a < n]


def fit(n: int, cap: int) -> int:
    return min(n, cap) // CHUNK * CHUNK


def build() -> None:
    import pyarrow.parquet as pq

    B = Builder()
    recs, prov = [], {"tokenizer": {"dir": str(PACK_TOK), "vocab.json": sha256(PACK_TOK / "vocab.json"),
                                    "merges.txt": sha256(PACK_TOK / "merges.txt"),
                                    "chat_template.jinja": sha256(PACK_TOK / "chat_template.jinja")},
                      "framing": "chat template enable_thinking=False; assistant content + <|im_end|> scored"}

    # chat: ultrachat_200k test_sft, the first rows (file order) with >= 2 user turns
    path = next((ULTRACHAT / "data").glob("test_sft-*.parquet"))
    rows = pq.read_table(path, columns=["prompt_id", "messages"]).to_pylist()
    taken = []
    for r_i, r in enumerate(rows):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        if sum(m["role"] == "user" for m in msgs) < 2 or any(m["role"] not in ("user", "assistant") for m in msgs):
            continue
        ids, score = B.chat(msgs)
        n = fit(len(ids), CHAT_MAX)
        if n < 512:
            continue
        ids, score = clip(ids, score, n)
        recs.append({"id": f"chat-ultrachat-test-{r_i}", "corpus": "chat", "source": f"ultrachat_200k test_sft row {r_i}",
                     "token_ids": ids, "score": score})
        taken.append(r_i)
        if len(taken) == N_CHAT:
            break
    prov["chat"] = {"dataset": "HuggingFaceH4/ultrachat_200k", "revision": ULTRACHAT.name, "file": str(path),
                    "sha256": sha256(path), "split": "test_sft", "rows": taken,
                    "rule": f"first rows with >= 2 user turns and >= 512 tokens; truncated to <= {CHAT_MAX} (x128)"}

    # code: raw text windows
    prov["code"] = []
    for p, lang in CODE:
        text = Path(p).read_text(encoding="utf-8")
        start = text.index("\n", len(text) // 4) + 1
        ids = B.raw(text[start:])
        assert len(ids) >= CODE_LEN, (p, len(ids))
        ids = ids[:CODE_LEN]
        recs.append({"id": f"code-{lang}-{Path(p).name}", "corpus": "code", "source": f"{p} from char {start}",
                     "token_ids": ids, "score": ranges([True] * len(ids))})
        prov["code"].append({"path": p, "language": lang, "sha256": sha256(p), "char_start": start, "tokens": CODE_LEN})

    # instruct: alpaca-cleaned single-turn chats packed back to back, every 211th row from row 7
    path = ALPACA / "alpaca_data_cleaned.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    used, r_i = [], 7
    for s in range(N_INSTRUCT):
        ids, score = [], []
        while len(ids) < INSTRUCT_LEN:
            r = data[r_i]
            used.append(r_i)
            r_i += 211
            user = r["instruction"].strip() + (("\n\n" + r["input"].strip()) if r["input"].strip() else "")
            one, sc = B.chat([{"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}])
            score += [[a + len(ids), b + len(ids)] for a, b in sc]
            ids += one
        ids, score = clip(ids, score, INSTRUCT_LEN)
        recs.append({"id": f"instruct-alpaca-{s}", "corpus": "instruct", "source": "alpaca-cleaned packed",
                     "token_ids": ids, "score": score})
    prov["instruct"] = {"dataset": "yahma/alpaca-cleaned", "revision": ALPACA.name, "file": str(path),
                        "sha256": sha256(path), "rows": used,
                        "rule": f"rows 7, 218, 429, ... (step 211) as single-turn chats packed into {INSTRUCT_LEN} tokens"}

    # book: three 2048-token passages of Newton's Opticks (raw text)
    text = OPTICKS.read_text(encoding="utf-8")
    body = text[text.index("Advertisement I"):]
    prov["book"] = {"source": "Project Gutenberg #33504, Isaac Newton, Opticks (4th ed. 1730)", "file": str(OPTICKS),
                    "sha256": sha256(OPTICKS), "passages": []}
    for k, f in enumerate(BOOK_STARTS):
        start = body.index("\n\n", int(len(body) * f)) + 2
        ids = B.raw(body[start:start + 40000])[:BOOK_LEN]
        assert len(ids) == BOOK_LEN
        recs.append({"id": f"book-opticks-{k}", "corpus": "book", "source": f"opticks.txt body char {start}",
                     "token_ids": ids, "score": ranges([True] * len(ids))})
        prov["book"]["passages"].append({"body_char_start": start, "tokens": BOOK_LEN})

    for r in recs:
        n = len(r["token_ids"])
        assert n % CHUNK == 0 and 0 < n <= MAX_LEN, (r["id"], n)
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(EVAL / "evalset.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # GSM8K: the first GSM_N test questions, one user turn, thinking off
    rows = [json.loads(l) for l in GSM8K.read_text().splitlines() if l.strip()][:GSM_N]
    with open(EVAL / "gsm8k.jsonl", "w") as f:
        for i, r in enumerate(rows):
            ids = B.tok.encode(B.render([{"role": "user", "content": r["question"] + GSM_SUFFIX}], gen=True),
                               parse_special=True)
            f.write(json.dumps({"idx": i, "question": r["question"], "gold": gold_answer(r["answer"]),
                                "prompt_ids": ids}) + "\n")
    prov["gsm8k"] = {"dataset": "openai/gsm8k main test", "file": str(GSM8K), "sha256": sha256(GSM8K),
                     "rows": f"0..{GSM_N - 1}", "suffix": GSM_SUFFIX, "framing": "chat template, enable_thinking=False"}
    (EVAL / "provenance.json").write_text(json.dumps(prov, indent=1))

    by = {}
    for r in recs:
        c = by.setdefault(r["corpus"], [0, 0, 0])
        c[0] += 1
        c[1] += len(r["token_ids"])
        c[2] += sum(b - a for a, b in r["score"])
    for c, (n, t, s) in by.items():
        print(f"{c:9s} {n:2d} sequences {t:6d} tokens {s:6d} scored")
    print(f"total {sum(v[1] for v in by.values())} tokens, {sum(v[2] for v in by.values())} scored; "
          f"gsm8k {len(rows)} prompts; wrote {EVAL}")


# ------------------------------------------------------------------------------- GSM8K answers (gemma4-12b extract.py)
def _normalize(s):
    from decimal import Decimal, InvalidOperation
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s or None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def extract_answer(text: str | None):
    """Canonical answer of a model output: the last "#### x", else the last number (commas / $ stripped)."""
    import re
    if text is None:
        return None
    m = re.findall(r"####[\s*$\\]*(-?[\d,]*\.?\d+)", text)
    if m:
        return _normalize(m[-1])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return _normalize(nums[-1]) if nums else None


def gold_answer(answer_field: str):
    return _normalize(answer_field.split("####")[-1])


def load(name: str = "evalset") -> list[dict]:
    return [json.loads(l) for l in (EVAL / f"{name}.jsonl").read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    build()
