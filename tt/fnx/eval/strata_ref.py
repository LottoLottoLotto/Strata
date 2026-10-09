"""GSM8K reference answers from Strata's own CUDA engine on the RTX 3090 (the published model as Strata serves it).

    cd /var/tmp/strata-tt/flashnext_tt && /var/tmp/strata-tt/venv/bin/python -m fnx.eval.strata_ref [--n 100]

Strata's production arguments for the Coder (setup.py: expert profile + VRAM expert cache auto, MTP drafts,
--spec 4 --spec-min-p 0.5) in --serve mode, --prompt-cache 0, greedy (Strata's default), max 512 new tokens, stop at
EOS; one GEN per prompt of FNX_EVAL/gsm8k.jsonl.  Writes FNX_EVAL/gsm8k-strata.jsonl (same fields as fnx.eval.run's
gsm8k.jsonl) and the exact command line to gsm8k-strata.cmd.  Needs the 3090 to itself (no other GPU job).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time

from . import data

W = "/var/tmp/strata-tt"
S1 = f"{W}/models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
S2 = f"{W}/models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
ARGV = [f"{W}/build-cuda/strata", "--serve", "--pack", f"{W}/packs/coder-iq1_m", "--native", S1, "--ple-gguf", S2,
        "--expert-profile", f"{W}/Strata/data/expert-profile-coder.bin", "--expert-cache", "auto",
        "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", f"{W}/mtp/rt",
        "--max-context", "4096", "--prompt-cache", "0"]
EOS = (248044, 248046)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args(argv)
    out = data.EVAL / "gsm8k-strata.jsonl"
    done = {json.loads(l)["idx"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    probs = [p for p in data.load("gsm8k")[:a.n] if p["idx"] not in done]
    if not probs:
        print("nothing to do")
        return 0
    tok = data.tokenizer()
    (data.EVAL / "gsm8k-strata.cmd").write_text(" ".join(shlex.quote(x) for x in ARGV) + "\n")
    elog = open(data.EVAL / "logs" / "strata-gsm8k.log", "a")
    p = subprocess.Popen(ARGV, cwd=W, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=elog, text=True, bufsize=1)

    def readline():
        line = p.stdout.readline()
        if not line:
            raise SystemExit(f"strata exited (rc={p.wait()}); see {elog.name}")
        return line.rstrip("\n")

    while not readline().startswith("READY"):
        pass
    try:
        for pr in probs:
            t0 = time.time()
            p.stdin.write(f"GEN {a.max_new} {','.join(map(str, pr['prompt_ids']))}\n")
            p.stdin.flush()
            got = []
            while True:
                line = readline()
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE"):
                    break
                elif line.startswith("ERR"):
                    raise SystemExit(f"problem {pr['idx']}: {line}")
            dt = time.time() - t0
            text = tok.decode([t for t in got if t not in EOS])
            pred = data.extract_answer(text)
            rec = {"idx": pr["idx"], "gold": pr["gold"], "pred": pred, "correct": pred == pr["gold"],
                   "n_new": len(got), "stop": "eos" if got and got[-1] in EOS else "length", "seconds": round(dt, 2),
                   "ids": got, "text": text}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"strata gsm8k {pr['idx']}: pred {pred} gold {pr['gold']} {'OK' if rec['correct'] else 'WRONG'}, "
                  f"{len(got)} tokens in {dt:.1f} s", flush=True)
    finally:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=300)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    print(f"Strata 3090: {sum(r['correct'] for r in rows)}/{len(rows)} correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
