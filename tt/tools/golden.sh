#!/bin/bash
# tools/golden.sh STEP [NAME...]   Strata CUDA reference ("golden") data for the Flash-Next TT port (RTX 3090).
#
# Steps
#   prompts          write the three prompts (chat ~30 tok, code ~300 tok, doc ~2000 tok) and their token ids
#   wait             block until both GGUF shards are fully downloaded (every 64 MiB chunk listed in <file>.done)
#   pack             native pack (tools/iq_pack.py, setup.py's call) -> $PACK
#   oneshot NAME..   one-shot CLI: greedy 64 tokens, STRATA_DUMP_FIRST_LOGITS (last prompt position; the prompt
#                    goes through the batched prefill path, experts on the GPU), --dump-routing, --window-hashes
#   serve NAME..     --serve, every prompt position read through verify windows (--short-read >= n, decode
#                    numerics, CPU experts): STRATA_LOGPOS top-256 teacher-forced per prompt position + greedy 64
#   full NAME..      --serve teacher-forced over prompt + the serve step's 64 greedy tokens (GEN 1): LOGPOS rows for
#                    every prompt and continuation position
#   all              prompts wait pack, then oneshot serve full for chat code doc
# Every engine command is appended verbatim (env + argv) to $OUT/commands.log before it runs.
# Determinism knobs (spec Part 8 §8.3): STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 1000000 --prompt-cache 0,
# no --mtp, --suffix-draft 0, greedy.  Expert placement: both the one-shot spec loop (generate.cpp:10470) and the
# serve verifier refuse to run without the profile-filled VRAM expert tier, so the smallest one is used:
# --expert-profile data/expert-profile-coder.bin --expert-cache 1 (one resident blob, computed on the GPU); every
# other routed expert of a verify window runs on the CPU pool (Strata's CPU kernels, the ones fnx.cpu_experts calls).
# Development machine only: the reference is the CUDA build $DATA/build-cuda/strata of the pinned Strata checkout
# $DATA/Strata (82f46a8; its tools/ and README.md, from which the prompts come), FNX_DATA's dev layout (fnx.paths).
set -euo pipefail
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
eval "$(python3 "$TTDIR/fnx/paths.py" --shell)"     # DATA ...
W=$DATA
SRC=$W/Strata
PY=$W/venv/bin/python
export STRATA_GGUF_PY=$W/third_party/llama.cpp/gguf-py
S1=$W/models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
PACK=${PACK:-$W/packs/coder-iq1_m}
OUT=${GOLDEN:-$W/golden}
STRATA=$W/build-cuda/strata
MAXNEW=${MAXNEW:-64}
POOL=${POOL:-12}          # CPU expert pool workers (shared host)
LOG=$OUT/commands.log
mkdir -p $OUT/prompts
COMMON=(--pack $PACK --native $S1 --spec ${SPEC:-2} --prefill auto --pcie-frac 0 --suffix-draft 0
        --adapt-every 1000000 --pool-workers $POOL --max-context 8192
        --expert-profile $SRC/data/expert-profile-coder.bin --expert-cache 1)

note() { echo "[$(date -Is)] $*" | tee -a $LOG; }
# logrun CMD...: log the exact command line (shell-quoted) then run it
logrun() { { printf '[%s] cwd=%s\n  ' "$(date -Is)" "$PWD"; printf '%q ' "$@"; printf '\n'; } >> $LOG; "$@"; }

step_prompts() {
  cd $SRC/tools
  logrun $PY - $S1 $OUT/prompts $SRC/README.md <<'EOF'
import sys, pathlib
from strata_tokenizer import Tokenizer
s1, out, readme = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
tk = Tokenizer.from_gguf(s1)
def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompts = {}
prompts["chat"] = chat("What is the capital of France, and what is it famous for? Answer in two sentences.")
prompts["code"] = '''import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(order=True)
class _Entry:
    """One scheduled job: ordered by deadline, then by insertion sequence."""
    deadline: float
    seq: int
    name: str = field(compare=False)
    cost: int = field(compare=False, default=1)


class DeadlineScheduler:
    """Earliest-deadline-first scheduler with a fixed capacity per tick.

    Jobs are added with a deadline and a cost.  Each call to `tick(budget)` runs the jobs with the earliest
    deadlines whose total cost fits in `budget`, and returns their names in execution order.  Jobs whose
    deadline has passed are dropped and reported by `expired()`.
    """

    def __init__(self) -> None:
        self._heap: List[_Entry] = []
        self._seq = 0
        self._now = 0.0
        self._expired: List[str] = []
        self._by_name: Dict[str, _Entry] = {}

    def add(self, name: str, deadline: float, cost: int = 1) -> None:
        if name in self._by_name:
            raise KeyError(f"job {name!r} already scheduled")
        entry = _Entry(deadline, self._seq, name, cost)
        self._seq += 1
        heapq.heappush(self._heap, entry)
        self._by_name[name] = entry

    def cancel(self, name: str) -> bool:
        """Remove a pending job; returns False if it was not scheduled."""
'''
# the document: README.md cut at a paragraph boundary so the whole chat prompt is <= 2000 tokens
text = pathlib.Path(readme).read_text()
paras = text.split("\n\n")
doc = ""
for p in paras:
    cand = doc + ("\n\n" if doc else "") + p
    if len(tk.encode(chat("Summarize the following document in three bullet points.\n\n" + cand), parse_special=True)) > 2000:
        break
    doc = cand
prompts["doc"] = chat("Summarize the following document in three bullet points.\n\n" + doc)
for name, s in prompts.items():
    ids = tk.encode(s, parse_special=True)
    assert tk.decode(ids) == s, name
    (out / f"{name}.txt").write_text(s)
    (out / f"{name}.ids").write_text(",".join(map(str, ids)) + "\n")
    print(f"{name}: {len(ids)} tokens")
EOF
}

step_wait() {
  local f
  for f in $W/models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf; do
    while [ -e $f.done ]; do
      local need have
      need=$(( ($(stat -c %s $f) + (64<<20) - 1) / (64<<20) ))
      have=$(sort -u $f.done | wc -l)
      [ "$have" -ge "$need" ] && break
      sleep 60
    done
    note "complete: $f"
  done
}

step_pack() {
  mkdir -p $(dirname $PACK)
  cd $W
  logrun $PY $SRC/tools/iq_pack.py --gguf $S1 --out $PACK 2>&1 | tee $OUT/pack.log
  test -s $PACK/native_experts.txt
}

step_oneshot() {
  local name=$1 D=$OUT/$1
  mkdir -p $D
  rm -f $D/first_logits.f32 $D/routing.bin $D/windows.txt
  cd $W
  logrun env STRATA_IQ_MT_MIN=1 STRATA_DUMP_FIRST_LOGITS=$D/first_logits.f32 \
    $STRATA "${COMMON[@]}" --tokens-file $OUT/prompts/$name.ids --max-new $MAXNEW --greedy \
    --dump-routing $D/routing.bin --window-hashes $D/windows.txt --stats > $D/oneshot.log 2>&1
  # "output  : id id ..." (generate.cpp:10944)
  grep -a '^output  :' $D/oneshot.log | sed 's/^output  ://' | tr -s ' ' '\n' | grep . | paste -sd, > $D/oneshot_gen.ids
  note "oneshot $name: $(tr ',' '\n' < $D/oneshot_gen.ids | wc -l) tokens, first_logits $(stat -c %s $D/first_logits.f32) B"
}

# serve_gen MAXNEW IDS_FILE LOGPOS_FILE OUT_IDS LOGFILE: one GEN request through a fresh --serve engine
serve_gen() {
  local max_new=$1 ids=$2 lp=$3 out_ids=$4 elog=$5
  rm -f $lp
  cd $W
  logrun env STRATA_IQ_MT_MIN=1 STRATA_LOGPOS=$lp STRATA_LOGPOS_TOPK=256 \
    $PY - $max_new $ids $out_ids $elog -- $STRATA --serve "${COMMON[@]}" --short-read 100000 --prompt-cache 0 <<'EOF'
import subprocess, sys, time
max_new, ids_path, out_ids, elog = sys.argv[1:5]
argv = sys.argv[sys.argv.index("--") + 1:]
ids = open(ids_path).read().strip()
with open(elog, "w") as err:
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
    lines = []
    def readline():
        l = p.stdout.readline()
        if not l:
            raise SystemExit(f"engine exited (rc={p.wait()}); see {elog}")
        lines.append(l.rstrip("\n"))
        return lines[-1]
    while not readline().startswith("READY"):
        pass
    t0 = time.time()
    p.stdin.write(f"GEN {max_new} {ids}\n"); p.stdin.flush()
    toks = []
    while True:
        l = readline()
        if l.startswith("T "):
            toks.append(int(l.split()[1]))
        elif l.startswith("DONE") or l.startswith("ERR"):
            break
    p.stdin.write("QUIT\n"); p.stdin.flush()
    p.wait(timeout=300)
open(out_ids, "w").write(",".join(map(str, toks)) + "\n")
open(elog, "a").write("\n# stdout\n" + "\n".join(lines) + "\n")
print(f"{len(toks)} tokens in {time.time() - t0:.1f} s; last line: {lines[-1]}")
if not lines[-1].startswith("DONE"):
    raise SystemExit(1)
EOF
}

step_serve() {
  local name=$1 D=$OUT/$1
  mkdir -p $D
  serve_gen $MAXNEW $OUT/prompts/$name.ids $D/logpos.tsv $D/serve_gen.ids $D/serve.log
  note "serve $name: $(wc -l < $D/logpos.tsv) logpos rows, gen $(cat $D/serve_gen.ids)"
}

step_full() {
  local name=$1 D=$OUT/$1
  paste -d, <(tr -d '\n' < $OUT/prompts/$name.ids) <(tr -d '\n' < $D/serve_gen.ids) > $D/full.ids
  serve_gen 1 $D/full.ids $D/logpos_full.tsv $D/full_gen.ids $D/full.log
  note "full $name: $(wc -l < $D/logpos_full.tsv) logpos rows"
}

step=${1:?usage: golden.sh prompts|wait|pack|oneshot NAME..|serve NAME..|full NAME..|all}
shift || true
case $step in
  prompts) step_prompts ;;
  wait) step_wait ;;
  pack) step_pack ;;
  oneshot|serve|full) for n in "$@"; do step_$step $n; done ;;
  all)
    step_prompts; step_wait; step_pack
    for n in chat code doc; do step_oneshot $n; step_serve $n; step_full $n; done ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
