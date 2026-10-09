#!/bin/bash
# validate_all.sh - every test of the TT port on this tree, in order; one summary line per test.
# Device tests go through devrun.sh (device lock); test_serve runs on the host (its engine takes the lock itself),
# with PY (default FNX_DATA/venv/bin/python, the dev host's venv).  Log: FNX_DATA/logs/validate/<stamp>/
set -u
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
eval "$(python3 "$TTDIR/fnx/paths.py" --shell)"     # DATA LOGS ...
PY=${PY:-$DATA/venv/bin/python}
cd "$TTDIR"
OUT=$LOGS/validate/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
run() {   # name, then the devrun.sh / host command
  local name=$1; shift
  local t0=$SECONDS
  "$@" > "$OUT/$name.out" 2>&1
  local rc=$?
  printf '%-14s rc=%d  %4ds  %s\n' "$name" "$rc" $((SECONDS - t0)) \
    "$(grep -E 'PASS|FAIL' "$OUT/$name.out" | tail -1 | cut -c1-120)" | tee -a "$OUT/summary.txt"
}
run cpu_experts env NODEV=1 tools/devrun.sh v-cpu -m fnx.tests.test_cpu_experts
run hc          tools/devrun.sh v-hc   -m fnx.tests.test_hc
run ple         tools/devrun.sh v-ple  -m fnx.tests.test_ple
run head        tools/devrun.sh v-head -m fnx.tests.test_head
run gdn         tools/devrun.sh v-gdn  fnx/tests/test_gdn.py
run qsa         tools/devrun.sh v-qsa  fnx/tests/test_qsa.py tt sparse
run moe         tools/devrun.sh v-moe  fnx/tests/test_moe.py
run mtp         tools/devrun.sh v-mtp  fnx/tests/test_mtp.py tt
run model       tools/devrun.sh v-model -m fnx.tests.test_model
run spec        tools/devrun.sh v-spec fnx/tests/test_spec.py qsa inv spec
run trace       tools/devrun.sh v-trace fnx/tests/test_trace.py plain spec sparse bench
run serve       "$PY" -m fnx.tests.test_serve
echo "summary: $OUT/summary.txt"
