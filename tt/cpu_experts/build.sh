#!/bin/bash
# tt/cpu_experts/build.sh   Build libfnx_cpu.so (fnx.cpu_experts: Strata's CPU expert kernels behind a C ABI) in the
# TT image FNX_IMAGE (fnx.paths.DEFAULT_IMAGE), into tt/cpu_experts/build-tt/ (fnx.paths.CPU_LIB's default).
#
# Every engine start loads it (the CPU expert path FNX_MOE=cpu, and the CpuExperts the engine always builds), so a
# fresh install builds it once.  The image's gcc 12 / glibc 2.35 build loads in the engine container and on any
# newer host alike; ggml-cpu is compiled -march=native (GGML_NATIVE), so build on the machine that serves.
# Sources: this repository's include/ and src/ (STRATA_SRC = the repository root) and ggml from STRATA_GGML_DIR
# (default: <repo>/third_party/llama.cpp, else FNX_DATA/third_party/llama.cpp, when one holds ggml/CMakeLists.txt;
# else empty: Strata's CMake fetches its pinned llama.cpp commit, third_party/ggml/VERSION.txt).  No device, no lock.
set -euo pipefail
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
REPO=$(dirname "$TTDIR")
eval "$(python3 "$TTDIR/fnx/paths.py" --shell)"     # DATA IMAGE ...
GG=${STRATA_GGML_DIR:-}
if [ -z "$GG" ]; then
  for d in "$REPO/third_party/llama.cpp" "$DATA/third_party/llama.cpp"; do
    if [ -f "$d/ggml/CMakeLists.txt" ]; then GG=$d; break; fi
  done
fi
B=$TTDIR/cpu_experts/build-tt
VOLS=(-v "$REPO:$REPO")
[ -n "$GG" ] && VOLS+=(-v "$GG:$GG")
echo "cpu_experts/build.sh: $IMAGE, STRATA_SRC $REPO, STRATA_GGML_DIR ${GG:-(fetch the pinned commit)} -> $B" >&2
docker run --rm "${VOLS[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -w "$TTDIR/cpu_experts" --entrypoint bash \
  "$IMAGE" -c 'cmake -S . -B "$1" -G Ninja ${2:+"-DSTRATA_GGML_DIR=$2"} && cmake --build "$1" --target fnx_cpu' \
  _ "$B" "$GG"
ls -l "$B/libfnx_cpu.so"
