#!/bin/bash
# devrun.sh NAME script.py [args...]   Run a Python script (or -m module) on the P150 in the TT image.
#
# Holds the shared device lock (FNX_DEVICE_LOCK, default /var/tmp/gemma4-12b/device.lock, used by every team on this
# card) for the whole run and first waits for any device user that bypassed it.  Mounted at their own paths: this
# repository (cwd <repo>/tt, PYTHONPATH tt-metal + <repo>/tt: import fnx), FNX_DATA (fnx.paths; default
# fnx.paths.DEV_DATA) with its override directories, and the parent of every absolute path argument.
# Log: FNX_DATA/logs/dev/NAME.log.
# Env: FNX_IMAGE (default fnx.paths.DEFAULT_IMAGE), FNX_DATA, FNX_DEVICE_LOCK; FNX_* and STRATA_* variables pass through.
# NODEV=1 runs without the device (no lock): for conversion and CPU-only work in the image.
# FNX_DEV_TIMEOUT=seconds (default 7200): watchdog. A run still alive then is treated as hung on the device: its
# container is removed and, for device runs, the board is reset with tt-smi -r 0 WHILE THE LOCK IS STILL HELD (a hung
# kernel otherwise leaves the shared card wedged for the next lock holder); rc 124.
set -uo pipefail
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
P=$(python3 "$TTDIR/fnx/paths.py" --shell "$@") || { echo "devrun: $TTDIR/fnx/paths.py --shell failed" >&2; exit 1; }
eval "$P"                                             # MODEL DATA KCACHE LOGS IMAGE LOCK TT_METAL CPU_LIB MOUNTS
export FNX_MODEL=$MODEL                               # FNX_MODEL, else the variant of a --native argument or FNX_GGUF
if [ -z "${NODEV:-}" ] && [ -z "${FNX_DEVICE_LOCKED:-}" ]; then
  mkdir -p "$(dirname "$LOCK")"
  exec env FNX_DEVICE_LOCKED=1 flock "$LOCK" "$0" "$@"
fi
if [ -z "${NODEV:-}" ]; then
  while grep -q . /proc/driver/tenstorrent/0/pids 2>/dev/null; do sleep 10; done
fi
NAME=$1; shift
TT_SMI=${TT_SMI:-$(command -v tt-smi || echo "$HOME/.local/bin/tt-smi")}
LOG=$LOGS/dev/$NAME.log
mkdir -p "$LOGS/dev"
CN=fnx-$(printf '%s' "$NAME" | tr -c 'a-zA-Z0-9_.-' '_')-$$     # docker names allow [a-zA-Z0-9_.-] only (e.g. no '+')
DEV=(--device /dev/tenstorrent:/dev/tenstorrent)
for d in /dev/hugepages /dev/hugepages-1G; do [ -d "$d" ] && DEV+=(-v "$d:$d"); done
[ -n "${NODEV:-}" ] && DEV=()
VOLS=()
for d in "${MOUNTS[@]}"; do VOLS+=(-v "$d:$d"); done
ENVS=(-e "FNX_DATA=$DATA")
while IFS= read -r kv; do ENVS+=(-e "$kv"); done < <(env | grep -E '^(FNX_|STRATA_|TT_METAL_LOGGER|TT_METAL_MOCK_CLUSTER_DESC_PATH|TTNN_)')
docker run -d --name "$CN" "${DEV[@]}" "${VOLS[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -w "$TTDIR" -e PYTHONPATH="$TT_METAL:$TTDIR" -e TT_METAL_CACHE="$KCACHE" -e TT_METAL_HOME="$TT_METAL" \
  "${ENVS[@]}" --entrypoint "$TT_METAL/python_env/bin/python" "$IMAGE" "$@" >/dev/null \
  || { echo "devrun: docker run failed for $CN" >&2; exit 1; }
TMO=${FNX_DEV_TIMEOUT:-7200}
if RC=$(timeout "$TMO" docker wait "$CN"); then
  docker logs "$CN" > "$LOG" 2>&1
  docker rm "$CN" >/dev/null
else
  docker logs "$CN" > "$LOG" 2>&1
  echo "devrun: $NAME still running after ${TMO}s (FNX_DEV_TIMEOUT): removing $CN" | tee -a "$LOG" >&2
  docker rm -f "$CN" >/dev/null
  if [ -z "${NODEV:-}" ]; then
    echo "devrun: resetting the board under the device lock" | tee -a "$LOG" >&2
    "$TT_SMI" -r 0 >> "$LOG" 2>&1 || echo "devrun: board reset FAILED" >&2
  fi
  RC=124
fi
grep -v -E 'Werror|pragma|DO_PRAGMA| info  *\|' "$LOG" | tail -"${TAILN:-40}"
echo "rc=$RC log=$LOG"
exit "$RC"
