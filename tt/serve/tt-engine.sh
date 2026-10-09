#!/bin/bash
# tt-engine.sh --serve [Strata engine args...]   Strata's engine protocol on the Tenstorrent P150 (fnx.serve_engine).
#
# The "exe" of a Strata run config with backend "tt" (setup.py --backend tt writes it; serve/server.py spawns it):
# stdin/stdout are the protocol pipe, stderr is the server's engine log.  Takes the shared device lock FNX_DEVICE_LOCK
# (default /var/tmp/gemma4-12b/device.lock: every team on this card uses it) and waits for any device user that
# bypassed it, then becomes `docker run -i --rm --init` of the TT image FNX_IMAGE (default fnx.paths.DEFAULT_IMAGE,
# pinned by digest): this process IS the docker client, run as this script's child; the script holds the lock's file
# descriptor until it exits.  QUIT or EOF on stdin ends the engine, which closes the device; the container ends and
# the lock is released.  The container is named tt-engine-<this pid> (printed on stderr).
# Mounts, each at its own path (no path translation: setup's host paths work as they are): this repository (cwd
# <repo>/tt, PYTHONPATH tt-metal + <repo>/tt), FNX_DATA (fnx.paths: the expert store, the drafter's BFP4 experts,
# caches, kcache = TT_METAL_CACHE, logs) and its override directories, and the parent directory of every absolute
# path argument (--native, --ple-gguf, --pack, --mtp).  The container runs as the caller's uid:gid.
# Signals: SIGTERM/SIGINT/SIGHUP (the server's close() after 20 s) are forwarded into the container (tini -> python
# closes the device and exits); if it is still there 20 s later (an engine hung in device code never reads a signal
# or EOF), it is removed with `docker rm -f` and the board is reset (tt-smi -r 0, still under the lock), so a hang
# never keeps the shared card.  Any other exit of this script removes a leftover container the same way.
# Start bound: from the moment the lock is held, the engine has FNX_READY_S seconds (default 900; 0 = none) to print
# READY (it touches FNX_READY_FILE then); otherwise this script ends it as on SIGTERM (container removed, board
# reset).  The lock wait itself is not bounded: run the server with STRATA_ENGINE_READY_S=0 (its #1317 READY timeout
# SIGKILLs this script, which skips the cleanup above).
# Env: FNX_DATA, FNX_IMAGE, FNX_DEVICE_LOCK, FNX_READY_S, TT_SMI (default: tt-smi on PATH, else ~/.local/bin/tt-smi);
# FNX_*, STRATA_*, TT_METAL_LOGGER* and TTNN_* pass into the container.
# Engine version: the engine prints INFO engine=<CMakeLists.txt project version>; setup writes BUILD.json beside this
# script with the same number.
set -uo pipefail
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
P=$(python3 "$TTDIR/fnx/paths.py" --shell "$@") || { echo "tt-engine: $TTDIR/fnx/paths.py --shell failed" >&2; exit 1; }
eval "$P"                                             # DATA KCACHE LOGS IMAGE LOCK TT_METAL CPU_LIB MOUNTS
if [ ! -e "$CPU_LIB" ]; then
  echo "tt-engine: $CPU_LIB is missing: build Strata's CPU expert kernels with $TTDIR/cpu_experts/build.sh" >&2
  exit 1
fi
TT_SMI=${TT_SMI:-$(command -v tt-smi || echo "$HOME/.local/bin/tt-smi")}
mkdir -p "$(dirname "$LOCK")"
exec 9<>"$LOCK" || { echo "tt-engine: cannot open the device lock $LOCK" >&2; exit 1; }
if ! flock -n 9; then
  echo "tt-engine: waiting for the device lock $LOCK (another device job is running)" >&2
  flock 9
fi
waited=0
while grep -q . /proc/driver/tenstorrent/0/pids 2>/dev/null; do
  [ $waited = 0 ] && echo "tt-engine: waiting for the device's current user to exit" >&2
  waited=1
  sleep 2
done
echo "tt-engine: device lock held by pid $$; starting the engine in $IMAGE (FNX_DATA $DATA)" >&2
NAME="tt-engine-$$"
READY_FILE=$LOGS/$NAME.ready                        # touched by fnx.serve_engine when it prints READY
rm -f "$READY_FILE"
export FNX_READY_FILE=$READY_FILE
ENVS=(-e "FNX_DATA=$DATA")
while IFS= read -r kv; do ENVS+=(-e "$kv"); done < <(env | grep -E '^(FNX_|STRATA_|TT_METAL_LOGGER|TTNN_)')
VOLS=()
for d in "${MOUNTS[@]}"; do VOLS+=(-v "$d:$d"); done
for d in /dev/hugepages /dev/hugepages-1G; do [ -d "$d" ] && VOLS+=(-v "$d:$d"); done
echo "tt-engine: container $NAME; mounts ${MOUNTS[*]}" >&2
WATCH=
if [ "${FNX_READY_S:-900}" != 0 ]; then
  (  # only stderr (the engine log) kept: no lock fd (a sleeping watchdog never holds the lock) and no protocol pipe
     # (an open stdout would keep the server from ever seeing the engine's EOF); its sleep dies with it
    exec 9>&- 1>/dev/null 0</dev/null
    sleep "${FNX_READY_S:-900}" &
    SP=$!
    trap 'kill $SP 2>/dev/null; exit 0' TERM
    wait $SP
    if [ ! -e "$READY_FILE" ]; then
      echo "tt-engine: no READY within ${FNX_READY_S:-900} s of holding the device lock: ending the engine" >&2
      kill -TERM $$
    fi
  ) &
  WATCH=$!
fi
alive() { [ -n "$(docker ps -q --filter "name=^${NAME}$")" ]; }
force_remove() {     # the engine did not end by itself: remove it and reset the board before releasing the lock
  if alive; then
    echo "tt-engine: removing the unresponsive container $NAME and resetting the board" >&2
    docker rm -f "$NAME" >/dev/null 2>&1
    "$TT_SMI" -r 0 >&2 2>&1 || echo "tt-engine: board reset failed" >&2
  fi
}
on_signal() {
  trap - TERM INT HUP
  docker kill --signal TERM "$NAME" >/dev/null 2>&1
  for _ in $(seq 20); do alive || exit 143; sleep 1; done
  force_remove
  exit 143
}
cleanup() {
  [ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
  rm -f "$READY_FILE"
  force_remove
}
trap cleanup EXIT
trap on_signal TERM INT HUP
docker run -i --rm --init --name "$NAME" --device /dev/tenstorrent:/dev/tenstorrent "${VOLS[@]}" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -w "$TTDIR" \
  -e PYTHONPATH="$TT_METAL:$TTDIR" -e TT_METAL_CACHE="$KCACHE" -e TT_METAL_HOME="$TT_METAL" -e PYTHONUNBUFFERED=1 \
  "${ENVS[@]}" --entrypoint "$TT_METAL/python_env/bin/python" "$IMAGE" -m fnx.serve_engine "$@" <&0 &
CHILD=$!
wait "$CHILD"
RC=$?
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; RC=$?; done   # a signal interrupted wait: the trap ran
exit $RC
