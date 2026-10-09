#!/bin/bash
# devrun.sh NAME script.py [args...]   Run a Python script on the P150 in the qwen36 TT image.
#
# Holds the shared device lock (/var/tmp/gemma4-12b/device.lock, used by every team on this card) for the whole run
# and first waits for any device user that bypassed it.  /var/tmp/strata-tt is mounted at the same path, and
# PYTHONPATH has tt-metal and flashnext_tt (import fnx).  Log: /var/tmp/strata-tt/logs/dev/NAME.log
# Env: IMAGE (default qwen36-p150:vllm-mtp-pc-x4), FNX_* and STRATA_* variables are passed through.
# NODEV=1 runs without the device (no lock): for conversion and CPU-only work in the image.
# FNX_DEV_TIMEOUT=seconds (default 7200): watchdog. A run still alive then is treated as hung on the device: its
# container is removed and, for device runs, the board is reset with tt-smi -r 0 WHILE THE LOCK IS STILL HELD (a hung
# kernel otherwise leaves the shared card wedged for the next lock holder); rc 124.
set -uo pipefail
if [ -z "${NODEV:-}" ] && [ -z "${FNX_DEVICE_LOCKED:-}" ]; then
  exec env FNX_DEVICE_LOCKED=1 flock /var/tmp/gemma4-12b/device.lock "$0" "$@"
fi
if [ -z "${NODEV:-}" ]; then
  while grep -q . /proc/driver/tenstorrent/0/pids 2>/dev/null; do sleep 10; done
fi
NAME=$1; shift
W=/var/tmp/strata-tt
IMAGE=${IMAGE:-qwen36-p150:vllm-mtp-pc-x4}
TT=/home/container_app_user/tt-metal
mkdir -p $W/logs/dev $W/kcache
CN=fnx-$(printf '%s' "$NAME" | tr -c 'a-zA-Z0-9_.-' '_')-$$     # docker names allow [a-zA-Z0-9_.-] only (e.g. no '+')
DEV=(--device /dev/tenstorrent:/dev/tenstorrent -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G)
[ -n "${NODEV:-}" ] && DEV=()
docker run -d --name $CN "${DEV[@]}" -v $W:$W --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -w $W/flashnext_tt -e PYTHONPATH=$TT:$W/flashnext_tt -e TT_METAL_CACHE=$W/kcache -e TT_METAL_HOME=$TT \
  $(env | grep -E '^(FNX_|STRATA_|TT_METAL_LOGGER|TT_METAL_MOCK_CLUSTER_DESC_PATH|TTNN_)' | sed 's/^/-e /') \
  --entrypoint $TT/python_env/bin/python $IMAGE "$@" >/dev/null || { echo "devrun: docker run failed for $CN" >&2; exit 1; }
TMO=${FNX_DEV_TIMEOUT:-7200}
if RC=$(timeout "$TMO" docker wait $CN); then
  docker logs $CN > $W/logs/dev/$NAME.log 2>&1
  docker rm $CN >/dev/null
else
  docker logs $CN > $W/logs/dev/$NAME.log 2>&1
  echo "devrun: $NAME still running after ${TMO}s (FNX_DEV_TIMEOUT): removing $CN" | tee -a $W/logs/dev/$NAME.log >&2
  docker rm -f $CN >/dev/null
  if [ -z "${NODEV:-}" ]; then
    echo "devrun: resetting the board under the device lock" | tee -a $W/logs/dev/$NAME.log >&2
    "${TT_SMI:-$HOME/.local/bin/tt-smi}" -r 0 >> $W/logs/dev/$NAME.log 2>&1 || echo "devrun: board reset FAILED" >&2
  fi
  RC=124
fi
grep -v -E 'Werror|pragma|DO_PRAGMA| info  *\|' $W/logs/dev/$NAME.log | tail -${TAILN:-40}
echo "rc=$RC log=$W/logs/dev/$NAME.log"
exit $RC
