#!/bin/bash
# fnx-serve.sh [server args...]   Strata's own server (serve/server.py, unchanged) answered by the P150.
#
# Runs `python serve/server.py --engine strata --config $CONFIG --port $PORT` with the host venv from the serving
# tree STRATA_SERVE (default /var/tmp/strata-tt/Strata-serve: a git worktree of Strata at the tag in STRATA_SERVE_TAG
# below; the main checkout /var/tmp/strata-tt/Strata stays at the port's pinned 82f46a8 for the reference engine,
# the CPU expert sources, tools and the eval corpus).  The config's engine is tools/fnx-engine (Strata's engine
# protocol, fnx.serve_engine in the TT image, holding the shared device lock for as long as the model is loaded).
# OpenAI API: http://127.0.0.1:$PORT/v1/chat/completions, Anthropic: /v1/messages, web chat: http://127.0.0.1:$PORT/.
# The config unloads the model after "idle_unload_s" (900 s) without requests so the shared card is given back; the
# next request loads it again (lock wait + ~2-4 min).  Ctrl+C / SIGTERM: the server sends QUIT and the engine
# releases the device.
# READY timeout: Strata >= 0.1.40.3 ends an engine that has not printed READY within STRATA_ENGINE_READY_S (900 s;
# #1317) with SIGKILL.  Our engine's start includes an unbounded wait for the shared device lock, and a SIGKILL skips
# tools/fnx-engine's container cleanup, so the server waits for ever here (STRATA_ENGINE_READY_S=0, the pre-#1317
# behaviour) and tools/fnx-engine bounds the start itself once it holds the lock (FNX_READY_S, default 900 s:
# container removed, board reset, the request fails and the next one starts it again).
# Env: PORT (default 8090), CONFIG (default /var/tmp/strata-tt/strata-fnx-coder.json), STRATA_SERVE,
# STRATA_ENGINE_READY_S, FNX_READY_S; engine log: the config's "log".
set -euo pipefail
W=/var/tmp/strata-tt
STRATA_SERVE_TAG=v0.1.40.4
STRATA_SERVE=${STRATA_SERVE:-$W/Strata-serve}
PORT=${PORT:-8090}
CONFIG=${CONFIG:-$W/strata-fnx-coder.json}
export STRATA_ENGINE_READY_S=${STRATA_ENGINE_READY_S:-0}
if [ ! -d "$STRATA_SERVE/serve" ]; then
  git -C $W/Strata fetch --quiet --tags origin
  git -C $W/Strata worktree add --detach "$STRATA_SERVE" "$STRATA_SERVE_TAG" >&2
fi
mkdir -p $W/logs
cd "$STRATA_SERVE"
exec $W/venv/bin/python serve/server.py --engine strata --config "$CONFIG" --port "$PORT" "$@"
