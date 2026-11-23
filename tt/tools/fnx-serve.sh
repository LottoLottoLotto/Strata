#!/bin/bash
# fnx-serve.sh [server args...]   Development launcher: this repository's Strata server (serve/server.py) answered by
# the P150, for a run config written by hand or by a test (fnx/tests/test_serve.py writes one).  An install made with
# `setup.py --backend tt` starts through its own run-<tag>.sh instead (tt/serve/server_tt.py); this script is for the
# development machine's data layout and venv.
#
# Runs `$PY serve/server.py --engine strata --config $CONFIG --port $PORT` from the repository root.  The config's
# engine is tt/serve/tt-engine.sh (Strata's engine protocol, fnx.serve_engine in the TT image, holding the shared
# device lock for as long as the model is loaded).  OpenAI API: http://127.0.0.1:$PORT/v1/chat/completions,
# Anthropic: /v1/messages, web chat: http://127.0.0.1:$PORT/.  A config with "idle_unload_s" (e.g. 900) unloads the
# model after that many idle seconds so the shared card is given back; the next request loads it again (lock wait +
# ~2-4 min).  Ctrl+C / SIGTERM: the server sends QUIT and the engine releases the device.
# READY timeout: Strata >= 0.1.40.3 ends an engine that has not printed READY within STRATA_ENGINE_READY_S (900 s;
# #1317) with SIGKILL.  Our engine's start includes an unbounded wait for the shared device lock, and a SIGKILL skips
# tt-engine.sh's container cleanup, so the server waits for ever here (STRATA_ENGINE_READY_S=0, the pre-#1317
# behaviour) and tt-engine.sh bounds the start itself once it holds the lock (FNX_READY_S, default 900 s: container
# removed, board reset, the request fails and the next one starts it again).
# Env: CONFIG (required), PORT (default 8090), PY (default FNX_DATA/venv/bin/python: the dev host's venv with
# Strata's requirements), STRATA_ENGINE_READY_S, FNX_READY_S, FNX_DATA; engine log: the config's "log".
set -euo pipefail
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
REPO=$(dirname "$TTDIR")
eval "$(python3 "$TTDIR/fnx/paths.py" --shell)"     # DATA ...
CONFIG=${CONFIG:?fnx-serve.sh: set CONFIG to a Strata run config whose exe is tt/serve/tt-engine.sh}
PORT=${PORT:-8090}
PY=${PY:-$DATA/venv/bin/python}
export STRATA_ENGINE_READY_S=${STRATA_ENGINE_READY_S:-0}
cd "$REPO"
exec "$PY" serve/server.py --engine strata --config "$CONFIG" --port "$PORT" "$@"
