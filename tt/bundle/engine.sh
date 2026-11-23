#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
#
# tt/bundle/engine.sh --serve [Strata engine args...]   The "exe" of the tt-model bundle's run config (tt/bundle/serve.py
# writes it; serve/server.py spawns it): fnx.serve_engine in this container, which tt-model started with the card.
# What tt/serve/tt-engine.sh does on a PC minus the container around it: no docker, no device lock, no board reset
# (the container is the device job; `docker stop` reaches the server, whose QUIT / EOF ends the engine, which closes
# the device).  stdin/stdout are the protocol pipe, stderr is the server's engine log.
# FNX_PROTOCOL_LOG=<prefix>: a copy of the protocol lines, requests to <prefix>.in and replies to <prefix>.out (the
# prompt and output token ids of every GEN), for token-level comparisons.
set -euo pipefail
TTDIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)      # <repo>/tt
cd "$TTDIR"
export PYTHONPATH="$TTDIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}     # tt-engine.sh's default (numpy's host math vs the CPU experts)
if [ -n "${FNX_PROTOCOL_LOG:-}" ]; then
  exec < <(tee -a "$FNX_PROTOCOL_LOG.in") > >(tee -a "$FNX_PROTOCOL_LOG.out")
fi
exec python -m fnx.serve_engine "$@"
