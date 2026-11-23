"""serve/server.py for a Tenstorrent P150: the same server, with the P150 pieces added from outside.

    python tt/serve/server_tt.py --engine strata --config strata-coder-iq1_m.json --port 8080 [...]

Every argument is serve/server.py's; the run script setup writes for a P150 (tt/setup_tt.py) starts this.  Nothing in
serve/ is edited (the TT engine keeps out of the shared files so upstream merges stay clean); this wrapper adds, at
run time:

  - the Monitor tab's card readings from the P150 (tt/serve/tt_telemetry.py: temperature, power, AI clock, link; the
    card reports no load or memory use), also on a PC whose NVIDIA card NVML would otherwise show;
  - no free-VRAM gate (min_free_vram_mib): the P150's memory is not readable from the host, and the shared device
    lock is what keeps two jobs off the card;
  - the start narration for the TT engine (tt/serve/tt-engine.sh): the wait for the shared device lock, the model
    built on the card, the warm-up - instead of the CUDA engine's "tens of GB into RAM" warning.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import serve.server as S  # noqa: E402
import serve.telemetry as T  # noqa: E402
from tt_telemetry import TtCard  # noqa: E402


def install_tt_reader():
    """telemetry.gpu_reader -> the P150's readings when the card is there (the model runs on it, not on a GPU NVML
    may also list)."""
    orig = T.gpu_reader

    def gpu_reader(index=0, amd=False):
        card = TtCard(index)
        return card if card.ok() else orig(index, amd)

    T.gpu_reader = gpu_reader


def install_no_vram_gate():
    """Service.free_vram_mib -> None for the TT engine: nothing is refused on another card's free VRAM."""
    orig = S.Service.free_vram_mib

    def free_vram_mib(self):
        return None if getattr(self, "backend", None) == "tt" else orig(self)

    S.Service.free_vram_mib = free_vram_mib


def start_phase(line: str):
    """(key, what the server window says) for an engine log line that starts a phase of the TT engine's start, else
    None.  The lines are tt-engine.sh's (the lock, the container) and fnx.serve_engine's (the model, the warm-up)."""
    if "waiting for the device lock" in line:
        return "lock", ("[strata] the P150 is in use by another job: waiting for the shared device lock; the model "
                        "loads as soon as the card is free ...")
    if "waiting for the device's current user" in line:
        return "user", "[strata] waiting for the card's current user to exit ..."
    if "device lock held" in line:
        return "held", ("[strata] device lock held: building the model on the P150 (the experts go into the card's "
                        "memory and a pinned host pool; 1-2 minutes) ...")
    if "TtModel built in" in line:
        return "built", "[strata] model built on the card (" + line.split("TtModel built in", 1)[1].split(":")[0].strip() + \
            "): warming up ..."
    if "warm-up in" in line:
        return "warm", "[strata] warm-up done: capturing the decode traces ..."
    if "decode traced" in line:
        return "traced", "[strata] almost ready ..."
    if "no READY within" in line:
        return "timeout", ("[strata] the engine did not get ready in time (FNX_READY_S after it held the lock): it was "
                           "stopped and the card reset; the next request starts it again")
    return None


def install_narrator():
    """The start narrator of serve/server.py says the CUDA engine's thing - tens of GB read into RAM, "YOUR PC CAN BE
    SLOW OR STOP RESPONDING" - at the experts' step.  The TT engine first waits for the shared device lock (possibly
    long: another team's job), then builds the model on the card: say those phases instead."""

    def narrate_start(log_path, offset, args, done, heartbeat=20.0):
        t0 = last = time.time()
        said = set()
        waiting = False

        def say(key, text):
            nonlocal last
            if key not in said:
                said.add(key)
                last = time.time()
                print(text, flush=True)

        say("start", "[strata] starting the TT engine (Docker, tt-metal) ...")
        pos = offset
        while not done.wait(0.5):
            try:
                with open(log_path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
            except OSError:
                chunk = b""
            if chunk.count(b"\n"):
                cut = chunk.rfind(b"\n") + 1
                pos += cut
                for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                    phase = start_phase(line)
                    if phase:
                        waiting = phase[0] in ("lock", "user")
                        say(*phase)
            if time.time() - last > heartbeat:
                last = time.time()
                print(f"[strata] still {'waiting for the card' if waiting else 'starting'} ({time.time() - t0:.0f} s) "
                      "- please wait ...", flush=True)
    S.narrate_start = narrate_start


if __name__ == "__main__":
    install_tt_reader()
    install_no_vram_gate()
    install_narrator()
    sys.exit(S.main())
