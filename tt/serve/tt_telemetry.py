"""Tenstorrent card readings for the server's Monitor tab, in the shape of serve/telemetry.py's readers.

Kept out of serve/telemetry.py so the TT engine touches no shared file: tt/serve/server_tt.py installs it as
telemetry.gpu_reader.  The tt-kmd driver's sysfs is read (no device access, so the reading never competes with a job
on the shared card): the card type and AI clock from /sys/class/tenstorrent/tenstorrent!N, the ASIC temperature and
the board power and its limit from the card's hwmon folder, the PCIe link from its PCI device.  On a driver without
hwmon, `tt-smi -s` (its JSON snapshot; it talks to the card) is asked at most every SMI_EVERY_S seconds.  The card
reports no load and no memory use: load stays empty, the memory total is the card's 32 GB.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

CARDS = {"p150": ("Tenstorrent Blackhole P150", 32), "p150a": ("Tenstorrent Blackhole P150a", 32),
         "p150b": ("Tenstorrent Blackhole P150b", 32), "p150c": ("Tenstorrent Blackhole P150c", 32)}
PCIE_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
SMI_EVERY_S = 15.0


def _rd(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _num(path, scale=1.0):
    v = _rd(path)
    try:
        return int(v) / scale if v is not None else None
    except ValueError:
        return None


class TtCard:
    """A Tenstorrent card's readings in NVML's shape (serve/telemetry.py: ok(), name(), read())."""

    def __init__(self, index=0, sysfs="/sys"):
        self.index = index
        self.cls = os.path.join(sysfs, "class", "tenstorrent", f"tenstorrent!{index}")
        self.dev = os.path.realpath(os.path.join(self.cls, "device")) if os.path.exists(
            os.path.join(self.cls, "device")) else None
        self.hwmon = None
        if self.dev and os.path.isdir(os.path.join(self.dev, "hwmon")):
            hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
            self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
        self.card = (_rd(os.path.join(self.cls, "tt_card_type")) or "").lower()
        self._smi = (0.0, None)                         # (when, the card's tt-smi entry)

    def ok(self):
        return sys.platform.startswith("linux") and os.path.isdir(self.cls)

    def name(self):
        card = self.card or self._smi_card()
        return CARDS.get(card, (f"Tenstorrent {card}" if card else "Tenstorrent card", 0))[0]

    def _smi_entry(self):
        """This card's entry of `tt-smi -s`, refreshed at most every SMI_EVERY_S seconds (None without tt-smi)."""
        t, entry = self._smi
        if time.time() - t < SMI_EVERY_S:
            return entry
        entry = None
        exe = shutil.which("tt-smi") or os.path.expanduser("~/.local/bin/tt-smi")
        if os.access(exe, os.X_OK):
            try:
                r = subprocess.run([exe, "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=30)
                text = r.stdout if r.returncode == 0 else ""
                infos = (json.loads(text[text.find("{"):]).get("device_info") or []) if "{" in text else []
                entry = infos[self.index] if self.index < len(infos) else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                entry = None
        self._smi = (time.time(), entry)
        return entry

    def _smi_card(self):
        e = self._smi_entry() if not self.hwmon else None
        return str(((e or {}).get("board_info") or {}).get("board_type") or "").lower().split(" ")[0]

    def _link(self):
        if not self.dev:
            return None, None, None
        cur = PCIE_GEN.get((_rd(os.path.join(self.dev, "current_link_speed")) or "").split(" ")[0])
        top = PCIE_GEN.get((_rd(os.path.join(self.dev, "max_link_speed")) or "").split(" ")[0])
        width = _num(os.path.join(self.dev, "current_link_width"))
        return cur, top, int(width) if width else None

    def read(self):
        card = self.card or self._smi_card()
        gb = CARDS.get(card, (None, 0))[1]
        out = {"util": None, "mem_used": None, "mem_total": gb * 2**30 if gb else None,
               "aiclk_mhz": _num(os.path.join(self.cls, "tt_aiclk"))}
        if self.hwmon:
            out["temp"] = _num(os.path.join(self.hwmon, "temp1_input"), 1000.0)
            out["power"] = _num(os.path.join(self.hwmon, "power1_input"), 1e6)
            out["power_limit"] = _num(os.path.join(self.hwmon, "power1_max"), 1e6)
        else:
            tel = (self._smi_entry() or {}).get("telemetry") or {}
            lim = (self._smi_entry() or {}).get("limits") or {}

            def f(d, k):
                try:
                    return float(str(d.get(k)).strip())
                except (TypeError, ValueError):
                    return None
            out["temp"], out["power"], out["power_limit"] = f(tel, "asic_temperature"), f(tel, "power"), f(lim, "tdp_limit")
            if out["aiclk_mhz"] is None:
                out["aiclk_mhz"] = f(tel, "aiclk")
        out["pcie_gen"], out["pcie_gen_max"], out["pcie_width"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None    # no PCIe traffic counters
        return out
