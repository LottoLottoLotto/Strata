"""Read Qwen3.8-Flash-Next GGUF tensors (all split shards) as float32 numpy arrays.

Shapes are numpy order (GGUF's ne reversed): a GGUF weight with ne = [n_in, n_out] comes back as (n_out, n_in),
an expert stack ne = [n_in, n_out, n_expert] as (n_expert, n_out, n_in).

Dequantization: gguf-py (llama.cpp's: fnx.paths.GGUF_PY when it exists, else the installed `gguf`) for every ggml
type it knows, and a vectorized Q2_0 (type 42, Strata's GSQ-RCO routed-expert encoding, absent from gguf-py): 64
values per 18-byte block, fp16 scale d, 2-bit codes LSB-first, value = (code - 1) * d - Strata's
tools/gguf_writer.py::dequantize_q2_0.
"""
from __future__ import annotations

import mmap
import re
import sys
from pathlib import Path

import numpy as np

from . import paths

STRATA_TOOLS = paths.STRATA_TOOLS
sys.path.insert(0, str(STRATA_TOOLS))
sys.path.insert(0, str(paths.GGUF_PY))

from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402  Strata's header reader (knows type 42)
import gguf.quants as gq  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def dequant_q2_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)          # (n, 1)
    qs = blocks[:, 2:]                                                     # (n, 16)
    codes = (qs[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3  # (n, 16, 4): value j = byte j//4, bits 2*(j%4)
    return ((codes.reshape(-1, 64).astype(np.float32) - 1.0) * d).reshape(-1)


class Model:
    """Every tensor of a split GGUF, by name: the shards named like `first_shard` (-00001-of-0000N), or exactly
    `first_shard` plus `others` when given (the engine's --ple-gguf)."""

    def __init__(self, first_shard: str | Path, others=None):
        first = Path(first_shard)
        m = SHARD.search(first.name)
        if others:
            files = [first] + [p for p in map(Path, others) if p != first]
        else:
            files = [first] if not m else [
                first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, int(m.group(2))))
                for i in range(1, int(m.group(2)) + 1)]
        self.files, self.maps, self.index = [], [], {}
        for p in files:
            g = GGUFFile(p)
            fh = open(p, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.files.append(g)
            self.maps.append((fh, mm))
            for t in g.tensors:
                if t.name in self.index:
                    raise ValueError(f"duplicate tensor {t.name}")
                self.index[t.name] = (len(self.files) - 1, t)
        self.metadata = self.files[0].metadata

    def info(self, name: str):
        return self.index[name][1]

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def raw(self, name: str) -> np.ndarray:
        fi, t = self.index[name]
        nbytes = t.expected_bytes()
        if nbytes is None:
            raise ValueError(f"{name}: unknown size for type {t.type_name}")
        off = self.files[fi].data_start + t.offset
        return np.frombuffer(self.maps[fi][1], dtype=np.uint8, count=nbytes, offset=off)

    def raw_expert(self, name: str, expert: int) -> np.ndarray:
        """One expert's bytes of an expert stack (ne = [n_in, n_out, n_expert]): experts are the slowest axis."""
        t = self.info(name)
        per = t.expected_bytes() // t.shape[2]
        return self.raw(name)[expert * per:(expert + 1) * per]

    def _dequant(self, type_name: str, raw: np.ndarray, n: int) -> np.ndarray:
        if type_name == "Q2_0":
            out = dequant_q2_0(raw)
        elif type_name == "F32":
            out = raw.view(np.float32).astype(np.float32)
        elif type_name == "F16":
            out = raw.view(np.float16).astype(np.float32)
        elif type_name == "BF16":
            out = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            out = np.asarray(gq.dequantize(raw, GGMLQuantizationType[type_name]), dtype=np.float32).reshape(-1)
        if out.size != n:
            raise ValueError(f"dequant produced {out.size} values, expected {n}")
        return out

    def available(self, name: str, expert: int | None = None) -> bool:
        """Whether the bytes of `name` (or of one expert of it) are on disk yet: a file still being fetched by
        tools/pdl.py has a `<file>.done` sidecar listing finished 64 MiB chunks; without one the file is complete."""
        fi, t = self.index[name]
        path = Path(self.files[fi].path)
        done_path = Path(str(path) + ".done")
        if not done_path.exists():
            return True
        lo = self.files[fi].data_start + t.offset
        n = t.expected_bytes()
        if expert is not None:
            per = n // t.shape[2]
            lo, n = lo + expert * per, per
        chunk = 64 << 20
        size = path.stat().st_size
        done = {int(x) for x in done_path.read_text().split()}
        if len(done) >= (size + chunk - 1) // chunk:
            return True
        return all(c in done for c in range(lo // chunk, (lo + n - 1) // chunk + 1))

    def get(self, name: str) -> np.ndarray:
        t = self.info(name)
        return self._dequant(t.type_name, self.raw(name), t.elements).reshape(tuple(reversed(t.shape)))

    def get_expert(self, name: str, expert: int) -> np.ndarray:
        """(n_out, n_in) float32 of one expert."""
        t = self.info(name)
        n_in, n_out = t.shape[0], t.shape[1]
        return self._dequant(t.type_name, self.raw_expert(name, expert), n_in * n_out).reshape(n_out, n_in)

    def type_of(self, name: str) -> str:
        return self.info(name).type_name


def selftest() -> None:
    # Q2_0 against Strata's scalar reference
    sys.path.insert(0, str(STRATA_TOOLS))
    from gguf_writer import dequantize_q2_0, quantize_q2_0
    w = np.random.default_rng(0).standard_normal(64 * 37).astype(np.float32)
    raw = quantize_q2_0(w)
    ref = dequantize_q2_0(raw)
    got = dequant_q2_0(np.frombuffer(raw, dtype=np.uint8))
    assert np.array_equal(ref, got), "Q2_0 vectorized dequant differs from the scalar reference"
    assert BLOCK_GEOMETRY["Q2_0"] == (64, 18)
    print("gguf_load selftest OK")


if __name__ == "__main__":
    selftest()
