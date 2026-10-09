"""Dequantized Flash-Next weights (float32 numpy, math orientation), cached on disk.

`Weights(model).get(name)` returns the tensor GGUF calls `name` (e.g. "blk.3.attn_q.weight") as float32 in numpy
order (GGUF ne reversed: a projection is (n_out, n_in)).  Quantized GGUF blocks are dequantized once (gguf-py is
slow) and kept as .npy under FNX_CACHE (default /var/tmp/strata-tt/cache/dequant/<shard1 stem>/); float tensors
are read straight from the mapping.  Routed experts (ffn_{gate,up,down}_exps) are NOT served here: they are 23 GiB;
use `model.get_expert` or fnx.cpu_experts.

Every value is exactly the GGUF's: no transforms (Strata applies none for the Coder: gammas are final multipliers,
ssm_a is already negative - spec §A.4).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .gguf_load import Model

DEFAULT_SHARD1 = "/var/tmp/strata-tt/models/coder/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
FLOAT = {"F32", "F16", "BF16"}


class Weights:
    def __init__(self, model: Model | str | None = None, cache_dir: str | None = None):
        self.model = model if isinstance(model, Model) else Model(model or os.environ.get("FNX_GGUF", DEFAULT_SHARD1))
        stem = Path(self.model.files[0].path).name.removesuffix(".gguf")
        self.cache = Path(cache_dir or os.environ.get("FNX_CACHE", "/var/tmp/strata-tt/cache")) / "dequant" / stem
        self.cache.mkdir(parents=True, exist_ok=True)

    def __contains__(self, name: str) -> bool:
        return name in self.model

    def type_of(self, name: str) -> str:
        return self.model.type_of(name)

    def get(self, name: str) -> np.ndarray:
        if "_exps." in name:
            raise ValueError(f"{name}: routed experts are not served by Weights (use model.get_expert / cpu_experts)")
        if not self.model.available(name):
            # a file still downloading has holes that read as zeros: never cache (or return) them
            raise ValueError(f"{name}: its bytes are not downloaded yet")
        if self.model.type_of(name) in FLOAT:
            return self.model.get(name)
        p = self.cache / (name + ".npy")
        if p.exists():
            return np.load(p, mmap_mode="r")
        a = self.model.get(name)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, a)
        os.replace(tmp, p)
        return np.load(p, mmap_mode="r")

    def layer(self, L: int, suffix: str) -> np.ndarray:
        return self.get(f"blk.{L}.{suffix}")
