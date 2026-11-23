"""Bit-exactness of fnx.quant.bfp (BFP2/BFP4/BFP8 emulator, store rows, tile bytes) against tt-metal itself.

ttnn has no bfloat2_b DataType, so the reference is tt-metal's own host packer (pack_as_bfp{2,4,8}_tiles /
unpack_..._into_float_vec, blockfloat_common.cpp) through a tiny C ABI shim (fnx/quant/ttbfp_shim.cpp, compiled here
with the image's g++ against libtt_metal.so). BFP4/BFP8 are additionally checked against ttnn.from_torch/to_torch.
No device: a mock cluster descriptor satisfies the HAL.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/third_party/tt-cluster-descriptors/\
blackhole/p100_cluster_desc/p100_cluster_desc.yaml NODEV=1 tools/devrun.sh qbfp fnx/tests/test_quant_bfp.py

Checks, per format and case (random wide-range, every bf16 pattern in [-1, 1], real Coder expert matrices, and
GPTQ-style snapped matrices whose blocks use exponents other than the RTN one):
  1. tile bytes: to_tiles(pack_codes(quantize_codes(x))) == tt-metal pack_as_bfp_tiles(tilize(x))   (0 bytes differ)
  2. values:     tt-metal unpack(pack(x)) == emulator rtn(x)                                         (0 bits differ)
  3. snapped:    for snapped s: tt-metal pack(s) == to_tiles(store.pack(s)), unpack(store row) == s == tt-metal unpack
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fnx import paths  # noqa: E402
from fnx.quant import bfp, methods  # noqa: E402

TT = Path(os.environ.get("TT_METAL_HOME", paths.TT_METAL))
SHIM_SRC = Path(__file__).resolve().parents[1] / "quant" / "ttbfp_shim.cpp"
SHIM_SO = paths.QUANT / "build" / "libttbfp.so"
FMT_ID = {"bfp2": 2, "bfp4": 4, "bfp8": 8}


def build_shim() -> ctypes.CDLL:
    if not SHIM_SO.exists() or SHIM_SO.stat().st_mtime < SHIM_SRC.stat().st_mtime:
        B = TT / "build_Release"
        inc = [TT / "tt_metal/api", TT / "tt_stl", B / "include", TT / "tt_metal/hostdevcommon/api", TT]
        inc += sorted(B.glob("_deps/*/include")) + sorted(TT.glob(".cpmcache/*/*/include"))
        SHIM_SO.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["g++", "-std=c++20", "-O2", "-fPIC", "-shared", *[f"-I{p}" for p in inc], str(SHIM_SRC), "-o",
               str(SHIM_SO), f"-L{B / 'lib'}", "-ltt_metal", "-ltt_stl", f"-Wl,-rpath,{B / 'lib'}"]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(SHIM_SO))
    lib.ttbfp_pack.restype = ctypes.c_int64
    lib.ttbfp_pack.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
    lib.ttbfp_unpack.restype = ctypes.c_int64
    lib.ttbfp_unpack.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
    return lib


def tilize(x: np.ndarray) -> np.ndarray:
    K, N = x.shape
    return np.ascontiguousarray(x.reshape(K // 32, 32, N // 32, 32).transpose(0, 2, 1, 3)).reshape(-1)


def untilize(v: np.ndarray, K: int, N: int) -> np.ndarray:
    return np.ascontiguousarray(v.reshape(K // 32, N // 32, 32, 32).transpose(0, 2, 1, 3)).reshape(K, N)


def tt_pack(lib, x: np.ndarray, fmt: str) -> np.ndarray:
    t = np.ascontiguousarray(tilize(np.asarray(x, np.float32)))
    cap = t.size  # words <= values
    out = np.zeros(cap, np.uint32)
    n = lib.ttbfp_pack(FMT_ID[fmt], t.ctypes.data, t.size, out.ctypes.data, cap)
    assert n > 0
    return out[:n].view(np.uint8)


def tt_unpack(lib, b: np.ndarray, fmt: str, K: int, N: int) -> np.ndarray:
    w = np.ascontiguousarray(b).view(np.uint32)
    out = np.zeros(K * N, np.float32)
    assert lib.ttbfp_unpack(FMT_ID[fmt], w.ctypes.data, w.size, out.ctypes.data, out.size) == out.size
    return untilize(out, K, N)


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = (a.view(np.int32) != b.view(np.int32)) & ~((a == 0) & (b == 0))
    return int(d.sum())


def real_cases():
    """A few real Coder expert matrices as ttnn sees them (W.T = [in, out]), across GGUF types."""
    if not os.environ.get("GGUF_PY"):
        paths.GGUF_PY = paths.LLAMA_CPP / "gguf-py"          # the pinned llama.cpp's gguf-py (before gguf_load)
    from fnx.gguf_load import Model
    from fnx.weights import DEFAULT_SHARD1
    m = Model(DEFAULT_SHARD1)
    for L, e in ((0, 3), (21, 100), (47, 255)):
        for r in ("gate", "down"):
            name = f"blk.{L}.ffn_{r}_exps.weight"
            yield f"{name}[{e}]({m.type_of(name)})", torch.from_numpy(m.get_expert(name, e).T.copy())


def cases():
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(512, 1024, generator=g) * torch.logspace(-8, 3, 512).unsqueeze(1)
    x[::5, ::3] *= 50.0
    x[7] = 0.0
    x[9, :16] = -0.0
    yield "random", x
    v = torch.arange(0, 65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    v = v[torch.isfinite(v) & (v.abs() <= 1.0)]
    v = v[: (v.numel() // 15) * 15].reshape(-1, 15)
    rows = torch.cat([torch.ones(v.shape[0], 1), v], 1).reshape(-1)
    rows = rows[: (rows.numel() // 1024) * 1024]
    yield "bf16-sweep", rows.reshape(-1, 32)
    yield from real_cases()


def check_ttnn(x: torch.Tensor, fmt: str) -> int:
    import ttnn
    dt = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[fmt]
    t = ttnn.to_torch(ttnn.from_torch(x.float(), dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
    return bit_mismatches(t.numpy(), bfp.rtn(x.float(), fmt).numpy())


def main() -> int:
    import ttnn  # noqa: F401  (sets tt-metal's runtime root, which the host packers' HAL lookup needs)
    lib = build_shim()
    total = 0
    for name, x in cases():
        x = x.float().contiguous()
        K, N = x.shape
        for fmt in ("bfp2", "bfp4", "bfp8"):
            q, E = bfp.quantize_codes(x, fmt)
            ours = bfp.to_tiles(bfp.pack_codes(q, E, fmt), fmt, K, N)
            theirs = tt_pack(lib, x.numpy(), fmt)
            n_bytes = int((ours != theirs).sum()) if ours.size == theirs.size else -1
            n_vals = bit_mismatches(tt_unpack(lib, theirs, fmt, K, N), bfp.rtn(x, fmt).numpy())
            n_ttnn = check_ttnn(x, fmt) if fmt != "bfp2" else 0
            # snapped: RTN, per-block exponent search, and one GPTQ-style column pass (blocks along N = last dim)
            snaps = [bfp.rtn(x, fmt), methods.blockq(x, fmt)]
            n_snap = 0
            for s in snaps:
                row = bfp.pack(s, fmt)
                tb = tt_pack(lib, s.numpy(), fmt)
                n_snap += int((bfp.to_tiles(row, fmt, K, N) != tb).sum())
                n_snap += bit_mismatches(bfp.unpack(row, fmt, K, N), s.numpy())
                n_snap += bit_mismatches(tt_unpack(lib, tb, fmt, K, N), s.numpy())
                if fmt != "bfp2":
                    n_snap += check_ttnn(s, fmt)
            bad = n_bytes + n_vals + n_ttnn + n_snap
            total += abs(bad)
            print(f"{name:42s} {fmt}: tile bytes {theirs.size} diff={n_bytes}  unpack bits diff={n_vals}  "
                  f"ttnn diff={n_ttnn}  snapped diff={n_snap}  {'OK' if bad == 0 else 'FAIL'}")
    print("PASS" if total == 0 else f"FAIL ({total} mismatches)")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
