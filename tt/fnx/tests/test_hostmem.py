"""Device kernels reading host memory directly over PCIe (fnx.tt.hostmem): exactness (2 MiB chunks, 1 GiB pages, the
4k region's paged items through their page tables, incl. a page table permuted so every 4 KiB boundary is a
discontinuity), bandwidth vs cores / request size / reads in flight against the same kernel reading DRAM, first-read
latency, the expert-page patterns (one cold expert per layer, ~130 MB per token, 2m vs 4k pools, a 10% host / 90%
DRAM mix) and trace replay (captured reads see host writes made after the capture).  Kernel:
fnx/tt/kernels/hostmem/hm_copy.cpp.

    tools/devrun.sh hostmem fnx/tests/test_hostmem.py [--quick | --smoke | --big GB [--big-mode 4k]]

Timing is the device wall clock (AICLK cycles; span = last core's end - first core's start) unless marked "host";
the host's load average is printed with every table because siblings load the CPU.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

from fnx.tt.hostmem import HostPool, free_hugepages_1g, system_limits

KDIR = Path(__file__).resolve().parents[1] / "tt" / "kernels" / "hostmem"
FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
MB = 1 << 20
MAT = 921600                   # one bfp4 expert matrix (1600 tiles x 576 B), DeviceMoE's host item
PAGE = 46080                   # DeviceMoE's gate/up page (80 bfp4 tiles)


def aiclk_hz() -> float:
    for p in glob.glob("/sys/bus/pci/devices/*/tenstorrent/*/tt_aiclk"):
        try:
            v = int(Path(p).read_text())
            if v > 0:
                return v * 1e6
        except (OSError, ValueError):
            pass
    return 1350e6


def load() -> str:
    return "load %.1f %.1f %.1f" % os.getloadavg()


class Bench:
    """hm_copy.cpp on the first `ncores` worker cores (row-major), cached per configuration."""

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.stats = ttnn.from_torch(torch.zeros(1, 1, len(self.grid), 16, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.hz = aiclk_hz()
        self.cache: dict = {}

    def table(self, chunks: np.ndarray):
        """DRAM tensor of the chunk (or page) table hm_copy reads: (lo, hi) u32 per 64-bit bus address."""
        tb = np.stack([chunks & np.uint64(0xFFFFFFFF), chunks >> np.uint64(32)], 1).astype(np.uint32).reshape(-1)
        tb = np.concatenate([tb, np.zeros((-tb.size) % 16, np.uint32)])
        t = ttnn.from_torch(torch.from_numpy(tb.view(np.int32).reshape(1, 1, 1, -1)), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return t, tb.size * 4

    def run(self, ncores: int, n: int, *, req: int, table, tbytes: int, src=None, out=None, host_every: int = 1,
            depth: int = 8, pages: int = 1, per_chunk: int = 1, stride: int = 0, relaxed: bool = False,
            reps: int = 1, riscv: int = 1, noc: int = 0, trace: bool = False, paged: bool = False, ppi: int = 0):
        """Run (or, trace=True, only issue) one launch; returns per-core stats [ncores, 16] uint32 (None if trace)."""
        src = src if src is not None else table
        dst = out if out is not None else table
        named = [("host_every", host_every), ("req", req), ("depth", depth), ("write_out", int(out is not None)),
                 ("pages", pages), ("per_chunk", per_chunk), ("stride", stride), ("table_bytes", tbytes),
                 ("relaxed", int(relaxed)), ("paged", int(paged)), ("ppi", ppi)]
        key = (ncores, riscv, noc, tuple(named), src is table, out is None)
        if key not in self.cache:
            cores = self.grid[:ncores]
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
            rt = ttnn.RuntimeArgs()
            for i, (x, y) in enumerate(cores):
                rt[x][y] = [i]
            acc = [a for t in (table, src, dst, self.stats) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
            cfg = ttnn.DataMovementConfigDescriptor(
                processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[riscv],
                noc=(ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default)[noc])
            k = ttnn.KernelDescriptor(kernel_source=str(KDIR / "hm_copy.cpp"), source_type=FP, core_ranges=crs,
                                      compile_time_args=acc, named_compile_time_args=named, runtime_args=rt,
                                      common_runtime_args=[0] * 6, config=cfg)
            size = tbytes + 64 + depth * req
            cb = ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=size)])
            h = int.from_bytes(hashlib.sha1(repr(("fnx.hostmem.bench", key)).encode()).digest()[:8], "little")
            self.cache[key] = (k, cb, h)
        k, cb, h = self.cache[key]
        k.common_runtime_args = [table.buffer_address(), src.buffer_address(), dst.buffer_address(),
                                 self.stats.buffer_address(), n, reps]
        pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
        pd.custom_program_hash = h
        io = [table] + ([src] if src is not table else []) + ([out] if out is not None else []) + [self.stats]
        ttnn.generic_op(io, pd)
        if trace:
            return None
        return ttnn.to_torch(self.stats).numpy().view(np.uint32).reshape(-1, 16)[:ncores]

    def span_s(self, st) -> float:
        t0 = st[:, 0].astype(np.uint64) | (st[:, 1].astype(np.uint64) << np.uint64(32))
        t1 = st[:, 2].astype(np.uint64) | (st[:, 3].astype(np.uint64) << np.uint64(32))
        return float(t1.max() - t0.min()) / self.hz


def dram_pages(mesh, npages: int, req: int, rng=None):
    a = (rng.integers(0, 2**31, (1, 1, npages, req // 4), dtype=np.int32) if rng is not None
         else np.zeros((1, 1, npages, req // 4), np.int32))
    return ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def readback(t) -> np.ndarray:
    return ttnn.to_torch(t).numpy().view(np.uint8).reshape(-1)


def pool_bytes(pool: HostPool, req: int, pages: int, npages: int) -> np.ndarray:
    """Expected bytes of requests 0..npages-1 (page r % pages of item r // pages)."""
    return np.concatenate([pool.item(r // pages)[(r % pages) * req:(r % pages + 1) * req] for r in range(npages)])


def big_pool(B: Bench, mesh, gb: float, mode: str = "2m") -> bool:
    """A cold-tier-sized pool: allocation + pinning time, where the chunks land, one device sweep over all of it
    (chunked modes), exactness of the items with the highest bus addresses (above 2^36 if any: the full 64-bit
    address path).  mode "4k": the 4k region (paged items: their physical runs reported; exactness through
    hostmem_read_paged)."""
    n = int(gb * 512)
    vm0 = meminfo_kb("VmallocUsed")
    t = time.perf_counter()
    pool = HostPool(n, 2 * MB, mode)
    alloc = time.perf_counter() - t
    try:
        if mode == "4k":
            return big_paged(B, mesh, pool, alloc, (meminfo_kb("VmallocUsed") - vm0) / 1024)
        above = int((pool.chunks >= np.uint64(1 << 36)).sum())
        print(f"big HostPool: {n} x 2 MiB = {pool.nbytes / 2**30:.2f} GiB pinned in {alloc:.1f} s, bus addresses "
              f"{int(pool.chunks.min()):#x} .. {int(pool.chunks.max()):#x}, {above} chunks above 2^36 "
              f"[{load()}]", flush=True)
        pat = (np.arange(2 * MB, dtype=np.uint32) * np.uint32(2654435761) >> np.uint32(13)).astype(np.uint8)
        t = time.perf_counter()
        for c in range(n):
            np.add(pat, np.uint8(c * 37 & 0xFF), out=pool.item(c))
        print(f"  filled in {time.perf_counter() - t:.1f} s")
        # one pass over the whole pool: 64 KiB requests on every core
        tab, tb = B.table(pool.chunks)
        R = n * 32
        c, k = best_split(R, len(B.grid))
        st = B.run(c, k, req=64 << 10, table=tab, tbytes=tb, pages=32, stride=2 * MB)
        print(f"  whole pool read once, {c} cores x {k} x 64 KiB: {B.span_s(st) * 1e3:.1f} ms = "
              f"{c * k * (64 << 10) / B.span_s(st) / 1e9:.2f} GB/s [{load()}]", flush=True)
        ttnn.deallocate(tab)
        # exactness on the 64 highest chunks
        sel = np.argsort(pool.chunks)[-64:]
        tab, tb = B.table(pool.chunks[sel])
        out = dram_pages(mesh, 64 * 128, 16 << 10)
        c, k = best_split(64 * 128, len(B.grid))
        B.run(c, k, req=16 << 10, table=tab, tbytes=tb, out=out, pages=128, stride=2 * MB)
        got = readback(out)
        exp = np.concatenate([pool.item(int(i)) for i in sel])
        good = np.array_equal(got, exp)
        print(f"  exact, 64 highest chunks ({int(pool.chunks[sel].min()):#x} .. {int(pool.chunks[sel].max()):#x}) "
              f"-> DRAM: {'PASS' if good else 'FAIL'}")
        ttnn.deallocate(out)
        ttnn.deallocate(tab)
        return good
    finally:
        pool.close()


def run_stats(pages: np.ndarray, ppi: int) -> str:
    """Physical runs of a page table (consecutive 4 KiB pages at consecutive bus addresses) within items of ppi
    pages."""
    brk = np.ones(len(pages), bool)
    brk[1:] = pages[1:] != pages[:-1] + np.uint64(4096)
    brk[::ppi] = True
    lens = np.diff(np.append(np.nonzero(brk)[0], len(pages)))
    q = np.percentile(lens, [10, 50, 90])
    return (f"{len(lens)} runs (pages per run: mean {lens.mean():.1f}, p10/p50/p90 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}, "
            f"{100 * float(lens[lens == 1].sum()) / len(pages):.1f}% of the pages alone)")


def meminfo_kb(key: str) -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k == key:
            return int(v.split()[0])
    return 0


def big_paged(B: Bench, mesh, pool: HostPool, alloc: float, kmem_mb: float) -> bool:
    """big_pool's 4k-region part: pin stats, runs, KMD memory (VmallocUsed growth), a pattern fill and the 64 items
    with the highest first page read back through hostmem_read_paged (16 KiB requests)."""
    n, ppi = pool.n_items, (2 * MB) // 4096
    pages = np.concatenate([pool.pages(i) for i in range(n)])
    print(f"big HostPool 4k: {n} x 2 MiB = {pool.nbytes / 2**30:.2f} GiB pinned in {alloc:.1f} s "
          f"({pool.alloc_stats}); {run_stats(pages, ppi)}; KMD vmalloc +{kmem_mb:.0f} MiB; "
          f"{int((pages >= np.uint64(1 << 36)).sum())} pages above 2^36 [{load()}]", flush=True)
    pat = (np.arange(2 * MB, dtype=np.uint32) * np.uint32(2654435761) >> np.uint32(13)).astype(np.uint8)
    t = time.perf_counter()
    for c in range(n):
        np.add(pat, np.uint8(c * 37 & 0xFF), out=pool.item(c))
    print(f"  filled in {time.perf_counter() - t:.1f} s")
    sel = np.argsort(pool.addrs)[-64:]
    tab, tb = B.table(np.concatenate([pool.pages(int(i)) for i in sel]))
    out = dram_pages(mesh, 64 * 128, 16 << 10)
    c, k = best_split(64 * 128, len(B.grid))
    st = B.run(c, k, req=16 << 10, table=tab, tbytes=tb, out=out, pages=128, paged=True, ppi=ppi)
    got = readback(out)
    exp = np.concatenate([pool.item(int(i)) for i in sel])
    good = np.array_equal(got, exp)
    print(f"  exact, 64 items with the highest first pages -> DRAM through the page tables (128 MiB, "
          f"{B.span_s(st) * 1e3:.1f} ms = {64 * 2 * MB / B.span_s(st) / 1e9:.2f} GB/s): {'PASS' if good else 'FAIL'}")
    ttnn.deallocate(out)
    ttnn.deallocate(tab)
    return good


def best_split(total: int, ncores: int) -> tuple[int, int]:
    """(cores, per-core requests) with cores * n == total, cores <= ncores, as many cores as possible."""
    for c in range(min(total, ncores), 0, -1):
        if total % c == 0:
            return c, total // c
    return 1, total


def main(quick: bool, smoke: bool = False, big: float = 0.0, big_mode: str = "2m") -> bool:
    lim = system_limits()
    print("system:", ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in lim.items()))
    print(load())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=8 << 20)
    ok = True
    pools = []
    try:
        B = Bench(mesh)
        NC = len(B.grid)
        print(f"worker cores {NC}, AICLK {B.hz / 1e6:.0f} MHz")
        rng = np.random.default_rng(0)
        if big:
            return big_pool(B, mesh, big, big_mode)

        # ---------------------------------------------------------------- 128 MiB pool of 2 MiB THP chunks
        TOT = (8 if smoke else 32 if quick else 128) * MB
        t = time.perf_counter()
        pool = HostPool(TOT // (2 * MB), 2 * MB, "2m")
        pools.append(pool)
        print(f"HostPool 2m: {len(pool.chunks)} chunks pinned in {time.perf_counter() - t:.2f} s, bus addresses "
              f"{int(pool.chunks.min()):#x} .. {int(pool.chunks.max()):#x} "
              f"({int((pool.chunks >= np.uint64(1 << 36)).sum())} above 2^36)")
        for i in range(pool.n_items):
            pool.item(i)[:] = rng.integers(0, 256, 2 * MB, dtype=np.uint8)
        tab, tb = B.table(pool.chunks)

        # exactness: host -> L1 -> DRAM copies, NOC0 (NCRISC) and NOC1 (BRISC), both PCIe windows, 1g pages
        def check(name, p, tab_, tb_, req, npages, exp=None, **kw):
            nonlocal ok
            c, n = best_split(npages, NC)
            out = dram_pages(mesh, npages, req)
            st = B.run(c, n, req=req, table=tab_, tbytes=tb_, out=out, **kw)
            got = readback(out)
            exp = pool_bytes(p, req, kw.get("pages", 1), npages) if exp is None else exp
            good = np.array_equal(got, exp)
            ok = ok and good
            print(f"exact {name:34s} {npages * req / MB:7.1f} MB on {c:3d} cores: "
                  f"{'PASS' if good else 'FAIL (%d bytes differ)' % int((got != exp).sum())}  "
                  f"(PCIE_NOC {st[0, 6]},{st[0, 7]})", flush=True)
            ttnn.deallocate(out)

        R16 = TOT // (16 << 10)
        check("2m pool, 16 KiB req, NOC0", pool, tab, tb, 16 << 10, R16, pages=128, stride=2 * MB)
        if smoke:
            return ok
        check("2m pool, 16 KiB req, NOC1 BRISC", pool, tab, tb, 16 << 10, R16, pages=128, stride=2 * MB,
              riscv=0, noc=1)
        check("2m pool, 64 KiB req, relaxed window", pool, tab, tb, 64 << 10, TOT // (64 << 10), pages=32,
              stride=2 * MB, relaxed=True)
        mpool = HostPool(144, MAT, "2m")       # 144 bfp4 matrices = 48 experts ~ the cold reads of one token
        pools.append(mpool)
        for i in range(mpool.n_items):
            mpool.item(i)[:] = rng.integers(0, 256, MAT, dtype=np.uint8)
        mtab, mtb = B.table(mpool.chunks)
        check("matrix pool (2/chunk), 46080 B pages", mpool, mtab, mtb, PAGE, 144 * 20, pages=20,
              per_chunk=mpool.per_chunk, stride=mpool.stride)
        # the 4k region: the same matrices in pinned 4 KiB pages, read through their page tables
        t = time.perf_counter()
        kpool = HostPool(144, MAT, "4k")
        pools.append(kpool)
        PPI = MAT // 4096
        kpages = np.concatenate([kpool.pages(i) for i in range(kpool.n_items)])
        print(f"HostPool 4k: 144 matrices pinned in {time.perf_counter() - t:.2f} s ({kpool.alloc_stats}), "
              f"{kpool.n_paged} items paged; {run_stats(kpages, PPI)}")
        for i in range(kpool.n_items):
            kpool.item(i)[:] = rng.integers(0, 256, MAT, dtype=np.uint8)
        ktab, ktb = B.table(kpages)
        check("4k pool, 46080 B pages (paged)", kpool, ktab, ktb, PAGE, 144 * 20, pages=20, paged=True, ppi=PPI)
        # worst case: every 4 KiB page boundary a discontinuity (the pool's pages permuted)
        perm = rng.permutation(len(kpages))
        xtab, xtb = B.table(kpages[perm])
        flat = np.concatenate([kpool.item(i) for i in range(kpool.n_items)]).reshape(-1, 4096)
        check("4k pool, pages permuted (paged)", kpool, xtab, xtb, PAGE, 144 * 20, exp=flat[perm].reshape(-1),
              pages=20, paged=True, ppi=PPI)
        check("4k pool, permuted, NOC1 BRISC, 4 KiB req", kpool, xtab, xtb, 4096, 144 * 225,
              exp=flat[perm].reshape(-1), pages=225, paged=True, ppi=PPI, riscv=0, noc=1)
        gpool = None
        if free_hugepages_1g() >= 1 and not quick:
            try:
                gpool = HostPool(1, 1 << 30, "1g")
            except (MemoryError, OSError) as e:
                print(f"1g pool unavailable: {e}")
        if gpool is not None:
            pools.append(gpool)
            gpool.item(0)[:] = rng.integers(0, 256, 1 << 30, dtype=np.uint8)
            gtab, gtb = B.table(gpool.chunks)
            print(f"HostPool 1g: bus address {int(gpool.chunks[0]):#x}")
            check("1g hugepage pool, 64 KiB req", gpool, gtab, gtb, 64 << 10, (256 * MB) // (64 << 10),
                  pages=(1 << 30) // (64 << 10), stride=1 << 30)
        else:
            print(f"1g pool skipped ({free_hugepages_1g()} free 1 GiB hugepages)")

        # ---------------------------------------------------------------- latency of a lone read (1 core, cold)
        lat = {"host": [], "dram": []}
        dsrc = dram_pages(mesh, 256, 16 << 10, rng)
        for i in range(8 if quick else 24):
            st = B.run(1, 1, req=16 << 10, table=tab, tbytes=tb, pages=128, stride=2 * MB)
            lat["host"].append(st[0, 4:6])
            st = B.run(1, 1, req=16 << 10, table=tab, tbytes=tb, src=dsrc, host_every=0)
            lat["dram"].append(st[0, 4:6])
        for k, v in lat.items():
            v = np.array(v) / B.hz * 1e6
            print(f"latency {k:4s}: 64 B read median {np.median(v[:, 0]):.2f} us (min {v[:, 0].min():.2f}, "
                  f"max {v[:, 0].max():.2f}); 16 KiB read median {np.median(v[:, 1]):.2f} us  [{load()}]")
        ttnn.deallocate(dsrc)

        # ---------------------------------------------------------------- bandwidth: cores x request size, host vs DRAM
        reqs = (4 << 10, 16 << 10, 64 << 10) if quick else (2 << 10, 4 << 10, 16 << 10, 64 << 10)
        ncs = sorted({1, 4, 16, 64, NC} if quick else {1, 2, 4, 8, 16, 32, 64, NC})
        print(f"\nbandwidth GB/s (1e9 B/s), {TOT // MB} MiB per launch, depth 8 reads in flight per core, NOC0 [{load()}]")
        print(f"{'req':>6s} {'src':>5s} " + " ".join(f"{c:>6d}c" for c in ncs))
        for req in reqs:
            R = TOT // req
            dsrc = dram_pages(mesh, R, req, rng)
            for name in ("host", "dram"):
                row = []
                for c in ncs:
                    n = R // c
                    kw = dict(pages=(2 * MB) // req, stride=2 * MB) if name == "host" else dict(src=dsrc, host_every=0)
                    reps = 1 if name == "host" else 4
                    st = B.run(c, n, req=req, table=tab, tbytes=tb, reps=reps, **kw)
                    row.append(c * n * req * reps / B.span_s(st) / 1e9)
                print(f"{req >> 10:4d}Ki {name:>5s} " + " ".join(f"{v:7.2f}" for v in row), flush=True)
            ttnn.deallocate(dsrc)

        print(f"\nreads in flight per core (all {NC} cores, 16 KiB req, host) [{load()}]")
        R = TOT // (16 << 10)
        c, n = best_split(R, NC)
        for depth in (1, 2, 4, 8, 15):
            st = B.run(c, n, req=16 << 10, table=tab, tbytes=tb, pages=128, stride=2 * MB, depth=depth)
            print(f"  depth {depth:2d}: {c * n * (16 << 10) / B.span_s(st) / 1e9:6.2f} GB/s")
        for relaxed in (False, True):
            for noc, riscv in ((0, 1), (1, 0)):
                st = B.run(c, n, req=16 << 10, table=tab, tbytes=tb, pages=128, stride=2 * MB, relaxed=relaxed,
                           noc=noc, riscv=riscv)
                print(f"  NOC{noc} {'relaxed window' if relaxed else 'default window'}: "
                      f"{c * n * (16 << 10) / B.span_s(st) / 1e9:6.2f} GB/s")
        if gpool is not None:
            R = (512 * MB) // (64 << 10)
            c, n = best_split(R, NC)
            st = B.run(c, n, req=64 << 10, table=gtab, tbytes=gtb, pages=(1 << 30) // (64 << 10), stride=1 << 30)
            print(f"  1g hugepage pool, 512 MiB, 64 KiB req: {c * n * (64 << 10) / B.span_s(st) / 1e9:6.2f} GB/s")

        # ---------------------------------------------------------------- DeviceMoE patterns (46080 B pages)
        print(f"\nexpert-page patterns (46080 B pages, bfp4 matrices of 921600 B) [{load()}]")
        mkw = dict(req=PAGE, table=mtab, tbytes=mtb, pages=20, per_chunk=mpool.per_chunk, stride=mpool.stride)
        kkw = dict(req=PAGE, table=ktab, tbytes=ktb, pages=20, paged=True, ppi=PPI)
        xkw = dict(kkw, table=xtab, tbytes=xtb)
        for label, npg in (("1 cold expert (3 matrices, 2.76 MB)", 60), ("48 cold experts = 1 token (132.7 MB)", 2880)):
            for pname, kw in (("2m", mkw), ("4k", kkw), ("4k permuted", xkw)):
                for depth in (2, 4):
                    c, n = best_split(npg, NC)
                    spans = [B.span_s(B.run(c, n, depth=depth, **kw)) for _ in range(5)]
                    print(f"  {label:38s} {pname:11s} {c:3d} cores x {n:2d} pages, depth {depth}: "
                          f"{np.median(spans) * 1e3:7.3f} ms ({npg * PAGE / np.median(spans) / 1e9:5.2f} GB/s)")
        # a layer: 10 experts x 3 matrices x 20 pages = 600 pages; every 10th page from the host vs all from DRAM
        dsrc = dram_pages(mesh, 2880, PAGE, rng)
        for npg, label in ((600, "1 layer (10 experts)"), (2880, "48 experts")):
            c, n = best_split(npg, NC)
            res = {}
            for he in (0, 10, 1):
                spans = [B.span_s(B.run(c, n, src=dsrc, host_every=he, depth=4, **mkw)) for _ in range(5)]
                res[he] = np.median(spans) * 1e3
            print(f"  {label:22s} {c} cores: all DRAM {res[0]:.3f} ms, 10% pages from host {res[10]:.3f} ms, "
                  f"all host {res[1]:.3f} ms")

        # ---------------------------------------------------------------- trace: capture, replay, host rewrite, replay
        print(f"\ntrace [{load()}]")
        c, n = best_split(2880, NC)
        out = dram_pages(mesh, 2880, PAGE)
        B.run(c, n, out=out, depth=4, **mkw)                       # compiled before the capture (ttnn #48536)
        B.run(c, n, depth=4, **mkw)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        B.run(c, n, out=out, depth=4, trace=True, **mkw)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for rnd in range(2):
            if rnd:
                for i in range(mpool.n_items):                    # new host contents after the capture
                    mpool.item(i)[:] = rng.integers(0, 256, MAT, dtype=np.uint8)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            good = np.array_equal(readback(out), pool_bytes(mpool, PAGE, 20, 2880))
            ok = ok and good
            print(f"  replay {rnd} ({'host rewritten after capture' if rnd else 'as captured'}): 132.7 MB host -> "
                  f"DRAM {'PASS' if good else 'FAIL'}")
        ttnn.release_trace(mesh, tid)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        B.run(c, n, depth=4, trace=True, **mkw)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        K = 20
        t = time.perf_counter()
        for _ in range(K):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t) / K * 1e3
        print(f"  traced 132.7 MB host read (one token's cold experts), host-timed: {ms:.3f} ms per replay "
              f"({2880 * PAGE / ms / 1e6:.2f} GB/s) [{load()}]")
        ttnn.release_trace(mesh, tid)
        ttnn.deallocate(out)
        ttnn.deallocate(dsrc)
    finally:
        ttnn.synchronize_device(mesh)
        ttnn.close_mesh_device(mesh)
        for p in pools:
            p.close()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller pools and sweeps")
    ap.add_argument("--smoke", action="store_true", help="only the first exactness check (8 MiB, NOC0)")
    ap.add_argument("--big", type=float, default=0.0, metavar="GB",
                    help="only the cold-tier-sized pool check: a GB-sized pool of 2 MiB items, pin time, exactness")
    ap.add_argument("--big-mode", default="2m", help="the big pool's chunk mode: 2m (one sweep) or 4k (paged)")
    a = ap.parse_args()
    good = main(a.quick, a.smoke, a.big, a.big_mode)
    print("test_hostmem", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
