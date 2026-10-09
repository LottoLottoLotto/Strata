"""Host-resident memory that device kernels read directly over PCIe: no host round trip, no CPU work, traceable.

Mechanism (Blackhole P150, KMD 2.11, this host: iommu=pt):
  - A NoC read sent to the PCIe tile carries a full 64-bit address; [0, 2^58) goes straight to the host bus
    (through the IOMMU when it translates; under iommu=pt the bus address IS the physical address).  The device side
    is fnx/tt/kernels/hostmem/hostmem.h: hostmem_read(addr64, l1, len) (the split coordinate + 64-bit address form
    of noc_read_with_state; plain noc_async_read only reaches 36-bit local addresses).
  - Without IOMMU translation one read must stay inside physically contiguous memory, and the KMD PIN_PAGES ioctl
    only pins a physically contiguous range (it returns the range's physical address; closing the fd that pinned it
    unpins).  A HostPool is made of
      * chunks: 1 GiB hugetlb pages (limited to the free ones: 3 on this host, the 4th is the command queues') and
        2 MiB transparent huge pages (no system configuration, but on a host with tens of GB of page cache the
        kernel often cannot make them: 2026-10-07, 4714 of 4749 chunks after 13594 tries), one pin each;
      * a "4k" region: ordinary anonymous memory (THPs where the kernel has them), pinned as physically contiguous
        runs of 4 KiB pages (each 2 MiB of it tried as one pin; in a region that fails every page's address is
        probed and each run pinned once), with a bus address per 4 KiB page (`pages(i)`).  It needs nothing but
        free RAM (plus ~4 KiB of KMD memory per run).
    Fixed-size items are packed into chunks without straddling one (`contig[i]`: True, one read may span the item)
    or laid out page aligned in the 4k region (`contig[i]` True only when its pages happen to be one run; else a
    reader splits a read at the 4 KiB page boundaries of the item: hostmem.h hostmem_read_paged with the item's
    page table).  `addrs[i]` / `addr(i, off)`: bus addresses (`lohi(i)` the two runtime-arg words of a contiguous
    item).
  - The addresses are fixed for the pool's life, so a captured trace that reads them replays correctly, and host
    writes into an item between replays are seen by the next replay (PCIe reads snoop the CPU caches).  Never
    close() a pool (or let it be collected) while a kernel or a captured trace may still read it.
  - KMD costs per pin (memory.c ioctl_pin_pages): a kmalloc + a vzalloc of 8 B per page (at least one page of
    kernel memory), and a walk of the fd's whole pin list (duplicate check).  So the 4k region pins runs, not pages,
    and spreads its pins over several fds (PINS_PER_FD).
  - Not tt-metal's experimental::PinnedMemory / BufferType::SYSTEM_MEMORY: the allocator rejects SYSTEM_MEMORY
    buffers, PinnedMemory reports 0 pins unless the IOMMU translates (iommu_group type DMA), and UMD refuses to map
    more than one 4 KiB page to the NoC without it.  The KMD pin used here is the same ioctl underneath.

Measured (fnx/tests/test_hostmem.py, P150 on a PCIe 4.0 x16 link, 16 GT/s): 26-29 GB/s for any request >= 4 KiB
from 1 to 110 cores (the link is the limit: one core with 8 reads in flight already reaches it; 2 KiB requests
~21 GB/s) vs 380-420 GB/s DRAM with the same kernel; a lone read 0.78 us (64 B) / 1.5 us (16 KiB) vs 0.35 / 0.60 us
DRAM; 132.7 MB (48 bfp4 experts, 46080 B pages) in 4.9-5.0 ms device-timed and host-timed as a trace replay; a
10 GiB pool of 2 MiB chunks pinned in 33 s and read whole at 28.8 GB/s, bytes exact above 2^36.
"""
from __future__ import annotations

import errno
import fcntl
import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np

_IOCTL_PIN_PAGES = 0xFA07      # _IO(TENSTORRENT_IOCTL_MAGIC 0xFA, 7)
_IOCTL_UNPIN_PAGES = 0xFA0A    # _IO(0xFA, 10)
_MAP_HUGETLB = 0x40000
_MAP_HUGE_1GB = 30 << 26
_MAP_POPULATE = 0x8000
_MADV_DONTFORK = 10            # children never inherit (a COW copy of a pinned page is not what the device reads)
_MADV_HUGEPAGE = 14
_MADV_NOHUGEPAGE = 15
_MADV_POPULATE_WRITE = 23
_MADV_COLLAPSE = 25
CHUNK = {"2m": 1 << 21, "1g": 1 << 30}
MODES = ("auto", "1g", "2m", "4k")
PAGE = 4096                    # the 4k region's pages: a paged item's reads split at these boundaries
ALIGN = 64                     # NOC_PCIE_READ_ALIGNMENT_BYTES
RELAXED = 1 << 58              # the same host addresses through the PCIe relaxed-ordering NoC window
DEVICE = "/dev/tenstorrent/0"
PINS_PER_FD = 1024             # the KMD walks an fd's whole pin list on every pin: keep each list short


def default_chunk() -> str:
    """FNX_HOST_CHUNK (auto | 1g | 2m | 4k), default "auto".  (FNX_HOST_4K_THP=0: the 4k region without THPs.)"""
    c = os.environ.get("FNX_HOST_CHUNK", "auto")
    if c not in MODES:
        raise ValueError(f"FNX_HOST_CHUNK={c!r}: one of {MODES}")
    return c


class HostPool:
    """`n_items` items of `item_bytes` (rounded up to 64 B) in pinned host memory the device reads over PCIe.

    chunk (None: default_chunk()):
      "2m"   2 MiB THP chunks (MemoryError when the kernel cannot give enough);
      "1g"   1 GiB hugetlb pages;
      "4k"   the 4k region only (module docstring; items page aligned; needs only free RAM);
      "auto" as many free 1 GiB hugetlb pages as whole pages of items fill, then 2 MiB THP chunks while the kernel
             gives them, the rest in the 4k region; never fails for lack of huge pages (MemoryError only for lack of
             RAM).  The 4k items are the FIRST ones (0 .. n_paged - 1): readers of the last items may rely on
             contiguity whenever chunks did not run short by more than those.
    `items[i]` / `item(i)`: writable np.uint8 [item_bytes] view of item i in place; `contig` np.bool_ [n_items]
    (item physically contiguous); `addrs` np.uint64 [n_items] the bus address of each item's byte 0; `addr(i, off)`
    of its byte off (contiguous to the end of off's 4 KiB page of the item, or of the item when contig[i]);
    `pages(i)` np.uint64 the bus addresses of item bytes 0, 4096, 8192, ... (the page table of a paged item);
    `chunks` np.uint64 [n_chunks] the chunk bus addresses; uniform chunk pools ("2m", "1g") keep item i in chunk
    i // per_chunk at (i % per_chunk) * stride (per_chunk None otherwise); `alloc_stats` what was pinned how.
    """

    def __init__(self, n_items: int, item_bytes: int, chunk: str | None = None, device: str = DEVICE):
        chunk = chunk or default_chunk()
        if chunk not in MODES:
            raise ValueError(f"chunk must be one of {MODES}")
        self.mode = chunk
        self.item_bytes = int(item_bytes)
        self.stride = -(-self.item_bytes // ALIGN) * ALIGN
        if chunk in CHUNK and self.stride > CHUNK[chunk]:
            raise ValueError(f"item of {item_bytes} B does not fit a {chunk} chunk")
        self.n_items = int(n_items)
        self._device = device
        self._fds: list[int] = []
        self._probe_fd = None
        self._fd_pins = PINS_PER_FD
        self._pinbuf = bytearray(40)      # tenstorrent_pin_pages: in (24 B), out_extended (16 B)
        self._maps: list[mmap.mmap] = []
        self.alloc_stats: dict = {}
        t0 = time.perf_counter()
        regions = []            # (view, chunk bus address or None, page bus addresses or None, items, item stride)
        try:
            left = self.n_items
            if chunk in ("1g", "auto") and self.stride <= CHUNK["1g"]:
                per = CHUNK["1g"] // self.stride
                n1 = -(-left // per)
                if chunk == "auto":
                    n1 = min(n1, free_hugepages_1g(), left // per)   # only whole 1 GiB pages of items
                if n1:
                    views, bus = self._alloc_1g(n1)
                    for v, b in zip(views, bus):
                        k = min(per, left)
                        regions.append((v, b, None, k, self.stride))
                        left -= k
                self.alloc_stats["1g"] = n1
            if left and (chunk == "2m" or (chunk == "auto" and self.stride <= CHUNK["2m"])):
                per = CHUNK["2m"] // self.stride
                views, bus = self._alloc_2m(-(-left // per), strict=chunk == "2m")
                for v, b in zip(views, bus):
                    k = min(per, left)
                    regions.append((v, b, None, k, self.stride))
                    left -= k
            if left:
                st = -(-self.item_bytes // PAGE) * PAGE
                view, pages = self._alloc_4k(left * st // PAGE)
                regions.insert(0, (view, None, pages, left, st))      # the 4k items first
        except BaseException:
            self.close()
            raise
        n_reg = len(regions)
        self._rview, self._rbase, self._rpages = [], np.zeros(n_reg, np.uint64), []
        ridx, offs = [], []
        for r, (v, b, pg, k, st) in enumerate(regions):
            self._rview.append(v)
            self._rpages.append(pg)
            if b is not None:
                self._rbase[r] = b
            ridx.append(np.full(k, r, np.int64))
            offs.append(np.arange(k, dtype=np.int64) * st)
        self._ridx = np.concatenate(ridx) if ridx else np.zeros(0, np.int64)
        self._off = np.concatenate(offs) if offs else np.zeros(0, np.int64)
        self.contig = np.ones(self.n_items, bool)
        self.addrs = np.zeros(self.n_items, np.uint64)
        npi = -(-self.item_bytes // PAGE)
        i0 = 0
        for r, (v, b, pg, k, st) in enumerate(regions):
            if pg is None:
                self.addrs[i0:i0 + k] = np.uint64(b) + self._off[i0:i0 + k].astype(np.uint64)
            else:
                tab = pg.reshape(k, st // PAGE)[:, :npi]
                self.addrs[i0:i0 + k] = tab[:, 0]
                self.contig[i0:i0 + k] = np.all(np.diff(tab, axis=1) == np.uint64(PAGE), axis=1)
            i0 += k
        self.chunks = np.array([b for v, b, pg, k, st in regions if pg is None], np.uint64)
        sizes = {len(v) for v, b, pg, k, st in regions if pg is None}
        uniform = len(sizes) == 1 and all(pg is None for v, b, pg, k, st in regions)
        self.chunk_bytes = sizes.pop() if uniform else None
        self.per_chunk = self.chunk_bytes // self.stride if uniform else None
        self.nbytes = sum(len(v) for v, b, pg, k, st in regions)
        self.n_paged = int((~self.contig).sum())
        self.alloc_stats.update({"fds": len(self._fds), "paged_items": self.n_paged,
                                 "s": round(time.perf_counter() - t0, 1)})
        self.items = _ItemView(self)

    # ------------------------------------------------------------------------------------------- allocation
    def _pin(self, va: int, size: int) -> int:
        """Pin [va, va + size) (physically contiguous, else OSError EINVAL) on an fd with room; its bus address."""
        if self._fd_pins >= PINS_PER_FD:
            self._fds.append(os.open(self._device, os.O_RDWR | os.O_CLOEXEC))
            self._fd_pins = 0
        buf = self._pinbuf
        struct.pack_into("<IIQQQQ", buf, 0, 16, 0, va, size, 0, 0)
        fcntl.ioctl(self._fds[-1], _IOCTL_PIN_PAGES, buf, True)
        self._fd_pins += 1
        return struct.unpack_from("<Q", buf, 24)[0]

    def _alloc_1g(self, n: int):
        size = n << 30
        try:
            m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | _MAP_HUGETLB | _MAP_HUGE_1GB | _MAP_POPULATE)
        except OSError as e:
            raise MemoryError(f"{n} x 1 GiB hugetlb pages unavailable ({free_hugepages_1g()} free): {e}") from e
        self._maps.append(m)
        m.madvise(_MADV_DONTFORK, 0, size)
        arr = np.frombuffer(m, np.uint8)
        va = arr.ctypes.data
        views = [arr[c << 30:(c + 1) << 30] for c in range(n)]
        return views, [self._pin(va + (c << 30), 1 << 30) for c in range(n)]

    def _alloc_2m(self, n: int, strict: bool = True):
        """Up to n THP-backed 2 MiB chunks: anonymous mappings advised MADV_HUGEPAGE; every 2 MiB-aligned chunk is
        faulted in, made one huge page (MADV_COLLAPSE: a no-op when the fault got one, else synchronous compaction)
        and pinned.  The pin fails on a chunk the kernel could not make one huge page (its 4 KiB pages are not
        contiguous: "pages discontiguous" in the kernel log); on this host, with tens of GB of page cache, about half
        the chunks fail that way (13.6 GB pools: 5311 of 11112 and 6129 of 17090 tries pinned, 2026-10-07); a failed
        chunk is given back (MADV_DONTNEED) and replaced from a further mapping (keeping failed chunks mapped did not
        help).  Gives up after 2 n + 4096 tries, or (strict=False) as soon as fewer than 1 in 8 of 512 or more tries
        succeed: strict raises MemoryError, else the chunks pinned so far are returned."""
        C = 1 << 21
        views, bus = [], []
        tried, failed, limit = 0, 0, 2 * n + 4096
        while len(views) < n:
            want = n - len(views)
            size = (want + want // 4 + 2) * C
            m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE)
            self._maps.append(m)
            arr = np.frombuffer(m, np.uint8)
            va0 = arr.ctypes.data
            off0 = (-va0) % C
            m.madvise(_MADV_DONTFORK, 0, size)
            m.madvise(_MADV_HUGEPAGE, 0, size)
            for k in range((size - off0) // C):
                if len(views) == n:
                    break
                if tried == limit or (not strict and tried >= 512 and 8 * len(views) < tried):
                    self.alloc_stats.update({"2m": len(views), "2m_tried": tried, "2m_failed": failed})
                    if strict:
                        raise MemoryError(f"only {len(views)} of {n} 2 MiB chunks could be pinned as huge pages "
                                          f"({tried} tried)")
                    return views, bus
                off = off0 + k * C
                tried += 1
                arr[off:off + C:PAGE] = 0           # fault the chunk in (one THP when available)
                try:
                    m.madvise(_MADV_COLLAPSE, off, C)
                    phys = self._pin(va0 + off, C)
                except OSError:
                    failed += 1
                    m.madvise(mmap.MADV_DONTNEED, off, C)
                    continue
                views.append(arr[off:off + C])
                bus.append(phys)
        self.alloc_stats.update({"2m": n, "2m_tried": tried, "2m_failed": failed})
        return views, bus

    def _probe(self, va: int) -> int:
        """Bus address of the 4 KiB page at va: pinned and at once unpinned on the probe fd (keeps no KMD memory)."""
        if self._probe_fd is None:
            self._probe_fd = os.open(self._device, os.O_RDWR | os.O_CLOEXEC)
        buf = self._pinbuf
        struct.pack_into("<IIQQQQ", buf, 0, 16, 0, va, PAGE, 0, 0)
        fcntl.ioctl(self._probe_fd, _IOCTL_PIN_PAGES, buf, True)
        phys = struct.unpack_from("<Q", buf, 24)[0]
        struct.pack_into("<QQQ", buf, 0, va, PAGE, 0)
        fcntl.ioctl(self._probe_fd, _IOCTL_UNPIN_PAGES, buf, True)
        return phys

    def _alloc_4k(self, n: int):
        """n pinned 4 KiB pages in one anonymous mapping -> (view, bus address per page).  Every 2 MiB of it is
        first tried as one pin (a THP, or 512 pages that happen to be one physical run); in a region that fails,
        every page's address is probed (pin + unpin on a separate fd) and each physical run is pinned once (a run
        that moved since its probe - compaction - is split in halves until its pieces pin).  One pin per run because
        the KMD keeps at least one page of kernel memory per pin (memory.c: vzalloc of the page array).
        The mapping is advised MADV_HUGEPAGE (a fault gets a THP when the kernel can make one; no MADV_COLLAPSE), or
        with FNX_HOST_4K_THP=0 MADV_NOHUGEPAGE (only 4 KiB pages: the fallback's worst case, for measurements).
        Measured 2026-10-07 (2 GiB, scratch/hp4k/probe.py): with free 2 MiB blocks every region one THP; 4 KiB pages
        alone came in runs of 2.3 pages on average (25% of the pages alone): 227k pins, 4.1 us and ~3.9 KiB of KMD
        vmalloc memory per pin.  MemoryError when the memory cannot be populated or a page cannot be pinned."""
        t0 = time.perf_counter()
        R = 1 << 21
        size = n * PAGE
        m = mmap.mmap(-1, size + R, flags=mmap.MAP_PRIVATE)
        self._maps.append(m)
        m.madvise(_MADV_DONTFORK, 0, size + R)
        m.madvise(_MADV_HUGEPAGE if os.environ.get("FNX_HOST_4K_THP", "1") == "1" else _MADV_NOHUGEPAGE, 0, size + R)
        arr = np.frombuffer(m, np.uint8)
        off = (-arr.ctypes.data) % R
        view = arr[off:off + size]
        va = view.ctypes.data
        try:
            m.madvise(_MADV_POPULATE_WRITE, off, size)
        except OSError as e:
            raise MemoryError(f"{size / 2**30:.2f} GiB of host memory for the 4k region unavailable: {e}") from e
        t_pop = time.perf_counter() - t0
        pages = np.empty(n, np.uint64)
        step = np.arange(R // PAGE, dtype=np.uint64) * np.uint64(PAGE)
        st = {"regions": 0, "regions_whole": 0, "probed": 0, "pins": 0, "split_pins": 0}

        def pin_run(p: int, k: int) -> None:      # pages [p, p + k), one pin, else halves
            stack = [(p, k)]
            while stack:
                q, c = stack.pop()
                try:
                    phys = self._pin(va + q * PAGE, c * PAGE)
                except OSError as e:
                    if c == 1 or e.errno != errno.EINVAL:       # EINVAL: the range is not one physical run
                        raise MemoryError(f"pinning {c} pages of the 4k region failed: {e}") from e
                    st["split_pins"] += 1
                    stack += [(q + c // 2, c - c // 2), (q, c // 2)]
                    continue
                pages[q:q + c] = np.uint64(phys) + step[:c]
                st["pins"] += 1

        try:
            for p0 in range(0, n, R // PAGE):
                k = min(R // PAGE, n - p0)
                st["regions"] += 1
                try:
                    pages[p0:p0 + k] = np.uint64(self._pin(va + p0 * PAGE, k * PAGE)) + step[:k]
                    st["pins"] += 1
                    st["regions_whole"] += 1
                    continue
                except OSError as e:
                    if e.errno != errno.EINVAL:
                        raise MemoryError(f"pinning the 4k region failed: {e}") from e
                ph = np.fromiter((self._probe(va + (p0 + i) * PAGE) for i in range(k)), np.uint64, k)
                st["probed"] += k
                brk = np.ones(k, bool)
                brk[1:] = ph[1:] != ph[:-1] + np.uint64(PAGE)
                starts = np.nonzero(brk)[0]
                for a, b in zip(starts, np.append(starts[1:], k)):
                    pin_run(p0 + int(a), int(b - a))
        finally:
            if self._probe_fd is not None:
                os.close(self._probe_fd)
                self._probe_fd = None
        m.madvise(_MADV_NOHUGEPAGE, 0, size + R)     # pinned: nothing for khugepaged to collapse (it would skip)
        self.alloc_stats.update({"4k_pages": n, "4k_regions": f"{st['regions_whole']}/{st['regions']} whole",
                                 "4k_probed": st["probed"], "4k_pins": st["pins"], "4k_split": st["split_pins"],
                                 "4k_populate_s": round(t_pop, 2), "4k_s": round(time.perf_counter() - t0, 1)})
        return view, pages

    # ------------------------------------------------------------------------------------------- access
    def item(self, i: int) -> np.ndarray:
        """Writable np.uint8 [item_bytes] view of item i."""
        o = int(self._off[i])
        return self._rview[int(self._ridx[i])][o:o + self.item_bytes]

    def write(self, first: int, data: np.ndarray) -> None:
        """Copy rows of data (uint8 [n, item_bytes] or a flat multiple) into items first, first + 1, ..."""
        d = np.ascontiguousarray(data).view(np.uint8).reshape(-1, self.item_bytes)
        for j in range(d.shape[0]):
            self.item(first + j)[:] = d[j]

    def addr(self, i: int, off=0):
        """Bus address of byte `off` (an int or an int array) of item i: physically contiguous to the end of the
        item when contig[i], else to the end of the 4 KiB page of the item it falls in."""
        r, o = int(self._ridx[i]), int(self._off[i])
        offs = np.asarray(off, np.int64)
        if np.any(offs < 0) or np.any(offs >= self.item_bytes):
            raise IndexError(f"offset outside item {i} ({self.item_bytes} B)")
        pg = self._rpages[r]
        if pg is None:
            a = self._rbase[r] + (o + offs).astype(np.uint64)
        else:
            b = o + offs
            a = pg[b // PAGE] + (b % PAGE).astype(np.uint64)
        return int(a) if a.ndim == 0 else a

    def pages(self, i: int) -> np.ndarray:
        """np.uint64 [ceil(item_bytes / 4096)]: the bus addresses of item i's bytes 0, 4096, 8192, ... (a paged
        item's page table; a read of [off, off + n) in it splits at the multiples of 4096)."""
        npi = -(-self.item_bytes // PAGE)
        r, o = int(self._ridx[i]), int(self._off[i])
        pg = self._rpages[r]
        if pg is None:
            return self._rbase[r] + np.uint64(o) + np.arange(npi, dtype=np.uint64) * np.uint64(PAGE)
        return pg[o // PAGE:o // PAGE + npi].copy()

    def lohi(self, i: int, relaxed: bool = False) -> tuple[int, int]:
        """A contiguous item's bus address as the two u32 runtime-arg words (kernel: hostmem_addr(lo, hi))."""
        if not self.contig[i]:
            raise ValueError(f"item {i} is not physically contiguous (4k region): use pages(i)")
        a = int(self.addrs[i]) | (RELAXED if relaxed else 0)
        return a & 0xFFFFFFFF, a >> 32

    def close(self) -> None:
        """Unpin (closing the KMD fds releases every pin) and unmap.  Only when no kernel/trace can read the pool."""
        for fd in self._fds:
            os.close(fd)
        self._fds = []
        self._rview = []
        self.items = None
        for m in self._maps:
            try:
                m.close()
            except BufferError:      # a caller still holds a view; the mapping goes with the last reference
                pass
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _ItemView:
    """pool.items[i] -> writable view of item i (rows of a chunked pool are not one strided array)."""

    def __init__(self, pool: HostPool):
        self._pool = pool

    def __getitem__(self, i: int) -> np.ndarray:
        return self._pool.item(i)

    def __len__(self) -> int:
        return self._pool.n_items


# ------------------------------------------------------------------------------------------------ system limits
def free_hugepages_1g() -> int:
    p = Path("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages")
    return int(p.read_text()) if p.exists() else 0


def system_limits() -> dict:
    """What bounds a host pool on this machine (read-only probes; nothing is changed)."""
    def rd(p, default=""):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return default
    pci = "/sys/bus/pci/devices/0000:01:00.0"
    thp = rd("/sys/kernel/mm/transparent_hugepage/enabled")
    mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in rd("/proc/meminfo").splitlines())}
    return {
        "hugepages_1g_total": int(rd("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages", "0")),
        "hugepages_1g_free": free_hugepages_1g(),
        "thp": thp[thp.find("[") + 1:thp.find("]")] if "[" in thp else thp,
        "iommu_group_type": rd(f"{pci}/iommu_group/type"),
        "mem_available_gb": mem.get("MemAvailable", 0) / 2**20,
        "mem_total_gb": mem.get("MemTotal", 0) / 2**20,
        "pcie_link": f"{rd(f'{pci}/current_link_speed')} x{rd(f'{pci}/current_link_width')}",
        "pcie_link_max": f"{rd(f'{pci}/max_link_speed')} x{rd(f'{pci}/max_link_width')}",
    }
