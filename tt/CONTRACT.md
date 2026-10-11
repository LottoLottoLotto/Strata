# Flash-Next on the P150: module contract

Package `fnx` at `tt/fnx` of the Strata repository (branch tt-p150; `import fnx` with `<repo>/tt` on PYTHONPATH; run
on the device with `tt/tools/devrun.sh`, from `tt/`).  Paths below without a root are relative to `tt/`.
Model: Qwen3.8-Flash-Next Coder IQ1_M GGUF (the engine: its `--native` / `--ple-gguf` arguments; tests and tools:
`fnx.paths.SHARD1/SHARD2`, `FNX_DATA/models/coder/`).  `Model.available(name[, expert])` reports holes of a file still
downloading.  Second variant (`FNX_MODEL=iq3s`, below): the full model's IQ3_S GGUF (512 routed experts per layer).

Model variants (`fnx.paths.VARIANTS`; FNX_MODEL, else read off the `--native` / FNX_GGUF file name, else "coder";
`paths.py --shell` prints it as MODEL and tt-engine.sh / devrun.sh export it): coder = IQ1_M, 256 experts,
`data/expert-profile-coder.bin`; iq3s = IQ3_S, 512 experts, `data/expert-profile.bin`.  Only `config.N_EXPERT`
differs in the geometry (`check_metadata` refuses a GGUF of the other variant).  Per-variant data: models/<model>/,
packs/<coder-iq1_m | iq3_s>, and for iq3s quant-iq3s/, cache/tt-iq3s, golden-iq3s/, eval-iq3s/ (the Coder's keep
quant/, cache/tt, golden/, eval/; cache/dequant is per GGUF stem; kcache and mtp/ are shared: the drafter is the full
model's MTP layer for both).  Shard 2 (the PLE table, sha256 316b46f3...) is the same file for every quant: the dev
host hardlinks the Coder's.  The tokenizer and chat template are byte-identical, so eval-iq3s reuses the Coder's
evalset.jsonl / gsm8k.jsonl.  iq3s has no GPTQ store (no CUDA GPU on the host): its device plans come from
`fnx.quant.datafree` (search-BFP4 rows of the GGUF values + profile-ranked residency, `search4-tiered-<budget>`), and
its quality reference is the cpu configuration's own teacher forcing (`fnx.eval.run --write-ref`).

Specification: `/home/lotto/.omp/agent/sessions/-tmp/2026-10-02T13-24-24-920Z_01a0fcc9-96d8-7578-b42f-1cffdb5ac6d2/local/flashnext-spec.md`
(`local://flashnext-spec.md`; Part 1 loading, 2 embedding+PLE, 3 HC+head, 4 GDN, 5 QSA, 6 MoE, 7 sampler+MTP,
8 serve protocol+oracles). TT reuse map: `local://tt-reuse.md` (same directory). Reusable TTNN code lives in the image
`qwen36-p150:vllm-mtp-pc-x4` under `/home/container_app_user/tt-metal/models/demos/blackhole/qwen36` (and an
extracted copy at `/var/tmp/qwen35-release/stage-x4/runtime/single/overlay/home/container_app_user/tt-metal`).

## Shared files (owner: Main; do not edit - message Main for changes)
- `fnx/gguf_load.py` - `Model(shard1, others=None)`: tensors of all shards (`others`: an explicit list of the other
  shards, the engine's `--ple-gguf`, instead of the `-0000N-of-` name pattern), `get(name)` f32 numpy (numpy order =
  GGUF ne reversed: a projection is `(n_out, n_in)`), `get_expert(name, e)`, `raw(name)`, `type_of`, `available`
  (`<shard>.done`: tools/pdl.py's finished-chunk list, or Strata setup's finish mark = the whole file is there).
- `fnx/weights.py` - `Weights(model=None, cache_dir=None, ple=None)`: `.get(name)` / `.layer(L, suffix)` dequantized
  f32, cached on disk under `fnx.paths.DEQUANT` (non-expert only).
- `fnx/config.py` - geometry constants, `is_qsa(L)`, `check_metadata`.
- `fnx/cpu_experts.py` - `CpuExperts(model, 48, n_threads).moe(L, x[T,2560], ids[T,10]) -> [T,10,2560]` UNWEIGHTED
  routed expert outputs on the host (Strata's own CPU kernels, libfnx_cpu.so). Negative id = skip.
  `capacity=` reserves extra layer slots; `add_layer(i, gate, up, down, gu_type, d_type)` registers in-memory expert
  stacks (e.g. the MTP layer's 512 Q2_0 experts). One instance per process. Test: `fnx/tests/test_cpu_experts.py`.
  Library: `cpu_experts/build.sh` builds `cpu_experts/build-tt/libfnx_cpu.so` (`fnx.paths.CPU_LIB`, FNX_CPU_LIB
  overrides) in the TT image (gcc 12 / glibc 2.35: loads in the container and on newer hosts; ggml-cpu
  `-march=native`: build on the serving machine) from this repository's include/ + src/ (CMake STRATA_SRC = the
  repository) and ggml from STRATA_GGML_DIR (`<repo>/third_party/llama.cpp`, else FNX_DATA/third_party/llama.cpp,
  else Strata's pinned fetch).  Every engine start needs it.  Strata 82f46a8 -> v0.1.40.4 changed the CPU kernels
  (AVX2 Q8_K activation quantizer #851, IQ3_S/IQ2_S AVX2 decode, pool task counts): outputs of the v0.1.40.4 build are
  bitwise the 82f46a8 build's for every layer at T=1/4/11 (2026-10-08).
- `tools/devrun.sh NAME script.py args` - runs in the TT image holding the shared device lock; `NODEV=1` for CPU-only.
  Device runs MUST go through it (another team shares the card). Keep device runs short.  Same mounts / image /
  lock as serve/tt-engine.sh; log FNX_DATA/logs/dev/NAME.log; image FNX_IMAGE (was IMAGE).

## Paths (fnx/paths.py: the only place a data location is written down)
- `REPO` = the repository containing tt/.  Strata resources come from it: `tools/` (gguf_reader, gguf_writer,
  strata_tokenizer; STRATA_TOOLS overrides), `data/expert-profile-coder.bin`, `src/kernels/ple_oracle_vectors.inc`,
  `CMakeLists.txt` (engine version).  The eval corpus file `src/kernels/cuda/qsa_select.cu` changed after 82f46a8:
  the gated version is pinned as `fnx/eval/corpus/qsa_select.cu` (sha256 997023ff..., the eval provenance's).
- `DATA` = FNX_DATA (absolute; default `DEV_DATA` = this dev machine's root, the only literal copy of it in fnx/,
  tools/, serve/).  Under it: `quant/store/{plans,pool}` (`STORE`; FNX_QUANT_DIR replaces quant/),
  `mtp/experts-bfp4-search.u8` (`MTP_BFP4`), `mtp/expert-profile.npy` (`MTP_PROFILE`), `cache/tt` (`TT_CACHE`;
  FNX_TT_CACHE), `cache/dequant` (`DEQUANT`; FNX_CACHE replaces cache/), `kcache` (TT_METAL_CACHE), `logs`.  Dev/test
  defaults resolve under it too (`SHARD1/SHARD2` models/coder, `PACK` packs/coder-iq1_m, `MTP_RT` mtp/rt, `GOLDEN`,
  `EVAL`, `VENV_PY`, `STRATA_REF` = the pinned 82f46a8 checkout of tools/golden.sh and fnx.eval.strata_ref, `LLAMA_CPP`).
  A fresh install fetches the artifact set into FNX_DATA (tools/tt_artifacts.py, artifacts.json); the caches and the
  kernel cache fill on the first start.
- `IMAGE` (FNX_IMAGE), `DEVICE_LOCK` (FNX_DEVICE_LOCK), `TT_METAL` (inside the image).  `python3 fnx/paths.py --shell
  [args]` prints them, DATA/KCACHE/LOGS/CPU_LIB and `MOUNTS=(...)` for the shell wrappers (devrun.sh, tt-engine.sh,
  validate_all.sh, golden.sh, fnx-serve.sh, cpu_experts/build.sh).

## Numerics target
Strata's DECODE (verify-window) semantics, compared with tolerances (spec §A.3, §A.5). The numpy references are
float32 with dequantized weights (no q8_1 activation emulation). TT matmuls for parity work: an explicit
`ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)`
(the default fidelity gave 1% error even with bf16 weights). Dense GGUF-quantized weights -> bfloat8_b; BF16 GGUF
weights -> bfloat16; norms/conv/small vectors -> bfloat16 or float32. The residual stack stays float32.
Measured decisions: the MoE input `mixed` stays float32 (bf16 flipped 4/320 routings); the shared expert runs in
bfloat16 (`Moe(..., shared_dtype=ttnn.bfloat16)`: bo error 2.4e-3 vs 1.03e-2 with bfloat8_b, +220 MB, hidden behind
the CPU experts). Device bf16 x bf16 -> f32 matmuls at HiFi4 + fp32 accumulation have a ~3e-4 relative floor.
CpuExperts: use 16 threads (persistent pool; T=1 ~0.48 ms per layer).

## Shapes and layouts (T tokens per call, Tp = T padded to 32)
- Numpy refs: `R` f32 `[T, 4, 2560]`; hidden `[T, 2560]`.
- TTNN: hidden `[1, 1, Tp, 2560]` bfloat16 TILE; residual `R` `[1, 1, Tp, 10240]` float32 TILE, column = stream*2560+d.
  Rows >= T are padding: contents unspecified, never read back as results.
- Every TT module takes `(mesh, w: fnx.weights.Weights, ...)`, builds its device weights in `__init__` (cache
  converted tensors with `ttnn.as_tensor(..., cache_file_name=f"{TT_CACHE}/{name}_{dtype}")`, `fnx.paths.TT_CACHE`:
  FNX_TT_CACHE, default FNX_DATA/cache/tt), and keeps its own per-sequence state with a `reset()` method.

## Modules (owner in brackets)
| ref (numpy) | TTNN | API |
|---|---|---|
| `ref/hc.py` [HcPle] | `tt/hc.py` | `hc_read(R, g, Wd, Wu, Wi|None) -> (mixed [T,2560], inj [T,4]|None)`, `hc_write(R, bo, inj) -> R`. TT: `HcRead(mesh, w, prefix)` with prefix `blk.{L}.hc_attn` / `blk.{L}.hc_ffn` / `output_hc`; `.forward(R) -> (mixed, inj)` (inj `[1,1,Tp,32]` f32, cols 0..3 valid); `hc_write(R, bo, inj) -> R` |
| `ref/ple.py` [HcPle] | `tt/ple.py` | Rollback: `PleState.snapshot(t)/commit(n_keep)`, `Ple.commit(n_keep)/drop_window()`. `ngram_rows(tokens, prev)`, `PleTable(model)` (host row gather, IQ4_NL 90 B rows), `PleState`, `ple_block(w, e_ple [T,2560], R, state) -> R`. TT: `Ple(mesh, w)`: `.forward(e_ple_dev, R) -> R`, host gathers rows |
| `ref/gdn.py` [Gdn] | `tt/gdn.py` | `GdnState`, `gdn(w, L, mixed [T,2560], state) -> bo [T,2560]` (T tokens in sequence). TT: `Gdn(mesh, w, L)`: `.forward(mixed, snap=False) -> bo`, `.rollback(n_keep)` (after a snap=True verify window: one kernel run from the row-0 snapshot + saved rows, no weight re-read; bitwise the n_keep-row forward), `.reset()`. Model commits T=4 1/2/3 rows: 0.96/1.42/1.99 ms (was 6.74/7.40/8.13). Kernel scalars go UNPACK->MATH only (`math_word`): `read_tile_value` also fed PACK, whose unconsumed mailbox loads could deadlock against its own queued c_gb pack (Blackhole load/store ordering; timing-dependent hang, fixed 2026-10-07). |
| `ref/qsa.py` [Qsa] | `tt/qsa.py` | `QsaState`, `qsa(w, L, mixed [T,2560], pos [T], state) -> bo`. TT: `Qsa(mesh, w, L, max_ctx)`: `.forward(mixed, pos) -> bo`, `.reset()`. Dense while n_kv <= 2051 (qsa.hpp:133-136); beyond that Strata's 2051-cell indexer selection, exact: device query + block scores, host-side exact indexer state and bounded refinement of undecided blocks, gathered sparse attention. One host round trip per call; all state is positional (`state_tensors()` returns []). Verified to 32,800 positions |
| `ref/moe.py` [Moe] | `tt/moe.py` | `moe(w, L, mixed [T,2560], experts: CpuExperts) -> bo`. TT: `Moe(mesh, w, L, experts)`: router + top-10 + shared expert on device, routed experts via `CpuExperts` (read back rows+ids+weights, compute, upload, combine) |
| `ref/mtp.py` [Mtp] | `tt/mtp.py` | MTP draft layer (weights from Strata's mtp_rt form: the engine's `--mtp <data>/mtp/rt`, default `fnx.paths.MTP_RT`; `mtp-q2_0.gguf` beside it). Ref: `MtpWeights(rt)`, `MtpRef(w, mw, experts).prefill/draft`. TT: `Mtp(mesh, w, cx, mw, max_ctx).prefill(R, next_tokens, cell0)` / `.draft(R, outv, p, a, steps, min_p) -> (drafts, probs)`; input R = the main model's final 4-stream residual before output_hc; positional KV (`state_tensors()` == []). Fully on device since wave 3: its 512 routed experts BFP4 (exponent search from Q2_0, `FNX_DATA/mtp/experts-bfp4-search.u8`, routing profile `FNX_DATA/mtp/expert-profile.npy`) through MoeDevice, the hottest `FNX_MTP_RESIDENT` (default 254) in DRAM, the rest + embedding rows in reserved items of TtModel's pool (`TtModel(..., host_reserve=fnx.tt.mtp.reserve_items())`, `make_drafter(mesh, w, max_ctx, k, min_p, model=model)`); a draft round is one captured trace (~10 ms device); ~1.2-1.3 GB DRAM |
| (`ref/moe.py`) | `tt/moe_device.py` [DeviceMoE] | All-device MoE, same API as Moe (`forward(mixed) -> bo`): router + fp32 top-10, gate/up, down + expert sum with the shared expert folded in; 3 programs, no host readback (single-trace capable). All 256 experts resident per layer, BFP4/BFP2 per matrix from a plan (`quant_layer(w, L, Plan(...))`; store `fnx/quant/store.py`). T=1 0.14 ms/layer. Host tier: experts not resident in DRAM live in one pinned host pool per TtModel (`fnx/tt/hostmem.py` HostPool, chunk="auto" (FNX_HOST_CHUNK): 1 GiB hugetlb pages, then 2 MiB THP chunks, then a 4 KiB-page region with a per-page bus-address table on the device (readers use `hostmem_read_paged`), so it fails only for lack of RAM, never for lack of huge pages; bitwise identical across modes, +0.17 ms/step with ~half the items paged; ~13.2 GB pinned; the readers use `kernels/hostmem/hostmem.h` hostmem_read over PCIe, 26-32 GB/s, trace-safe); weights load straight from the GPTQ store rows (no converted cache copy). **Default since 2026-10-07** (`FNX_MOE=device`, plan `bfp4-tiered-2.2e+10`: 7957 resident experts, 81.6% of routings; 24.4e9 does not fit beside 32K + MTP). Gate: FNX_DATA/eval/runs/device-gptq (all dPPL +0.10%, chat +0.27%, code -0.16%; top-1 .938, KL .038; GSM8K-50 44/50 = Strata). `FNX_MOE=cpu` = the Strata-exact path Data movement (MoeDualNoc, 2026-10-08, bitwise): the router's route core writes each row's entry page (route row + its experts' table entries, `route()` 4th value) so stages A/B read T small pages instead of every core reading the expert table from one bank; stage A/B DRAM weight blocks are split over NOC0 + NOC1 (DM_DYNAMIC_NOC, stream.h `BlockStream<DEPTH, true>`; the router stays single-NoC). DRAM-only layer T=1 0.170 -> 0.133 ms, T=4 0.382 -> 0.318 ms; host-tier layers stay PCIe-bound. |
| `ref/model.py` [Model] | `tt/model.py` [Model] | embedding (IQ4_XS row gather on host), PLE wiring, 48 layers, `output_hc` read, Q6_K head, greedy; `forward(tokens, positions) -> logits`; per-layer residual dumps for bisection |
| (matmul) | `tt/packed_matmul.py` [PackedRead] | Bank-packed dual-NoC decode matmul (port of the Qwen3.6-35B x6 PackedDecodeMatmul). `PackedMatmul(w, program_config, ckc, dtype, memory_config)(x [1,1,T<=32,K]) -> [1,1,T,N]`: the stock 1D-mcast program (MatmulMultiCoreReuseMcast1DProgramFactory descriptor: same compute kernel, compile args, CBs, in0 multicast) with only the in1 reader swapped for `kernels/packed/packed_in1_reader.cpp` (the image's packed LM-head reader; K-group size generalized, K = 80 tiles uses groups of 2), every read split over both NoCs (DM_DYNAMIC_NOC); descriptors cached by (input, output) address (eager host cost +20 us/call). `w` holds its TILE pages worker-major / bank-rotating (`PackedLayout(k_tiles, n_tiles, per_core_N[, group])`), placed by exact on-device page copies (`place_packed(part, dst, col0, layout)`; no extra DRAM: dram_budget TtModel/drafter allocation unchanged, `layout.unpack(host values)` for parity hooks). Bitwise the stock matmul (`fnx/tests/test_packed.py`, rows 1-4/32, 33-128 as 32-row slices). Requires per_core_M 1, out block = per_core_N, K tiles % (8 group) == 0, interleaved DRAM BFP8/BFP4 weight. Users (traced us/call, stock -> packed): LM head 2001 -> 1653 (gptq4 1543 -> 965), MTP draft head 932 -> 707 (one joined [2560, 106304] matmul, ttnn's auto config, replaces 4 column-group matmuls + concat), GDN mega 119.7 -> 110.7 (MEGA_PC 11x10 x 5 columns). test_trace bench 2026-10-07 (load ~6): MTP 45.1/49.5/48.5 -> 46.6/50.8/49.4 tok/s, T=1 38.54 -> 37.62 ms, T=4 59.22 -> 58.61 ms, same tokens/acceptance. Single-NoC packed reads were slower than stock on the 35B; out_proj-sized matmuls (per_core_N 2) lose. |
| (argmax) | `tt/tile_argmax.py` [TileArgmax] | `TileArgmax(mesh)(logits[, keep_rm])`: row argmax of float32 TILE logits [1,1,T,V] (T <= 32) without the untilize (port of the Qwen3.6-35B x6 TileArgmax). One program on the 11x10 rectangle (BRISC + NCRISC scan contiguous tile ranges reading only the T face rows; row r gathered on core r/2). ids uint32 [1,1,T,1] ROW_MAJOR, bitwise ttnn.argmax's (first maximum, NaN never, -0 == +0, 0 if nothing exceeds -inf). keep_rm also returns the rows row-major as [1,1,T*k,V/k] (k = 8 head / 11 draft; read flattened). Used by Head.argmax (T <= 32; stock to_layout + argmax above) and Mtp.rest. Traced us/call stock -> new: head T=1 169.7 -> 23.9 (keep_rm 29.6), T=4 415.5 -> 37.5 (keep_rm 50.2), draft T=1 77.5 -> 12.3. Test: fnx/tests/test_tile_argmax.py. |

Tests: `fnx/tests/test_<module>.py`, runnable as a script through `tools/devrun.sh`, comparing TT against the numpy
reference on REAL weights (layers available on disk) with realistic inputs (not just N(0,1): e.g. scale of real
activations). Print max/rel error and the pass threshold used. No mocks.

## Per-sequence state
Every module with per-sequence device state exposes `state_tensors() -> list` of the in-place device tensors that
`reset()` zeroes (fixed addresses). TtModel.snapshot()/restore() copy exactly those to/from the host.

## Serving (owner: Serve, wave 2)
- `fnx/serve_engine.py`: Strata's engine stdin/stdout protocol on TtModel (READY <ctx>, GEN/PP/T/DONE, STOP, QUIT, ERR);
  context up to 32768 (config `--max-context`; DRAM allows ~64K with the default expert placement; YaRN above 262144 not implemented), prompt chunks of 128 rows, conversation reuse + up to 6 prompt snapshots, greedy on
  device, Strata's sampled chain (incl. its Philox draw) on the host from read-back logits.
- Engine invocation (what `setup.py --backend tt` writes into the run config; `tt/setup_tt.py` owns that side):
  `exe` = `<repo>/tt/serve/tt-engine.sh`, `cwd` = `<repo>`, `args` = `--pack <data>/packs/coder-iq1_m --native <shard1>
  --ple-gguf <shard2> --mtp <data>/mtp/rt --max-context N (<= 32768) --prefill <number> --spec 4 --spec-min-p 0.5`,
  `env` = `{FNX_DATA: <data>/tt[, FNX_IMAGE]}`, `lib_dirs` [], `backend` "tt".  The engine reads shard 1 + the PLE shard
  (an explicit shard list), the drafter from `--mtp` (its `mtp-q2_0.gguf` beside it only to rebuild the BFP4 file),
  everything of its own from FNX_DATA; `--pack` is the server's tokenizer pack (accepted; the engine reads no tensors
  from it); any other Strata flag is logged as ignored.  `INFO ... engine=<version>`: the repository's CMakeLists.txt
  `project(strata VERSION ...)` (today 0.1.40.4, numeric as setup's `engine_version` and the server's Monitor parse
  it); setup writes `tt/serve/BUILD.json` {"backend":"tt","source":"docker","image","version","lib_dirs":[]} beside the
  wrapper.
- `serve/tt-engine.sh` (successor of tools/fnx-engine): the engine exe for serve/server.py; it takes the device lock
  ITSELF (never run it inside devrun.sh): `FNX_DEVICE_LOCK` (default /var/tmp/gemma4-12b/device.lock), then waits for
  device users that bypassed it, then `docker run -i --rm --init` of `FNX_IMAGE` (default
  `lottolabs/qwen36-35b-tt-p150@sha256:118e0d90...`, the image tagged qwen36-p150:vllm-mtp-pc-x4 here) as its child,
  container `tt-engine-<pid>`, as the caller's uid:gid.  Mounts at the same path: the repository (cwd `<repo>/tt`,
  PYTHONPATH tt-metal + `<repo>/tt`), FNX_DATA and its override dirs, the parent of every absolute path argument
  (`fnx.paths.mounts`).  Forwards FNX_*/STRATA_*/TT_METAL_LOGGER*/TTNN_*.  TERM/HUP -> `docker kill TERM`, after 20 s
  `docker rm -f` + `tt-smi -r 0` under the lock; SIGINT is ignored once the engine runs and the docker client runs
  under setsid (a terminal's Ctrl+C reaches the server, whose QUIT/EOF ends the engine).  Refuses to start without
  `libfnx_cpu.so` (`cpu_experts/build.sh`).  Start bound: the server's READY timeout (#1317, `STRATA_ENGINE_READY_S`,
  SIGKILLs the engine) must be OFF (0: setup's run script, `tools/fnx-serve.sh`) because the start includes an
  unbounded wait for the shared lock; tt-engine.sh bounds the start after the lock is held (`FNX_READY_S`, 900 s; the
  engine touches `FNX_READY_FILE` = FNX_DATA/logs/tt-engine-<pid>.ready at READY), then ends the engine like a SIGTERM
  (container removed, board reset).
- `tools/fnx-serve.sh`: development launcher, `CONFIG=<run config> [PORT=8090] tools/fnx-serve.sh` runs this
  repository's serve/server.py with the dev venv (`PY`, default FNX_DATA/venv/bin/python) and STRATA_ENGINE_READY_S=0.
  A setup install starts through its run-<tag>.sh (tt/serve/server_tt.py) instead.  The dev machine's live server
  config `/var/tmp/strata-tt/strata-fnx-coder.json` (port 8090, idle_unload_s 900) is this tree's dev config
  (`fnx/tests/test_serve.py` dev_config form); `/var/tmp/strata-tt/flashnext_tt` is a symlink to this `tt/` since
  2026-10-08 (the pre-import copy: flashnext_tt.retired-20261008).
- TtModel additions: `forward(..., head=True, read_logits=True, keep_R=False, verify=False) -> (logits, ids[, R])` (R = final 4-stream residual before output_hc), `commit(n_keep)` (GDN shadow restore + `GdnFused.replay` of the kept rows, `Ple.commit`, pos), `snapshot()` / `restore()` (restore drops the PLE window; snapshots only at committed states).
- Speculative decoding (MTP) is ON by default: engine `--spec N` (default 4 = next token + 3 drafts; `FNX_SPEC`; 1 = plain),
  `--spec-min-p`; READY INFO reports spec/mtp_max. Exact-match acceptance; output equals plain decode bitwise because every
  module is row-invariant (QSA attention matmuls use `packer_l1_acc=False`; checked by `fnx/tests/test_spec.py` qsa/inv).
  Measured (CPU experts): chat 7.2 -> 12.3, code 7.0 -> 17.5, doc 6.7 -> 14.8 tok/s.
- `fnx/tests/test_serve.py` runs on the HOST (venv python, cwd tt/), not via devrun.sh: `--config` a Strata run config
  (default: written to FNX_DATA/logs/test-serve-config.json from fnx.paths' dev data with serve/tt-engine.sh and
  explicit --pack/--native/--ple-gguf/--mtp), protocol part = the config's exe + model-file args + env, HTTP part =
  this repository's serve/server.py via tools/fnx-serve.sh; 17 checks (P1 checks INFO engine= the repo version).
- Serving default MoE is `device` (above). Measured traced tok/s, chat / code / doc, final wave-3 tree 2026-10-07
  (load 3-8): device plain 29.3 / 28.0 / 25.7, MTP 45.1 / 49.5 / 48.5 (T=1 38.5 ms, T=4 59.2 ms); CPU path plain
  16.6 / 16.3 / 15.7, MTP 23.3 / 26.4 / 26.0.  Quality gate (device-final = default): all dPPL +0.09%, chat +0.37%,
  code -0.18%, top-1 .937, KL .038, GSM8K-50 45/50 (Strata 3090 44/50).  validate_all 12/12: logs/validate/20261007-153853.
  After the 35B-x6 ports (packed dual-NoC head/draft head/GDN mega, MoE entry pages + dual-NoC streams, TileArgmax;
  all bitwise, 2026-10-08, load 7-17): device plain 30.9 / 29.2 / 26.6, MTP 46.9 / 50.9 / 50.0 (T=1 37.47 ms, T=4 58.01 ms).
  Profile T=1 37.6 ms / T=4 49.1 ms (logs/dev/ta-prof-t*.log): the host-tier experts over PCIe cost 8.4 / 14.1 ms of it,
  the largest remaining item (residency is DRAM-bound; see tools/dram_budget.py).
  Moved into the Strata repository (tt-p150 44ff376, 2026-10-08; logs FNX_DATA/logs/tt-tree-verify.txt): FNX_MOE=cpu
  test_trace plain spec sparse PASS with result lines identical to the pre-move run (logs/dev/tt-tree-trace-cpu.log vs
  ta-trace-cpu.log, timings aside); device test_trace bench PASS, tokens equal, acceptance 26/45, 44/57, 45/54 as
  ta-bench.log, traced plain 30.6 / 29.1 / 25.8, MTP 47.0 / 51.0 / 49.7 tok/s, T=1 37.54 ms, T=4 58.16 ms; test_serve
  PASS 17/17 through serve/tt-engine.sh with explicit model args (P3 decode 47.6 tok/s, H0 ready after 108 s).
- Quality configs in `fnx/eval/run.py`: `cpu`, `bfp8` (both FNX_MOE=cpu), `device-gptq`.
- Data-free BFP4 (`fnx.quant.datafree`, no calibration) gated on the Coder 2026-10-10 (run `device-search4`, plan
  search4-tiered-2.4e+10 in a scratch FNX_QUANT_DIR, since deleted): all dPPL +0.38% (GPTQ +0.09%), chat +2.04%
  (+0.37%), code -0.27% (-0.18%), top-1 .927 (.937), KL .0475 (.038; limit .057), GSM8K-50 44/50 (Strata 44/50).  The
  Coder keeps GPTQ; data-free is the iq3s variant's device store (no CUDA host for GPTQ).
- Default expert plan `bfp4-tiered-2.4e+10` (fnx/tt/model.py DEFAULT_PLAN): 8680 resident experts (24.0e9 B, 85.9% of
  routings). Size any new default with `tools/devrun.sh dram tools/dram_budget.py [--plan P]` (serving-exact build: 32K
  ctx, MTP drafter, warm-up, traces; prints the budget that keeps 1.5 GB free). Trace region 480 MiB (FNX_TRACE_MB;
  serving traces use 0.200 GB). Placement changes are numerics-neutral (all experts BFP4 either side).
- h2 plans (BFP2 host-tier experts, `fnx.quant.alloc --tiered 2.4e10 --h2 F`): OPT-IN only. h2-25 gated 2026-10-07:
  all dPPL +2.08% (default +0.10%), chat +1.13%, code +1.23%, book +4.33%, KL .079 vs .038 (limit .057), GSM8K 46/50 ->
  fails. The alloc KL predictor underestimates BFP2 damage ~7x (pred +0.006, measured +0.041). Max reachable cut 44%
  (BFP2 rows exist for each layer's coldest 40% only). h2-40: all +3.07%, chat +2.70%, KL .110, GSM8K 40/50 -> fails.
  Curve (24.0 GB plan, new GDN, traced MTP chat/code/doc tok/s): h2-0 45.0/49.2/48.4 (T=1 38.7 ms), h2-25 47.0/59.3/51.7
  (35.4 ms), h2-40 48.0/56.1/54.3 (34.1 ms).  Since 2026-10-10 the dev store holds no BFP2 rows and no h2 / a-28.5 /
  a-30.0 plans (deleted for disk space); rebuilding them needs `fnx.quant.pipeline quantize --pool2 F` (CUDA) and then
  `fnx.quant.alloc --h2`.
- iq3s (FNX_MODEL=iq3s), gated 2026-10-10 against its own cpu-config reference (`fnx.eval.run --config cpu
  --write-ref`, FNX_DATA/eval-iq3s; the TT cpu path matched fnx.ref.model.RefModel on chat-ultrachat-test-1, mean NLL
  .844 vs .849).  Default MoE `cpu` (fnx.paths.DEFAULT_MOE) with fnx.tt.moe.DEFAULT_RESIDENT = 4671 bfloat8_b experts
  on the device (Strata's full-model profile, hottest first; `expert_placement.py --measure`: the iq3s model takes
  5.78 GB of DRAM): all dPPL +0.63%, top-1 .974, KL .0067, GSM8K-50 46 (exact cpu 47).  `FNX_MOE=device`
  (search4-tiered-2.4e+10, `fnx.quant.datafree`; DRAM budget 2.403e10 measured, 50.6 GB host RAM pinned): all -1.55%
  (chat -6.45%, code +0.33%, instruct +0.09%, book +0.17%), top-1 .916, KL .061, GSM8K-50 46.  Traced bench (golden
  prompts, MTP chat/code/doc, load 2-20): cpu 33.8/29.7/33.5 tok/s (plain 19.0/18.6/18.3), device 28.0/28.8/33.5
  (plain 18.1/18.0/23.7); served general chat (warm, MTP): cpu ~20-22, device ~25-28 tok/s; both test_trace bench PASS.
  The full model vs the Coder (whose 256 experts are byte-identical to 256 of iq3s's 512; dense tensors identical):
  better on code and prose (book NLL 1.40/1.15/0.90 vs 1.90/1.60/1.44), GSM8K-20 19 vs 17, worse on ultrachat replies
  (chat PPL 5.39 vs 3.43: it puts more mass on ending the turn mid-reply, e.g. `<|im_end|>` after "to get you").
  The device MoE drops the page cache of the store and GGUF before pinning (fnx.tt.hostmem.drop_page_cache): with tens
  of GB cached, the 44 GB host tier fell back to 4 KiB pins and ran out of KMD fds.
- iq3s speed, 2026-10-11 (cpu MoE).  CPU experts are DRAM-bandwidth-bound (scratch bench: ~46 GB/s of IQ3_S expert
  bytes at 12-24 threads, no gain past 12), so only fewer CPU bytes help.  (1) Residency by measured routing counts
  (tools/route_counts.py: 64 calibration + 16 held-out sequences through the cpu MoE, FNX_QUANT_DIR/stats{,-held}):
  held-out hit rate at 4671 experts 0.41 vs 0.25 with Strata's profile; datafree plans and the cpu default use it.
  (2) Native routing (cpu_experts/fnx_route.cpp via fnx.cpu_experts.route, used by Moe.plan; also writes the
  device-expert metadata row): identical ids / weights / words to the numpy version on 2000 synthetic calls incl. ties
  and near ties; traced route+upload host time per token 11.4 -> 2.8 ms (T=4 rounds); the Coder's cpu path still
  reproduces Strata's 50 golden tokens (test_serve P3).  (3) BFP4 residents (placement "dtype": "bfloat4_b";
  DeviceExperts snaps them with fnx.tt.moe.snap_bfp4, min-MSE blocks along ttnn's 16-element rows): 8823 experts in
  the same 24.4 GB, held-out hit 0.62.  Traced MTP chat/code/doc: bfp8-4671 33.8/34.9/29.9 -> bfp4-8823 42.6/39.8/36.7
  tok/s (plain 19.5/19.4/18.3 -> 23.2/22.8/21.9; load 19-25); device time per T=1 forward 38 ms (layers 25.7, shared
  + resident experts 10.6, head 1.8).  Gate (run cpu-s4): all dPPL -2.21% (chat -4.58%, code -0.00%, instruct -6.44%,
  book -1.02%), top-1 .925, KL .0437, GSM8K-50 46 (bfp8-profile .0067 / 46; exact 47): KL within .057, perplexity
  moves beyond 2% - downward, against the exact experts.  The default stays bfp8-4671 by counts, gated 2026-10-11 (run
  cpu-b8-counts): all +0.05% (chat +1.73%, code -0.18%, instruct -3.01%, book +0.03%), top-1 .974, KL .0066, GSM8K-50
  46.
- LM head: `FNX_HEAD=bfp8` (default) | `gptq4` (GPTQ-BFP4 head from `fnx/quant/head.py`; 0.69 ms/token (packed heads:
  1.65 vs 0.97 ms traced) + 0.34 GB DRAM saved). Gated 2026-10-07 on the cpu path: all dPPL +0.17% -> +0.73% (chat +1.19%), top-1 .970 -> .951, KL .008 -> .014;
  GSM8K-50 47/50 (bfp8 head 45/50, Strata 44/50: within n=50 noise); kept OPT-IN (cost/benefit). Every head mode places its 80 uploaded blocks by exact page copies into the bank-packed weight (`fnx.tt.packed_matmul.place_packed`; matmul `PackedMatmul`, bitwise the stock decode matmul and, for rows > 32, the old auto-config call); the old bfp8 concat + typecast
  join re-blocked through BF16 and moved the device weights off their bfp8 values (logits 6.2e-3 off; fixed 2026-10-07). Quantized experts: `fnx/quant/*`,
  store `fnx.paths.STORE` = FNX_DATA/quant/store (GPTQ BFP4 rows for all 12,288 experts, `Plan(name)`).

## Tracing (owner: Tracing)
- `fnx/tt/trace.py`: the decode step runs as captured device segments between the host round trips (MoE router readback +
  CPU experts per layer, QSA indexer readback per QSA layer, head, PLE row upload); the MTP draft round is traced too.
  Default in serve_engine; `FNX_TRACE=0` forces eager. Outputs are bitwise equal to eager (`fnx/tests/test_trace.py`).
- `TtModel.enable_trace(Ts, spans, verbose, drafter)`, `prepare_decode()` (call after a prompt read), `free_R()`
  (residuals from `keep_R=True` MUST be released through it: a recapture may own them).
- Rule (ttnn #48536): a program compiled after a trace capture can corrupt the trace and hang the next replay. The tracer
  recaptures every trace whenever the program cache grew since the last capture (~1.4 s, only after a prompt read with a
  new shape); `serve_engine.warm_up` compiles every decode-time shape (windows 1..4, all partial commits, drafter
  catch-up/resume) before the first capture, so decode never recaptures.
- QSA KV append kernel: `fnx/tt/kernels/qsa/kv_write.cpp`.
- With `FNX_MOE=device` a dense decode step / MTP verify window is ONE captured segment (CPU path: 97); sparse positions
  (> 2050) have 13 (one exact QSA host-indexer cut per QSA layer). The MTP drafter's 512 Q2_0 experts stay on the CPU.
- `tools/profile_step.py`: per-segment and per-op-type device profile of the traced T=1 step (use it before/after any
  device optimization; HcNoc's ranked profile: gdn 22.6%, moe_shared 16.1%, hc_read 15.3%, qsa 16.4%, head 5.8%).
- Device hygiene: `serve/tt-engine.sh` runs the container as a child under the device lock, forwards signals, and on a hang
  force-removes it and resets the board (`tt-smi -r 0`) while still holding the lock. Never reset the board without the lock.
- `tools/devrun.sh` watchdog: `FNX_DEV_TIMEOUT` seconds (default 7200). A run still alive then is removed and, for device
  runs, the board is reset under the still-held lock (rc 124). Set it close to your run's expected length for new device
  paths (a hung kernel otherwise holds the shared card until someone notices).
