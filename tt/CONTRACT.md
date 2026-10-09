# Flash-Next on the P150: module contract

Package `fnx` at `/var/tmp/strata-tt/flashnext_tt/fnx` (`import fnx`; run on the device with `tools/devrun.sh`).
Model: Qwen3.8-Flash-Next Coder IQ1_M GGUF, `/var/tmp/strata-tt/models/coder/` (still downloading: check
`Model.available(name[, expert])`; layers 0-9, `output.*`, `token_embd` are complete; shard 2's PLE table is not yet).

Specification: `/home/lotto/.omp/agent/sessions/-tmp/2026-10-02T13-24-24-920Z_01a0fcc9-96d8-7578-b42f-1cffdb5ac6d2/local/flashnext-spec.md`
(`local://flashnext-spec.md`; Part 1 loading, 2 embedding+PLE, 3 HC+head, 4 GDN, 5 QSA, 6 MoE, 7 sampler+MTP,
8 serve protocol+oracles). TT reuse map: `local://tt-reuse.md` (same directory). Reusable TTNN code lives in the image
`qwen36-p150:vllm-mtp-pc-x4` under `/home/container_app_user/tt-metal/models/demos/blackhole/qwen36` (and an
extracted copy at `/var/tmp/qwen35-release/stage-x4/runtime/single/overlay/home/container_app_user/tt-metal`).

## Shared files (owner: Main; do not edit - message Main for changes)
- `fnx/gguf_load.py` - `Model(shard1)`: tensors of all shards, `get(name)` f32 numpy (numpy order = GGUF ne reversed:
  a projection is `(n_out, n_in)`), `get_expert(name, e)`, `raw(name)`, `type_of`, `available`.
- `fnx/weights.py` - `Weights()`: `.get(name)` / `.layer(L, suffix)` dequantized f32, cached on disk (non-expert only).
- `fnx/config.py` - geometry constants, `is_qsa(L)`, `check_metadata`.
- `fnx/cpu_experts.py` - `CpuExperts(model, 48, n_threads).moe(L, x[T,2560], ids[T,10]) -> [T,10,2560]` UNWEIGHTED
  routed expert outputs on the host (Strata's own CPU kernels, libfnx_cpu.so). Negative id = skip.
  `capacity=` reserves extra layer slots; `add_layer(i, gate, up, down, gu_type, d_type)` registers in-memory expert
  stacks (e.g. the MTP layer's 512 Q2_0 experts). One instance per process. Test: `fnx/tests/test_cpu_experts.py`.
- `tools/devrun.sh NAME script.py args` - runs in the TT image holding the shared device lock; `NODEV=1` for CPU-only.
  Device runs MUST go through it (another team shares the card). Keep device runs short.

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
  converted tensors with `ttnn.as_tensor(..., cache_file_name=f"{FNX_TT_CACHE}/{name}_{dtype}")`, FNX_TT_CACHE default
  `/var/tmp/strata-tt/cache/tt`), and keeps its own per-sequence state with a `reset()` method.

## Modules (owner in brackets)
| ref (numpy) | TTNN | API |
|---|---|---|
| `ref/hc.py` [HcPle] | `tt/hc.py` | `hc_read(R, g, Wd, Wu, Wi|None) -> (mixed [T,2560], inj [T,4]|None)`, `hc_write(R, bo, inj) -> R`. TT: `HcRead(mesh, w, prefix)` with prefix `blk.{L}.hc_attn` / `blk.{L}.hc_ffn` / `output_hc`; `.forward(R) -> (mixed, inj)` (inj `[1,1,Tp,32]` f32, cols 0..3 valid); `hc_write(R, bo, inj) -> R` |
| `ref/ple.py` [HcPle] | `tt/ple.py` | Rollback: `PleState.snapshot(t)/commit(n_keep)`, `Ple.commit(n_keep)/drop_window()`. `ngram_rows(tokens, prev)`, `PleTable(model)` (host row gather, IQ4_NL 90 B rows), `PleState`, `ple_block(w, e_ple [T,2560], R, state) -> R`. TT: `Ple(mesh, w)`: `.forward(e_ple_dev, R) -> R`, host gathers rows |
| `ref/gdn.py` [Gdn] | `tt/gdn.py` | `GdnState`, `gdn(w, L, mixed [T,2560], state) -> bo [T,2560]` (T tokens in sequence). TT: `Gdn(mesh, w, L)`: `.forward(mixed, snap=False) -> bo`, `.rollback(n_keep)` (after a snap=True verify window: one kernel run from the row-0 snapshot + saved rows, no weight re-read; bitwise the n_keep-row forward), `.reset()`. Model commits T=4 1/2/3 rows: 0.96/1.42/1.99 ms (was 6.74/7.40/8.13). Kernel scalars go UNPACK->MATH only (`math_word`): `read_tile_value` also fed PACK, whose unconsumed mailbox loads could deadlock against its own queued c_gb pack (Blackhole load/store ordering; timing-dependent hang, fixed 2026-10-07). |
| `ref/qsa.py` [Qsa] | `tt/qsa.py` | `QsaState`, `qsa(w, L, mixed [T,2560], pos [T], state) -> bo`. TT: `Qsa(mesh, w, L, max_ctx)`: `.forward(mixed, pos) -> bo`, `.reset()`. Dense while n_kv <= 2051 (qsa.hpp:133-136); beyond that Strata's 2051-cell indexer selection, exact: device query + block scores, host-side exact indexer state and bounded refinement of undecided blocks, gathered sparse attention. One host round trip per call; all state is positional (`state_tensors()` returns []). Verified to 32,800 positions |
| `ref/moe.py` [Moe] | `tt/moe.py` | `moe(w, L, mixed [T,2560], experts: CpuExperts) -> bo`. TT: `Moe(mesh, w, L, experts)`: router + top-10 + shared expert on device, routed experts via `CpuExperts` (read back rows+ids+weights, compute, upload, combine) |
| `ref/mtp.py` [Mtp] | `tt/mtp.py` | MTP draft layer (weights from `/var/tmp/strata-tt/mtp/rt`, Strata's mtp_rt form). Ref: `MtpWeights`, `MtpRef(w, mw, experts).prefill/draft`. TT: `Mtp(mesh, w, cx, mw, max_ctx).prefill(R, next_tokens, cell0)` / `.draft(R, outv, p, a, steps, min_p) -> (drafts, probs)`; input R = the main model's final 4-stream residual before output_hc; positional KV (`state_tensors()` == []). Fully on device since wave 3: its 512 routed experts BFP4 (exponent search from Q2_0, `/var/tmp/strata-tt/mtp/experts-bfp4-search.u8`) through MoeDevice, the hottest `FNX_MTP_RESIDENT` (default 254) in DRAM, the rest + embedding rows in reserved items of TtModel's pool (`TtModel(..., host_reserve=fnx.tt.mtp.reserve_items())`, `make_drafter(mesh, w, max_ctx, k, min_p, model=model)`); a draft round is one captured trace (~10 ms device); ~1.2-1.3 GB DRAM |
| (`ref/moe.py`) | `tt/moe_device.py` [DeviceMoE] | All-device MoE, same API as Moe (`forward(mixed) -> bo`): router + fp32 top-10, gate/up, down + expert sum with the shared expert folded in; 3 programs, no host readback (single-trace capable). All 256 experts resident per layer, BFP4/BFP2 per matrix from a plan (`quant_layer(w, L, Plan(...))`; store `fnx/quant/store.py`). T=1 0.14 ms/layer. Host tier: experts not resident in DRAM live in one pinned host pool per TtModel (`fnx/tt/hostmem.py` HostPool, chunk="auto" (FNX_HOST_CHUNK): 1 GiB hugetlb pages, then 2 MiB THP chunks, then a 4 KiB-page region with a per-page bus-address table on the device (readers use `hostmem_read_paged`), so it fails only for lack of RAM, never for lack of huge pages; bitwise identical across modes, +0.17 ms/step with ~half the items paged; ~13.2 GB pinned; the readers use `kernels/hostmem/hostmem.h` hostmem_read over PCIe, 26-32 GB/s, trace-safe); weights load straight from the GPTQ store rows (no converted cache copy). **Default since 2026-10-07** (`FNX_MOE=device`, plan `bfp4-tiered-2.2e+10`: 7957 resident experts, 81.6% of routings; 24.4e9 does not fit beside 32K + MTP). Gate: /var/tmp/strata-tt/eval/runs/device-gptq (all dPPL +0.10%, chat +0.27%, code -0.16%; top-1 .938, KL .038; GSM8K-50 44/50 = Strata). `FNX_MOE=cpu` = the Strata-exact path Data movement (MoeDualNoc, 2026-10-08, bitwise): the router's route core writes each row's entry page (route row + its experts' table entries, `route()` 4th value) so stages A/B read T small pages instead of every core reading the expert table from one bank; stage A/B DRAM weight blocks are split over NOC0 + NOC1 (DM_DYNAMIC_NOC, stream.h `BlockStream<DEPTH, true>`; the router stays single-NoC). DRAM-only layer T=1 0.170 -> 0.133 ms, T=4 0.382 -> 0.318 ms; host-tier layers stay PCIe-bound. |
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
- `tools/fnx-engine`: the engine exe for serve/server.py; it takes the device lock ITSELF (never run it inside
  devrun.sh). `tools/fnx-serve.sh` starts Strata's unchanged serve/server.py on port 8090 with
  `/var/tmp/strata-tt/strata-fnx-coder.json` (idle_unload_s 900 gives the shared card back after 15 idle minutes; while
  the model is loaded the server holds the device lock).  Serving tree: `/var/tmp/strata-tt/Strata-serve`, a git
  worktree of Strata at v0.1.40.4 (`STRATA_SERVE`, tag in fnx-serve.sh; created on first start); the main checkout
  `/var/tmp/strata-tt/Strata` stays at 82f46a8 (CUDA reference engine, CPU expert sources, tools, eval corpus file).
  Start bound: the server's READY timeout (#1317, `STRATA_ENGINE_READY_S`, SIGKILLs the engine) is OFF in fnx-serve.sh
  because the start includes an unbounded wait for the shared lock; fnx-engine bounds the start after the lock is held
  (`FNX_READY_S`, 900 s; the engine touches `FNX_READY_FILE` at READY), then ends the engine like a SIGTERM (container
  removed, board reset).
- TtModel additions: `forward(..., head=True, read_logits=True, keep_R=False, verify=False) -> (logits, ids[, R])` (R = final 4-stream residual before output_hc), `commit(n_keep)` (GDN shadow restore + `GdnFused.replay` of the kept rows, `Ple.commit`, pos), `snapshot()` / `restore()` (restore drops the PLE window; snapshots only at committed states).
- Speculative decoding (MTP) is ON by default: engine `--spec N` (default 4 = next token + 3 drafts; `FNX_SPEC`; 1 = plain),
  `--spec-min-p`; READY INFO reports spec/mtp_max. Exact-match acceptance; output equals plain decode bitwise because every
  module is row-invariant (QSA attention matmuls use `packer_l1_acc=False`; checked by `fnx/tests/test_spec.py` qsa/inv).
  Measured (CPU experts): chat 7.2 -> 12.3, code 7.0 -> 17.5, doc 6.7 -> 14.8 tok/s.
- `fnx/tests/test_serve.py` runs on the HOST (venv python), not via devrun.sh.
- Serving default MoE is `device` (above). Measured traced tok/s, chat / code / doc, final wave-3 tree 2026-10-07
  (load 3-8): device plain 29.3 / 28.0 / 25.7, MTP 45.1 / 49.5 / 48.5 (T=1 38.5 ms, T=4 59.2 ms); CPU path plain
  16.6 / 16.3 / 15.7, MTP 23.3 / 26.4 / 26.0.  Quality gate (device-final = default): all dPPL +0.09%, chat +0.37%,
  code -0.18%, top-1 .937, KL .038, GSM8K-50 45/50 (Strata 3090 44/50).  validate_all 12/12: logs/validate/20261007-153853.
  After the 35B-x6 ports (packed dual-NoC head/draft head/GDN mega, MoE entry pages + dual-NoC streams, TileArgmax;
  all bitwise, 2026-10-08, load 7-17): device plain 30.9 / 29.2 / 26.6, MTP 46.9 / 50.9 / 50.0 (T=1 37.47 ms, T=4 58.01 ms).
  Profile T=1 37.6 ms / T=4 49.1 ms (logs/dev/ta-prof-t*.log): the host-tier experts over PCIe cost 8.4 / 14.1 ms of it,
  the largest remaining item (residency is DRAM-bound; see tools/dram_budget.py).
- Quality configs in `fnx/eval/run.py`: `cpu`, `bfp8` (both FNX_MOE=cpu), `device-gptq`.
- Default expert plan `bfp4-tiered-2.4e+10` (fnx/tt/model.py DEFAULT_PLAN): 8680 resident experts (24.0e9 B, 85.9% of
  routings). Size any new default with `tools/devrun.sh dram tools/dram_budget.py [--plan P]` (serving-exact build: 32K
  ctx, MTP drafter, warm-up, traces; prints the budget that keeps 1.5 GB free). Trace region 480 MiB (FNX_TRACE_MB;
  serving traces use 0.200 GB). Placement changes are numerics-neutral (all experts BFP4 either side).
- h2 plans (BFP2 host-tier experts, `fnx.quant.alloc --tiered 2.4e10 --h2 F`): OPT-IN only. h2-25 gated 2026-10-07:
  all dPPL +2.08% (default +0.10%), chat +1.13%, code +1.23%, book +4.33%, KL .079 vs .038 (limit .057), GSM8K 46/50 ->
  fails. The alloc KL predictor underestimates BFP2 damage ~7x (pred +0.006, measured +0.041). Max reachable cut 44%
  (BFP2 rows exist for each layer's coldest 40% only). h2-40: all +3.07%, chat +2.70%, KL .110, GSM8K 40/50 -> fails.
  Curve (24.0 GB plan, new GDN, traced MTP chat/code/doc tok/s): h2-0 45.0/49.2/48.4 (T=1 38.7 ms), h2-25 47.0/59.3/51.7
  (35.4 ms), h2-40 48.0/56.1/54.3 (34.1 ms).
- LM head: `FNX_HEAD=bfp8` (default) | `gptq4` (GPTQ-BFP4 head from `fnx/quant/head.py`; 0.69 ms/token (packed heads:
  1.65 vs 0.97 ms traced) + 0.34 GB DRAM saved). Gated 2026-10-07 on the cpu path: all dPPL +0.17% -> +0.73% (chat +1.19%), top-1 .970 -> .951, KL .008 -> .014;
  GSM8K-50 47/50 (bfp8 head 45/50, Strata 44/50: within n=50 noise); kept OPT-IN (cost/benefit). Every head mode places its 80 uploaded blocks by exact page copies into the bank-packed weight (`fnx.tt.packed_matmul.place_packed`; matmul `PackedMatmul`, bitwise the stock decode matmul and, for rows > 32, the old auto-config call); the old bfp8 concat + typecast
  join re-blocked through BF16 and moved the device weights off their bfp8 values (logits 6.2e-3 off; fixed 2026-10-07). Quantized experts: `fnx/quant/*`,
  store `/var/tmp/strata-tt/quant/store` (GPTQ BFP4 rows for all 12,288 experts, `Plan(name)`).

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
- Device hygiene: `tools/fnx-engine` runs the container as a child under the device lock, forwards signals, and on a hang
  force-removes it and resets the board (`tt-smi -r 0`) while still holding the lock. Never reset the board without the lock.
- `tools/devrun.sh` watchdog: `FNX_DEV_TIMEOUT` seconds (default 7200). A run still alive then is removed and, for device
  runs, the board is reset under the still-held lock (rc 124). Set it close to your run's expected length for new device
  paths (a hung kernel otherwise holds the shared card until someone notices).
