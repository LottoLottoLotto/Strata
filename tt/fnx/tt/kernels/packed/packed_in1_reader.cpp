// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// fnx copy of the image's ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/
// reader_bmm_tile_layout_qwen_lmhead_packed.cpp (md5 1d8020a06ea70d4151b893f8aa3200ed), the in1 reader that
// fnx.tt.packed_matmul swaps into the stock 1D-mcast matmul program.  Changes (QWEN_LMHEAD_PACKED_LAYOUT path only):
// the K-group size is QWEN_PACKED_GROUP_K_TILES (was 4: K % 32 tiles; group 2 admits K = 80 tiles) and the grid width
// QWEN_PACKED_GRID_X (was 11); the one-outstanding stage reads end at group boundaries (bank barrier per group end).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/remote_circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

#ifdef QWEN_PACKED_DUAL_NOC
#if !defined(QWEN_LMHEAD_PACKED_LAYOUT) || !defined(SKIP_MCAST) || !defined(ARCH_BLACKHOLE)
#error "QWEN_PACKED_DUAL_NOC requires the Blackhole non-multicast packed weight reader"
#endif
static_assert(QWEN_PACKED_DUAL_NOC == 1);
static_assert(noc_mode == DM_DYNAMIC_NOC, "Both data-movement processors must use dynamic NoC mode");
#endif

void kernel_main() {
    // READER
    uint32_t rt_args_idx = 0;
    // in1 tensor args
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args (READER)
    const uint32_t last_block_w = get_arg_val<uint32_t>(rt_args_idx++);
    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_next_w_dim_block_stride = get_compile_time_arg_val(3);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);

    // in1 mcast args
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(12);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(13);
    // batch args
    constexpr uint32_t KtNt = get_compile_time_arg_val(14);
    constexpr uint32_t batch = get_compile_time_arg_val(15);
    constexpr uint32_t bcast_B = get_compile_time_arg_val(16);
    // sparsity args
    constexpr uint32_t batchB = get_compile_time_arg_val(17);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(18);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(19);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(20);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(21);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(22);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_compile_time_arg_val(23);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_compile_time_arg_val(24);
    // out subblock args
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(27);
    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(28);  // if 0
    // Don't need batch; same as batch from READER args

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

#ifdef FUSE_BIAS
    // in3 mcast args
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in3_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in3_tensor_stride_w = get_compile_time_arg_val(29);

    constexpr uint32_t dfb_id_in3 = get_named_compile_time_arg_val("cb_bias");
    // Use the CB page size (padded to the DRAM alignment by the factory) for DRAM reads
    // and L1 write strides, NOT the raw tile size. On Blackhole, the DRAM read alignment
    // is 64B, so a sub-64B tile (e.g. 32B for a (1,16) bf16 bias tile) cannot be read
    // directly from DRAM, and 32B-strided L1 writes land at non-64B-aligned addresses
    // that disagree with the 64B-aligned DRAM source. The factory pads the CB page to
    // 64B; the unpacker still reads the actual 32B tile from the padded page via tile
    // dims. For tiles already >= dram_alignment (e.g. 32x32 bf16 = 2048B), the page size
    // equals the tile size, so this is a no-op. Mirrors how in0/in1 readers walk at the
    // aligned stride.
    const uint32_t bias_single_tile_size_bytes = get_local_cb_interface(dfb_id_in3).fifo_page_size;
    constexpr const uint32_t in3_tile_hw = get_tile_hw(dfb_id_in3);

#ifndef BIAS_SHARDED
    uint32_t l1_write_addr_in3;
    // Bias accessor will be defined later after TensorAccessor args
#endif  // BIAS_SHARDED
#else
    rt_args_idx += 2;  // Skip over placeholders
#endif  // FUSE_BIAS
#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);
#endif  // OUT_SHARDED

    constexpr bool fuse_op_all_gather = (bool)get_compile_time_arg_val(30);
    constexpr bool fuse_op_reduce_scatter = (bool)get_compile_time_arg_val(31);

    MatmulOpReceiver fused_op_receiver;
    OpSignaler op_signaler;
    if constexpr (fuse_op_all_gather) {
        fused_op_receiver = MatmulOpReceiver(
            false, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in1_block_h /* tiles_per_block (in the same dimension */
        );
    } else if constexpr (fuse_op_reduce_scatter) {
        op_signaler = OpSignaler(rt_args_idx);
    }

    constexpr auto in1_args = TensorAccessorArgs<32>();
    constexpr auto sparsity_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sparsity_args.next_compile_time_args_offset()>();
#ifdef FUSE_BIAS
    constexpr auto bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto after_bias_offset = bias_args.next_compile_time_args_offset();
#else
    constexpr auto after_bias_offset = out_args.next_compile_time_args_offset();
#endif  // FUSE_BIAS

// RT and COMPILE TIME ARGS for DRAM sharded weights
#ifdef IN1_DRAM_WIDTH_SHARDED
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_dram_shards_to_read = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_tensor_start_offset = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in1_block_w_dram_stride_bytes = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);
    tt_l1_ptr uint32_t* current_dram_bank_id = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);

    constexpr uint32_t in1_dram_block_num_tiles = get_compile_time_arg_val(after_bias_offset);
    constexpr uint32_t in1_block_w_dram_bytes = get_compile_time_arg_val(after_bias_offset + 1);
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_KtNt_per_batch = get_compile_time_arg_val(after_bias_offset);        // K*N tiles per batch
    constexpr uint32_t in1_batches_per_bank = get_compile_time_arg_val(after_bias_offset + 1);  // batches per DRAM bank
#endif  // IN1_DRAM_HEIGHT_SHARDED

#ifdef FUSE_BIAS
#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(bias_args, in3_tensor_addr);
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS

    constexpr uint32_t dfb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(dfb_id_in1);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM, and the
    // interleaved in1 CB pages are sized to match (see the program factory). On the plain interleaved
    // path the NOC reads the unpadded tile of data into each padded slot and tiles are laid out /
    // multicast at the padded stride. No-op when already aligned. The sharded / DRAM-sharded paths
    // keep their natural (unpadded) stride.
    constexpr uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#if !defined(IN1_SHARDED) && !defined(IN1_DRAM_WIDTH_SHARDED) && !defined(IN1_DRAM_HEIGHT_SHARDED) && \
    !defined(ENABLE_GLOBAL_CB)
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_aligned_tile_size_bytes;
#else
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
#endif

    constexpr uint32_t dfb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(dfb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(dfb_id_out0);

    Noc noc;
    DataflowBuffer dfb_in1(dfb_id_in1);
    DataflowBuffer dfb_out(dfb_id_out0);
    Semaphore<> sender_sem(get_compile_time_arg_val(10));
    Semaphore<> receiver_sem(get_compile_time_arg_val(11));
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in3(dfb_id_in3);
#endif

//  READER
#ifdef IN1_SHARDED
    dfb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    dfb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#elif !defined(ENABLE_GLOBAL_CB)
    uint32_t l1_write_addr_in1;

    [[maybe_unused]] const auto s1 = TensorAccessor(in1_args, in1_tensor_addr);
#endif  // IN1_SHARDED / ENABLE_GLOBAL_CB

#ifdef ENABLE_GLOBAL_CB
    constexpr uint32_t remote_cb_id = tt::CBIndex::c_31;
    const uint32_t in1_fifo_tiles = get_local_cb_interface(dfb_id_in1).fifo_num_pages;
#endif

    //  WRITER
    const auto s = TensorAccessor(out_args, out_tensor_addr);
    // `s` is only consumed inside the `#ifndef OUT_SHARDED` write path below; mark it used so
    // sharded builds don't warn (-Wunused-but-set-variable).
    (void)s;

    // sparsity accessor
    constexpr uint32_t dfb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    DataflowBuffer dfb_sparsity(dfb_id_sparsity);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr);

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

#ifdef IN1_SHARDED
    uint64_t in1_start_address = dfb_in1.get_write_ptr();
#endif  // IN1_SHARDED
#endif  // SKIP_MCAST

    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        dfb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = dfb_sparsity.get_write_ptr();
    }

#ifdef IN1_DRAM_WIDTH_SHARDED
    constexpr uint32_t in1_dram_block_size_bytes = in1_dram_block_num_tiles * in1_single_tile_size_bytes;
    uint32_t in1_block_w_bytes = in1_block_w * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_batch_stride_bytes = in1_KtNt_per_batch * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

#ifdef IN1_DRAM_HEIGHT_SHARDED
        // Compute DRAM bank and offset for this batch
        uint32_t in1_dram_bank_id = b / in1_batches_per_bank;
        uint32_t in1_batch_in_shard = b % in1_batches_per_bank;
        AllocatorBank<AllocatorBankType::DRAM> dram_src;
        uint32_t in1_dram_batch_offset = in1_batch_in_shard * in1_batch_stride_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

        if constexpr (batchB > 0) {
            noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            if constexpr (batchB > 0) {
                if (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB] == 0) {
                    out_tensor_start_tile_id += MtNt;
                    in1_batch_tile_id += KtNt;
                    continue;
                }
            }

            uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
            uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                uint32_t in1_tensor_current_w_dim_block_tile_id = in1_tensor_current_h_dim_block_tile_id;
                uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
#ifdef FUSE_BIAS
                uint32_t in3_tensor_current_w_dim_block_tile_id = in3_tensor_start_tile_id;
#endif  // FUSE_BIAS
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                    uint32_t in1_tensor_current_inner_dim_block_start_tile_id = in1_tensor_current_w_dim_block_tile_id;
#ifdef IN1_DRAM_WIDTH_SHARDED
                    // Reset DRAM read offset for each bh block — the inner dim loop
                    // advances through K, and each output row block re-reads the same
                    // in1 columns from K=0. (bw is always 1 for DRAM-sharded senders.)
                    uint32_t l1_read_addr_in1_offset = 0;
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef QWEN_LMHEAD_PIPELINE_INSTRUMENT
#define QWEN_LMHEAD_ZONE(name) DeviceZoneScopedN(name)
#else
#define QWEN_LMHEAD_ZONE(name)
#endif
#ifdef QWEN_LMHEAD_PACKED_LAYOUT
                    constexpr uint32_t qwen_lmhead_outstanding = QWEN_LMHEAD_OUTSTANDING;
                    constexpr uint32_t qwen_lmhead_cb_slots = QWEN_LMHEAD_CB_SLOTS;
                    static_assert(qwen_lmhead_outstanding >= 1 && qwen_lmhead_outstanding <= 4);
                    static_assert(qwen_lmhead_cb_slots > qwen_lmhead_outstanding && qwen_lmhead_cb_slots <= 8);
                    constexpr uint32_t qwen_lmhead_trids = qwen_lmhead_outstanding;
                    uint32_t qwen_lmhead_issue_trid = 1;
                    uint32_t qwen_lmhead_wait_trid = 1;
                    uint32_t qwen_lmhead_pending = 0;
                    // Gate/up M1/M2 already share one tiled M row and one weight pass.
                    // Consecutive K blocks in a packed four-row group use the same
                    // DRAM bank. Keep the read command's source-coordinate/control
                    // state across those blocks instead of reprogramming it for
                    // every burst. Addresses, burst splitting, VC, CB credits and
                    // publication/barrier order are unchanged.
                    constexpr bool gateup_stateful_read =
                        in1_block_h == 2 && in1_block_w == 10 && num_blocks_inner_dim == 80 &&
                        QWEN_PACKED_LAST_WORKER == 108 && in1_single_tile_size_bytes == 576 &&
                        qwen_lmhead_outstanding == 1 && !fuse_op_all_gather && !fuse_op_reduce_scatter;
#ifdef QWEN_PACKED_DUAL_NOC
                    static_assert(!fuse_op_all_gather && !fuse_op_reduce_scatter && batchB == 0);
                    static_assert(in1_single_tile_size_bytes % DRAM_ALIGNMENT == 0);
                    static_assert(in1_single_tile_size_bytes >= 2 * DRAM_ALIGNMENT);
                    // Dynamic mode partitions command buffers by processor on BOTH NoCs.
                    // Use only this BRISC's SDK-selected read_cmd_buf; never borrow NCRISC's
                    // activation read/atomic buffer. Output writes keep the original `noc`.
                    const Noc packed_nocs[2] = {noc, Noc(static_cast<uint8_t>(noc.get_noc_id() ^ 1u))};
                    uint32_t active_read_bank[2] = {8, 8};
                    auto read_packed_rows = [&](uint32_t bank, uint32_t page, uint32_t dst, uint32_t bytes) {
                        AllocatorBank<AllocatorBankType::DRAM> packed_bank;
                        const uint32_t addr = in1_tensor_addr + page * in1_single_tile_size_bytes;
                        // Split bytes, not tiles: even a one-row/partial worker uses both NoCs.
                        // Both source and destination offsets retain DRAM alignment, and the
                        // two ranges exactly partition the original read, without padding reads.
                        const uint32_t first_bytes = (bytes / 2) & ~(DRAM_ALIGNMENT - 1);
                        for (uint32_t lane = 0; lane < 2; ++lane) {
                            const uint32_t offset = lane == 0 ? 0 : first_bytes;
                            const uint32_t size = lane == 0 ? first_bytes : bytes - first_bytes;
                            const auto& read_noc = packed_nocs[lane];
                            // AllocatorBank computes the bank coordinates for THIS NoC.
                            if constexpr (qwen_lmhead_outstanding > 1) {
                                read_noc.async_read<NocOptions::TXN_ID>(
                                    packed_bank, CoreLocalMem<uint32_t>(dst + offset), size,
                                    {.bank_id = bank, .addr = addr + offset}, {},
                                    NocOptVals{.trid = qwen_lmhead_issue_trid});
                            } else if constexpr (gateup_stateful_read) {
                                // No atomics, bias reads or output writes intervene in this K
                                // loop. Each NoC has independent sticky coordinate/control state.
                                if (bank != active_read_bank[lane]) {
                                    read_noc.set_async_read_state(
                                        packed_bank, size, {.bank_id = bank, .addr = addr + offset});
                                    active_read_bank[lane] = bank;
                                }
                                read_noc.async_read_with_state(
                                    packed_bank, CoreLocalMem<uint32_t>(dst + offset), size,
                                    {.bank_id = bank, .addr = addr + offset}, {});
                            } else {
                                read_noc.async_read(
                                    packed_bank, CoreLocalMem<uint32_t>(dst + offset), size,
                                    {.bank_id = bank, .addr = addr + offset}, {});
                            }
                        }
                    };
#else
                    uint32_t active_read_bank = 8;
                    auto read_packed_rows = [&](uint32_t bank, uint32_t page, uint32_t dst, uint32_t bytes) {
                        AllocatorBank<AllocatorBankType::DRAM> packed_bank;
                        const uint32_t addr = in1_tensor_addr + page * in1_single_tile_size_bytes;
                        if constexpr (gateup_stateful_read) {
                            if (bank != active_read_bank) {
                                noc.set_async_read_state(packed_bank, bytes, {.bank_id = bank, .addr = addr});
                                active_read_bank = bank;
                            }
                            // Default max_page_size retains multi-packet splitting:
                            // a full two-row BF4 block is 11520 bytes, not one packet.
                            noc.async_read_with_state(
                                packed_bank, CoreLocalMem<uint32_t>(dst), bytes,
                                {.bank_id = bank, .addr = addr}, {});
                        } else {
                            noc.async_read(
                                packed_bank, CoreLocalMem<uint32_t>(dst), bytes,
                                {.bank_id = bank, .addr = addr}, {});
                        }
                    };
#endif
                    auto wait_packed_reads = [&]() {
                        if constexpr (qwen_lmhead_outstanding > 1) {
                            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = qwen_lmhead_wait_trid});
#ifdef QWEN_PACKED_DUAL_NOC
                            packed_nocs[1].async_read_barrier<NocOptions::TXN_ID>(
                                {.trid = qwen_lmhead_wait_trid});
#endif
                        } else {
                            noc.async_read_barrier();
#ifdef QWEN_PACKED_DUAL_NOC
                            packed_nocs[1].async_read_barrier();
#endif
                        }
                    };
#endif
                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        if constexpr (fuse_op_all_gather) {
                            fused_op_receiver.update_current_block_start_tile_id(
                                block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
                        }
#if defined(ENABLE_GLOBAL_CB)
                        // The tensor prefetcher pushes this receiver's K-blocks in natural order.
                        // Keep one block of lookahead: publish the current block to compute, then
                        // wait for the unpack engine to drain the previous block before returning
                        // its remote-CB credit to the prefetcher.
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        experimental::remote_cb_wait_front(remote_cb_id, block == 0 ? 1u : 2u);
#elif defined(IN1_DRAM_WIDTH_SHARDED)
                        // Operand 1 - DRAM width sharded
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        uint32_t l1_write_addr_in1_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        AllocatorBank<AllocatorBankType::DRAM> dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t shard_bank_id = current_dram_bank_id[next_bank_id_and_dram_stride_index];
                            uint32_t shard_base_addr = in1_tensor_addr;
                            if (i == 0) {
                                shard_base_addr += dram_tensor_start_offset;
                            }
                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                dram_bank,
                                in1_single_tile_size_bytes,
                                {.bank_id = shard_bank_id, .addr = shard_base_addr},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                            uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr() + l1_write_addr_in1_offset;
                            uint32_t in1_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t m = 0; m < in1_block_h; ++m) {
                                uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                                uint32_t l1_write_addr_in1_temp = l1_write_addr_in1;
                                for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                                    noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                        dram_bank,
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1_temp),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = shard_bank_id, .addr = shard_base_addr + l1_read_addr_in1_temp},
                                        {},
                                        NocOptVals{.vc = vc});
                                    l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                                    l1_write_addr_in1_temp += in1_single_tile_size_bytes;
                                }
                                l1_read_addr_in1 += in1_block_w_dram_bytes;
                                l1_write_addr_in1 += in1_block_w_bytes;
                            }
                            l1_write_addr_in1_offset +=
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc.async_read_barrier();
#elif defined(IN1_DRAM_HEIGHT_SHARDED)
                        // Operand 1 - DRAM height sharded (batched)
                        // Each DRAM bank holds batches_per_bank complete [K, N] matrices
                        // Bank and offset computed at start of batch loop
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        l1_write_addr_in1 = dfb_in1.get_write_ptr();
                        uint64_t in1_start_address =
                            l1_write_addr_in1;  // copy start address of block, to be used for mcasting

                        // Read in1 block from the correct DRAM bank
                        // Tile layout within a batch: row-major [K, N], same strides as interleaved
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    uint32_t tile_byte_offset =
                                        in1_dram_batch_offset + in1_tensor_tile_id * in1_single_tile_size_bytes;
                                    noc.async_read(
                                        dram_src,
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = in1_dram_bank_id, .addr = in1_tensor_addr + tile_byte_offset},
                                        {});
                                }
                                l1_write_addr_in1 += in1_single_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#elif defined(QWEN_LMHEAD_PACKED_LAYOUT)
                        // Worker-major packed layout. K stages rotate over all eight DRAM banks;
                        // within a bank every stage is contiguous and remains K-major.
                        constexpr uint32_t packed_grid_x = QWEN_PACKED_GRID_X;
                        constexpr uint32_t packed_banks = 8;
                        constexpr uint32_t packed_last_worker = QWEN_PACKED_LAST_WORKER;
                        constexpr uint32_t packed_pages_per_worker_per_bank =
                            QWEN_PACKED_PAGES_PER_WORKER_PER_BANK;
                        const uint32_t worker =
                            get_absolute_logical_x() + packed_grid_x * get_absolute_logical_y();
                        const uint32_t valid_cols = worker == packed_last_worker ? last_block_w : in1_block_w;
                        constexpr uint32_t packed_group_k_tiles = QWEN_PACKED_GROUP_K_TILES;
                        static_assert(packed_group_k_tiles >= 1 &&
                                      (num_blocks_inner_dim * in1_block_h) % (packed_banks * packed_group_k_tiles) == 0);

                        if constexpr (qwen_lmhead_outstanding == 1) {
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_CB_RESERVE");
                                dfb_in1.reserve_back(in1_block_num_tiles);
                            }
                            const uint32_t write_start = dfb_in1.get_write_ptr();
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_READ_ISSUE");
                                if (valid_cols == in1_block_w) {
                                    constexpr uint32_t packed_read_stage_rows = 2;
                                    for (uint32_t h = 0; h < in1_block_h;) {
                                        const uint32_t packed_k_row = block * in1_block_h + h;
                                        const uint32_t packed_group = packed_k_row / packed_group_k_tiles;
                                        const uint32_t packed_row = packed_k_row % packed_group_k_tiles;
                                        uint32_t rows = packed_read_stage_rows;
                                        if (rows > in1_block_h - h) {
                                            rows = in1_block_h - h;
                                        }
                                        if (rows > packed_group_k_tiles - packed_row) {
                                            rows = packed_group_k_tiles - packed_row;
                                        }
                                        const uint32_t bank = (worker + packed_group) % packed_banks;
                                        const uint32_t bank_page =
                                            worker * packed_pages_per_worker_per_bank +
                                            (packed_group / packed_banks) * packed_group_k_tiles * valid_cols +
                                            packed_row * valid_cols;
                                        read_packed_rows(
                                            bank, bank_page,
                                            write_start + h * in1_block_w * in1_single_tile_size_bytes,
                                            rows * in1_block_w * in1_single_tile_size_bytes);
                                        h += rows;
                                        if (packed_row + rows == packed_group_k_tiles && h < in1_block_h) {
                                            QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_BANK_BARRIER");
                                            wait_packed_reads();
                                        }
                                    }
                                } else {
                                    for (uint32_t h = 0; h < in1_block_h; ++h) {
                                        const uint32_t packed_k_row = block * in1_block_h + h;
                                        const uint32_t packed_group = packed_k_row / packed_group_k_tiles;
                                        const uint32_t packed_row = packed_k_row % packed_group_k_tiles;
                                        const uint32_t bank = (worker + packed_group) % packed_banks;
                                        const uint32_t bank_page =
                                            worker * packed_pages_per_worker_per_bank +
                                            (packed_group / packed_banks) * packed_group_k_tiles * valid_cols +
                                            packed_row * valid_cols;
                                        read_packed_rows(
                                            bank, bank_page,
                                            write_start + h * in1_block_w * in1_single_tile_size_bytes,
                                            valid_cols * in1_single_tile_size_bytes);
                                        if (packed_row + 1 == packed_group_k_tiles && h + 1 < in1_block_h) {
                                            QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_BANK_BARRIER");
                                            wait_packed_reads();
                                        }
                                    }
                                }
                            }
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_READ_BARRIER");
                                wait_packed_reads();
                            }
                        } else {
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_CB_RESERVE");
                                // reserve_back does not move the producer cursor or account for
                                // DMA already issued. Protect every unpublished block plus this one.
                                dfb_in1.reserve_back(in1_block_num_tiles * (qwen_lmhead_pending + 1));
                            }
                            uint32_t write_start =
                                dfb_in1.get_write_ptr() + qwen_lmhead_pending * in1_block_size_bytes;
                            // The producer cursor can already be partway through the ring on
                            // later output blocks/batches. Wrap against the actual CB limit.
                            if (write_start >= get_local_cb_interface(dfb_id_in1).fifo_limit) {
                                write_start -= qwen_lmhead_cb_slots * in1_block_size_bytes;
                            }
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_READ_ISSUE");
#ifndef QWEN_PACKED_DUAL_NOC
                                AllocatorBank<AllocatorBankType::DRAM> packed_bank;
#endif
                                const uint32_t stage_rows = valid_cols == in1_block_w ? 2u : 1u;
                                for (uint32_t h = 0; h < in1_block_h;) {
                                    const uint32_t packed_k_row = block * in1_block_h + h;
                                    const uint32_t packed_group = packed_k_row / packed_group_k_tiles;
                                    const uint32_t packed_row = packed_k_row % packed_group_k_tiles;
                                    uint32_t rows = stage_rows;
                                    if (rows > in1_block_h - h) {
                                        rows = in1_block_h - h;
                                    }
                                    if (rows > packed_group_k_tiles - packed_row) {
                                        rows = packed_group_k_tiles - packed_row;
                                    }
                                    const uint32_t bank = (worker + packed_group) % packed_banks;
                                    const uint32_t bank_page =
                                        worker * packed_pages_per_worker_per_bank +
                                        (packed_group / packed_banks) * packed_group_k_tiles * valid_cols +
                                        packed_row * valid_cols;
                                    // All row/bank requests for one K block share a transaction ID.
                                    // Default burst splitting is required for these multi-packet reads.
                                    // Partial workers read one valid row into each full-width CB row.
#ifdef QWEN_PACKED_DUAL_NOC
                                    read_packed_rows(
                                        bank, bank_page,
                                        write_start + h * in1_block_w * in1_single_tile_size_bytes,
                                        rows * valid_cols * in1_single_tile_size_bytes);
#else
                                    noc.async_read<NocOptions::TXN_ID>(
                                        packed_bank,
                                        CoreLocalMem<uint32_t>(
                                            write_start + h * in1_block_w * in1_single_tile_size_bytes),
                                        rows * valid_cols * in1_single_tile_size_bytes,
                                        {.bank_id = bank,
                                         .addr = in1_tensor_addr + bank_page * in1_single_tile_size_bytes},
                                        {},
                                        NocOptVals{.trid = static_cast<uint8_t>(qwen_lmhead_issue_trid)});
#endif
                                    h += rows;
                                }
                            }
                            ++qwen_lmhead_pending;
                            qwen_lmhead_issue_trid =
                                qwen_lmhead_issue_trid == qwen_lmhead_trids ? 1 : qwen_lmhead_issue_trid + 1;
                            if (qwen_lmhead_pending == qwen_lmhead_outstanding) {
                                {
                                    QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_READ_BARRIER");
                                    wait_packed_reads();
                                }
                                {
                                    QWEN_LMHEAD_ZONE("QWEN_LMHEAD_CB_PUSH");
                                    dfb_in1.push_back(in1_block_num_tiles);
                                }
                                // Retire in K order; this ID is now safe to reuse on the next issue.
                                --qwen_lmhead_pending;
                                qwen_lmhead_wait_trid =
                                    qwen_lmhead_wait_trid == qwen_lmhead_trids ? 1 : qwen_lmhead_wait_trid + 1;
                            }
                        }
#elif defined(QWEN_GATEUP_PACKED_LAYOUT)
                        // Fixed Qwen3.5-9B gate/up layout: each output worker owns one contiguous
                        // 8-column region, striped over all eight DRAM banks. Each 8x8 K block is
                        // already in the exact K-major order consumed by the compute kernel.
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        constexpr uint32_t packed_grid_x = 11;
                        constexpr uint32_t packed_banks = 8;
                        constexpr uint32_t packed_k_tiles = 128;
                        const uint32_t packed_block_tiles = in1_block_h * in1_block_w;
                        const uint32_t packed_blocks = packed_k_tiles / in1_block_h;
                        const uint32_t worker =
                            get_absolute_logical_x() + packed_grid_x * get_absolute_logical_y();
                        const uint32_t bank = worker % packed_banks;
                        const uint32_t region = worker / packed_banks;
                        const uint32_t block_bytes = packed_block_tiles * in1_single_tile_size_bytes;
                        const uint32_t region_bytes = packed_blocks * block_bytes;
                        const uint64_t src = get_noc_addr_from_bank_id<true>(
                            bank, in1_tensor_addr + region * region_bytes + block * block_bytes);
                        noc_async_read(src, dfb_in1.get_write_ptr(), block_bytes);
                        noc_async_read_barrier();
#elif !defined(IN1_SHARDED)
                        // Operand 1 - interleaved
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_write_offset = 0;
                        uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        // Copy in1 block into CB, as the default kernel
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    noc.async_read(
                                        s1,
                                        dfb_in1,
                                        in1_single_tile_size_bytes,
                                        {.page_id = in1_tensor_tile_id},
                                        {.offset_bytes = in1_write_offset});
                                }
                                in1_write_offset += in1_aligned_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_DRAM_WIDTH_SHARDED / IN1_DRAM_HEIGHT_SHARDED / IN1_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(static_cast<uint32_t>(in1_start_address)),
                            mcast_dst,
                            in1_block_size_bytes,
                            in1_mcast_num_cores,
                            {},
                            {.noc_x_start = in1_mcast_dest_noc_start_x,
                             .noc_y_start = in1_mcast_dest_noc_start_y,
                             .noc_x_end = in1_mcast_dest_noc_end_x,
                             .noc_y_end = in1_mcast_dest_noc_end_y,
                             .addr = static_cast<uint32_t>(in1_start_address)},
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency
                        // which means data could be changed before
                        //  write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in1_mcast_dest_noc_start_x,
                            in1_mcast_dest_noc_start_y,
                            in1_mcast_dest_noc_end_x,
                            in1_mcast_dest_noc_end_y,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

#if !defined(IN1_SHARDED) && !defined(QWEN_LMHEAD_PIPELINED)
                        dfb_in1.push_back(in1_block_num_tiles);
#endif
#ifdef ENABLE_GLOBAL_CB
                        if (block >= 1) {
                            while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles - in1_block_num_tiles)) {
                                invalidate_l1_cache();
                            }
                            experimental::remote_cb_pop_front(remote_cb_id, 1);
                        }
#endif
                    }
#ifdef QWEN_LMHEAD_PACKED_LAYOUT
                    if constexpr (qwen_lmhead_outstanding > 1) {
                        // Drain only issued blocks, including a sequence shorter than lookahead.
                        while (qwen_lmhead_pending > 0) {
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_DRAM_READ_BARRIER");
                                wait_packed_reads();
                            }
                            {
                                QWEN_LMHEAD_ZONE("QWEN_LMHEAD_CB_PUSH");
                                dfb_in1.push_back(in1_block_num_tiles);
                            }
                            --qwen_lmhead_pending;
                            qwen_lmhead_wait_trid =
                                qwen_lmhead_wait_trid == qwen_lmhead_trids ? 1 : qwen_lmhead_wait_trid + 1;
                        }
                    }
#ifdef QWEN_PACKED_DUAL_NOC
                    if constexpr (qwen_lmhead_outstanding > 1) {
                        // Transaction tags are sticky on the read/atomic command buffer.
                        // All IDs have retired on both NoCs before bias/output processing.
                        noc_async_read_set_trid(0, packed_nocs[0].get_noc_id());
                        noc_async_read_set_trid(0, packed_nocs[1].get_noc_id());
                    }
#endif
#endif  // QWEN_LMHEAD_PACKED_LAYOUT
#ifdef ENABLE_GLOBAL_CB
                    if (num_blocks_inner_dim > 0) {
                        while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles)) {
                            invalidate_l1_cache();
                        }
                        experimental::remote_cb_pop_front(remote_cb_id, 1);
                    }
#endif
#ifdef FUSE_BIAS
                    // Only read bias on first batch, or we have multiple output blocks
                    if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                        // Operand 1
#ifndef BIAS_SHARDED
                        dfb_in3.reserve_back(in1_block_w);
                        uint32_t in3_write_offset = 0;

                        uint64_t in3_start_address =
                            dfb_in3.get_write_ptr();        // copy start address of block, to be used for mcasting
                        uint32_t in3_block_size_bytes = 0;  // can be optimized later, pass it to kernel

#ifdef IN1_DRAM_WIDTH_SHARDED
                        uint32_t l1_write_addr_in3_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        AllocatorBank<AllocatorBankType::DRAM> bias_dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t bias_shard_bank_id = current_dram_bank_id[next_bank_id_and_dram_stride_index];
                            uint32_t bias_shard_base_addr = in3_tensor_addr;
                            if (i == 0) {
                                // dram_tensor_start_offset is in in1 tile bytes; convert to
                                // bias tile bytes since bias_dtype may differ from in1_dtype.
                                bias_shard_base_addr += (dram_tensor_start_offset / in1_single_tile_size_bytes) *
                                                        bias_single_tile_size_bytes;
                            }

                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                bias_dram_bank,
                                bias_single_tile_size_bytes,
                                {.bank_id = bias_shard_bank_id, .addr = bias_shard_base_addr},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in3 = 0;
                            l1_write_addr_in3 = dfb_in3.get_write_ptr() + l1_write_addr_in3_offset;
                            // in1_block_w_dram_stride_bytes is in in1 tile bytes, so divide
                            // by in1_single_tile_size_bytes (not bias) to get the tile count.
                            uint32_t in3_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                                noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                    bias_dram_bank,
                                    CoreLocalMem<uint32_t>(l1_write_addr_in3),
                                    bias_single_tile_size_bytes,
                                    {.bank_id = bias_shard_bank_id, .addr = bias_shard_base_addr + l1_read_addr_in3},
                                    {},
                                    NocOptVals{.vc = vc});
                                l1_read_addr_in3 += bias_single_tile_size_bytes;
                                l1_write_addr_in3 += bias_single_tile_size_bytes;
                                in3_block_size_bytes += bias_single_tile_size_bytes;
                            }
                            // Advance L1 offset in bias tile bytes, not in1 stride bytes.
                            l1_write_addr_in3_offset += in3_block_w_dram * bias_single_tile_size_bytes;
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        noc.async_read_barrier();
#else
                        // Copy in1 block into CB, as the default kernel
                        uint32_t in3_tensor_tile_id = in3_tensor_current_w_dim_block_tile_id;
                        for (uint32_t w = 0; w < in1_block_w; ++w) {
                            if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                noc.async_read(
                                    s3,
                                    dfb_in3,
                                    bias_single_tile_size_bytes,
                                    {.page_id = in3_tensor_tile_id},
                                    {.offset_bytes = in3_write_offset});
                            }
                            in3_write_offset += bias_single_tile_size_bytes;
                            in3_tensor_tile_id += in3_tensor_stride_w;
                            in3_block_size_bytes += bias_single_tile_size_bytes;
                        }
                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifndef SKIP_MCAST

                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(static_cast<uint32_t>(in3_start_address)),
                            mcast_dst,
                            in3_block_size_bytes,
                            in1_mcast_num_cores,
                            {},
                            {.noc_x_start = in1_mcast_dest_noc_start_x,
                             .noc_y_start = in1_mcast_dest_noc_start_y,
                             .noc_x_end = in1_mcast_dest_noc_end_x,
                             .noc_y_end = in1_mcast_dest_noc_end_y,
                             .addr = static_cast<uint32_t>(in3_start_address)},
                            true);
                        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same
                        // vc, same cmd_buf Also, this only works because we are setting VCs statically (using
                        // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higherthan L1 <-> RISCV
                        // latency which means data could be changed before write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in1_mcast_dest_noc_start_x,
                            in1_mcast_dest_noc_start_y,
                            in1_mcast_dest_noc_end_x,
                            in1_mcast_dest_noc_end_y,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

                        dfb_in3.push_back(in1_block_w);
#else
                        dfb_in3.reserve_back(in1_block_w);
                        dfb_in3.push_back(in1_block_w);
#endif  // BIAS_SHARDED
                    }
#endif  // FUSE_BIAS

#ifndef OUT_SHARDED
                    // WRITER
                    uint32_t num_blocks_w_dim_ =
                        bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
                    uint32_t out_num_nonzero_subblocks_h_ = out_num_nonzero_subblocks_h;
                    uint32_t out_num_nonzero_subblocks_w_ = out_num_nonzero_subblocks_w;
                    if (bw == num_blocks_w_dim_ - 1) {
                        out_num_nonzero_subblocks_w_ = out_last_num_nonzero_subblocks_w;
                    }
                    uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
                    for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
                        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                        for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
                            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                            uint32_t out_subblock_h_ = out_subblock_h;
                            uint32_t out_subblock_w_ = out_subblock_w;
                            uint32_t subblock_tiles_addr_skip = 0;
                            if (bh == num_blocks_h_dim - 1 && sbh == out_num_nonzero_subblocks_h - 1) {
                                out_subblock_h_ = out_last_subblock_h;
                            }
                            if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                                out_subblock_w_ = out_last_subblock_w;
                                subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                            }

                            dfb_out.wait_front(out_subblock_tile_count);
                            uint32_t out_read_offset = 0;

                            for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                                for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                    if (bw < num_blocks_w_dim_) {
                                        noc.async_write(
                                            dfb_out,
                                            s,
                                            output_single_tile_size_bytes,
                                            {.offset_bytes = out_read_offset},
                                            {.page_id = out_tensor_tile_id});
                                    }

                                    out_read_offset += output_single_tile_size_bytes;

                                    out_tensor_tile_id += out_tensor_stride_w;
                                }
                                // Skip padded tiles in subblock along row
                                out_read_offset += subblock_tiles_addr_skip;
                                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                            }

                            noc.async_write_barrier();
                            dfb_out.pop_front(out_subblock_tile_count);
                            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                        }
                        // Pop fully padded subblocks along the row
                        if (bw == num_blocks_w_dim_ - 1) {
                            dfb_out.wait_front(padded_block_tiles_w_skip);
                            dfb_out.pop_front(padded_block_tiles_w_skip);
                        }
                        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                    }
                    // Pop row(s) of fully padded subblocks
                    if (bh == num_blocks_h_dim - 1) {
                        dfb_out.wait_front(padded_block_tiles_h_skip);
                        dfb_out.pop_front(padded_block_tiles_h_skip);
                    }

#endif
                    in1_tensor_current_w_dim_block_tile_id += in1_tensor_next_w_dim_block_stride;
                    out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
#ifdef FUSE_BIAS
                    in3_tensor_current_w_dim_block_tile_id += in1_block_w;
#endif
                }
                out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
            }
            out_tensor_start_tile_id += MtNt;
            in1_batch_tile_id += KtNt;
        }
        if constexpr (bcast_B == 0) {
#ifndef IN1_DRAM_HEIGHT_SHARDED
            // For height-sharded DRAM, tile IDs are relative within a batch;
            // batch offset is handled by switching DRAM banks
            in1_tensor_start_tile_id += KtNt;
#endif
        }

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }

#if OUT_SHARDED
    dfb_out.wait_front(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
#endif
#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc.async_atomic_barrier();
#endif
    noc.async_write_barrier();
}
