"""Qwen3.8-Flash-Next geometry (Coder IQ1_M), as Strata hard-codes it (spec Part 1 §1); checked against GGUF metadata."""
from __future__ import annotations

N_EMBD = 2560
N_LAYERS = 48
QSA_INTERVAL = 4                 # layer % 4 == 3 is full attention (QSA); the rest are GDN
HC = 4                           # hyper-connection streams
HC_DIM = HC * N_EMBD             # 10240, index = stream * 2560 + d
HC_LR = 320
EPS = 1e-6                       # every RMSNorm

# GDN
SSM_STATE = 128
SSM_K_HEADS = 16
SSM_V_HEADS = 48
SSM_D_CONV = 4
SSM_CONV_CH = 10240              # q 2048 | k 2048 | v 6144
SSM_VALUE_DIM = 6144

# QSA
N_HEAD = 24
N_HEAD_KV = 2
HEAD_DIM = 256
ROPE_DIM = 64                    # partial NEOX rope over the first 64 of 256 dims
ROPE_BASE = 1e7
IDX_HEADS = 4
IDX_DIM = 128
IDX_TOP_K = 2048
COMPRESS = 4

# MoE (Coder: 256 of the original 512 experts kept)
N_EXPERT = 256
N_USED = 10
N_FF = 640

# PLE (layer 1 only)
PLE_LAYER = 1
PLE_HEAD_DIM = 160
NGRAM = 3
PLE_HEADS = 16
PLE_CONV = 4
PLE_EOS = 248044

N_VOCAB = 248320
EOS_IDS = (248044, 248046)


def is_qsa(layer: int) -> bool:
    return layer % QSA_INTERVAL == QSA_INTERVAL - 1


def check_metadata(md: dict) -> None:
    """Refuse a GGUF whose metadata disagrees with the constants above."""
    want = {
        "general.architecture": "qwen4exp", "qwen4exp.block_count": N_LAYERS, "qwen4exp.embedding_length": N_EMBD,
        "qwen4exp.attention.head_count": N_HEAD, "qwen4exp.attention.head_count_kv": N_HEAD_KV,
        "qwen4exp.attention.key_length": HEAD_DIM, "qwen4exp.expert_count": N_EXPERT,
        "qwen4exp.expert_used_count": N_USED, "qwen4exp.expert_feed_forward_length": N_FF,
        "qwen4exp.ssm.conv_kernel": SSM_D_CONV, "qwen4exp.ssm.state_size": SSM_STATE,
        "qwen4exp.ssm.group_count": SSM_K_HEADS, "qwen4exp.ssm.time_step_rank": SSM_V_HEADS,
        "qwen4exp.full_attention_interval": QSA_INTERVAL, "qwen4exp.rope.dimension_count": ROPE_DIM,
        "qwen4exp.hyper_connection.count": HC, "qwen4exp.hyper_connection.low_rank": HC_LR,
        "qwen4exp.attention.indexer.head_count": IDX_HEADS, "qwen4exp.attention.indexer.key_length": IDX_DIM,
        "qwen4exp.attention.indexer.top_k": IDX_TOP_K, "qwen4exp.ple.layers": [PLE_LAYER],
        "qwen4exp.rope.freq_base": ROPE_BASE,
    }
    bad = [f"{k}: GGUF {md.get(k)!r}, expected {v!r}" for k, v in want.items() if md.get(k) != v]
    if bad:
        raise ValueError("GGUF metadata differs from the Flash-Next geometry: " + "; ".join(bad))
