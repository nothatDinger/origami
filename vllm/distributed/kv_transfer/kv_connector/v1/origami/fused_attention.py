# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FusedLayerPayload:
    """GPU-resident compressed prefix for one Origami cache layer."""

    request_id: str
    cache_key: str
    layer_name: str
    quantizer: str
    symbols: torch.Tensor
    metadata: dict[str, Any]
    ready_event: torch.cuda.Event
    ref_count: int = 1

    @property
    def prefix_tokens(self) -> int:
        return int(self.metadata["token_count"])


def run_fused_prefix_attention(
    payload: FusedLayerPayload,
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_position: int,
    sequence_length: int,
    softmax_scale: float,
    output: torch.Tensor,
) -> None:
    """Launch an Origami fused online-softmax attention kernel."""
    from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
        native_gpu,
    )

    metadata = payload.metadata
    if payload.quantizer == "kivi":
        native_gpu.kivi_fused_prefix_attention(
            payload.symbols,
            metadata=metadata,
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            query_start_position=query_start_position,
            sequence_length=sequence_length,
            softmax_scale=softmax_scale,
            output=output,
        )
        return
    if payload.quantizer == "kvquant":
        native_gpu.kvquant_fused_prefix_attention(
            payload.symbols,
            metadata=metadata,
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            query_start_position=query_start_position,
            sequence_length=sequence_length,
            softmax_scale=softmax_scale,
            output=output,
        )
        return
    raise ValueError(f"Unsupported Origami fused quantizer {payload.quantizer!r}")


def reference_fused_prefix_attention(
    payload: FusedLayerPayload,
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_position: int,
    sequence_length: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Slow explicit-dequantization reference used only by tests."""
    from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kivi_adapter import (
        KiviAdapter,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kvquant_adapter import (
        KVQuantAdapter,
    )

    metadata = payload.metadata
    symbols_cpu = payload.symbols.detach().cpu()
    if payload.quantizer == "kivi":
        adapter = KiviAdapter({
            "bits": int(metadata["bits"]),
            "group_size": int(metadata["group_size"]),
            "sink_tokens": int(metadata["sink_tokens"]),
            "dequant_device": "cpu",
        })
    elif payload.quantizer == "kvquant":
        adapter = KVQuantAdapter({
            "bits": int(metadata["bits"]),
            "dequant_device": "cpu",
        })
    else:
        raise ValueError(f"Unsupported Origami fused quantizer {payload.quantizer!r}")
    restored = adapter.dequantize(symbols_cpu, metadata)
    canonical, _ = KiviAdapter._canonicalize(restored)
    prefix_key = canonical[0].to(device=query.device, dtype=query.dtype)
    prefix_value = canonical[1].to(device=query.device, dtype=query.dtype)

    if kv_cache.dim() != 5 or kv_cache.shape[0] != 2:
        raise ValueError("reference path expects [2, blocks, block, heads, dim]")
    block_size = int(kv_cache.shape[2])
    suffix_keys = []
    suffix_values = []
    for token in range(payload.prefix_tokens, sequence_length):
        logical_block = token // block_size
        block_offset = token % block_size
        physical_block = int(block_table[logical_block].item())
        suffix_keys.append(kv_cache[0, physical_block, block_offset])
        suffix_values.append(kv_cache[1, physical_block, block_offset])
    if suffix_keys:
        key = torch.cat([prefix_key, torch.stack(suffix_keys)], dim=0)
        value = torch.cat([prefix_value, torch.stack(suffix_values)], dim=0)
    else:
        key = prefix_key
        value = prefix_value

    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1).float()
    value = value.repeat_interleave(repeats, dim=1).float()
    scores = torch.einsum("qhd,thd->qht", query.float(), key)
    scores.mul_(float(softmax_scale))
    key_positions = torch.arange(key.shape[0], device=query.device)
    causal_ends = query_start_position + torch.arange(
        query.shape[0], device=query.device
    )
    scores.masked_fill_(key_positions[None, None, :] > causal_ends[:, None, None],
                        float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("qht,thd->qhd", probabilities, value).to(query.dtype)
