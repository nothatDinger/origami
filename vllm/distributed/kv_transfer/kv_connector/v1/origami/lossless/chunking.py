# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    ChunkLayout,
)


@dataclass(frozen=True)
class PlannedChunk:
    chunk_id: int
    layout: ChunkLayout


def _bytes_for_shape(tokens: int, channels: int, bytes_per_symbol: int) -> int:
    return max(0, int(tokens)) * max(0, int(channels)) * max(1, int(bytes_per_symbol))


def plan_head_channel_chunks(
    *,
    layer_index: int,
    num_heads: int,
    head_dim: int,
    token_count: int,
    bytes_per_symbol: int,
    min_bytes: int,
    target_bytes: int,
    max_bytes: int,
) -> list[PlannedChunk]:
    """Plan QAT-friendly chunks inside attention heads along channels.

    Chunks prefer a single head and contiguous channel ranges. If a full head is
    too small for the minimum chunk size, adjacent heads are grouped while the
    recorded layout still preserves head/channel boundaries.
    """
    if num_heads <= 0 or head_dim <= 0 or token_count <= 0:
        return []

    channel_bytes = max(1, token_count * bytes_per_symbol)
    target_channels = max(1, target_bytes // channel_bytes)
    max_channels = max(1, max_bytes // channel_bytes)
    min_channels = max(1, min_bytes // channel_bytes)

    chunks: list[PlannedChunk] = []
    chunk_id = 0
    head = 0
    while head < num_heads:
        remaining_heads = num_heads - head
        full_head_bytes = _bytes_for_shape(token_count, head_dim, bytes_per_symbol)

        if full_head_bytes < min_bytes and remaining_heads > 1:
            heads_per_chunk = min(
                remaining_heads,
                max(1, min(max_channels // head_dim, remaining_heads)),
            )
            while (
                heads_per_chunk < remaining_heads
                and _bytes_for_shape(
                    token_count, heads_per_chunk * head_dim, bytes_per_symbol
                ) < min_bytes
            ):
                heads_per_chunk += 1
            channels = heads_per_chunk * head_dim
            chunks.append(
                PlannedChunk(
                    chunk_id=chunk_id,
                    layout=ChunkLayout(
                        layer_index=layer_index,
                        head_start=head,
                        head_end=head + heads_per_chunk,
                        channel_start=0,
                        channel_end=head_dim,
                        token_start=0,
                        token_end=token_count,
                        unpacked_bytes=_bytes_for_shape(
                            token_count, channels, bytes_per_symbol
                        ),
                    ),
                )
            )
            chunk_id += 1
            head += heads_per_chunk
            continue

        channel = 0
        while channel < head_dim:
            remaining_channels = head_dim - channel
            width = min(remaining_channels, max(1, min(target_channels, max_channels)))
            if remaining_channels > width:
                width = max(width, min_channels)
                width = min(width, remaining_channels)
            chunks.append(
                PlannedChunk(
                    chunk_id=chunk_id,
                    layout=ChunkLayout(
                        layer_index=layer_index,
                        head_start=head,
                        head_end=head + 1,
                        channel_start=channel,
                        channel_end=channel + width,
                        token_start=0,
                        token_end=token_count,
                        unpacked_bytes=_bytes_for_shape(
                            token_count, width, bytes_per_symbol
                        ),
                    ),
                )
            )
            chunk_id += 1
            channel += width
        head += 1

    return chunks

