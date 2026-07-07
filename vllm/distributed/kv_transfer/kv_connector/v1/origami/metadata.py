# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

if os.environ.get("ORIGAMI_LIGHT_IMPORT") != "1":
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
else:

    class KVConnectorMetadata:  # noqa: D101
        pass

LosslessPath = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class ChunkLayout:
    layer_index: int
    head_start: int
    head_end: int
    channel_start: int
    channel_end: int
    token_start: int
    token_end: int
    unpacked_bytes: int


@dataclass
class ChunkRecord:
    chunk_id: int
    layout: ChunkLayout
    codec: str
    compressed: torch.Tensor
    compressed_bytes: int
    unpacked_bytes: int


@dataclass
class LayerPayload:
    layer_name: str
    quantizer: str
    quant_metadata: dict[str, Any]
    chunks: list[ChunkRecord] = field(default_factory=list)


@dataclass
class OrigamiPayload:
    cache_key: str
    request_id: str
    quantizer: str
    quantizer_config_hash: str
    token_start: int
    token_count: int
    layer_payloads: dict[str, LayerPayload] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrigamiRestoreRequest:
    request_id: str
    cache_key: str
    block_ids_per_group: tuple[tuple[int, ...], ...]
    num_tokens: int
    lossless_path: LosslessPath = "cpu"
    metric_request_id: str | None = None


@dataclass(frozen=True)
class OrigamiSaveRequest:
    request_id: str
    cache_key: str
    block_ids_per_group: tuple[tuple[int, ...], ...]
    num_tokens: int
    token_start: int = 0


@dataclass
class OrigamiConnectorMetadata(KVConnectorMetadata):
    reqs_to_restore: dict[str, OrigamiRestoreRequest] = field(default_factory=dict)
    reqs_to_save: dict[str, OrigamiSaveRequest] = field(default_factory=dict)
    gpu_lossless_ratio: int = 0
