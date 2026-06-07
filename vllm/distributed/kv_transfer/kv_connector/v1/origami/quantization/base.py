# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class QuantizedKV:
    symbols: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantizerAdapter(ABC):
    """Interface for transmission-oriented KV quantizers.

    Implementations may call third-party GPU quantization code, but the
    connector only depends on this byte-symbol contract.
    """

    quantizer_id: str

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    @abstractmethod
    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        raise NotImplementedError

    @abstractmethod
    def dequantize(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
        *,
        dst_cache: torch.Tensor | None = None,
        block_ids: list[int] | tuple[int, ...] | None = None,
        stream: Any | None = None,
    ) -> torch.Tensor | Any:
        raise NotImplementedError

    def supports_gpu_dequantize(self) -> bool:
        return True

    def metadata_schema(self) -> dict[str, Any]:
        return {"quantizer": self.quantizer_id, "version": 1}

