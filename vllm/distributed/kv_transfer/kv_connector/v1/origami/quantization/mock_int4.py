# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.native_cpu import (
    normalize_symbol_layout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.base import (
    QuantizedKV,
    QuantizerAdapter,
)


class MockInt4Adapter(QuantizerAdapter):
    """Shaped reversible mock quantizer for Origami layout tests.

    Despite the historical name, this mock uses 8-bit symbols so tests can
    roundtrip small numeric tensors exactly while still exercising arbitrary
    source-layout metadata.
    """

    quantizer_id = "mock_int4"

    def _source_layout(self) -> list[str]:
        return normalize_symbol_layout(
            self.config.get("origami_symbol_layout", ["head", "token", "head_dim"])
        )

    @staticmethod
    def _permute_from_token_head_dim(layout: list[str]) -> list[int]:
        canonical = ["token", "head", "head_dim"]
        return [canonical.index(axis) for axis in layout]

    @staticmethod
    def _permute_to_token_head_dim(layout: list[str]) -> list[int]:
        return [layout.index(axis) for axis in ["token", "head", "head_dim"]]

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, layer_group, stream
        if kv.dim() != 3:
            raise ValueError("MockInt4Adapter expects [token, head, head_dim] input")
        layout = self._source_layout()
        if "layer" in layout:
            source = kv.detach().cpu().to(torch.uint8).unsqueeze(layout.index("layer"))
            rank3_layout = [axis for axis in layout if axis != "layer"]
            permute = self._permute_from_token_head_dim(rank3_layout)
            source = kv.detach().cpu().to(torch.uint8).permute(permute).contiguous()
            source = source.unsqueeze(layout.index("layer")).contiguous()
        else:
            permute = self._permute_from_token_head_dim(layout)
            source = kv.detach().cpu().to(torch.uint8).permute(permute).contiguous()
        return QuantizedKV(
            symbols=source.reshape(-1).contiguous(),
            metadata={
                "format": "mock_uint8_shaped",
                "dtype": str(kv.dtype).replace("torch.", ""),
                "shape": list(kv.shape),
                "origami_bits": 8,
                "origami_symbol_shape": list(source.shape),
                "origami_symbol_layout": layout,
            },
        )

    def dequantize(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
        *,
        dst_cache: torch.Tensor | None = None,
        block_ids: list[int] | tuple[int, ...] | None = None,
        stream: Any | None = None,
    ) -> torch.Tensor | Any:
        del stream
        layout = normalize_symbol_layout(metadata["origami_symbol_layout"])
        source_shape = [int(dim) for dim in metadata["origami_symbol_shape"]]
        source = symbols.detach().cpu().to(torch.uint8).reshape(source_shape)
        if "layer" in layout:
            source = source.squeeze(layout.index("layer"))
            layout = [axis for axis in layout if axis != "layer"]
        tensor = source.permute(self._permute_to_token_head_dim(layout)).contiguous()
        dtype_name = metadata.get("dtype", "float32")
        dtype = getattr(torch, str(dtype_name), torch.float32)
        tensor = tensor.to(dtype=dtype)
        if dst_cache is not None and block_ids is not None:
            block_ids_tensor = torch.tensor(
                list(block_ids), dtype=torch.long, device=dst_cache.device
            )
            dst_cache[block_ids_tensor] = tensor.to(
                dtype=dst_cache.dtype, device=dst_cache.device
            )
            if dst_cache.device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                return event
        return tensor
