# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import io
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.base import (
    QuantizedKV,
    QuantizerAdapter,
)


class TorchSerializedTensorAdapter(QuantizerAdapter):
    """Reversible adapter used by tests and by algorithm stubs.

    It preserves the QuantizerAdapter boundary while avoiding a dependency on a
    particular research quantizer in environments where those packages or model
    calibration artifacts are unavailable.
    """

    quantizer_id = "torch_serialized"

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, layer_group, stream
        buffer = io.BytesIO()
        torch.save(kv.detach().cpu().contiguous(), buffer)
        data = buffer.getvalue()
        symbols = torch.tensor(list(data), dtype=torch.uint8)
        return QuantizedKV(
            symbols=symbols,
            metadata={
                "format": "torch.save",
                "dtype": str(kv.dtype).replace("torch.", ""),
                "shape": list(kv.shape),
                "device": str(kv.device),
                "origami_bits": 8,
                "origami_symbol_shape": [1, 1, len(data)],
                "origami_symbol_layout": ["token", "head", "head_dim"],
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
        del metadata, stream
        data = bytes(symbols.detach().cpu().tolist())
        tensor = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
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

