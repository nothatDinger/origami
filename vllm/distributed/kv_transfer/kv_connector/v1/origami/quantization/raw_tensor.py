# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import io
from functools import reduce
from operator import mul
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
        symbols = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
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


def _dtype_from_metadata(metadata: dict[str, Any]) -> torch.dtype:
    dtype_name = str(metadata.get("dtype", "float32")).replace("torch.", "")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported raw-bytes tensor dtype {dtype_name!r}")
    return dtype


class RawBytesTensorAdapter(QuantizerAdapter):
    """Lossless byte-symbol adapter for Origami benchmark artifacts.

    This is the benchmark path for raw lossless Origami reuse: QAT/zlib restores
    packed bytes, the native bitunpack step reassembles this uint8 stream, and
    dequantize materializes the original tensor via a dtype view. It avoids the
    legacy torch.save/torch.load serializer so profiling measures the intended
    QAT + AVX512 bitunpack + KV restore pipeline.
    """

    quantizer_id = "raw_bytes"

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, layer_group, stream
        tensor = kv.detach().cpu().contiguous()
        symbols = tensor.view(torch.uint8).reshape(-1).clone()
        shape = [int(dim) for dim in tensor.shape]
        return QuantizedKV(
            symbols=symbols,
            metadata={
                "format": "raw_bytes",
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": shape,
                "byte_count": int(symbols.numel()),
                "origami_bits": 8,
                "origami_symbol_shape": [1, 1, int(symbols.numel())],
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
        del stream
        raw = symbols.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        dtype = _dtype_from_metadata(metadata)
        shape = [int(dim) for dim in metadata["shape"]]
        element_size = torch.empty((), dtype=dtype).element_size()
        expected_bytes = reduce(mul, shape, 1) * element_size
        if int(raw.numel()) != int(expected_bytes):
            raise RuntimeError(
                "Origami raw-bytes payload size mismatch: "
                f"got {int(raw.numel())}, expected {int(expected_bytes)}"
            )
        tensor = raw.view(dtype).reshape(shape).contiguous()
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
