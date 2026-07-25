# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.base import (
    QuantizedKV,
    QuantizerAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kivi_adapter import (
    KiviAdapter,
    _pack_ordered_lowbit,
    _unpack_ordered_lowbit,
)

KVQUANT_KEY_ORDER = ("head_dim", "layer", "head", "token")
KVQUANT_VALUE_ORDER = ("layer", "head", "token", "head_dim")


def _dtype_from_metadata(metadata: dict[str, Any]) -> torch.dtype:
    dtype_name = str(metadata.get("dtype", "float16")).replace("torch.", "")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported KVQuant tensor dtype {dtype_name!r}")
    return dtype


def _quant_minmax_codes(
    matrix: torch.Tensor,
    *,
    dim: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = float((1 << int(bits)) - 1)
    minv = torch.amin(matrix, dim=dim, keepdim=True)
    maxv = torch.amax(matrix, dim=dim, keepdim=True)
    scale = torch.clamp((maxv - minv) / levels, min=torch.finfo(torch.float32).tiny)
    q = torch.round((matrix - minv) / scale).clamp_(0, levels).to(torch.uint8)
    return (
        q.contiguous(),
        minv.squeeze(dim).to(torch.float16).contiguous(),
        scale.squeeze(dim).to(torch.float16).contiguous(),
    )


def _fp16_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return (
        tensor.detach()
        .cpu()
        .to(torch.float16)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
    )


def _read_fp16(raw: torch.Tensor, offset: int, numel: int) -> torch.Tensor:
    return (
        raw.narrow(0, int(offset), int(numel) * 2)
        .contiguous()
        .view(torch.float16)
    )


def _canonical_to_matrices(canonical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = int(canonical.shape[1])
    heads = int(canonical.shape[2])
    head_dim = int(canonical.shape[3])
    key = canonical[0].reshape(tokens, heads * head_dim).float()
    value = canonical[1].reshape(tokens, heads * head_dim).float()
    return key, value


class KVQuantAdapter(QuantizerAdapter):
    """KVQuant-style 2-bit min/max KV quantizer.

    This adapter intentionally stops after quantization.  It emits one flat
    uint8 payload containing packed low-bit key/value streams and fp16 metadata;
    any lossless stage is outside this adapter and can be bypassed with the raw
    Origami backend for quality-only tests.
    """

    quantizer_id = "kvquant"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.bits = int(self.config.get("bits", self.config.get("kvquant_bits", 2)))
        if self.bits not in {2, 3, 4, 8}:
            raise ValueError("KVQuantAdapter bits must be one of {2, 3, 4, 8}")

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, layer_group, stream
        canonical, layout = KiviAdapter._canonicalize(kv)
        canonical_cpu = canonical.detach().cpu().to(torch.float16).contiguous()
        tokens = int(canonical_cpu.shape[1])
        heads = int(canonical_cpu.shape[2])
        head_dim = int(canonical_cpu.shape[3])
        cols = heads * head_dim

        key_matrix, value_matrix = _canonical_to_matrices(canonical_cpu)
        key_q, key_min, key_scale = _quant_minmax_codes(
            key_matrix, dim=0, bits=self.bits
        )
        value_q, value_min, value_scale = _quant_minmax_codes(
            value_matrix, dim=1, bits=self.bits
        )

        data_shape = (1, tokens, heads, head_dim)
        key_stream = _pack_ordered_lowbit(
            key_q.reshape(data_shape), KVQUANT_KEY_ORDER, self.bits
        )
        value_stream = _pack_ordered_lowbit(
            value_q.reshape(data_shape), KVQUANT_VALUE_ORDER, self.bits
        )

        parts: list[torch.Tensor] = []
        offsets: dict[str, int] = {}
        cursor = 0
        for name, part in (
            ("key_stream", key_stream),
            ("value_stream", value_stream),
            ("key_min", _fp16_bytes(key_min)),
            ("key_scale", _fp16_bytes(key_scale)),
            ("value_min", _fp16_bytes(value_min)),
            ("value_scale", _fp16_bytes(value_scale)),
        ):
            if name.endswith(("min", "scale")) and cursor % 2:
                parts.append(torch.zeros((1,), dtype=torch.uint8))
                cursor += 1
            offsets[name] = cursor
            flat = part.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
            parts.append(flat)
            cursor += int(flat.numel())
        symbols = torch.cat(parts, dim=0).contiguous()

        metadata: dict[str, Any] = {
            "format": "kvquant_structured_blob",
            "version": 1,
            "bits": self.bits,
            "dtype": str(kv.dtype).replace("torch.", ""),
            "shape": [int(dim) for dim in kv.shape],
            "kv_layout": layout,
            "token_count": tokens,
            "num_heads": heads,
            "head_dim": head_dim,
            "channel_count": cols,
            "data_shape": list(data_shape),
            "key_order": list(KVQUANT_KEY_ORDER),
            "value_order": list(KVQUANT_VALUE_ORDER),
            "key_stream_bytes": int(key_stream.numel()),
            "value_stream_bytes": int(value_stream.numel()),
            "key_min_numel": int(key_min.numel()),
            "key_scale_numel": int(key_scale.numel()),
            "value_min_numel": int(value_min.numel()),
            "value_scale_numel": int(value_scale.numel()),
            "symbol_byte_count": int(symbols.numel()),
            "origami_bits": 8,
            "origami_symbol_shape": [1, 1, int(symbols.numel())],
            "origami_symbol_layout": ["token", "head", "head_dim"],
        }
        for name, offset in offsets.items():
            metadata[f"{name}_offset"] = int(offset)
        return QuantizedKV(symbols=symbols, metadata=metadata)

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
        if str(metadata.get("format")) != "kvquant_structured_blob":
            raise ValueError("KVQuantAdapter can only dequantize KVQuant payloads")
        raw = symbols.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        expected = int(metadata["symbol_byte_count"])
        if int(raw.numel()) != expected:
            raise RuntimeError(
                f"KVQuant payload size mismatch: got {int(raw.numel())}, expected {expected}"
            )

        tokens = int(metadata["token_count"])
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        cols = heads * head_dim
        bits = int(metadata["bits"])
        data_shape = tuple(int(dim) for dim in metadata["data_shape"])

        key_stream = raw.narrow(
            0, int(metadata["key_stream_offset"]), int(metadata["key_stream_bytes"])
        )
        value_stream = raw.narrow(
            0, int(metadata["value_stream_offset"]), int(metadata["value_stream_bytes"])
        )
        key_q = _unpack_ordered_lowbit(
            key_stream,
            shape=data_shape,
            order=tuple(metadata["key_order"]),
            bits=bits,
        ).reshape(tokens, cols)
        value_q = _unpack_ordered_lowbit(
            value_stream,
            shape=data_shape,
            order=tuple(metadata["value_order"]),
            bits=bits,
        ).reshape(tokens, cols)

        key_min = _read_fp16(
            raw, int(metadata["key_min_offset"]), int(metadata["key_min_numel"])
        ).reshape(cols).float()
        key_scale = _read_fp16(
            raw, int(metadata["key_scale_offset"]), int(metadata["key_scale_numel"])
        ).reshape(cols).float()
        value_min = _read_fp16(
            raw, int(metadata["value_min_offset"]), int(metadata["value_min_numel"])
        ).reshape(tokens).float()
        value_scale = _read_fp16(
            raw, int(metadata["value_scale_offset"]), int(metadata["value_scale_numel"])
        ).reshape(tokens).float()

        key = key_q.float() * key_scale[None, :] + key_min[None, :]
        value = value_q.float() * value_scale[:, None] + value_min[:, None]
        canonical = torch.stack(
            [
                key.reshape(tokens, heads, head_dim),
                value.reshape(tokens, heads, head_dim),
            ],
            dim=0,
        ).to(dtype=_dtype_from_metadata(metadata))
        tensor = KiviAdapter._restore_layout(canonical, metadata)
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
