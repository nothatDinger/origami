# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.base import (
    QuantizedKV,
    QuantizerAdapter,
)

KIVI_KEY_ORDER = ("head_dim", "layer", "head", "token")
KIVI_VALUE_ORDER = ("head", "layer", "head_dim", "token")


def _dtype_from_metadata(metadata: dict[str, Any]) -> torch.dtype:
    dtype_name = str(metadata.get("dtype", "float16")).replace("torch.", "")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported KIVI tensor dtype {dtype_name!r}")
    return dtype


def _pack_lowbit_values(values: torch.Tensor, bits: int) -> torch.Tensor:
    vals = values.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
    if vals.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8)
    if bits == 8:
        return vals.clone()
    if bits == 4:
        if vals.numel() % 2:
            vals = torch.cat([vals, torch.zeros(1, dtype=torch.uint8)])
        pairs = vals.reshape(-1, 2)
        return (pairs[:, 0] | (pairs[:, 1] << 4)).contiguous()
    if bits == 2:
        pad = (-int(vals.numel())) % 4
        if pad:
            vals = torch.cat([vals, torch.zeros(pad, dtype=torch.uint8)])
        groups = vals.reshape(-1, 4)
        return (
            (groups[:, 0] & 0x03)
            | ((groups[:, 1] & 0x03) << 2)
            | ((groups[:, 2] & 0x03) << 4)
            | ((groups[:, 3] & 0x03) << 6)
        ).contiguous()
    raise ValueError("KIVI adapter supports only 2, 4, or 8 bit packing")


def _unpack_lowbit_values(
    data: torch.Tensor,
    *,
    bits: int,
    count: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    count = int(count)
    src = data.detach().to(torch.uint8).reshape(-1).contiguous()
    if count == 0:
        return torch.empty((0,), dtype=torch.uint8, device=device)
    if bits == 8:
        return src[:count].clone().to(device=device)
    if bits == 4:
        out = torch.empty((int(src.numel()) * 2,), dtype=torch.uint8)
        out[0::2] = src.cpu() & 0x0F
        out[1::2] = (src.cpu() >> 4) & 0x0F
        return out[:count].contiguous().to(device=device)
    if bits == 2:
        out = torch.empty((int(src.numel()) * 4,), dtype=torch.uint8)
        src_cpu = src.cpu()
        out[0::4] = src_cpu & 0x03
        out[1::4] = (src_cpu >> 2) & 0x03
        out[2::4] = (src_cpu >> 4) & 0x03
        out[3::4] = (src_cpu >> 6) & 0x03
        return out[:count].contiguous().to(device=device)
    raise ValueError("KIVI adapter supports only 2, 4, or 8 bit packing")


def _pack_ordered_lowbit(
    tensor: torch.Tensor,
    order: tuple[str, ...],
    bits: int,
) -> torch.Tensor:
    axes = ("layer", "token", "head", "head_dim")
    dims = [axes.index(axis) for axis in order]
    ordered = tensor.permute(*dims).contiguous()
    return _pack_lowbit_values(ordered.reshape(-1), bits)


def _unpack_ordered_lowbit(
    data: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    order: tuple[str, ...],
    bits: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    axes = ("layer", "token", "head", "head_dim")
    dims = [axes.index(axis) for axis in order]
    ordered_shape = tuple(shape[dim] for dim in dims)
    values = _unpack_lowbit_values(
        data, bits=bits, count=reduce(mul, shape, 1), device=device
    )
    ordered = values.reshape(ordered_shape)
    inverse = [0] * len(dims)
    for idx, dim in enumerate(dims):
        inverse[dim] = idx
    return ordered.permute(*inverse).contiguous()


class KiviAdapter(QuantizerAdapter):
    """KIVI-style KV quantizer for Origami artifacts.

    The adapter preserves KIVI's layout-friendly byte streams for transmission:
    keys are ordered as head_dim > layer > head > token, values are ordered as
    head > layer > head_dim > token.  Lossless backends see these packed byte
    streams directly instead of a generic tensor order.
    """

    quantizer_id = "kivi"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.bits = int(self.config.get("bits", self.config.get("kivi_bits", 2)))
        self.group_size = int(
            self.config.get("group_size", self.config.get("kivi_group_size", 64))
        )
        self.sink_tokens = int(
            self.config.get("sink_tokens", self.config.get("kivi_sink_tokens", 128))
        )
        self.dequant_device = str(self.config.get("dequant_device", "auto")).lower()
        self.last_profile: list[dict[str, Any]] = []
        if self.bits not in {2, 4, 8}:
            raise ValueError("KiviAdapter bits must be one of {2, 4, 8}")
        if self.group_size <= 0:
            raise ValueError("KiviAdapter group_size must be positive")
        if self.sink_tokens < 0:
            raise ValueError("KiviAdapter sink_tokens must be non-negative")
        if self.dequant_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError(
                "KiviAdapter dequant_device must be one of "
                "{'auto', 'cpu', 'gpu', 'cuda'}"
            )

    @staticmethod
    def _canonicalize(kv: torch.Tensor) -> tuple[torch.Tensor, str]:
        tensor = kv.detach().contiguous()
        if tensor.dim() == 5 and int(tensor.shape[0]) == 2:
            return (
                tensor.reshape(
                    2,
                    int(tensor.shape[1]) * int(tensor.shape[2]),
                    int(tensor.shape[3]),
                    int(tensor.shape[4]),
                ),
                "kv_blocks_block_heads_dim",
            )
        if tensor.dim() == 5 and int(tensor.shape[1]) == 2:
            return (
                tensor.permute(1, 0, 2, 3, 4).contiguous().reshape(
                    2,
                    int(tensor.shape[0]) * int(tensor.shape[2]),
                    int(tensor.shape[3]),
                    int(tensor.shape[4]),
                ),
                "blocks_kv_block_heads_dim",
            )
        if tensor.dim() == 5 and int(tensor.shape[2]) == 2:
            return (
                tensor.permute(2, 0, 1, 3, 4).contiguous().reshape(
                    2,
                    int(tensor.shape[0]) * int(tensor.shape[1]),
                    int(tensor.shape[3]),
                    int(tensor.shape[4]),
                ),
                "blocks_block_kv_heads_dim",
            )
        if tensor.dim() == 4 and int(tensor.shape[0]) == 2:
            if int(tensor.shape[1]) <= 128 and int(tensor.shape[2]) > int(tensor.shape[1]):
                return (
                    tensor.permute(0, 2, 1, 3).contiguous(),
                    "kv_heads_tokens_dim",
                )
            return tensor, "kv_tokens_heads_dim"
        if tensor.dim() == 4 and int(tensor.shape[1]) == 2:
            return tensor.permute(1, 0, 2, 3).contiguous(), "tokens_kv_heads_dim"
        raise ValueError(
            "KiviAdapter expects KV cache layer with a key/value axis; "
            f"got shape {tuple(kv.shape)}"
        )

    @staticmethod
    def _restore_layout(canonical: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        shape = [int(dim) for dim in metadata["shape"]]
        layout = str(metadata["kv_layout"])
        if layout == "kv_blocks_block_heads_dim":
            return canonical.reshape(shape)
        if layout == "blocks_kv_block_heads_dim":
            return canonical.reshape(
                2, shape[0], shape[2], shape[3], shape[4]
            ).permute(1, 0, 2, 3, 4).contiguous()
        if layout == "blocks_block_kv_heads_dim":
            return canonical.reshape(
                2, shape[0], shape[1], shape[3], shape[4]
            ).permute(1, 2, 0, 3, 4).contiguous()
        if layout == "kv_tokens_heads_dim":
            return canonical.reshape(shape)
        if layout == "kv_heads_tokens_dim":
            return canonical.permute(0, 2, 1, 3).contiguous().reshape(shape)
        if layout == "tokens_kv_heads_dim":
            return canonical.permute(1, 0, 2, 3).contiguous().reshape(shape)
        raise ValueError(f"Unsupported KIVI KV layout {layout!r}")

    @staticmethod
    def _quantize_key_per_channel(
        matrix: torch.Tensor,
        group_size: int,
        bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        groups = (rows + group_size - 1) // group_size
        if rows == 0:
            return (
                torch.empty((0, cols), dtype=torch.uint8),
                torch.empty((0, cols), dtype=torch.float16),
                torch.empty((0, cols), dtype=torch.float16),
            )
        pad = groups * group_size - rows
        padded = matrix if pad == 0 else torch.cat(
            [matrix, matrix[-1:].expand(pad, cols)], dim=0
        )
        grouped = padded.reshape(groups, group_size, cols)
        zero = torch.amin(grouped, dim=1)
        maxv = torch.amax(grouped, dim=1)
        levels = float((1 << bits) - 1)
        scale = torch.clamp((maxv - zero) / levels, min=torch.finfo(torch.float32).tiny)
        q = torch.round((grouped - zero[:, None, :]) / scale[:, None, :])
        q = q.clamp(0, levels).to(torch.uint8)
        return (
            q.reshape(groups * group_size, cols)[:rows].contiguous(),
            scale.to(torch.float16).contiguous(),
            zero.to(torch.float16).contiguous(),
        )

    @staticmethod
    def _quantize_value_per_token(
        matrix: torch.Tensor,
        group_size: int,
        bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        groups = (cols + group_size - 1) // group_size
        if rows == 0:
            return (
                torch.empty((0, cols), dtype=torch.uint8),
                torch.empty((0, groups), dtype=torch.float16),
                torch.empty((0, groups), dtype=torch.float16),
            )
        pad = groups * group_size - cols
        padded = matrix if pad == 0 else torch.cat(
            [matrix, matrix[:, -1:].expand(rows, pad)], dim=1
        )
        grouped = padded.reshape(rows, groups, group_size)
        zero = torch.amin(grouped, dim=2)
        maxv = torch.amax(grouped, dim=2)
        levels = float((1 << bits) - 1)
        scale = torch.clamp((maxv - zero) / levels, min=torch.finfo(torch.float32).tiny)
        q = torch.round((grouped - zero[:, :, None]) / scale[:, :, None])
        q = q.clamp(0, levels).to(torch.uint8)
        return (
            q.reshape(rows, groups * group_size)[:, :cols].contiguous(),
            scale.to(torch.float16).contiguous(),
            zero.to(torch.float16).contiguous(),
        )

    @staticmethod
    def _fp16_bytes(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.detach()
            .cpu()
            .to(torch.float16)
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)
        )

    @staticmethod
    def _append_part(
        parts: list[torch.Tensor],
        offsets: dict[str, int],
        cursor: int,
        name: str,
        data: torch.Tensor,
        *,
        align: int = 1,
    ) -> int:
        if align > 1:
            pad = (-cursor) % align
            if pad:
                parts.append(torch.zeros((pad,), dtype=torch.uint8))
                cursor += pad
        offsets[name] = cursor
        part = data.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        parts.append(part)
        return cursor + int(part.numel())

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, layer_group, stream
        canonical, layout = self._canonicalize(kv)
        canonical_cpu = canonical.detach().cpu().to(torch.float16).contiguous()
        tokens = int(canonical_cpu.shape[1])
        heads = int(canonical_cpu.shape[2])
        head_dim = int(canonical_cpu.shape[3])
        sink_tokens = min(self.sink_tokens, tokens)
        body_tokens = tokens - sink_tokens
        cols = heads * head_dim

        key_sink = canonical_cpu[0, :sink_tokens].contiguous()
        value_sink = canonical_cpu[1, :sink_tokens].contiguous()
        key_body = canonical_cpu[0, sink_tokens:].reshape(body_tokens, cols).float()
        value_body = canonical_cpu[1, sink_tokens:].reshape(body_tokens, cols).float()

        key_q, key_scale, key_zero = self._quantize_key_per_channel(
            key_body, self.group_size, self.bits
        )
        value_q, value_scale, value_zero = self._quantize_value_per_token(
            value_body, self.group_size, self.bits
        )
        data_shape = (1, body_tokens, heads, head_dim)
        key_stream = _pack_ordered_lowbit(
            key_q.reshape(data_shape), KIVI_KEY_ORDER, self.bits
        )
        value_stream = _pack_ordered_lowbit(
            value_q.reshape(data_shape), KIVI_VALUE_ORDER, self.bits
        )

        parts: list[torch.Tensor] = []
        offsets: dict[str, int] = {}
        cursor = 0
        cursor = self._append_part(parts, offsets, cursor, "key_stream", key_stream)
        cursor = self._append_part(parts, offsets, cursor, "value_stream", value_stream)
        cursor = self._append_part(
            parts, offsets, cursor, "key_scale", self._fp16_bytes(key_scale), align=2
        )
        cursor = self._append_part(
            parts, offsets, cursor, "key_zero", self._fp16_bytes(key_zero), align=2
        )
        cursor = self._append_part(
            parts,
            offsets,
            cursor,
            "value_scale",
            self._fp16_bytes(value_scale),
            align=2,
        )
        cursor = self._append_part(
            parts, offsets, cursor, "value_zero", self._fp16_bytes(value_zero), align=2
        )
        cursor = self._append_part(
            parts, offsets, cursor, "key_sink", self._fp16_bytes(key_sink), align=2
        )
        cursor = self._append_part(
            parts,
            offsets,
            cursor,
            "value_sink",
            self._fp16_bytes(value_sink),
            align=2,
        )
        symbols = torch.cat(parts, dim=0).contiguous() if parts else torch.empty(
            (0,), dtype=torch.uint8
        )

        key_groups = (body_tokens + self.group_size - 1) // self.group_size
        value_groups = (cols + self.group_size - 1) // self.group_size
        metadata = {
            "format": "kivi_structured_blob",
            "version": 1,
            "bits": self.bits,
            "group_size": self.group_size,
            "sink_tokens": sink_tokens,
            "body_tokens": body_tokens,
            "token_count": tokens,
            "num_heads": heads,
            "head_dim": head_dim,
            "channel_count": cols,
            "key_groups": key_groups,
            "value_groups": value_groups,
            "data_shape": list(data_shape),
            "key_order": list(KIVI_KEY_ORDER),
            "value_order": list(KIVI_VALUE_ORDER),
            "dtype": str(kv.dtype).replace("torch.", ""),
            "shape": [int(dim) for dim in kv.shape],
            "kv_layout": layout,
            "symbol_byte_count": int(symbols.numel()),
            "origami_bits": 8,
            "origami_symbol_shape": [1, 1, int(symbols.numel())],
            "origami_symbol_layout": ["token", "head", "head_dim"],
        }
        for name, offset in offsets.items():
            metadata[f"{name}_offset"] = int(offset)
        metadata.update({
            "key_stream_bytes": int(key_stream.numel()),
            "value_stream_bytes": int(value_stream.numel()),
            "key_scale_numel": int(key_scale.numel()),
            "key_zero_numel": int(key_zero.numel()),
            "value_scale_numel": int(value_scale.numel()),
            "value_zero_numel": int(value_zero.numel()),
            "key_sink_numel": int(key_sink.numel()),
            "value_sink_numel": int(value_sink.numel()),
        })
        return QuantizedKV(symbols=symbols, metadata=metadata)

    @staticmethod
    def _read_fp16(
        raw: torch.Tensor,
        offset: int,
        numel: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        view = raw.narrow(0, int(offset), int(numel) * 2).contiguous()
        if device is not None:
            view = view.to(device=device, non_blocking=True)
        return view.view(torch.float16)

    def _dequantize_cpu(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        raw = symbols.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        expected = int(metadata["symbol_byte_count"])
        if int(raw.numel()) != expected:
            raise RuntimeError(
                f"KIVI payload size mismatch: got {int(raw.numel())}, expected {expected}"
            )
        tokens = int(metadata["token_count"])
        body_tokens = int(metadata["body_tokens"])
        sink_tokens = int(metadata["sink_tokens"])
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        cols = heads * head_dim
        bits = int(metadata["bits"])
        group_size = int(metadata["group_size"])
        data_shape = tuple(int(dim) for dim in metadata["data_shape"])

        key_stream = raw.narrow(
            0, int(metadata["key_stream_offset"]), int(metadata["key_stream_bytes"])
        )
        value_stream = raw.narrow(
            0,
            int(metadata["value_stream_offset"]),
            int(metadata["value_stream_bytes"]),
        )
        key_q = _unpack_ordered_lowbit(
            key_stream, shape=data_shape, order=tuple(metadata["key_order"]), bits=bits
        ).reshape(body_tokens, cols)
        value_q = _unpack_ordered_lowbit(
            value_stream,
            shape=data_shape,
            order=tuple(metadata["value_order"]),
            bits=bits,
        ).reshape(body_tokens, cols)
        key_scale = self._read_fp16(
            raw, int(metadata["key_scale_offset"]), int(metadata["key_scale_numel"])
        ).reshape(int(metadata["key_groups"]), cols).float()
        key_zero = self._read_fp16(
            raw, int(metadata["key_zero_offset"]), int(metadata["key_zero_numel"])
        ).reshape(int(metadata["key_groups"]), cols).float()
        value_scale = self._read_fp16(
            raw, int(metadata["value_scale_offset"]), int(metadata["value_scale_numel"])
        ).reshape(body_tokens, int(metadata["value_groups"])).float()
        value_zero = self._read_fp16(
            raw, int(metadata["value_zero_offset"]), int(metadata["value_zero_numel"])
        ).reshape(body_tokens, int(metadata["value_groups"])).float()
        key_sink = self._read_fp16(
            raw, int(metadata["key_sink_offset"]), int(metadata["key_sink_numel"])
        ).reshape(sink_tokens, heads, head_dim)
        value_sink = self._read_fp16(
            raw, int(metadata["value_sink_offset"]), int(metadata["value_sink_numel"])
        ).reshape(sink_tokens, heads, head_dim)

        if body_tokens > 0:
            row_groups = torch.arange(body_tokens, dtype=torch.long) // group_size
            col_groups = torch.arange(cols, dtype=torch.long) // group_size
            key_body = key_q.float() * key_scale[row_groups, :] + key_zero[row_groups, :]
            value_body = (
                value_q.float() * value_scale[:, col_groups]
                + value_zero[:, col_groups]
            )
            key = torch.cat(
                [
                    key_sink,
                    key_body.reshape(body_tokens, heads, head_dim).to(torch.float16),
                ],
                dim=0,
            )
            value = torch.cat(
                [
                    value_sink,
                    value_body.reshape(body_tokens, heads, head_dim).to(torch.float16),
                ],
                dim=0,
            )
        else:
            key = key_sink
            value = value_sink
        if int(key.shape[0]) != tokens or int(value.shape[0]) != tokens:
            raise RuntimeError("KIVI dequantized token count mismatch")
        canonical = torch.stack([key, value], dim=0).to(dtype=_dtype_from_metadata(metadata))
        return self._restore_layout(canonical, metadata)

    def _use_gpu_dequantize(self) -> bool:
        if self.dequant_device == "cpu":
            return False
        if not torch.cuda.is_available():
            if self.dequant_device in {"gpu", "cuda"}:
                raise RuntimeError("KiviAdapter GPU dequant requested but CUDA is unavailable")
            return False
        return True

    def can_dequantize_to_cache(
        self,
        metadata: dict[str, Any],
        dst_cache: torch.Tensor,
        block_ids: list[int] | tuple[int, ...],
        *,
        tokens: int | None = None,
    ) -> bool:
        if str(metadata.get("format")) != "kivi_structured_blob":
            return False
        if not self._use_gpu_dequantize():
            return False
        if dst_cache.device.type != "cuda" or dst_cache.dtype not in {
            torch.float16,
            torch.bfloat16,
        }:
            return False
        if dst_cache.dim() != 5 or not block_ids:
            return False
        if int(dst_cache.shape[1]) == 2:
            block_size = int(dst_cache.shape[2])
            heads = int(dst_cache.shape[3])
            head_dim = int(dst_cache.shape[4])
        elif int(dst_cache.shape[0]) == 2:
            block_size = int(dst_cache.shape[2])
            heads = int(dst_cache.shape[3])
            head_dim = int(dst_cache.shape[4])
        else:
            return False
        restore_tokens = int(tokens if tokens is not None else metadata["token_count"])
        if restore_tokens <= 0 or restore_tokens % block_size != 0:
            return False
        if restore_tokens > len(block_ids) * block_size:
            return False
        return (
            restore_tokens == int(metadata["token_count"])
            and heads == int(metadata["num_heads"])
            and head_dim == int(metadata["head_dim"])
        )

    def _dequantize_gpu_to_cache(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
        dst_cache: torch.Tensor,
        block_ids: list[int] | tuple[int, ...],
        *,
        tokens: int,
    ) -> torch.cuda.Event:
        from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
            native_gpu,
        )

        self.last_profile = []
        raw = symbols.detach().reshape(-1)
        h2d_bytes = int(raw.numel() * raw.element_size())
        h2d_events = None
        if raw.device.type != "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            raw = raw.to(device=dst_cache.device, dtype=torch.uint8, non_blocking=True)
            end_event.record()
            h2d_events = (start_event, end_event, str(raw.device))
        elif raw.device != dst_cache.device:
            raw = raw.to(device=dst_cache.device, dtype=torch.uint8, non_blocking=True)
        elif raw.dtype != torch.uint8:
            raw = raw.to(torch.uint8)
        if not raw.is_contiguous():
            raw = raw.contiguous()

        def read_fp16(name: str, numel_name: str) -> torch.Tensor:
            return self._read_fp16(
                raw,
                int(metadata[f"{name}_offset"]),
                int(metadata[numel_name]),
                device=dst_cache.device,
            )

        key_scale = read_fp16("key_scale", "key_scale_numel")
        key_zero = read_fp16("key_zero", "key_zero_numel")
        value_scale = read_fp16("value_scale", "value_scale_numel")
        value_zero = read_fp16("value_zero", "value_zero_numel")
        key_sink = read_fp16("key_sink", "key_sink_numel")
        value_sink = read_fp16("value_sink", "value_sink_numel")
        block_ids_tensor = torch.tensor(
            list(block_ids), dtype=torch.long, device=dst_cache.device
        )

        pushed = False
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        try:
            torch.cuda.nvtx.range_push("origami:kivi_dequantize_to_kv_cuda")
            pushed = True
        except Exception:
            pushed = False
        try:
            start_event.record()
            native_gpu.kivi_dequantize_to_kv_cache(
                raw,
                key_offset=int(metadata["key_stream_offset"]),
                key_bytes=int(metadata["key_stream_bytes"]),
                value_offset=int(metadata["value_stream_offset"]),
                value_bytes=int(metadata["value_stream_bytes"]),
                bits=int(metadata["bits"]),
                group_size=int(metadata["group_size"]),
                sink_tokens=int(metadata["sink_tokens"]),
                key_scale=key_scale,
                key_zero=key_zero,
                value_scale=value_scale,
                value_zero=value_zero,
                key_sink=key_sink,
                value_sink=value_sink,
                kv_cache=dst_cache,
                block_ids=block_ids_tensor,
                tokens=int(tokens),
                heads=int(metadata["num_heads"]),
                head_dim=int(metadata["head_dim"]),
            )
            end_event.record()
            end_event.synchronize()
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()

        if h2d_events is not None:
            self.last_profile.append({
                "type": "kivi_bitstream_h2d_cuda",
                "device": h2d_events[2],
                "input_bytes": h2d_bytes,
                "output_bytes": h2d_bytes,
                "bytes": h2d_bytes,
                "_cuda_start_event": h2d_events[0],
                "_cuda_end_event": h2d_events[1],
            })
        output_bytes = int(tokens * 2 * int(metadata["num_heads"]) *
                           int(metadata["head_dim"]) * dst_cache.element_size())
        input_bytes = int(metadata["symbol_byte_count"])
        self.last_profile.append({
            "type": "origami_kivi_dequantize_to_kv_cuda",
            "device": str(dst_cache.device),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "bytes": output_bytes,
            "_cuda_start_event": start_event,
            "_cuda_end_event": end_event,
        })
        return end_event

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
        if str(metadata.get("format")) != "kivi_structured_blob":
            raise ValueError("KiviAdapter can only dequantize KIVI Origami payloads")
        if (
            dst_cache is not None
            and block_ids is not None
            and self.can_dequantize_to_cache(
                metadata, dst_cache, block_ids, tokens=int(metadata["token_count"])
            )
        ):
            return self._dequantize_gpu_to_cache(
                symbols,
                metadata,
                dst_cache,
                block_ids,
                tokens=int(metadata["token_count"]),
            )

        tensor = self._dequantize_cpu(symbols, metadata)
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
