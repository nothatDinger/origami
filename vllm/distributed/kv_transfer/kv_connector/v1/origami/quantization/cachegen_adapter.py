# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
    native_cpu,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.base import (
    QuantizedKV,
    QuantizerAdapter,
)


@dataclass(frozen=True)
class _CacheGenBins:
    key_first_layers: int
    key_second_layers: int
    key_third_layers: int
    key_first_bins: int
    key_second_bins: int
    key_third_bins: int
    value_first_layers: int
    value_first_bins: int
    value_second_bins: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[8]


def _setup_cachegen_basics_path() -> None:
    root = _repo_root() / "thrid_party" / "CacheGen"
    for path in (root, root / "LMCache"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _fallback_bins(model_name: str, quant_level: str) -> _CacheGenBins:
    total_layers = 36 if "Qwen3-8B" in model_name or "Qwen3_8B" in model_name else 32
    if quant_level == "1":
        return _CacheGenBins(10, 20, total_layers, 16, 16, 12, 2, 16, 12)
    if quant_level == "2":
        return _CacheGenBins(10, 20, total_layers, 32, 16, 16, 2, 32, 16)
    if quant_level == "3":
        return _CacheGenBins(10, 20, total_layers, 32, 32, 32, 2, 32, 32)
    raise ValueError(f"Unsupported CacheGen quant_level {quant_level!r}")


def _load_cachegen_bins(model_name: str, quant_level: str) -> _CacheGenBins:
    try:
        _setup_cachegen_basics_path()
        old = os.environ.get("QUANT_LEVEL")
        os.environ["QUANT_LEVEL"] = str(quant_level)
        try:
            from lmcache.storage_backend.serde.cachegen_basics import (
                CacheGenConfig,
            )

            cfg = CacheGenConfig.from_model_name(model_name)
        finally:
            if old is None:
                os.environ.pop("QUANT_LEVEL", None)
            else:
                os.environ["QUANT_LEVEL"] = old
        return _CacheGenBins(
            key_first_layers=int(cfg.key_first_layers),
            key_second_layers=int(cfg.key_second_layers),
            key_third_layers=int(cfg.key_third_layers),
            key_first_bins=int(cfg.key_first_bins),
            key_second_bins=int(cfg.key_second_bins),
            key_third_bins=int(cfg.key_third_bins),
            value_first_layers=int(cfg.value_first_layers),
            value_first_bins=int(cfg.value_first_bins),
            value_second_bins=int(cfg.value_second_bins),
        )
    except Exception:
        return _fallback_bins(model_name, quant_level)


def _dtype_from_metadata(metadata: dict[str, Any]) -> torch.dtype:
    dtype_name = str(metadata.get("dtype", "float16")).replace("torch.", "")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported CacheGen tensor dtype {dtype_name!r}")
    return dtype


def _layer_index(layer_group: Any | None) -> int:
    if layer_group is None:
        return 0
    digits = ""
    for ch in reversed(str(layer_group)):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else 0


def _bits_for_bins(num_bins: int) -> int:
    if num_bins <= 1:
        return 1
    return max(1, min(8, int(num_bins - 1).bit_length()))


def _pack_linear_bits(symbols: torch.Tensor, bits: int) -> torch.Tensor:
    flat = symbols.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
    if int(flat.numel()) == 0:
        return torch.empty((0,), dtype=torch.uint8)
    specs = [[0, 1, 0, int(flat.numel()), 0, 1]]
    return native_cpu.pack_canonical_storage_chunks(
        flat,
        int(bits),
        [1, 1, int(flat.numel())],
        ["token", "head", "head_dim"],
        specs,
    )[0]


def _unpack_linear_bits(raw: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    flat = raw.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
    if int(count) == 0:
        return torch.empty((0,), dtype=torch.uint8)
    specs = [[0, 1, 0, int(count), 0, 1]]
    return native_cpu.unpack_canonical_storage_chunks(
        [flat],
        int(bits),
        [1, 1, int(count)],
        ["token", "head", "head_dim"],
        specs,
    ).reshape(-1)[: int(count)].contiguous()


class CacheGenAdapter(QuantizerAdapter):
    """CacheGen-style per-layer quantizer for Origami artifacts.

    This reuses CacheGen's layer-dependent bin schedule and max-abs affine
    quantization, but leaves the lossless entropy/deflate stage to Origami's
    QAT pipeline. Each layer payload stores quantized K/V symbols plus fp16
    per-token max tensors in a compact byte blob.
    """

    quantizer_id = "cachegen"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model_name = str(
            self.config.get("model_name", "/data/llm/Mistral-7B-Instruct-v0.2")
        )
        self.quant_level = str(self.config.get("quant_level", "2"))
        self.dequant_device = str(self.config.get("dequant_device", "auto")).lower()
        self.last_profile: list[dict[str, Any]] = []
        self.bins = _load_cachegen_bins(self.model_name, self.quant_level)
        if self.dequant_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError(
                "CacheGenAdapter dequant_device must be one of "
                "{'auto', 'cpu', 'gpu', 'cuda'}"
            )

    def _key_bins(self, layer: int) -> int:
        if layer < self.bins.key_first_layers:
            return self.bins.key_first_bins
        if layer < self.bins.key_second_layers:
            return self.bins.key_second_bins
        return self.bins.key_third_bins

    def _value_bins(self, layer: int) -> int:
        if layer < self.bins.value_first_layers:
            return self.bins.value_first_bins
        return self.bins.value_second_bins

    @staticmethod
    def _canonicalize(kv: torch.Tensor) -> tuple[torch.Tensor, str]:
        tensor = kv.detach().contiguous()
        if tensor.dim() == 5 and int(tensor.shape[0]) == 2:
            return (
                tensor.reshape(
                    2, int(tensor.shape[1]) * int(tensor.shape[2]),
                    int(tensor.shape[3]), int(tensor.shape[4])
                ),
                "kv_blocks_block_heads_dim",
            )
        if tensor.dim() == 5 and int(tensor.shape[1]) == 2:
            return (
                tensor.permute(1, 0, 2, 3, 4).contiguous().reshape(
                    2, int(tensor.shape[0]) * int(tensor.shape[2]),
                    int(tensor.shape[3]), int(tensor.shape[4])
                ),
                "blocks_kv_block_heads_dim",
            )
        if tensor.dim() == 5 and int(tensor.shape[2]) == 2:
            return (
                tensor.permute(2, 0, 1, 3, 4).contiguous().reshape(
                    2, int(tensor.shape[0]) * int(tensor.shape[1]),
                    int(tensor.shape[3]), int(tensor.shape[4])
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
            "CacheGenAdapter expects KV cache layer with a key/value axis; "
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
        raise ValueError(f"Unsupported CacheGen KV layout {layout!r}")

    @staticmethod
    def _quantize_plane(
        plane: torch.Tensor,
        bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = int(plane.shape[0])
        flat = plane.reshape(tokens, -1).to(torch.float32)
        max_abs = torch.amax(torch.abs(flat), dim=-1, keepdim=True).clamp_min(1e-6)
        center = bins // 2 - 1
        quantized = torch.round(flat * (center / max_abs) + center)
        quantized = quantized.clamp(0, center * 2).to(torch.uint8)
        return quantized, max_abs.to(torch.float16)

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

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, stream
        canonical, layout = self._canonicalize(kv)
        layer = _layer_index(layer_group)
        key_bins = self._key_bins(layer)
        value_bins = self._value_bins(layer)
        key_bits = _bits_for_bins(key_bins)
        value_bits = _bits_for_bins(value_bins)
        key_q, key_max = self._quantize_plane(canonical[0], key_bins)
        value_q, value_max = self._quantize_plane(canonical[1], value_bins)

        key_numel = int(key_q.numel())
        value_numel = int(value_q.numel())
        key_symbols = key_q.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        value_symbols = value_q.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        key_max_bytes = self._fp16_bytes(key_max)
        value_max_bytes = self._fp16_bytes(value_max)
        parts = [key_symbols, value_symbols, key_max_bytes, value_max_bytes]
        offsets = []
        cursor = 0
        for part in parts:
            offsets.append(cursor)
            cursor += int(part.numel())
        symbols = torch.cat(parts, dim=0).contiguous()

        token_count = int(canonical.shape[1])
        num_heads = int(canonical.shape[2])
        head_dim = int(canonical.shape[3])
        channel_count = num_heads * head_dim
        return QuantizedKV(
            symbols=symbols,
            metadata={
                "format": "cachegen_structured_blob",
                "version": 2,
                "model_name": self.model_name,
                "quant_level": self.quant_level,
                "layer_index": layer,
                "key_bins": key_bins,
                "value_bins": value_bins,
                "key_bits": key_bits,
                "value_bits": value_bits,
                "dtype": str(kv.dtype).replace("torch.", ""),
                "shape": [int(dim) for dim in kv.shape],
                "kv_layout": layout,
                "token_count": token_count,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "channel_count": channel_count,
                "key_q_offset": offsets[0],
                "key_q_numel": key_numel,
                "value_q_offset": offsets[1],
                "value_q_numel": value_numel,
                "key_max_offset": offsets[2],
                "key_max_numel": int(key_max.numel()),
                "value_max_offset": offsets[3],
                "value_max_numel": int(value_max.numel()),
                "symbol_byte_count": int(symbols.numel()),
                "origami_bits": 8,
                "origami_symbol_shape": [1, 1, int(symbols.numel())],
                "origami_symbol_layout": ["token", "head", "head_dim"],
                "origami_data_layout_policy": "layer>head>head_dim>token",
            },
        )

    @staticmethod
    def _read_u8(
        raw: torch.Tensor,
        offset: int,
        numel: int,
    ) -> torch.Tensor:
        return raw.narrow(0, int(offset), int(numel)).contiguous()

    @staticmethod
    def _read_fp16(
        raw: torch.Tensor,
        offset: int,
        numel: int,
    ) -> torch.Tensor:
        return (
            raw.narrow(0, int(offset), int(numel) * 2)
            .contiguous()
            .view(torch.float16)
            .to(torch.float32)
            .reshape(int(numel), 1)
        )

    @staticmethod
    def _dequantize_plane(
        quantized: torch.Tensor,
        max_abs: torch.Tensor,
        bins: int,
    ) -> torch.Tensor:
        center = bins // 2 - 1
        return (quantized.to(torch.float32) - center) / center * max_abs

    def _use_gpu_dequantize(self) -> bool:
        if self.dequant_device == "cpu":
            return False
        if not torch.cuda.is_available():
            if self.dequant_device in {"gpu", "cuda"}:
                raise RuntimeError("CacheGenAdapter GPU dequant requested but CUDA is unavailable")
            return False
        return True

    @staticmethod
    def _read_fp16_from_u8(
        raw: torch.Tensor,
        offset: int,
        numel: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        view = raw.narrow(0, int(offset), int(numel) * 2).contiguous()
        if device is not None:
            view = view.to(device=device, non_blocking=True)
        return view.view(torch.float16).to(torch.float32).reshape(int(numel), 1)

    def _dequantize_gpu(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        if str(metadata.get("format")) == "cachegen_bitpacked_blob":
            return self._dequantize_bitpacked_gpu(symbols, metadata)
        raw = symbols.detach().reshape(-1)
        if raw.device.type != "cuda":
            raw = raw.to(device="cuda", dtype=torch.uint8, non_blocking=True)
        elif raw.dtype != torch.uint8:
            raw = raw.to(torch.uint8)
        if not raw.is_contiguous():
            raw = raw.contiguous()

        tokens = int(metadata["token_count"])
        channels = int(metadata["channel_count"])
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        key_q = self._read_u8(
            raw, int(metadata["key_q_offset"]), int(metadata["key_q_numel"])
        ).reshape(tokens, channels)
        value_q = self._read_u8(
            raw, int(metadata["value_q_offset"]), int(metadata["value_q_numel"])
        ).reshape(tokens, channels)
        key_max = self._read_fp16_from_u8(
            raw,
            int(metadata["key_max_offset"]),
            int(metadata["key_max_numel"]),
        )
        value_max = self._read_fp16_from_u8(
            raw,
            int(metadata["value_max_offset"]),
            int(metadata["value_max_numel"]),
        )
        nvtx_pushed = False
        try:
            torch.cuda.nvtx.range_push("origami:cachegen_gpu_dequant")
            nvtx_pushed = True
        except Exception:
            nvtx_pushed = False
        try:
            key = self._dequantize_plane(key_q, key_max, int(metadata["key_bins"]))
            value = self._dequantize_plane(value_q, value_max, int(metadata["value_bins"]))
            dtype = _dtype_from_metadata(metadata)
            canonical = torch.stack(
                [
                    key.reshape(tokens, heads, head_dim),
                    value.reshape(tokens, heads, head_dim),
                ],
                dim=0,
            ).to(dtype=dtype)
            return self._restore_layout(canonical, metadata)
        finally:
            if nvtx_pushed:
                torch.cuda.nvtx.range_pop()

    def _dequantize_bitpacked_gpu(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
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
            raw = raw.to(device="cuda", dtype=torch.uint8, non_blocking=True)
            end_event.record()
            h2d_events = (start_event, end_event, str(raw.device))
        elif raw.dtype != torch.uint8:
            raw = raw.to(torch.uint8)
        if not raw.is_contiguous():
            raw = raw.contiguous()

        tokens = int(metadata["token_count"])
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        key_offset = int(metadata["key_stream_offset"])
        key_bytes = int(metadata["key_stream_bytes"])
        value_offset = int(metadata["value_stream_offset"])
        value_bytes = int(metadata["value_stream_bytes"])
        key_max = (
            raw.narrow(0, int(metadata["key_max_offset"]),
                       int(metadata["key_max_numel"]) * 2)
            .contiguous()
            .view(torch.float16)
            .reshape(tokens, 1)
        )
        value_max = (
            raw.narrow(0, int(metadata["value_max_offset"]),
                       int(metadata["value_max_numel"]) * 2)
            .contiguous()
            .view(torch.float16)
            .reshape(tokens, 1)
        )
        key_stream = raw.narrow(0, key_offset, key_bytes).contiguous()
        value_stream = raw.narrow(0, value_offset, value_bytes).contiguous()

        def _time_cuda_stage(stage: str, fn):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            pushed = False
            try:
                torch.cuda.nvtx.range_push(f"origami:{stage}")
                pushed = True
            except Exception:
                pushed = False
            try:
                start_event.record()
                result = fn()
                end_event.record()
                return result, (start_event, end_event)
            finally:
                if pushed:
                    torch.cuda.nvtx.range_pop()

        def _unpack():
            key_q_flat = native_gpu.unpack_canonical_storage_chunks(
                [key_stream],
                int(metadata["key_bits"]),
                [1, 1, int(metadata["key_q_numel"])],
                ["token", "head", "head_dim"],
                [[0, 1, 0, int(metadata["key_q_numel"]), 0, 1]],
                device=raw.device,
            )
            value_q_flat = native_gpu.unpack_canonical_storage_chunks(
                [value_stream],
                int(metadata["value_bits"]),
                [1, 1, int(metadata["value_q_numel"])],
                ["token", "head", "head_dim"],
                [[0, 1, 0, int(metadata["value_q_numel"]), 0, 1]],
                device=raw.device,
            )
            return (
                key_q_flat.reshape(tokens, heads * head_dim),
                value_q_flat.reshape(tokens, heads * head_dim),
            )

        (key_q, value_q), unpack_events = _time_cuda_stage(
            "cachegen_unpack_cuda",
            _unpack,
        )
        unpack_input_bytes = key_bytes + value_bytes
        unpack_output_bytes = int(key_q.numel() + value_q.numel())

        def _dequant():
            key = self._dequantize_plane(key_q, key_max, int(metadata["key_bins"]))
            value = self._dequantize_plane(
                value_q, value_max, int(metadata["value_bins"])
            )
            return torch.stack(
                [
                    key.reshape(tokens, heads, head_dim),
                    value.reshape(tokens, heads, head_dim),
                ],
                dim=0,
            ).to(dtype=_dtype_from_metadata(metadata))

        canonical, dequant_events = _time_cuda_stage(
            "cachegen_dequant_cuda",
            _dequant,
        )
        dequant_input_bytes = (
            int(key_q.numel() * key_q.element_size())
            + int(value_q.numel() * value_q.element_size())
            + int(key_max.numel() * key_max.element_size())
            + int(value_max.numel() * value_max.element_size())
        )
        dequant_output_bytes = int(canonical.numel() * canonical.element_size())

        restored, layout_events = _time_cuda_stage(
            "layout_restore_cuda",
            lambda: self._restore_layout(canonical, metadata),
        )
        layout_events[1].synchronize()
        layout_bytes = int(restored.numel() * restored.element_size())
        if h2d_events is not None:
            h2d_ms = float(h2d_events[0].elapsed_time(h2d_events[1]))
            self.last_profile.append({
                "type": "cachegen_bitstream_h2d_cuda",
                "device": h2d_events[2],
                "input_bytes": h2d_bytes,
                "output_bytes": h2d_bytes,
                "bytes": h2d_bytes,
                "ms": h2d_ms,
                "gbps": (
                    h2d_bytes * 8.0 / h2d_ms / 1e6 if h2d_ms > 0 else 0.0
                ),
            })
        unpack_ms = float(unpack_events[0].elapsed_time(unpack_events[1]))
        self.last_profile.append({
            "type": "cachegen_unpack_cuda",
            "device": str(raw.device),
            "input_bytes": unpack_input_bytes,
            "output_bytes": unpack_output_bytes,
            "bytes": unpack_output_bytes,
            "ms": unpack_ms,
            "input_gbps": (
                unpack_input_bytes * 8.0 / unpack_ms / 1e6
                if unpack_ms > 0 else 0.0
            ),
            "output_gbps": (
                unpack_output_bytes * 8.0 / unpack_ms / 1e6
                if unpack_ms > 0 else 0.0
            ),
            "gbps": (
                unpack_output_bytes * 8.0 / unpack_ms / 1e6
                if unpack_ms > 0 else 0.0
            ),
        })
        dequant_ms = float(dequant_events[0].elapsed_time(dequant_events[1]))
        self.last_profile.append({
            "type": "cachegen_dequant_cuda",
            "device": str(canonical.device),
            "input_bytes": dequant_input_bytes,
            "output_bytes": dequant_output_bytes,
            "bytes": dequant_output_bytes,
            "ms": dequant_ms,
            "input_gbps": (
                dequant_input_bytes * 8.0 / dequant_ms / 1e6
                if dequant_ms > 0 else 0.0
            ),
            "output_gbps": (
                dequant_output_bytes * 8.0 / dequant_ms / 1e6
                if dequant_ms > 0 else 0.0
            ),
            "gbps": (
                dequant_output_bytes * 8.0 / dequant_ms / 1e6
                if dequant_ms > 0 else 0.0
            ),
        })
        layout_ms = float(layout_events[0].elapsed_time(layout_events[1]))
        self.last_profile.append({
            "type": "layout_restore_cuda",
            "device": str(restored.device),
            "input_bytes": dequant_output_bytes,
            "output_bytes": layout_bytes,
            "bytes": layout_bytes,
            "ms": layout_ms,
            "input_gbps": (
                dequant_output_bytes * 8.0 / layout_ms / 1e6
                if layout_ms > 0 else 0.0
            ),
            "output_gbps": (
                layout_bytes * 8.0 / layout_ms / 1e6
                if layout_ms > 0 else 0.0
            ),
            "gbps": (
                layout_bytes * 8.0 / layout_ms / 1e6
                if layout_ms > 0 else 0.0
            ),
        })
        return restored

    def can_dequantize_to_cache(
        self,
        metadata: dict[str, Any],
        dst_cache: torch.Tensor,
        block_ids: list[int] | tuple[int, ...],
        *,
        tokens: int | None = None,
    ) -> bool:
        if str(metadata.get("format")) != "cachegen_bitpacked_blob":
            return False
        if not self._use_gpu_dequantize():
            return False
        if dst_cache.device.type != "cuda" or dst_cache.dtype not in {
            torch.float16,
            torch.bfloat16,
        }:
            return False
        if dst_cache.dim() != 5:
            return False
        if not block_ids:
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
        if restore_tokens <= 0:
            return False
        # The direct path writes only the quantized tokens.  Partial last blocks
        # require explicit zero-fill of padding tokens, so keep the old padded
        # materialize path for that rare case.
        if restore_tokens % block_size != 0:
            return False
        if restore_tokens > len(block_ids) * block_size:
            return False
        return (
            heads == int(metadata["num_heads"])
            and head_dim == int(metadata["head_dim"])
        )

    def _dequantize_bitpacked_gpu_to_cache(
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

        token_count = int(tokens)
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        key_max = (
            raw.narrow(
                0,
                int(metadata["key_max_offset"]),
                int(metadata["key_max_numel"]) * 2,
            )
            .view(torch.float16)
            .reshape(int(metadata["token_count"]))
        )
        value_max = (
            raw.narrow(
                0,
                int(metadata["value_max_offset"]),
                int(metadata["value_max_numel"]) * 2,
            )
            .view(torch.float16)
            .reshape(int(metadata["token_count"]))
        )
        block_ids_tensor = torch.tensor(
            list(block_ids),
            dtype=torch.long,
            device=dst_cache.device,
        )

        pushed = False
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        try:
            torch.cuda.nvtx.range_push("origami:cachegen_fused_restore_to_kv_cuda")
            pushed = True
        except Exception:
            pushed = False
        try:
            start_event.record()
            native_gpu.cachegen_unpack_dequantize_to_kv_cache(
                raw,
                key_offset=int(metadata["key_stream_offset"]),
                key_bytes=int(metadata["key_stream_bytes"]),
                value_offset=int(metadata["value_stream_offset"]),
                value_bytes=int(metadata["value_stream_bytes"]),
                key_bits=int(metadata["key_bits"]),
                value_bits=int(metadata["value_bits"]),
                key_bins=int(metadata["key_bins"]),
                value_bins=int(metadata["value_bins"]),
                max_key=key_max,
                max_value=value_max,
                kv_cache=dst_cache,
                block_ids=block_ids_tensor,
                tokens=token_count,
                heads=heads,
                head_dim=head_dim,
            )
            end_event.record()
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()

        if h2d_events is not None:
            self.last_profile.append({
                "type": "cachegen_bitstream_h2d_cuda",
                "device": h2d_events[2],
                "input_bytes": h2d_bytes,
                "output_bytes": h2d_bytes,
                "bytes": h2d_bytes,
                "_cuda_start_event": h2d_events[0],
                "_cuda_end_event": h2d_events[1],
            })
        output_bytes = int(token_count * 2 * heads * head_dim * dst_cache.element_size())
        input_bytes = (
            int(metadata["key_stream_bytes"])
            + int(metadata["value_stream_bytes"])
            + int(metadata["key_max_numel"]) * 2
            + int(metadata["value_max_numel"]) * 2
        )
        self.last_profile.append({
            "type": "origami_cachegen_fused_restore_to_kv_cuda",
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
        fmt = str(metadata.get("format"))
        if fmt not in {
            "cachegen_quantized_blob",
            "cachegen_bitpacked_blob",
            "cachegen_structured_blob",
        }:
            raise ValueError(
                "CacheGenAdapter can only dequantize CacheGen Origami payloads"
            )
        if (
            fmt == "cachegen_bitpacked_blob"
            and dst_cache is not None
            and block_ids is not None
            and self.can_dequantize_to_cache(metadata, dst_cache, block_ids)
        ):
            return self._dequantize_bitpacked_gpu_to_cache(
                symbols,
                metadata,
                dst_cache,
                block_ids,
                tokens=int(metadata["token_count"]),
            )
        if self._use_gpu_dequantize():
            tensor = self._dequantize_gpu(symbols, metadata)
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

        raw = symbols.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        expected = int(metadata["symbol_byte_count"])
        if int(raw.numel()) != expected:
            raise RuntimeError(
                "CacheGen quantized payload size mismatch: "
                f"got {int(raw.numel())}, expected {expected}"
            )

        tokens = int(metadata["token_count"])
        channels = int(metadata["channel_count"])
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        if fmt == "cachegen_bitpacked_blob":
            key_q = _unpack_linear_bits(
                raw.narrow(
                    0,
                    int(metadata["key_stream_offset"]),
                    int(metadata["key_stream_bytes"]),
                ),
                int(metadata["key_bits"]),
                int(metadata["key_q_numel"]),
            ).reshape(tokens, channels)
            value_q = _unpack_linear_bits(
                raw.narrow(
                    0,
                    int(metadata["value_stream_offset"]),
                    int(metadata["value_stream_bytes"]),
                ),
                int(metadata["value_bits"]),
                int(metadata["value_q_numel"]),
            ).reshape(tokens, channels)
        else:
            key_q = self._read_u8(
                raw, int(metadata["key_q_offset"]), int(metadata["key_q_numel"])
            ).reshape(tokens, channels)
            value_q = self._read_u8(
                raw, int(metadata["value_q_offset"]), int(metadata["value_q_numel"])
            ).reshape(tokens, channels)
        key_max = self._read_fp16(
            raw, int(metadata["key_max_offset"]), int(metadata["key_max_numel"])
        )
        value_max = self._read_fp16(
            raw, int(metadata["value_max_offset"]), int(metadata["value_max_numel"])
        )
        key = self._dequantize_plane(key_q, key_max, int(metadata["key_bins"]))
        value = self._dequantize_plane(value_q, value_max, int(metadata["value_bins"]))
        dtype = _dtype_from_metadata(metadata)
        canonical = torch.stack(
            [
                key.reshape(tokens, heads, head_dim),
                value.reshape(tokens, heads, head_dim),
            ],
            dim=0,
        ).to(dtype=dtype)
        tensor = self._restore_layout(canonical, metadata)
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
        if reduce(mul, tensor.shape, 1) != reduce(
            mul, [int(dim) for dim in metadata["shape"]], 1
        ):
            raise RuntimeError("CacheGen dequantized tensor shape mismatch")
        return tensor
