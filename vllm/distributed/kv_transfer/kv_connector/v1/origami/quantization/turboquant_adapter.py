# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
import importlib.util
import json
import sys
import types
from pathlib import Path
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

TURBO_DATA_ORDER = ("layer", "head", "head_dim", "token")


def _ensure_dserve_path() -> None:
    for candidate in (Path("/home/td/dpucomp"), Path(__file__).resolve().parents[8]):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _load_dserve_helpers() -> dict[str, Any]:
    _ensure_dserve_path()
    try:
        from dserve_kv_compress.origami_adapters import (  # type: ignore
            _bytes_to_tensor,
            _load_turbo_modules,
            _pack_last_dim_lowbit,
            _tensor_bytes,
        )
    except Exception as exc:  # pragma: no cover - dependency is optional.
        raise RuntimeError(
            "TurboQuantAdapter requires /home/td/dpucomp/dserve_kv_compress"
        ) from exc
    try:
        modules = _load_turbo_modules()
    except Exception:
        modules = _load_turbo_modules_without_scipy()
    modules.update(
        {
            "_bytes_to_tensor": _bytes_to_tensor,
            "_pack_last_dim_lowbit": _pack_last_dim_lowbit,
            "_tensor_bytes": _tensor_bytes,
        }
    )
    return modules


def _load_module_from_path(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_turbo_modules_without_scipy() -> dict[str, Any]:
    """Load TurboQuant modules using precomputed JSON codebooks only."""
    root = Path("/home/td/dpucomp/turboquant/turboquant")
    if not root.exists():
        root = Path("/home/td/Origami/third_party/turboquant/turboquant")
    if not root.exists():
        raise RuntimeError("TurboQuant source tree not found")

    package = sys.modules.get("turboquant")
    if package is None:
        package = types.ModuleType("turboquant")
        package.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules["turboquant"] = package

    codebook_name = "turboquant.codebook"
    if codebook_name not in sys.modules:
        codebook = types.ModuleType(codebook_name)

        def get_codebook_tensors(
            d: int,
            bits: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
        ):
            path = root / "codebooks" / f"codebook_d{int(d)}_b{int(bits)}.json"
            if not path.exists():
                raise RuntimeError(
                    f"missing precomputed TurboQuant codebook {path}; "
                    "install scipy to compute it on demand"
                )
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            centroids = torch.tensor(payload["centroids"], device=device, dtype=dtype)
            boundaries = torch.tensor(payload["boundaries"], device=device, dtype=dtype)
            return centroids, boundaries

        codebook.get_codebook_tensors = get_codebook_tensors
        sys.modules[codebook_name] = codebook

    _load_module_from_path("turboquant.rotation", root / "rotation.py")
    quantizer = _load_module_from_path("turboquant.quantizer", root / "quantizer.py")
    kv_cache = _load_module_from_path("turboquant.kv_cache", root / "kv_cache.py")
    return {
        "TurboQuantKVCache": kv_cache.TurboQuantKVCache,
        "ValueQuantized": kv_cache.ValueQuantized,
        "dequantize_values": kv_cache.dequantize_values,
        "unpack_values": kv_cache.unpack_values,
        "ProdQuantized": quantizer.ProdQuantized,
        "TurboQuantProd": quantizer.TurboQuantProd,
        "_pack_indices": quantizer._pack_indices,
        "_unpack_indices": quantizer._unpack_indices,
    }


def _dtype_from_metadata(metadata: dict[str, Any]) -> torch.dtype:
    dtype_name = str(metadata.get("dtype", "float16")).replace("torch.", "")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported TurboQuant tensor dtype {dtype_name!r}")
    return dtype


def _fp16_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return (
        tensor.detach()
        .cpu()
        .to(torch.float16)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
    )


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.uint8).contiguous().reshape(-1)


def _bytes_to_u8(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def _read_fp16(raw: torch.Tensor, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
    if not shape:
        return torch.empty((0,), dtype=torch.float16)
    count = int(math.prod(shape))
    return (
        raw.narrow(0, int(offset), count * 2)
        .contiguous()
        .view(torch.float16)
        .reshape(shape)
    )


class TurboQuantAdapter(QuantizerAdapter):
    """TurboQuant K3V2 adapter without a lossless compression stage."""

    quantizer_id = "turboquant"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.key_bits = int(self.config.get("key_bits", self.config.get("turbo_key_bits", 3)))
        self.value_bits = int(
            self.config.get("value_bits", self.config.get("turbo_value_bits", 2))
        )
        self.value_group_size = int(
            self.config.get(
                "value_group_size", self.config.get("turbo_value_group_size", 32)
            )
        )
        self.buffer_size = int(
            self.config.get("buffer_size", self.config.get("turbo_buffer_size", 128))
        )
        self.device = str(self.config.get("device", self.config.get("turbo_device", "cuda:0")))
        if self.key_bits < 2:
            raise ValueError("TurboQuantAdapter key_bits must be >= 2")
        if self.value_bits not in {2, 4, 8}:
            raise ValueError("TurboQuantAdapter value_bits must be one of {2, 4, 8}")

    @staticmethod
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

    def quantize(
        self,
        kv: torch.Tensor,
        *,
        request: Any | None = None,
        layer_group: Any | None = None,
        stream: Any | None = None,
    ) -> QuantizedKV:
        del request, stream
        modules = _load_dserve_helpers()
        TurboQuantKVCache = modules["TurboQuantKVCache"]
        unpack_values = modules["unpack_values"]
        unpack_indices = modules["_unpack_indices"]
        tensor_bytes = modules["_tensor_bytes"]

        canonical, layout = KiviAdapter._canonicalize(kv)
        canonical_cpu = canonical.detach().cpu().to(torch.float16).contiguous()
        tokens = int(canonical_cpu.shape[1])
        heads = int(canonical_cpu.shape[2])
        head_dim = int(canonical_cpu.shape[3])
        layer_idx = self._layer_index(layer_group)
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TurboQuantAdapter CUDA device requested but CUDA is unavailable")

        key = canonical_cpu[0].permute(1, 0, 2).unsqueeze(0).to(device=device)
        value = canonical_cpu[1].permute(1, 0, 2).unsqueeze(0).to(device=device)
        tq_cache = TurboQuantKVCache(
            head_dim=head_dim,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            value_group_size=self.value_group_size,
            buffer_size=self.buffer_size,
            device=device,
            dtype=torch.float16,
            layer_idx=layer_idx,
        )
        tq_cache.prefill(key.contiguous(), value.contiguous())
        quant_tokens = max(0, tokens - self.buffer_size)
        key_mse_bits = max(1, self.key_bits - 1)

        parts: list[torch.Tensor] = []
        offsets: dict[str, int] = {}
        cursor = 0

        def append(name: str, data: torch.Tensor, *, align: int = 1) -> None:
            nonlocal cursor
            if align > 1:
                pad = (-cursor) % align
                if pad:
                    parts.append(torch.zeros((pad,), dtype=torch.uint8))
                    cursor += pad
            offsets[name] = cursor
            flat = data.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
            parts.append(flat)
            cursor += int(flat.numel())

        qjl_shape: tuple[int, ...] = ()
        key_norms_shape: tuple[int, ...] = ()
        key_residual_norms_shape: tuple[int, ...] = ()
        if tq_cache.key_quantized is not None:
            key_q = tq_cache.key_quantized
            key_mse_bits = int(key_q.mse_bits)
            mse = unpack_indices(key_q.mse_indices, key_q.mse_bits, head_dim).to(torch.uint8)
            mse_layer = mse.squeeze(0).permute(1, 0, 2).cpu().contiguous()
            key_mse_data = _pack_ordered_lowbit(
                mse_layer.unsqueeze(0), TURBO_DATA_ORDER, key_mse_bits
            )
            qjl_cpu = key_q.qjl_signs.detach().cpu().to(torch.uint8).contiguous()
            qjl_shape = tuple(int(v) for v in qjl_cpu.shape)
            key_norms = key_q.norms.detach().cpu().to(torch.float16).contiguous()
            key_residual_norms = (
                key_q.residual_norms.detach().cpu().to(torch.float16).contiguous()
            )
            key_norms_shape = tuple(int(v) for v in key_norms.shape)
            key_residual_norms_shape = tuple(int(v) for v in key_residual_norms.shape)
        else:
            key_mse_data = torch.empty((0,), dtype=torch.uint8)
            qjl_cpu = torch.empty((0,), dtype=torch.uint8)
            key_norms = torch.empty((0,), dtype=torch.float16)
            key_residual_norms = torch.empty((0,), dtype=torch.float16)

        value_scales_shape: tuple[int, ...] = ()
        value_zeros_shape: tuple[int, ...] = ()
        if tq_cache.value_quantized is not None:
            value_q = tq_cache.value_quantized
            codes = unpack_values(value_q).to(torch.uint8)
            value_layer = codes.squeeze(0).permute(1, 0, 2).cpu().contiguous()
            value_data = _pack_ordered_lowbit(
                value_layer.unsqueeze(0), TURBO_DATA_ORDER, self.value_bits
            )
            value_scales = value_q.scales.detach().cpu().to(torch.float16).contiguous()
            value_zeros = value_q.zeros.detach().cpu().to(torch.float16).contiguous()
            value_scales_shape = tuple(int(v) for v in value_scales.shape)
            value_zeros_shape = tuple(int(v) for v in value_zeros.shape)
        else:
            value_data = torch.empty((0,), dtype=torch.uint8)
            value_scales = torch.empty((0,), dtype=torch.float16)
            value_zeros = torch.empty((0,), dtype=torch.float16)

        if tq_cache.key_buffer is not None:
            key_recent = tq_cache.key_buffer.detach().cpu().to(torch.float16).contiguous()
            value_recent = tq_cache.value_buffer.detach().cpu().to(torch.float16).contiguous()
        else:
            key_recent = torch.empty((1, heads, 0, head_dim), dtype=torch.float16)
            value_recent = torch.empty((1, heads, 0, head_dim), dtype=torch.float16)
        key_recent_shape = tuple(int(v) for v in key_recent.shape)
        value_recent_shape = tuple(int(v) for v in value_recent.shape)

        append("key_mse_stream", key_mse_data)
        append("key_qjl", qjl_cpu)
        append("value_stream", value_data)
        append("key_norms", _fp16_bytes(key_norms), align=2)
        append("key_residual_norms", _fp16_bytes(key_residual_norms), align=2)
        append("value_scales", _fp16_bytes(value_scales), align=2)
        append("value_zeros", _fp16_bytes(value_zeros), align=2)
        append("key_recent", _bytes_to_u8(tensor_bytes(key_recent)), align=2)
        append("value_recent", _bytes_to_u8(tensor_bytes(value_recent)), align=2)
        symbols = torch.cat(parts, dim=0).contiguous()

        metadata: dict[str, Any] = {
            "format": "turboquant_structured_blob",
            "version": 1,
            "dtype": str(kv.dtype).replace("torch.", ""),
            "shape": [int(dim) for dim in kv.shape],
            "kv_layout": layout,
            "layer_idx": layer_idx,
            "token_count": tokens,
            "num_heads": heads,
            "head_dim": head_dim,
            "quant_tokens": quant_tokens,
            "key_bits": self.key_bits,
            "key_mse_bits": key_mse_bits,
            "value_bits": self.value_bits,
            "value_group_size": self.value_group_size,
            "buffer_size": self.buffer_size,
            "device": str(device),
            "key_mse_shape": [1, quant_tokens, heads, head_dim],
            "value_shape": [1, quant_tokens, heads, head_dim],
            "data_order": list(TURBO_DATA_ORDER),
            "qjl_shape": list(qjl_shape),
            "key_norms_shape": list(key_norms_shape),
            "key_residual_norms_shape": list(key_residual_norms_shape),
            "value_scales_shape": list(value_scales_shape),
            "value_zeros_shape": list(value_zeros_shape),
            "key_recent_shape": list(key_recent_shape),
            "value_recent_shape": list(value_recent_shape),
            "symbol_byte_count": int(symbols.numel()),
            "origami_bits": 8,
            "origami_symbol_shape": [1, 1, int(symbols.numel())],
            "origami_symbol_layout": ["token", "head", "head_dim"],
        }
        for name, offset in offsets.items():
            metadata[f"{name}_offset"] = int(offset)
        for name in offsets:
            next_offsets = [value for key, value in offsets.items() if value > offsets[name]]
            end = min(next_offsets) if next_offsets else int(symbols.numel())
            metadata[f"{name}_bytes"] = int(end - offsets[name])
        del tq_cache, key, value
        if device.type == "cuda":
            torch.cuda.empty_cache()
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
        if str(metadata.get("format")) != "turboquant_structured_blob":
            raise ValueError("TurboQuantAdapter can only dequantize TurboQuant payloads")
        modules = _load_dserve_helpers()
        ValueQuantized = modules["ValueQuantized"]
        dequantize_values = modules["dequantize_values"]
        ProdQuantized = modules["ProdQuantized"]
        TurboQuantProd = modules["TurboQuantProd"]
        pack_indices = modules["_pack_indices"]
        bytes_to_tensor = modules["_bytes_to_tensor"]
        pack_last_dim_lowbit = modules["_pack_last_dim_lowbit"]

        raw = symbols.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        expected = int(metadata["symbol_byte_count"])
        if int(raw.numel()) != expected:
            raise RuntimeError(
                f"TurboQuant payload size mismatch: got {int(raw.numel())}, expected {expected}"
            )

        tokens = int(metadata["token_count"])
        heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        quant_tokens = int(metadata["quant_tokens"])
        device = torch.device(str(metadata.get("device", self.device)))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        key_shape = tuple(int(v) for v in metadata["key_mse_shape"])
        value_shape = tuple(int(v) for v in metadata["value_shape"])
        order = tuple(metadata["data_order"])
        key_mse_data = raw.narrow(
            0, int(metadata["key_mse_stream_offset"]), int(metadata["key_mse_stream_bytes"])
        )
        value_data = raw.narrow(
            0, int(metadata["value_stream_offset"]), int(metadata["value_stream_bytes"])
        )
        key_mse_tensor = (
            _unpack_ordered_lowbit(
                key_mse_data,
                shape=key_shape,
                order=order,
                bits=int(metadata["key_mse_bits"]),
            )
            if math.prod(key_shape)
            else torch.empty(key_shape, dtype=torch.uint8)
        )
        value_tensor = (
            _unpack_ordered_lowbit(
                value_data,
                shape=value_shape,
                order=order,
                bits=int(metadata["value_bits"]),
            )
            if math.prod(value_shape)
            else torch.empty(value_shape, dtype=torch.uint8)
        )

        qjl_bytes = bytes(
            raw.narrow(0, int(metadata["key_qjl_offset"]), int(metadata["key_qjl_bytes"]))
            .tolist()
        )
        key_norms = _read_fp16(
            raw, int(metadata["key_norms_offset"]), tuple(metadata["key_norms_shape"])
        )
        key_residual_norms = _read_fp16(
            raw,
            int(metadata["key_residual_norms_offset"]),
            tuple(metadata["key_residual_norms_shape"]),
        )
        value_scales = _read_fp16(
            raw, int(metadata["value_scales_offset"]), tuple(metadata["value_scales_shape"])
        )
        value_zeros = _read_fp16(
            raw, int(metadata["value_zeros_offset"]), tuple(metadata["value_zeros_shape"])
        )
        key_recent_bytes = bytes(
            raw.narrow(
                0, int(metadata["key_recent_offset"]), int(metadata["key_recent_bytes"])
            ).tolist()
        )
        value_recent_bytes = bytes(
            raw.narrow(
                0,
                int(metadata["value_recent_offset"]),
                int(metadata["value_recent_bytes"]),
            ).tolist()
        )
        key_recent = bytes_to_tensor(
            key_recent_bytes,
            dtype=torch.float16,
            shape=tuple(metadata["key_recent_shape"]),
        )
        value_recent = bytes_to_tensor(
            value_recent_bytes,
            dtype=torch.float16,
            shape=tuple(metadata["value_recent_shape"]),
        )

        if quant_tokens > 0:
            mse_symbols = key_mse_tensor[0].permute(1, 0, 2).unsqueeze(0).to(device=device)
            mse_indices = pack_indices(mse_symbols, int(metadata["key_mse_bits"]))
            qjl = bytes_to_tensor(
                qjl_bytes,
                dtype=torch.uint8,
                shape=tuple(metadata["qjl_shape"]),
            ).to(device=device)
            key_quantizer = TurboQuantProd(
                dim=head_dim,
                bits=int(metadata["key_bits"]),
                device=device,
                seed=42 + int(metadata["layer_idx"]) * 7,
            )
            prod = ProdQuantized(
                mse_indices=mse_indices,
                qjl_signs=qjl,
                residual_norms=key_residual_norms.to(device=device),
                norms=key_norms.to(device=device),
                mse_bits=int(metadata["key_mse_bits"]),
            )
            key_quant = key_quantizer.dequantize(prod).to(torch.float16).cpu()
            value_codes = value_tensor[0].permute(1, 0, 2).unsqueeze(0)
            value_packed = pack_last_dim_lowbit(
                value_codes.cpu(), int(metadata["value_bits"])
            ).to(device=device)
            value_quant = ValueQuantized(
                data=value_packed,
                scales=value_scales.to(device=device),
                zeros=value_zeros.to(device=device),
                bits=int(metadata["value_bits"]),
            )
            value_quant_decoded = dequantize_values(
                value_quant,
                group_size=int(metadata["value_group_size"]),
            ).to(torch.float16).cpu()
        else:
            key_quant = torch.empty((1, heads, 0, head_dim), dtype=torch.float16)
            value_quant_decoded = torch.empty((1, heads, 0, head_dim), dtype=torch.float16)

        key = torch.cat([key_quant, key_recent.cpu()], dim=-2).contiguous()
        value = torch.cat([value_quant_decoded, value_recent.cpu()], dim=-2).contiguous()
        if int(key.shape[-2]) != tokens or int(value.shape[-2]) != tokens:
            raise RuntimeError("TurboQuant dequantized token count mismatch")
        canonical = torch.stack(
            [
                key.squeeze(0).permute(1, 0, 2),
                value.squeeze(0).permute(1, 0, 2),
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
