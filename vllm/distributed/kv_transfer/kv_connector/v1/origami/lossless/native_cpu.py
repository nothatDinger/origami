# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence

import torch
from torch.utils.cpp_extension import load

_REPO_ROOT = Path(__file__).resolve().parents[7]
_CSRC = _REPO_ROOT / "csrc" / "origami"
_BUILD_ROOT = Path(
    os.environ.get(
        "ORIGAMI_NATIVE_BUILD_DIR",
        str(Path(tempfile.gettempdir()) / "origami_native_cpu"),
    )
)
_VERBOSE = bool(int(os.environ.get("ORIGAMI_NATIVE_VERBOSE", "0")))
_DEFAULT_QAT_CODEC = "qat_codec"
_LAST_QAT_PROFILE: dict[str, Any] = {}
_LAST_QAT_PREPARE_PROFILE: dict[str, Any] = {}
_QAT_PROFILE_FALLBACK_LOCK = Lock()
_REQUIRED_AXES = ("token", "head", "head_dim")
_AXIS_ALIASES = {
    "token": "token",
    "tokens": "token",
    "seq": "token",
    "sequence": "token",
    "head": "head",
    "heads": "head",
    "kv_head": "head",
    "kv_heads": "head",
    "head_dim": "head_dim",
    "channel": "head_dim",
    "channels": "head_dim",
    "dim": "head_dim",
    "layer": "layer",
    "layers": "layer",
}
STORAGE_LAYOUT = ["layer", "head", "head_dim", "token"]


def _build_dir(name: str) -> str:
    path = _BUILD_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@lru_cache(maxsize=1)
def load_bitpack_extension() -> Any:
    return load(
        name="origami_bitpack_cpu_v2",
        sources=[str(_CSRC / "bitpack_layout.cpp")],
        extra_cflags=["-O3"],
        build_directory=_build_dir("bitpack_v2"),
        verbose=_VERBOSE,
    )


@lru_cache(maxsize=1)
def load_qat_extension() -> Any:
    return load(
        name="origami_qat_cpu",
        sources=[str(_CSRC / "qat_deflate_dp.cpp")],
        extra_cflags=["-O3"],
        extra_include_paths=["/usr/local/include"],
        extra_ldflags=["-L/usr/local/lib", "-lqat", "-lusdm", "-lcrypto"],
        build_directory=_build_dir("qat"),
        verbose=_VERBOSE,
    )


@lru_cache(maxsize=1)
def load_qat_cpa_extension() -> Any:
    return load(
        name="origami_qat_cpa_cpu",
        sources=[str(_CSRC / "qat_deflate_cpa.cpp")],
        extra_cflags=["-O3", "-march=native", "-I/usr/local/include/qat"],
        extra_ldflags=[
            "-L/usr/local/lib",
            "-Wl,-rpath,/usr/local/lib",
            "-lqat",
            "-lusdm",
            "-lpthread",
            "-ldl",
        ],
        build_directory=_build_dir("qat_cpa"),
        verbose=_VERBOSE,
    )


def qat_codec() -> str:
    value = os.environ.get("ORIGAMI_QAT_CODEC", _DEFAULT_QAT_CODEC)
    value = value.strip().lower()
    if value == "cpa":
        return "cpa"
    if value in {"qat_codec", "qat"}:
        return "qat_codec"
    if value in {"dp", "dp_prepared", "prepared"}:
        return "dp_prepared"
    raise ValueError(
        "ORIGAMI_QAT_CODEC must be one of {'qat_codec', 'cpa', 'dp_prepared'}"
    )


def qat_uses_prepared_restore() -> bool:
    return qat_codec() in {"qat_codec", "dp_prepared"}


def normalize_symbol_layout(layout: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for axis in layout:
        key = str(axis).lower()
        if key not in _AXIS_ALIASES:
            raise ValueError(f"Unsupported Origami symbol layout axis {axis!r}")
        normalized.append(_AXIS_ALIASES[key])
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Origami symbol layout axes must be unique: {layout!r}")
    for required in _REQUIRED_AXES:
        if required not in normalized:
            raise ValueError(
                "Origami symbol layout must include token, head, and head_dim axes"
            )
    extra = set(normalized) - set(_REQUIRED_AXES) - {"layer"}
    if extra:
        raise ValueError(f"Unsupported Origami symbol layout axes: {sorted(extra)}")
    if len(normalized) not in {3, 4}:
        raise ValueError("Origami symbol layout supports rank 3 or rank 4")
    return normalized


def canonical_axis_sizes(
    source_shape: Sequence[int],
    source_layout: Sequence[str],
) -> tuple[int, int, int, list[int], list[str]]:
    shape = [int(dim) for dim in source_shape]
    layout = normalize_symbol_layout(source_layout)
    if len(shape) != len(layout):
        raise ValueError("origami_symbol_shape rank must match origami_symbol_layout")
    if len(shape) not in {3, 4}:
        raise ValueError("origami_symbol_shape supports rank 3 or rank 4")
    if any(dim <= 0 for dim in shape):
        raise ValueError("origami_symbol_shape dimensions must be positive")
    if "layer" in layout and shape[layout.index("layer")] != 1:
        raise ValueError("per-layer Origami worker requires layer axis size 1")
    token_count = shape[layout.index("token")]
    num_heads = shape[layout.index("head")]
    head_dim = shape[layout.index("head_dim")]
    return token_count, num_heads, head_dim, shape, layout


def axis_positions(source_layout: Sequence[str]) -> list[int]:
    layout = normalize_symbol_layout(source_layout)
    return [
        layout.index("token"),
        layout.index("head"),
        layout.index("head_dim"),
        layout.index("layer") if "layer" in layout else -1,
    ]


def _cpu_u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.uint8).reshape(-1).contiguous()


def _shape_tensor(source_shape: Sequence[int]) -> torch.Tensor:
    return torch.tensor([int(dim) for dim in source_shape], dtype=torch.int64).contiguous()


def _axis_tensor(source_layout: Sequence[str]) -> torch.Tensor:
    return torch.tensor(axis_positions(source_layout), dtype=torch.int64).contiguous()


def _chunk_specs_tensor(chunk_specs: torch.Tensor | Iterable[Iterable[int]]) -> torch.Tensor:
    if isinstance(chunk_specs, torch.Tensor):
        specs = chunk_specs.detach().cpu().to(torch.int64).contiguous()
    else:
        rows = list(chunk_specs)
        if not rows:
            return torch.empty((0, 6), dtype=torch.int64)
        specs = torch.tensor(rows, dtype=torch.int64).contiguous()
    if specs.numel() == 0:
        return torch.empty((0, 6), dtype=torch.int64)
    if specs.dim() == 1:
        specs = specs.reshape(1, 6)
    return specs.contiguous()


def pack_canonical_storage_chunks(
    symbols: torch.Tensor,
    bits: int,
    source_shape: Sequence[int],
    source_layout: Sequence[str],
    chunk_specs: torch.Tensor | Iterable[Iterable[int]],
) -> list[torch.Tensor]:
    token_count, num_heads, head_dim, shape, layout = canonical_axis_sizes(
        source_shape, source_layout
    )
    ext = load_bitpack_extension()
    specs = _chunk_specs_tensor(chunk_specs)
    return list(
        ext.pack_canonical_storage_chunks(
            _cpu_u8(symbols),
            int(bits),
            _shape_tensor(shape),
            _axis_tensor(layout),
            int(token_count),
            int(num_heads),
            int(head_dim),
            specs,
        )
    )


def unpack_canonical_storage_chunks(
    chunks: Iterable[torch.Tensor],
    bits: int,
    source_shape: Sequence[int],
    source_layout: Sequence[str],
    chunk_specs: torch.Tensor | Iterable[Iterable[int]],
) -> torch.Tensor:
    token_count, num_heads, head_dim, shape, layout = canonical_axis_sizes(
        source_shape, source_layout
    )
    ext = load_bitpack_extension()
    specs = _chunk_specs_tensor(chunk_specs)
    cpu_chunks = [_cpu_u8(chunk) for chunk in chunks]
    return ext.unpack_canonical_storage_chunks(
        cpu_chunks,
        int(bits),
        _shape_tensor(shape),
        _axis_tensor(layout),
        int(token_count),
        int(num_heads),
        int(head_dim),
        specs,
    )


def pack_head_channel_chunks(
    symbols: torch.Tensor,
    bits: int,
    token_count: int,
    num_heads: int,
    head_dim: int,
    chunk_specs: torch.Tensor | Iterable[Iterable[int]],
) -> list[torch.Tensor]:
    return pack_canonical_storage_chunks(
        symbols,
        bits,
        [int(token_count), int(num_heads), int(head_dim)],
        ["token", "head", "head_dim"],
        chunk_specs,
    )


def unpack_head_channel_chunks(
    chunks: Iterable[torch.Tensor],
    bits: int,
    token_count: int,
    num_heads: int,
    head_dim: int,
    chunk_specs: torch.Tensor | Iterable[Iterable[int]],
) -> torch.Tensor:
    return unpack_canonical_storage_chunks(
        chunks,
        bits,
        [int(token_count), int(num_heads), int(head_dim)],
        ["token", "head", "head_dim"],
        chunk_specs,
    )


def cpu_isa() -> str:
    return str(load_bitpack_extension().cpu_isa())


def qat_instance_count(max_instances: int = 0) -> int:
    if qat_codec() == "cpa":
        ext = load_qat_cpa_extension()
        count = int(ext.qat_deflate_instance_count())
        if max_instances and max_instances > 0:
            return min(count, int(max_instances))
        return count
    ext = load_qat_extension()
    count = int(ext.qat_deflate_dp_instance_count())
    if max_instances and max_instances > 0:
        return min(count, int(max_instances))
    return count


def qat_instance_nodes(max_instances: int = 0) -> list[int]:
    if qat_codec() == "cpa":
        count = qat_instance_count(max_instances)
        return [-1 for _ in range(count)]
    ext = load_qat_extension()
    return [int(node) for node in ext.qat_deflate_dp_instance_nodes(int(max_instances))]


def qat_available() -> bool:
    try:
        return qat_instance_count() > 0
    except Exception:
        return False


def last_qat_profile() -> dict[str, Any]:
    return dict(_LAST_QAT_PROFILE)


def _set_last_qat_profile(profile: dict[str, Any]) -> None:
    global _LAST_QAT_PROFILE
    _LAST_QAT_PROFILE = dict(profile)


def last_qat_prepare_profile() -> dict[str, Any]:
    return dict(_LAST_QAT_PREPARE_PROFILE)


def _set_last_qat_prepare_profile(profile: dict[str, Any]) -> None:
    global _LAST_QAT_PREPARE_PROFILE
    _LAST_QAT_PREPARE_PROFILE = dict(profile)


def _empty_qat_profile() -> dict[str, Any]:
    return {
        "total_ms": 0.0,
        "worker_wall_ms_max": 0.0,
        "thread_launch_join_ms": 0.0,
        "slot_alloc_ms_sum": 0.0,
        "qat_enqueue_ms_sum": 0.0,
        "qat_poll_wait_ms_sum": 0.0,
        "qat_submit_poll_ms_sum": 0.0,
        "qat_submit_poll_ms_critical": 0.0,
        "qae_output_copy_ms_sum": 0.0,
        "chunks": 0,
        "workers": 0,
        "inflight": 0,
        "batch": 0,
        "compressed_bytes": 0,
        "unpacked_bytes": 0,
        "compressed_gbps_qat_critical": 0.0,
        "unpacked_gbps_qat_critical": 0.0,
        "persistent_workers": False,
    }


def _add_qat_profiles(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = dict(left)
    summed = {
        "total_ms",
        "worker_wall_ms_max",
        "thread_launch_join_ms",
        "slot_alloc_ms_sum",
        "qat_enqueue_ms_sum",
        "qat_poll_wait_ms_sum",
        "qat_submit_poll_ms_sum",
        "qat_submit_poll_ms_critical",
        "qae_output_copy_ms_sum",
        "chunks",
        "compressed_bytes",
        "unpacked_bytes",
    }
    for key in summed:
        out[key] = out.get(key, 0) + right.get(key, 0)
    out["workers"] = max(int(out.get("workers", 0)), int(right.get("workers", 0)))
    out["inflight"] = max(int(out.get("inflight", 0)), int(right.get("inflight", 0)))
    out["batch"] = max(int(out.get("batch", 0)), int(right.get("batch", 0)))
    if "output_staging_pinned" in right:
        out["output_staging_pinned"] = bool(
            out.get("output_staging_pinned", True)
        ) and bool(right.get("output_staging_pinned"))
    if "persistent_workers" in right:
        if int(out.get("chunks", 0) or 0) == int(right.get("chunks", 0) or 0):
            out["persistent_workers"] = bool(right.get("persistent_workers"))
        else:
            out["persistent_workers"] = bool(
                out.get("persistent_workers", True)
            ) and bool(right.get("persistent_workers"))
    critical_ms = float(out.get("qat_submit_poll_ms_critical", 0.0))
    compressed = int(out.get("compressed_bytes", 0))
    unpacked = int(out.get("unpacked_bytes", 0))
    out["compressed_gbps_qat_critical"] = (
        compressed * 8.0 / critical_ms / 1e6 if critical_ms > 0 else 0.0
    )
    out["unpacked_gbps_qat_critical"] = (
        unpacked * 8.0 / critical_ms / 1e6 if critical_ms > 0 else 0.0
    )
    return out


def _allocate_restore_staging(output_bytes: int) -> torch.Tensor:
    output_bytes = int(output_bytes)
    if output_bytes <= 0:
        return torch.empty((0,), dtype=torch.uint8)
    if torch.cuda.is_available():
        try:
            return torch.empty((output_bytes,), dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty((output_bytes,), dtype=torch.uint8)


def compress_raw_deflate_many(
    chunks: Iterable[torch.Tensor],
    dynamic_huffman: bool = True,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> list[torch.Tensor]:
    if qat_codec() == "cpa":
        return _compress_raw_deflate_many_cpa(chunks, dynamic_huffman=dynamic_huffman)
    return _compress_raw_deflate_many_dp(
        chunks,
        dynamic_huffman=dynamic_huffman,
        inflight=inflight,
        batch=batch,
        max_instances=max_instances,
    )


def _compress_raw_deflate_many_cpa(
    chunks: Iterable[torch.Tensor],
    dynamic_huffman: bool = True,
) -> list[torch.Tensor]:
    ext = load_qat_cpa_extension()
    flat_chunks = [_cpu_u8(chunk) for chunk in chunks]
    outputs: list[torch.Tensor | None] = [None] * len(flat_chunks)
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for idx, flat in enumerate(flat_chunks):
        size = int(flat.numel())
        if size == 0:
            outputs[idx] = torch.empty((0,), dtype=torch.uint8)
        else:
            groups.setdefault(size, []).append((idx, flat))

    for chunk_bytes, group in groups.items():
        merged = torch.cat([flat for _, flat in group]).contiguous()
        bytestream, lengths = ext.qat_deflate_compress_opts(
            merged,
            int(chunk_bytes),
            bool(dynamic_huffman),
        )
        lengths = lengths.detach().cpu().to(torch.int32).contiguous()
        if int(lengths.numel()) != len(group):
            raise RuntimeError("Origami QAT CPA compress returned an unexpected length count")
        cursor = 0
        stream = _cpu_u8(bytestream)
        for (idx, _), length in zip(group, lengths.tolist()):
            length = int(length)
            outputs[idx] = stream[cursor:cursor + length].clone()
            cursor += length
        if cursor != int(stream.numel()):
            raise RuntimeError("Origami QAT CPA lengths do not sum to bytestream size")

    return [output for output in outputs if output is not None]


def _compress_raw_deflate_many_dp(
    chunks: Iterable[torch.Tensor],
    dynamic_huffman: bool = True,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> list[torch.Tensor]:
    ext = load_qat_extension()
    flat_chunks = [_cpu_u8(chunk) for chunk in chunks]
    outputs: list[torch.Tensor | None] = [None] * len(flat_chunks)
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for idx, flat in enumerate(flat_chunks):
        size = int(flat.numel())
        if size == 0:
            outputs[idx] = torch.empty((0,), dtype=torch.uint8)
        else:
            groups.setdefault(size, []).append((idx, flat))

    for chunk_bytes, group in groups.items():
        merged = torch.cat([flat for _, flat in group]).contiguous()
        bytestream, lengths = ext.qat_deflate_compress_dp(
            merged,
            int(chunk_bytes),
            bool(dynamic_huffman),
            int(inflight),
            int(batch),
            int(max_instances),
        )
        lengths = lengths.detach().cpu().to(torch.int32).contiguous()
        if int(lengths.numel()) != len(group):
            raise RuntimeError("Origami QAT compress returned an unexpected length count")
        cursor = 0
        stream = _cpu_u8(bytestream)
        for (idx, _), length in zip(group, lengths.tolist()):
            length = int(length)
            outputs[idx] = stream[cursor:cursor + length].clone()
            cursor += length
        if cursor != int(stream.numel()):
            raise RuntimeError("Origami QAT compressed lengths do not sum to bytestream size")

    return [output for output in outputs if output is not None]


def decompress_raw_deflate_many(
    chunks: Iterable[torch.Tensor],
    output_bytes: Iterable[int],
    dynamic_huffman: bool = True,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> list[torch.Tensor]:
    if qat_codec() == "cpa":
        return _decompress_raw_deflate_many_cpa(
            chunks,
            output_bytes,
            dynamic_huffman=dynamic_huffman,
        )
    return _decompress_raw_deflate_many_dp(
        chunks,
        output_bytes,
        dynamic_huffman=dynamic_huffman,
        inflight=inflight,
        batch=batch,
        max_instances=max_instances,
    )


def _decompress_raw_deflate_many_cpa(
    chunks: Iterable[torch.Tensor],
    output_bytes: Iterable[int],
    dynamic_huffman: bool = True,
) -> list[torch.Tensor]:
    ext = load_qat_cpa_extension()
    compressed_chunks = [_cpu_u8(chunk) for chunk in chunks]
    output_sizes = [int(size) for size in output_bytes]
    if len(compressed_chunks) != len(output_sizes):
        raise ValueError("chunks and output_bytes must have the same length")

    outputs: list[torch.Tensor | None] = [None] * len(compressed_chunks)
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for idx, (flat, out_bytes) in enumerate(zip(compressed_chunks, output_sizes)):
        if out_bytes == 0:
            outputs[idx] = torch.empty((0,), dtype=torch.uint8)
        else:
            groups.setdefault(out_bytes, []).append((idx, flat))

    for chunk_bytes, group in groups.items():
        bytestream = torch.cat([flat for _, flat in group]).contiguous()
        lengths = torch.tensor([int(flat.numel()) for _, flat in group], dtype=torch.int32)
        restored = ext.qat_deflate_decompress_opts(
            bytestream,
            lengths,
            int(chunk_bytes) * len(group),
            int(chunk_bytes),
            bool(dynamic_huffman),
        )
        restored = _cpu_u8(restored)
        cursor = 0
        for idx, _ in group:
            outputs[idx] = restored[cursor:cursor + chunk_bytes].clone()
            cursor += chunk_bytes
        if cursor != int(restored.numel()):
            raise RuntimeError("Origami QAT CPA restored byte size does not match chunk plan")

    return [output for output in outputs if output is not None]


def _decompress_raw_deflate_many_dp(
    chunks: Iterable[torch.Tensor],
    output_bytes: Iterable[int],
    dynamic_huffman: bool = True,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> list[torch.Tensor]:
    ext = load_qat_extension()
    compressed_chunks = [_cpu_u8(chunk) for chunk in chunks]
    output_sizes = [int(size) for size in output_bytes]
    if len(compressed_chunks) != len(output_sizes):
        raise ValueError("chunks and output_bytes must have the same length")

    outputs: list[torch.Tensor | None] = [None] * len(compressed_chunks)
    groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for idx, (flat, out_bytes) in enumerate(zip(compressed_chunks, output_sizes)):
        if out_bytes == 0:
            outputs[idx] = torch.empty((0,), dtype=torch.uint8)
        else:
            groups.setdefault(out_bytes, []).append((idx, flat))

    for chunk_bytes, group in groups.items():
        bytestream = torch.cat([flat for _, flat in group]).contiguous()
        lengths = torch.tensor([int(flat.numel()) for _, flat in group], dtype=torch.int32)
        restored = ext.qat_deflate_decompress_dp(
            bytestream,
            lengths,
            int(chunk_bytes) * len(group),
            int(chunk_bytes),
            bool(dynamic_huffman),
            int(inflight),
            int(batch),
            int(max_instances),
        )
        restored = _cpu_u8(restored)
        cursor = 0
        for idx, _ in group:
            outputs[idx] = restored[cursor:cursor + chunk_bytes].clone()
            cursor += chunk_bytes
        if cursor != int(restored.numel()):
            raise RuntimeError("Origami QAT restored byte size does not match chunk plan")

    return [output for output in outputs if output is not None]


@dataclass(frozen=True)
class QatPreparedPayload:
    payloads: tuple[Any, ...]


@dataclass(frozen=True)
class CpaPreparedPayloadItem:
    bytestream: torch.Tensor
    lengths: torch.Tensor
    output_bytes: int
    chunk_bytes: int
    dynamic_huffman: bool


def prepare_raw_deflate_many(
    chunks: Iterable[torch.Tensor],
    output_bytes: Iterable[int],
    dynamic_huffman: bool = True,
    max_instances: int = 16,
) -> QatPreparedPayload:
    if qat_codec() == "cpa":
        payloads: list[CpaPreparedPayloadItem] = []
        for chunk, out_bytes in zip(chunks, output_bytes):
            out_bytes = int(out_bytes)
            flat = _cpu_u8(chunk)
            payloads.append(
                CpaPreparedPayloadItem(
                    bytestream=flat,
                    lengths=torch.tensor([int(flat.numel())], dtype=torch.int32),
                    output_bytes=out_bytes,
                    chunk_bytes=max(1, out_bytes),
                    dynamic_huffman=bool(dynamic_huffman),
                )
            )
        return QatPreparedPayload(tuple(payloads))
    ext = load_qat_extension()
    payloads: list[Any] = []
    for chunk, out_bytes in zip(chunks, output_bytes):
        out_bytes = int(out_bytes)
        flat = _cpu_u8(chunk)
        lengths = torch.tensor([int(flat.numel())], dtype=torch.int32)
        payloads.append(
            ext.qat_deflate_prepare_dp(
                flat,
                lengths,
                out_bytes,
                max(1, out_bytes),
                bool(dynamic_huffman),
                int(max_instances),
            )
        )
    return QatPreparedPayload(tuple(payloads))


def prepare_raw_deflate_bytestream(
    bytestream: torch.Tensor,
    lengths: torch.Tensor | Iterable[int],
    output_bytes: int,
    chunk_bytes: int,
    dynamic_huffman: bool = True,
    max_instances: int = 16,
) -> QatPreparedPayload:
    wrapper_start = time.perf_counter()
    tensor_start = time.perf_counter()
    stream = _cpu_u8(bytestream)
    stream_to_tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
    lengths_start = time.perf_counter()
    if isinstance(lengths, torch.Tensor):
        lengths_tensor = lengths.detach().cpu().to(torch.int32).contiguous()
    else:
        lengths_tensor = torch.tensor(
            [int(length) for length in lengths], dtype=torch.int32
        )
    if lengths_tensor.dim() != 1:
        lengths_tensor = lengths_tensor.reshape(-1).contiguous()
    lengths_to_tensor_ms = (time.perf_counter() - lengths_start) * 1000.0
    if qat_codec() == "cpa":
        _set_last_qat_prepare_profile(
            {
                "prepare_python_wrapper_ms": (time.perf_counter() - wrapper_start)
                * 1000.0,
                "prepare_python_stream_to_tensor_ms": stream_to_tensor_ms,
                "prepare_python_lengths_to_tensor_ms": lengths_to_tensor_ms,
                "prepare_chunks": int(lengths_tensor.numel()),
                "prepare_compressed_bytes": int(stream.numel()),
                "prepare_unpacked_bytes": int(output_bytes),
            }
        )
        return QatPreparedPayload(
            (
                CpaPreparedPayloadItem(
                    bytestream=stream,
                    lengths=lengths_tensor,
                    output_bytes=int(output_bytes),
                    chunk_bytes=int(chunk_bytes),
                    dynamic_huffman=bool(dynamic_huffman),
                ),
            )
        )
    load_start = time.perf_counter()
    ext = load_qat_extension()
    load_extension_ms = (time.perf_counter() - load_start) * 1000.0
    payload = ext.qat_deflate_prepare_dp(
        stream,
        lengths_tensor,
        int(output_bytes),
        int(chunk_bytes),
        bool(dynamic_huffman),
        int(max_instances),
    )
    try:
        profile = dict(ext.qat_deflate_last_prepare_profile_dp())
    except AttributeError:
        profile = {}
    profile.update(
        {
            "prepare_python_wrapper_ms": (time.perf_counter() - wrapper_start)
            * 1000.0,
            "prepare_python_stream_to_tensor_ms": stream_to_tensor_ms,
            "prepare_python_lengths_to_tensor_ms": lengths_to_tensor_ms,
            "prepare_python_load_extension_ms": load_extension_ms,
        }
    )
    _set_last_qat_prepare_profile(profile)
    return QatPreparedPayload((payload,))


def prepare_raw_deflate_file(
    path: str,
    lengths: torch.Tensor | Iterable[int],
    output_bytes: int,
    chunk_bytes: int,
    dynamic_huffman: bool = True,
    max_instances: int = 16,
    file_offset: int = 0,
) -> QatPreparedPayload:
    if isinstance(lengths, torch.Tensor):
        lengths_tensor = lengths.detach().cpu().to(torch.int32).contiguous()
    else:
        lengths_tensor = torch.tensor(
            [int(length) for length in lengths], dtype=torch.int32
        )
    if lengths_tensor.dim() != 1:
        lengths_tensor = lengths_tensor.reshape(-1).contiguous()
    if qat_codec() == "cpa":
        total = int(lengths_tensor.sum().item())
        with open(path, "rb") as handle:
            if file_offset:
                handle.seek(int(file_offset))
            data = handle.read(total)
        if len(data) != total:
            raise RuntimeError("unexpected EOF while reading QAT CPA payload")
        return prepare_raw_deflate_bytestream(
            torch.frombuffer(bytearray(data), dtype=torch.uint8),
            lengths_tensor,
            output_bytes,
            chunk_bytes,
            dynamic_huffman=dynamic_huffman,
            max_instances=max_instances,
        )
    ext = load_qat_extension()
    payload = ext.qat_deflate_prepare_from_file_dp(
        str(path),
        lengths_tensor,
        int(output_bytes),
        int(chunk_bytes),
        bool(dynamic_huffman),
        int(max_instances),
        int(file_offset),
    )
    return QatPreparedPayload((payload,))


def prepare_raw_deflate_bundle(
    bundle: torch.Tensor | bytes | bytearray | memoryview,
    dynamic_huffman: bool = True,
    max_instances: int = 16,
) -> QatPreparedPayload:
    if qat_codec() != "qat_codec":
        raise RuntimeError(
            "Origami request-level bundle restore requires "
            "ORIGAMI_QAT_CODEC=qat_codec"
        )
    wrapper_start = time.perf_counter()
    tensor_start = time.perf_counter()
    if isinstance(bundle, torch.Tensor):
        bundle_tensor = _cpu_u8(bundle)
    else:
        bundle_tensor = torch.frombuffer(bytearray(bundle), dtype=torch.uint8)
    bundle_to_tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
    load_start = time.perf_counter()
    ext = load_qat_extension()
    load_extension_ms = (time.perf_counter() - load_start) * 1000.0
    payload = ext.qat_deflate_prepare_bundle_dp(
        bundle_tensor,
        bool(dynamic_huffman),
        int(max_instances),
    )
    try:
        profile = dict(ext.qat_deflate_last_prepare_profile_dp())
    except AttributeError:
        profile = {}
    profile.update(
        {
            "prepare_python_wrapper_ms": (time.perf_counter() - wrapper_start)
            * 1000.0,
            "prepare_python_bundle_to_tensor_ms": bundle_to_tensor_ms,
            "prepare_python_load_extension_ms": load_extension_ms,
        }
    )
    _set_last_qat_prepare_profile(profile)
    return QatPreparedPayload((payload,))


def prepare_raw_deflate_bundle_file(
    path: str | Path,
    dynamic_huffman: bool = True,
    max_instances: int = 16,
    bandwidth_gbps: float = 0.0,
    file_offset: int = 0,
) -> QatPreparedPayload:
    if qat_codec() != "qat_codec":
        raise RuntimeError(
            "Origami request-level bundle restore requires "
            "ORIGAMI_QAT_CODEC=qat_codec"
        )
    wrapper_start = time.perf_counter()
    load_start = time.perf_counter()
    ext = load_qat_extension()
    load_extension_ms = (time.perf_counter() - load_start) * 1000.0
    payload = ext.qat_deflate_prepare_bundle_from_file_dp(
        str(path),
        bool(dynamic_huffman),
        int(max_instances),
        float(bandwidth_gbps),
        int(file_offset),
    )
    try:
        profile = dict(ext.qat_deflate_last_prepare_profile_dp())
    except AttributeError:
        profile = {}
    profile.update(
        {
            "prepare_python_wrapper_ms": (time.perf_counter() - wrapper_start)
            * 1000.0,
            "prepare_python_load_extension_ms": load_extension_ms,
            "prepare_python_direct_file": True,
        }
    )
    _set_last_qat_prepare_profile(profile)
    return QatPreparedPayload((payload,))


def decompress_prepared_raw_deflate_many(
    payload: QatPreparedPayload,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> list[torch.Tensor]:
    if any(isinstance(item, CpaPreparedPayloadItem) for item in payload.payloads):
        ext = load_qat_cpa_extension()
        outputs = []
        profile = _empty_qat_profile()
        for item in payload.payloads:
            if not isinstance(item, CpaPreparedPayloadItem):
                raise TypeError("mixed CPA and DP prepared payloads are not supported")
            restored = _cpu_u8(
                ext.qat_deflate_decompress_opts(
                    item.bytestream,
                    item.lengths,
                    int(item.output_bytes),
                    int(item.chunk_bytes),
                    bool(item.dynamic_huffman),
                )
            )
            outputs.append(restored)
            compressed = int(item.bytestream.numel())
            unpacked = int(item.output_bytes)
            profile = _add_qat_profiles(
                profile,
                {
                    "total_ms": 0.0,
                    "chunks": int(item.lengths.numel()),
                    "compressed_bytes": compressed,
                    "unpacked_bytes": unpacked,
                },
            )
        _set_last_qat_profile(profile)
        return outputs
    ext = load_qat_extension()
    outputs = []
    profile = _empty_qat_profile()
    for item in payload.payloads:
        output = _allocate_restore_staging(int(item.output_bytes))
        if hasattr(ext, "qat_deflate_decompress_prepared_into_dp"):
            restored = ext.qat_deflate_decompress_prepared_into_dp(
                item,
                output,
                int(inflight),
                int(batch),
                int(max_instances),
            )
        else:
            restored = ext.qat_deflate_decompress_prepared_dp(
                item,
                int(inflight),
                int(batch),
                int(max_instances),
            )
        restored = _cpu_u8(restored)
        outputs.append(restored)
        try:
            item_profile = dict(ext.qat_deflate_last_profile_dp())
        except AttributeError:
            item_profile = _empty_qat_profile()
        item_profile["output_staging_pinned"] = bool(
            hasattr(restored, "is_pinned") and restored.is_pinned()
        )
        profile = _add_qat_profiles(profile, item_profile)
    _set_last_qat_profile(profile)
    return outputs


def decompress_prepared_raw_deflate_window_with_profile(
    payload: QatPreparedPayload,
    indices: Iterable[int],
    output_bytes: int,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decompress a chunk-index window from a prepared DP payload.

    The returned tensor is a contiguous CPU uint8 staging buffer containing the
    selected chunks in the same order as ``indices``. Indices are local to the
    single prepared DP payload item.
    """
    if any(isinstance(item, CpaPreparedPayloadItem) for item in payload.payloads):
        raise RuntimeError("prepared window restore is only implemented for DP payloads")
    if len(payload.payloads) != 1:
        raise RuntimeError(
            "prepared window restore expects exactly one DP payload item"
        )
    ext = load_qat_extension()
    index_list = [int(index) for index in indices]
    indices_tensor = torch.tensor(index_list, dtype=torch.int64).contiguous()
    output = _allocate_restore_staging(int(output_bytes))
    if hasattr(ext, "qat_deflate_decompress_prepared_window_with_profile_into_dp"):
        restored, profile = ext.qat_deflate_decompress_prepared_window_with_profile_into_dp(
            payload.payloads[0],
            indices_tensor,
            output,
            int(inflight),
            int(batch),
            int(max_instances),
        )
        profile = dict(profile)
    elif hasattr(ext, "qat_deflate_decompress_prepared_window_into_dp"):
        with _QAT_PROFILE_FALLBACK_LOCK:
            restored = ext.qat_deflate_decompress_prepared_window_into_dp(
                payload.payloads[0],
                indices_tensor,
                output,
                int(inflight),
                int(batch),
                int(max_instances),
            )
            try:
                profile = dict(ext.qat_deflate_last_profile_dp())
            except AttributeError:
                profile = _empty_qat_profile()
    else:
        with _QAT_PROFILE_FALLBACK_LOCK:
            restored = ext.qat_deflate_decompress_prepared_window_dp(
                payload.payloads[0],
                indices_tensor,
                int(inflight),
                int(batch),
                int(max_instances),
            )
            try:
                profile = dict(ext.qat_deflate_last_profile_dp())
            except AttributeError:
                profile = _empty_qat_profile()
    restored = _cpu_u8(restored)
    profile["output_staging_pinned"] = bool(
        hasattr(restored, "is_pinned") and restored.is_pinned()
    )
    _set_last_qat_profile(profile)
    return restored, profile


def decompress_prepared_raw_deflate_window(
    payload: QatPreparedPayload,
    indices: Iterable[int],
    output_bytes: int,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> torch.Tensor:
    restored, _ = decompress_prepared_raw_deflate_window_with_profile(
        payload,
        indices,
        output_bytes,
        inflight=inflight,
        batch=batch,
        max_instances=max_instances,
    )
    return restored


def profile_prepared_raw_deflate_window(
    payload: QatPreparedPayload,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
    loops: int = 1,
) -> dict[str, Any]:
    """Run a CPA-style no-copy prepared decompress profile.

    This intentionally does not return restored bytes. It mirrors the window
    benchmark by measuring enqueue/poll throughput on already prepared
    QAT DMA input and reusable worker-local destination slots.
    """
    if any(isinstance(item, CpaPreparedPayloadItem) for item in payload.payloads):
        raise RuntimeError("CPA-style window profile is only implemented for DP payloads")
    ext = load_qat_extension()
    profile = _empty_qat_profile()
    for item in payload.payloads:
        item_profile = dict(
            ext.qat_deflate_profile_prepared_window_dp(
                item,
                int(inflight),
                int(batch),
                int(max_instances),
                int(loops),
            )
        )
        profile = _add_qat_profiles(profile, item_profile)
    _set_last_qat_profile(profile)
    return profile
