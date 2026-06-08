# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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
    ext = load_qat_extension()
    count = int(ext.qat_deflate_dp_instance_count())
    if max_instances and max_instances > 0:
        return min(count, int(max_instances))
    return count


def qat_instance_nodes(max_instances: int = 0) -> list[int]:
    ext = load_qat_extension()
    return [int(node) for node in ext.qat_deflate_dp_instance_nodes(int(max_instances))]


def qat_available() -> bool:
    try:
        return qat_instance_count() > 0
    except Exception:
        return False


def compress_raw_deflate_many(
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


def prepare_raw_deflate_many(
    chunks: Iterable[torch.Tensor],
    output_bytes: Iterable[int],
    dynamic_huffman: bool = True,
    max_instances: int = 16,
) -> QatPreparedPayload:
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


def decompress_prepared_raw_deflate_many(
    payload: QatPreparedPayload,
    inflight: int = 32,
    batch: int = 32,
    max_instances: int = 16,
) -> list[torch.Tensor]:
    ext = load_qat_extension()
    return [
        _cpu_u8(
            ext.qat_deflate_decompress_prepared_dp(
                item,
                int(inflight),
                int(batch),
                int(max_instances),
            )
        )
        for item in payload.payloads
    ]
