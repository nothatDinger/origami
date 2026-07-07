# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.cpp_extension import load

from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
    native_cpu,
)

_REPO_ROOT = Path(__file__).resolve().parents[7]
_CSRC = _REPO_ROOT / "csrc" / "origami"
_BUILD_ROOT = Path(
    os.environ.get(
        "ORIGAMI_NATIVE_GPU_BUILD_DIR",
        str(Path(tempfile.gettempdir()) / "origami_native_gpu"),
    )
)
_VERBOSE = bool(int(os.environ.get("ORIGAMI_NATIVE_VERBOSE", "0")))


def _build_dir(name: str) -> str:
    path = _BUILD_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@lru_cache(maxsize=1)
def load_bitpack_cuda_extension() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("Origami CUDA bitpack extension requires CUDA")
    return load(
        name="origami_bitpack_cuda_v5",
        sources=[
            str(_CSRC / "bitpack_layout_cuda.cpp"),
            str(_CSRC / "bitpack_layout_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=_build_dir("bitpack_cuda_v5"),
        verbose=_VERBOSE,
    )


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        load_bitpack_cuda_extension()
        return True
    except Exception:
        return False


def _cpu_i64(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor([int(value) for value in values],
                        dtype=torch.int64).contiguous()


def _cuda_i64(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([int(value) for value in values],
                        dtype=torch.int64,
                        device=device).contiguous()


def _chunk_specs_tensor(
    chunk_specs: torch.Tensor | Iterable[Iterable[int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(chunk_specs, torch.Tensor):
        specs = chunk_specs.detach().to(dtype=torch.int64, device=device).contiguous()
    else:
        rows = list(chunk_specs)
        if not rows:
            return torch.empty((0, 6), dtype=torch.int64, device=device)
        specs = torch.tensor(rows, dtype=torch.int64, device=device).contiguous()
    if specs.numel() == 0:
        return torch.empty((0, 6), dtype=torch.int64, device=device)
    if specs.dim() == 1:
        specs = specs.reshape(1, 6)
    return specs.contiguous()


def _flat_chunks_and_offsets(
    chunks: Iterable[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_chunks: list[torch.Tensor] = []
    offsets = [0]
    total = 0
    for chunk in chunks:
        flat = chunk.detach().to(device=device, dtype=torch.uint8,
                                 non_blocking=True).reshape(-1).contiguous()
        flat_chunks.append(flat)
        total += int(flat.numel())
        offsets.append(total)
    if flat_chunks:
        flat = torch.cat(flat_chunks, dim=0).contiguous()
    else:
        flat = torch.empty((0,), dtype=torch.uint8, device=device)
    return flat, _cuda_i64(offsets, device)


def unpack_canonical_storage_chunks(
    chunks: Iterable[torch.Tensor],
    bits: int | Sequence[int],
    source_shape: Sequence[int],
    source_layout: Sequence[str],
    chunk_specs: torch.Tensor | Iterable[Iterable[int]],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    target = torch.device(device or "cuda")
    token_count, num_heads, head_dim, shape, layout = (
        native_cpu.canonical_axis_sizes(source_shape, source_layout)
    )
    specs = _chunk_specs_tensor(chunk_specs, device=target)
    num_chunks = int(specs.shape[0])
    if isinstance(bits, int):
        bits_per_chunk = [int(bits)] * num_chunks
    else:
        bits_per_chunk = [int(value) for value in bits]
    if len(bits_per_chunk) != num_chunks:
        raise ValueError("bits_per_chunk length must match chunk_specs")
    flat_chunks, offsets = _flat_chunks_and_offsets(chunks, device=target)
    ext = load_bitpack_cuda_extension()
    return ext.unpack_canonical_storage_chunks_cuda(
        flat_chunks,
        offsets,
        _cuda_i64(bits_per_chunk, target),
        _cpu_i64(shape),
        _cpu_i64(native_cpu.axis_positions(layout)),
        int(token_count),
        int(num_heads),
        int(head_dim),
        specs,
    )


def cachegen_unpack_dequantize(
    bytestream: torch.Tensor,
    offsets: torch.Tensor,
    bits_per_layer: torch.Tensor,
    key_bins: torch.Tensor,
    value_bins: torch.Tensor,
    max_key: torch.Tensor,
    max_value: torch.Tensor,
    *,
    tokens: int,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    ext = load_bitpack_cuda_extension()
    return ext.cachegen_unpack_dequantize_cuda(
        bytestream.reshape(-1).contiguous(),
        offsets.to(device=bytestream.device, dtype=torch.int64).contiguous(),
        bits_per_layer.to(device=bytestream.device,
                          dtype=torch.int64).contiguous(),
        key_bins.to(device=bytestream.device, dtype=torch.int64).contiguous(),
        value_bins.to(device=bytestream.device, dtype=torch.int64).contiguous(),
        max_key.contiguous(),
        max_value.contiguous(),
        int(tokens),
        int(heads),
        int(head_dim),
    )


def cachegen_unpack_dequantize_to_kv_cache(
    bytestream: torch.Tensor,
    *,
    key_offset: int,
    key_bytes: int,
    value_offset: int,
    value_bytes: int,
    key_bits: int,
    value_bits: int,
    key_bins: int,
    value_bins: int,
    max_key: torch.Tensor,
    max_value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    tokens: int,
    heads: int,
    head_dim: int,
) -> None:
    ext = load_bitpack_cuda_extension()
    ext.cachegen_unpack_dequantize_to_kv_cache_cuda(
        bytestream.reshape(-1).contiguous(),
        int(key_offset),
        int(key_bytes),
        int(value_offset),
        int(value_bytes),
        int(key_bits),
        int(value_bits),
        int(key_bins),
        int(value_bins),
        max_key.contiguous(),
        max_value.contiguous(),
        kv_cache,
        block_ids.to(device=kv_cache.device, dtype=torch.int64).contiguous(),
        int(tokens),
        int(heads),
        int(head_dim),
    )


def kivi_dequantize_to_kv_cache(
    bytestream: torch.Tensor,
    *,
    key_offset: int,
    key_bytes: int,
    value_offset: int,
    value_bytes: int,
    bits: int,
    group_size: int,
    sink_tokens: int,
    key_scale: torch.Tensor,
    key_zero: torch.Tensor,
    value_scale: torch.Tensor,
    value_zero: torch.Tensor,
    key_sink: torch.Tensor,
    value_sink: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    tokens: int,
    heads: int,
    head_dim: int,
) -> None:
    ext = load_bitpack_cuda_extension()
    ext.kivi_dequantize_to_kv_cache_cuda(
        bytestream.reshape(-1).contiguous(),
        int(key_offset),
        int(key_bytes),
        int(value_offset),
        int(value_bytes),
        int(bits),
        int(group_size),
        int(sink_tokens),
        key_scale.contiguous(),
        key_zero.contiguous(),
        value_scale.contiguous(),
        value_zero.contiguous(),
        key_sink.contiguous(),
        value_sink.contiguous(),
        kv_cache,
        block_ids.to(device=kv_cache.device, dtype=torch.int64).contiguous(),
        int(tokens),
        int(heads),
        int(head_dim),
    )
