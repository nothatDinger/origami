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
        name="origami_bitpack_cuda_v6",
        sources=[
            str(_CSRC / "bitpack_layout_cuda.cpp"),
            str(_CSRC / "bitpack_layout_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=_build_dir("bitpack_cuda_v6"),
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


def _fp16_blob_view(
    bytestream: torch.Tensor,
    metadata: dict[str, Any],
    name: str,
) -> torch.Tensor:
    offset = int(metadata[f"{name}_offset"])
    numel = int(metadata[f"{name}_numel"])
    byte_count = numel * torch.empty((), dtype=torch.float16).element_size()
    if offset < 0 or offset + byte_count > int(bytestream.numel()):
        raise ValueError(f"{name} range exceeds the Origami artifact")
    if offset % 2:
        raise ValueError(f"{name} must be aligned for float16 access")
    return bytestream.narrow(0, offset, byte_count).view(torch.float16)


def _fused_prefix_attention(
    bytestream: torch.Tensor,
    *,
    codec: int,
    metadata: dict[str, Any],
    key_scale: torch.Tensor,
    key_zero: torch.Tensor,
    value_scale: torch.Tensor,
    value_zero: torch.Tensor,
    key_sink: torch.Tensor,
    value_sink: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_position: int,
    sequence_length: int,
    softmax_scale: float,
    output: torch.Tensor,
) -> None:
    if bytestream.device != query.device or query.device != kv_cache.device:
        raise ValueError("compressed payload, query, and KV cache must share a device")
    if query.ndim != 3 or output.shape != query.shape:
        raise ValueError("query and output must be matching rank-3 tensors")
    if kv_cache.ndim != 5 or int(kv_cache.shape[0]) != 2:
        raise ValueError(
            "fused attention requires NHD KV cache shape "
            "[2, blocks, block, heads, dim]"
        )
    if int(metadata["head_dim"]) != int(query.shape[2]):
        raise ValueError("artifact head_dim does not match query")
    if int(metadata["num_heads"]) != int(kv_cache.shape[3]):
        raise ValueError("artifact KV head count does not match KV cache")
    if int(kv_cache.shape[4]) != int(query.shape[2]):
        raise ValueError("KV cache head_dim does not match query")
    declared_bytes = int(metadata.get("symbol_byte_count", bytestream.numel()))
    if declared_bytes != int(bytestream.numel()):
        raise ValueError("artifact byte count does not match compressed payload")
    if int(query_start_position) < int(metadata["token_count"]):
        raise ValueError("query starts inside the compressed prefix")
    ext = load_bitpack_cuda_extension()
    ext.fused_prefix_attention_cuda(
        bytestream.reshape(-1).contiguous(),
        int(metadata["key_stream_offset"]),
        int(metadata["key_stream_bytes"]),
        int(metadata["value_stream_offset"]),
        int(metadata["value_stream_bytes"]),
        int(codec),
        int(metadata["bits"]),
        int(metadata.get("group_size", 1)),
        int(metadata.get("sink_tokens", 0)),
        key_scale.contiguous(),
        key_zero.contiguous(),
        value_scale.contiguous(),
        value_zero.contiguous(),
        key_sink.contiguous(),
        value_sink.contiguous(),
        query.contiguous(),
        kv_cache,
        block_table.to(device=kv_cache.device, dtype=torch.int32).contiguous(),
        int(metadata["token_count"]),
        int(query_start_position),
        int(sequence_length),
        float(softmax_scale),
        output,
    )


def kivi_fused_prefix_attention(
    bytestream: torch.Tensor,
    *,
    metadata: dict[str, Any],
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_position: int,
    sequence_length: int,
    softmax_scale: float,
    output: torch.Tensor,
) -> None:
    if str(metadata.get("format")) != "kivi_structured_blob":
        raise ValueError("KIVI fused attention requires a structured KIVI artifact")
    if tuple(metadata.get("key_order", ())) != (
        "head_dim", "layer", "head", "token"
    ) or tuple(metadata.get("value_order", ())) != (
        "head", "layer", "head_dim", "token"
    ):
        raise ValueError("KIVI artifact has an unsupported packed layout")
    _fused_prefix_attention(
        bytestream,
        codec=0,
        metadata=metadata,
        key_scale=_fp16_blob_view(bytestream, metadata, "key_scale"),
        key_zero=_fp16_blob_view(bytestream, metadata, "key_zero"),
        value_scale=_fp16_blob_view(bytestream, metadata, "value_scale"),
        value_zero=_fp16_blob_view(bytestream, metadata, "value_zero"),
        key_sink=_fp16_blob_view(bytestream, metadata, "key_sink"),
        value_sink=_fp16_blob_view(bytestream, metadata, "value_sink"),
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        query_start_position=query_start_position,
        sequence_length=sequence_length,
        softmax_scale=softmax_scale,
        output=output,
    )


def kvquant_fused_prefix_attention(
    bytestream: torch.Tensor,
    *,
    metadata: dict[str, Any],
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_position: int,
    sequence_length: int,
    softmax_scale: float,
    output: torch.Tensor,
) -> None:
    if str(metadata.get("format")) != "kvquant_structured_blob":
        raise ValueError("KVQuant fused attention requires a structured artifact")
    if tuple(metadata.get("key_order", ())) != (
        "head_dim", "layer", "head", "token"
    ) or tuple(metadata.get("value_order", ())) != (
        "layer", "head", "token", "head_dim"
    ):
        raise ValueError("KVQuant artifact has an unsupported packed layout")
    empty = torch.empty((0,), dtype=torch.float16, device=bytestream.device)
    _fused_prefix_attention(
        bytestream,
        codec=1,
        metadata=metadata,
        key_scale=_fp16_blob_view(bytestream, metadata, "key_scale"),
        key_zero=_fp16_blob_view(bytestream, metadata, "key_min"),
        value_scale=_fp16_blob_view(bytestream, metadata, "value_scale"),
        value_zero=_fp16_blob_view(bytestream, metadata, "value_min"),
        key_sink=empty,
        value_sink=empty,
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        query_start_position=query_start_position,
        sequence_length=sequence_length,
        softmax_scale=softmax_scale,
        output=output,
    )
