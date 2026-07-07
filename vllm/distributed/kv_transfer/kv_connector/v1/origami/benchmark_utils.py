# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import torch


@dataclass
class ReadMetric:
    request_id: str
    cache_key: str
    system: str
    path: str
    bytes_read: int
    start_ts: float
    end_ts: float
    bandwidth_gbps: float


@dataclass
class H2DMetric:
    request_id: str
    cache_key: str
    system: str
    layer_name: str
    bytes_copied: int
    h2d_ms: float
    h2d_gbps: float
    start_ts: float
    end_ts: float
    device: str
    source_device: str = ""
    copy_kind: str = ""
    timing_complete: bool = True


@dataclass
class KernelRange:
    name: str
    start_ns: int
    end_ns: int


@dataclass
class KernelRecord:
    name: str
    start_ns: int
    end_ns: int


_JSONL_LOCK = Lock()
_ASYNC_METRIC_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_ARTIFACT_MEMORY_CACHE: dict[str, bytes] = {}
_ARTIFACT_MEMORY_LOCK = Lock()


class TokenBucket:
    """Simple process-local token bucket for deterministic read throttling."""

    def __init__(self, bytes_per_second: float):
        self.bytes_per_second = float(bytes_per_second)
        self._next_available_ts = time.perf_counter()

    @classmethod
    def from_gbps(cls, gbps: float) -> "TokenBucket | None":
        gbps = float(gbps)
        if gbps <= 0:
            return None
        return cls(gbps * 1e9 / 8.0)

    def consume(self, num_bytes: int) -> float:
        if self.bytes_per_second <= 0 or num_bytes <= 0:
            return 0.0
        now = time.perf_counter()
        start = max(now, self._next_available_ts)
        duration = float(num_bytes) / self.bytes_per_second
        self._next_available_ts = start + duration
        delay = self._next_available_ts - now
        if delay > 0:
            time.sleep(delay)
            return delay
        return 0.0


@contextmanager
def nvtx_range(name: str):
    pushed = False
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


def write_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_LOCK:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")


def safe_cache_key(cache_key: str) -> str:
    return str(cache_key).replace("/", "_").replace(":", "_")


def safe_layer_name(layer_name: str) -> str:
    return str(layer_name).replace("/", "_").replace(":", "_")


def controlled_read(
    path: str | Path,
    *,
    request_id: str,
    cache_key: str,
    system: str,
    bandwidth_gbps: float,
    metrics_path: str | Path | None = None,
    chunk_bytes: int = 4 << 20,
    ingress_mode: str = "file",
) -> bytes:
    path = Path(path)
    ingress_mode = str(ingress_mode or "file").lower()
    if ingress_mode == "memory":
        data = get_memory_artifact_bytes(
            path,
            system=system,
            metrics_path=metrics_path,
        )
        return controlled_memory_ingress(
            data,
            path=path,
            request_id=request_id,
            cache_key=cache_key,
            system=system,
            bandwidth_gbps=bandwidth_gbps,
            metrics_path=metrics_path,
        )
    if ingress_mode != "file":
        raise ValueError(
            "artifact ingress mode must be one of {'file', 'memory'}, "
            f"got {ingress_mode!r}"
        )
    bucket = TokenBucket.from_gbps(float(bandwidth_gbps))
    chunks: list[bytes] = []
    bytes_read = 0
    read_io_ms = 0.0
    read_sleep_ms = 0.0
    start_ts = time.time()
    with nvtx_range(f"{system}:ssd_read"):
        with path.open("rb") as handle:
            while True:
                io_start = time.perf_counter()
                chunk = handle.read(chunk_bytes)
                read_io_ms += (time.perf_counter() - io_start) * 1000.0
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bucket is not None:
                    read_sleep_ms += bucket.consume(len(chunk)) * 1000.0
    end_ts = time.time()
    elapsed_ms = max(0.0, (end_ts - start_ts) * 1000.0)
    source_read_io_gbps = (
        bytes_read * 8.0 / read_io_ms / 1e6 if read_io_ms > 0.0 else 0.0
    )
    effective_limited_read_gbps = (
        bytes_read * 8.0 / elapsed_ms / 1e6 if elapsed_ms > 0.0 else 0.0
    )
    metric = ReadMetric(
        request_id=request_id,
        cache_key=cache_key,
        system=system,
        path=str(path),
        bytes_read=bytes_read,
        start_ts=start_ts,
        end_ts=end_ts,
        bandwidth_gbps=float(bandwidth_gbps),
    )
    write_jsonl(
        metrics_path,
        {
            "type": "ssd_read",
            **asdict(metric),
            "ms": elapsed_ms,
            "read_io_ms": read_io_ms,
            "read_sleep_ms": read_sleep_ms,
            "source_read_io_gbps": source_read_io_gbps,
            "effective_limited_read_gbps": effective_limited_read_gbps,
            "storage_source": "dram"
            if str(path).startswith("/dev/shm/")
            else "ssd",
            "limiter_scope": "artifact_ingress"
            if float(bandwidth_gbps) > 0.0
            else "none",
        },
    )
    return b"".join(chunks)


def _memory_cache_key(path: str | Path) -> str:
    return str(Path(path))


def get_memory_artifact_bytes(
    path: str | Path,
    *,
    system: str = "",
    metrics_path: str | Path | None = None,
) -> bytes:
    """Return a process-local resident copy of an artifact file.

    This is intentionally outside the per-request throttled ingress path. It is
    used by high-bandwidth DRAM-backed experiments where the measured variable
    is the controlled movement from resident artifact bytes into restore/QAT
    logic, not Python file I/O.
    """
    path = Path(path)
    key = _memory_cache_key(path)
    with _ARTIFACT_MEMORY_LOCK:
        cached = _ARTIFACT_MEMORY_CACHE.get(key)
    if cached is not None:
        return cached

    start_ts = time.time()
    start = time.perf_counter()
    data = path.read_bytes()
    end_ts = time.time()
    ms = (time.perf_counter() - start) * 1000.0
    with _ARTIFACT_MEMORY_LOCK:
        cached = _ARTIFACT_MEMORY_CACHE.setdefault(key, data)
    if cached is data:
        write_jsonl(
            metrics_path,
            {
                "type": "artifact_memory_stage_load",
                "system": system,
                "path": str(path),
                "bytes_read": len(data),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "ms": ms,
                "source": "file_to_process_memory",
                "storage_source": "dram"
                if str(path).startswith("/dev/shm/")
                else "ssd",
                "source_read_io_gbps": (
                    len(data) * 8.0 / ms / 1e6 if ms > 0.0 else 0.0
                ),
            },
        )
    return cached


def preload_memory_artifacts(
    paths: Iterable[str | Path],
    *,
    system: str = "",
    metrics_path: str | Path | None = None,
) -> None:
    for path in paths:
        get_memory_artifact_bytes(path, system=system, metrics_path=metrics_path)


def controlled_memory_ingress(
    data: bytes,
    *,
    path: str | Path,
    request_id: str,
    cache_key: str,
    system: str,
    bandwidth_gbps: float,
    metrics_path: str | Path | None = None,
) -> bytes:
    bucket = TokenBucket.from_gbps(float(bandwidth_gbps))
    bytes_read = len(data)
    start_ts = time.time()
    read_start = time.perf_counter()
    # The artifact is already resident; this records only Python lookup cost.
    read_io_ms = (time.perf_counter() - read_start) * 1000.0
    read_sleep_ms = 0.0
    with nvtx_range(f"{system}:artifact_ingress"):
        if bucket is not None:
            read_sleep_ms = bucket.consume(bytes_read) * 1000.0
    end_ts = time.time()
    elapsed_ms = max(0.0, (end_ts - start_ts) * 1000.0)
    source_read_io_gbps = (
        bytes_read * 8.0 / read_io_ms / 1e6 if read_io_ms > 0.0 else 0.0
    )
    effective_limited_read_gbps = (
        bytes_read * 8.0 / elapsed_ms / 1e6 if elapsed_ms > 0.0 else 0.0
    )
    write_jsonl(
        metrics_path,
        {
            "type": "ssd_read",
            "request_id": request_id,
            "cache_key": cache_key,
            "system": system,
            "path": str(path),
            "bytes_read": bytes_read,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "bandwidth_gbps": float(bandwidth_gbps),
            "ms": elapsed_ms,
            "read_io_ms": read_io_ms,
            "read_sleep_ms": read_sleep_ms,
            "source_read_io_gbps": source_read_io_gbps,
            "effective_limited_read_gbps": effective_limited_read_gbps,
            "storage_source": "process_memory",
            "limiter_scope": "memory_artifact_ingress"
            if float(bandwidth_gbps) > 0.0
            else "none",
            "artifact_ingress_mode": "memory",
        },
    )
    return data


def torch_load_from_bytes(data: bytes, *, map_location: str = "cpu") -> Any:
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _normalize_kv_value(
    kv_cache: torch.Tensor,
    block_count: int,
    value: torch.Tensor,
) -> torch.Tensor:
    if block_count <= 0:
        return value
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        block_size = int(kv_cache.shape[2])
        heads = int(kv_cache.shape[3])
        head_dim = int(kv_cache.shape[4])
        if (
            value.dim() == 5
            and value.shape[0] == 2
            and value.shape[1] == block_count
            and value.shape[2] == block_size
        ):
            return value
        if value.dim() == 4 and value.shape[0] == 2:
            if value.shape[1] == heads and value.shape[3] == head_dim:
                token_major = value.permute(0, 2, 1, 3).contiguous()
            elif value.shape[2] == heads and value.shape[3] == head_dim:
                token_major = value.contiguous()
            else:
                raise ValueError(
                    "Unsupported HF KV artifact layer shape "
                    f"{tuple(value.shape)} for cache {tuple(kv_cache.shape)}"
                )
            needed_tokens = block_count * block_size
            if token_major.shape[1] < needed_tokens:
                pad = torch.zeros(
                    (
                        2,
                        needed_tokens - int(token_major.shape[1]),
                        heads,
                        head_dim,
                    ),
                    dtype=token_major.dtype,
                    device=token_major.device,
                )
                token_major = torch.cat((token_major, pad), dim=1)
            token_major = token_major[:, :needed_tokens, :, :]
            return token_major.reshape(2, block_count, block_size, heads, head_dim)
    if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
        block_size = int(kv_cache.shape[2])
        heads = int(kv_cache.shape[3])
        head_dim = int(kv_cache.shape[4])
        if (
            value.dim() == 5
            and value.shape[0] == block_count
            and value.shape[1] == 2
            and value.shape[2] == block_size
        ):
            return value
        if value.dim() == 4 and value.shape[0] == 2:
            if value.shape[1] == heads and value.shape[3] == head_dim:
                token_major = value.permute(0, 2, 1, 3).contiguous()
            elif value.shape[2] == heads and value.shape[3] == head_dim:
                token_major = value.contiguous()
            else:
                raise ValueError(
                    "Unsupported HF KV artifact layer shape "
                    f"{tuple(value.shape)} for cache {tuple(kv_cache.shape)}"
                )
            needed_tokens = block_count * block_size
            if token_major.shape[1] < needed_tokens:
                pad = torch.zeros(
                    (
                        2,
                        needed_tokens - int(token_major.shape[1]),
                        heads,
                        head_dim,
                    ),
                    dtype=token_major.dtype,
                    device=token_major.device,
                )
                token_major = torch.cat((token_major, pad), dim=1)
            token_major = token_major[:, :needed_tokens, :, :]
            return token_major.reshape(2, block_count, block_size, heads, head_dim).permute(
                1, 0, 2, 3, 4
            ).contiguous()
    return value


def _assign_blocks(kv_cache: torch.Tensor, block_ids: list[int], value: torch.Tensor) -> None:
    block_ids_tensor = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    value = _normalize_kv_value(kv_cache, len(block_ids), value)
    if value.device != kv_cache.device or value.dtype != kv_cache.dtype:
        value = value.to(device=kv_cache.device, dtype=kv_cache.dtype, non_blocking=False)
    if kv_cache.dim() >= 2 and kv_cache.shape[0] == 2 and value.shape[0] == 2:
        kv_cache.index_copy_(1, block_ids_tensor, value)
    elif kv_cache.dim() >= 2 and kv_cache.shape[1] == 2 and value.shape[1] == 2:
        kv_cache.index_copy_(0, block_ids_tensor, value)
    else:
        kv_cache.index_copy_(0, block_ids_tensor, value)


def _finalize_cuda_copy_metric(
    metric: H2DMetric,
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
    metrics_path: str | Path | None,
) -> None:
    end_event.synchronize()
    metric.h2d_ms = float(start_event.elapsed_time(end_event))
    metric.h2d_gbps = (
        metric.bytes_copied * 8.0 / metric.h2d_ms / 1e6
        if metric.h2d_ms > 0
        else 0.0
    )
    metric.end_ts = time.time()
    metric.timing_complete = True
    write_jsonl(metrics_path, {"type": "h2d_copy", **asdict(metric)})


def copy_kv_to_hbm(
    *,
    kv_cache: torch.Tensor,
    block_ids: Iterable[int],
    value: torch.Tensor,
    request_id: str,
    cache_key: str,
    system: str,
    layer_name: str,
    metrics_path: str | Path | None = None,
    synchronize: bool = True,
    return_event: bool = False,
) -> H2DMetric | tuple[H2DMetric, torch.cuda.Event | None]:
    block_ids_list = [int(block_id) for block_id in block_ids]
    bytes_copied = tensor_nbytes(value)
    source_device = str(value.device)
    start_ts = time.time()
    completion_event: torch.cuda.Event | None = None
    if kv_cache.device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with nvtx_range(f"{system}:h2d_copy"):
            start_event.record()
            _assign_blocks(kv_cache, block_ids_list, value)
            end_event.record()
            completion_event = end_event
            if synchronize:
                end_event.synchronize()
        h2d_ms = float(start_event.elapsed_time(end_event)) if synchronize else 0.0
    else:
        start_perf = time.perf_counter()
        with nvtx_range(f"{system}:h2d_copy"):
            _assign_blocks(kv_cache, block_ids_list, value)
        h2d_ms = (time.perf_counter() - start_perf) * 1000.0
    end_ts = time.time()
    h2d_gbps = (bytes_copied * 8.0 / h2d_ms / 1e6) if h2d_ms > 0 else 0.0
    metric = H2DMetric(
        request_id=request_id,
        cache_key=cache_key,
        system=system,
        layer_name=layer_name,
        bytes_copied=bytes_copied,
        h2d_ms=h2d_ms,
        h2d_gbps=h2d_gbps,
        start_ts=start_ts,
        end_ts=end_ts,
        device=str(kv_cache.device),
        source_device=source_device,
        copy_kind=(
            "hbm_to_hbm" if source_device.startswith("cuda")
            and kv_cache.device.type == "cuda" else "dram_to_hbm"
            if kv_cache.device.type == "cuda" else "host_copy"
        ),
        timing_complete=bool(synchronize or kv_cache.device.type != "cuda"),
    )
    if kv_cache.device.type == "cuda" and not synchronize and completion_event is not None:
        _ASYNC_METRIC_EXECUTOR.submit(
            _finalize_cuda_copy_metric,
            metric,
            start_event,
            completion_event,
            metrics_path,
        )
    else:
        write_jsonl(metrics_path, {"type": "h2d_copy", **asdict(metric)})
    if return_event:
        return metric, completion_event
    return metric


def classify_kernel_records(
    kernels: Iterable[KernelRecord],
    ranges: Iterable[KernelRange],
) -> dict[str, float]:
    """Classify kernel time by enclosing NVTX range name.

    Returns milliseconds by category: attention, dequant, decompress, other_gpu_kernel.
    """
    totals = {
        "attention": 0.0,
        "dequant": 0.0,
        "decompress": 0.0,
        "other_gpu_kernel": 0.0,
    }
    ordered_ranges = list(ranges)
    for kernel in kernels:
        category = "other_gpu_kernel"
        for rng in ordered_ranges:
            if rng.start_ns <= kernel.start_ns and kernel.end_ns <= rng.end_ns:
                lowered = rng.name.lower()
                if "attention" in lowered:
                    category = "attention"
                elif "dequant" in lowered:
                    category = "dequant"
                elif "decompress" in lowered:
                    category = "decompress"
                break
        totals[category] += max(0.0, (kernel.end_ns - kernel.start_ns) / 1e6)
    return totals

def windows_overlap(
    first_start_ts: float,
    first_end_ts: float,
    second_start_ts: float,
    second_end_ts: float,
) -> bool:
    return max(first_start_ts, second_start_ts) < min(first_end_ts, second_end_ts)
