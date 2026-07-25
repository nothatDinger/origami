# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.benchmark_utils import (
    copy_kv_to_hbm,
    nvtx_range,
    write_jsonl,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import OrigamiConfig
from vllm.distributed.kv_transfer.kv_connector.v1.origami.fused_attention import (
    FusedLayerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import native_cpu
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import native_gpu
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.chunking import (
    PlannedChunk,
    plan_head_channel_chunks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.cpu_qat import (
    CpuLosslessCodec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.gpu_nvcomp import (
    GpuLosslessCodec,
    NvcompCodec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    ChunkLayout,
    ChunkRecord,
    LayerPayload,
    OrigamiConnectorMetadata,
    OrigamiPayload,
    OrigamiRestoreRequest,
    OrigamiSaveRequest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization import (
    QuantizerAdapter,
    create_quantizer_adapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.store import (
    OrigamiStore,
    PreparedLayerPayload,
    PreparedOrigamiPayload,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata


_NATIVE_LAYOUT_KEY = "origami_native_layout"


def _layer_index(layer_name: str) -> int:
    digits = ""
    for ch in reversed(layer_name):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else 0


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _product(values: list[int] | tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


@dataclass
class _LayerLoadResult:
    event: torch.cuda.Event | None = None
    pending_metrics: list[tuple[str, OrigamiRestoreRequest, LayerPayload,
                                dict[str, Any]]] | None = None
    wait_on_current_stream: bool = True


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _restore_metric_id(restore: OrigamiRestoreRequest) -> str:
    return str(restore.metric_request_id or restore.request_id)


def _token_axis_from_metadata(metadata: dict[str, Any]) -> int | None:
    layout = metadata.get("layout") or metadata.get("source_layout")
    if isinstance(layout, (list, tuple)):
        try:
            return [str(item) for item in layout].index("token")
        except ValueError:
            return None
    # CacheGen restored tensors are normally [2, token, head, head_dim].
    shape = metadata.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        return 1
    return None


def _slice_kv_tensor_to_tokens(
    kv_tensor: torch.Tensor,
    metadata: dict[str, Any],
    token_count: int,
) -> torch.Tensor:
    token_count = int(token_count)
    if token_count <= 0:
        return kv_tensor
    axis = _token_axis_from_metadata(metadata)
    if axis is None or axis >= kv_tensor.dim():
        return kv_tensor
    available = int(kv_tensor.shape[axis])
    if token_count >= available:
        return kv_tensor
    return kv_tensor.narrow(axis, 0, token_count).contiguous()


def _merge_qat_profile(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(left or {})
    if not right:
        return out
    summed_keys = {
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
    for key in summed_keys:
        out[key] = out.get(key, 0) + right.get(key, 0)
    for key in ("workers", "inflight", "batch"):
        out[key] = max(int(out.get(key, 0)), int(right.get(key, 0)))
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


def _qat_profile_metric_fields(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    mapping = {
        "qat_profile_total_ms": "total_ms",
        "qat_worker_wall_ms_max": "worker_wall_ms_max",
        "qat_thread_launch_join_ms": "thread_launch_join_ms",
        "qat_slot_alloc_ms_sum": "slot_alloc_ms_sum",
        "qat_enqueue_ms_sum": "qat_enqueue_ms_sum",
        "qat_poll_wait_ms_sum": "qat_poll_wait_ms_sum",
        "qat_device_op_ms_sum": "qat_submit_poll_ms_sum",
        "qat_device_op_ms_critical": "qat_submit_poll_ms_critical",
        "qat_qae_output_copy_ms_sum": "qae_output_copy_ms_sum",
        "qat_workers": "workers",
        "qat_inflight": "inflight",
        "qat_batch": "batch",
        "qat_compressed_gbps_critical": "compressed_gbps_qat_critical",
        "qat_unpacked_gbps_critical": "unpacked_gbps_qat_critical",
        "qat_output_staging_pinned": "output_staging_pinned",
        "qat_persistent_workers": "persistent_workers",
    }
    return {dst: profile[src] for dst, src in mapping.items() if src in profile}


class OrigamiConnectorWorker:

    def __init__(
        self,
        config: OrigamiConfig,
        store: OrigamiStore,
        kv_cache_config: Any | None = None,
        *,
        fused_attention_enabled: bool = False,
    ):
        self.config = config
        self.store = store
        self.kv_cache_config = kv_cache_config
        self.fused_attention_enabled = bool(fused_attention_enabled)
        self.quantizer: QuantizerAdapter = create_quantizer_adapter(
            config.quantizer, config.quantizer_config
        )
        self.cpu_codec = CpuLosslessCodec(
            backend=config.lossless_cpu_backend,
            allow_zlib_fallback=config.allow_zlib_fallback,
            dynamic_huffman=config.qat_dynamic_huffman,
            qat_inflight=config.qat_inflight,
            qat_batch=config.qat_batch,
            qat_max_instances=config.qat_max_instances,
        )
        self._gpu_codec: GpuLosslessCodec | None = None
        self._nvcomp_codec: NvcompCodec | None = None
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.layer_to_cache_group: dict[str, int] = {}
        self._executor = ThreadPoolExecutor(max_workers=max(1, config.qat_threads))
        self._qat_executor = ThreadPoolExecutor(max_workers=max(1, config.qat_threads))
        self._pending_layer_futures: dict[str, list[Future[Any]]] = {}
        self._pending_request_layer_futures: dict[str, list[Future[Any]]] = {}
        self._restore_futures: list[Future[Any]] = []
        self._restore_future_to_request_id: dict[Future[Any], str] = {}
        self._save_futures: list[Future[tuple[OrigamiSaveRequest, LayerPayload]]] = []
        self._save_payloads: dict[str, OrigamiPayload] = {}
        self._save_lock = Lock()
        self._fused_lock = Lock()
        self._fused_payloads: dict[
            tuple[str, str, str], FusedLayerPayload
        ] = {}
        self._request_fused_keys: dict[
            str, dict[str, tuple[str, str, str]]
        ] = {}
        self._fused_upload_stream: torch.cuda.Stream | None = None
        self._fused_attention_stream: torch.cuda.Stream | None = None
        self._fused_qkv_ready_event: torch.cuda.Event | None = None
        self._fused_done_event: torch.cuda.Event | None = None
        self._metadata: OrigamiConnectorMetadata | None = None
        self.metrics_path = (
            str(Path(config.metrics_dir) / "origami_metrics.jsonl")
            if config.metrics_dir
            else None
        )
        gpu_extension_requested = (
            config.bitunpack_device in {"auto", "gpu", "cuda"}
            or config.dequant_device in {"auto", "gpu", "cuda"}
        )
        if torch.cuda.is_available() and gpu_extension_requested:
            try:
                native_gpu.load_bitpack_cuda_extension()
            except Exception:
                if (
                    config.bitunpack_device in {"gpu", "cuda"}
                    or config.dequant_device in {"gpu", "cuda"}
                ):
                    raise
        if self.fused_attention_enabled:
            capability = torch.cuda.get_device_capability()
            if capability < (8, 0):
                if config.fused_attention == "required":
                    raise RuntimeError(
                        "Origami fused attention requires CUDA compute capability 8.0+"
                    )
                self.fused_attention_enabled = False
            else:
                try:
                    native_gpu.load_bitpack_cuda_extension()
                except Exception:
                    if config.fused_attention == "required":
                        raise
                    self.fused_attention_enabled = False
            if self.fused_attention_enabled:
                self._fused_upload_stream = torch.cuda.Stream(priority=0)
                self._fused_attention_stream = torch.cuda.Stream(
                    priority=config.fused_stream_priority
                )
                self._fused_qkv_ready_event = torch.cuda.Event()
                self._fused_done_event = torch.cuda.Event()

    def _store_fused_layer_payload(
        self,
        restore: OrigamiRestoreRequest,
        layer_payload: LayerPayload,
        symbols: torch.Tensor,
    ) -> _LayerLoadResult | None:
        metadata = layer_payload.quant_metadata
        fmt = str(metadata.get("format", ""))
        expected_format = {
            "kivi": "kivi_structured_blob",
            "kvquant": "kvquant_structured_blob",
        }.get(self.quantizer.quantizer_id)
        artifact_compatible = (
            expected_format is not None
            and fmt == expected_format
            and int(metadata.get("token_count", 0)) == int(restore.num_tokens)
        )
        if not self.fused_attention_enabled or self._fused_upload_stream is None:
            return None
        if not artifact_compatible:
            if self.config.fused_attention == "required":
                raise RuntimeError(
                    "Origami fused attention requires a full structured "
                    f"{self.quantizer.quantizer_id} prefix artifact"
                )
            return None

        key = (restore.cache_key, layer_payload.layer_name, self.quantizer.quantizer_id)
        with self._fused_lock:
            existing = self._fused_payloads.get(key)
            if existing is not None:
                request_keys = self._request_fused_keys.setdefault(
                    restore.request_id, {}
                )
                if request_keys.get(layer_payload.layer_name) != key:
                    existing.ref_count += 1
                    request_keys[layer_payload.layer_name] = key
                return _LayerLoadResult(
                    event=existing.ready_event,
                    wait_on_current_stream=False,
                )

        kv_cache = self.kv_caches[layer_payload.layer_name]
        producer_event = None
        if symbols.is_cuda:
            producer_event = torch.cuda.Event()
            producer_event.record(torch.cuda.current_stream(symbols.device))
        with torch.cuda.stream(self._fused_upload_stream):
            if producer_event is not None:
                self._fused_upload_stream.wait_event(producer_event)
            raw = symbols.detach().reshape(-1)
            raw_gpu = raw.to(
                device=kv_cache.device,
                dtype=torch.uint8,
                non_blocking=True,
            ).contiguous()
            ready_event = torch.cuda.Event()
            ready_event.record(self._fused_upload_stream)
        payload = FusedLayerPayload(
            request_id=restore.request_id,
            cache_key=restore.cache_key,
            layer_name=layer_payload.layer_name,
            quantizer=self.quantizer.quantizer_id,
            symbols=raw_gpu,
            metadata=dict(metadata),
            ready_event=ready_event,
        )
        with self._fused_lock:
            raced = self._fused_payloads.get(key)
            if raced is None:
                self._fused_payloads[key] = payload
                selected = payload
            else:
                selected = raced
            request_keys = self._request_fused_keys.setdefault(
                restore.request_id, {}
            )
            if request_keys.get(layer_payload.layer_name) != key:
                if raced is not None:
                    raced.ref_count += 1
                request_keys[layer_payload.layer_name] = key
        return _LayerLoadResult(
            event=selected.ready_event,
            wait_on_current_stream=False,
        )

    def get_fused_layer_payloads(
        self,
        layer_name: str,
        request_ids: tuple[str, ...] | list[str],
    ) -> dict[str, FusedLayerPayload]:
        result: dict[str, FusedLayerPayload] = {}
        with self._fused_lock:
            for request_id in request_ids:
                key = self._request_fused_keys.get(request_id, {}).get(layer_name)
                if key is not None and key in self._fused_payloads:
                    result[request_id] = self._fused_payloads[key]
        return result

    def get_fused_attention_stream_state(self):
        if not self.fused_attention_enabled:
            return None
        if (
            self._fused_attention_stream is None
            or self._fused_qkv_ready_event is None
            or self._fused_done_event is None
        ):
            return None
        return (
            self._fused_attention_stream,
            self._fused_qkv_ready_event,
            self._fused_done_event,
        )

    def release_fused_requests(self, request_ids: set[str]) -> None:
        with self._fused_lock:
            for request_id in request_ids:
                keys = self._request_fused_keys.pop(request_id, {})
                for key in keys.values():
                    payload = self._fused_payloads.get(key)
                    if payload is None:
                        continue
                    payload.ref_count -= 1
                    if payload.ref_count <= 0:
                        self._fused_payloads.pop(key, None)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = dict(kv_caches)
        self.layer_to_cache_group = self._build_layer_to_cache_group()

    def _build_layer_to_cache_group(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        groups = getattr(self.kv_cache_config, "kv_cache_groups", None)
        if groups:
            for idx, group in enumerate(groups):
                for layer_name in getattr(group, "layer_names", ()):
                    mapping[str(layer_name)] = idx
        for layer_name in self.kv_caches:
            mapping.setdefault(layer_name, 0)
        return mapping

    def start_load_kv(self, metadata: OrigamiConnectorMetadata) -> None:
        self._metadata = metadata
        if not metadata.reqs_to_restore:
            return
        submit_start = time.perf_counter()
        restores = list(metadata.reqs_to_restore.values())
        write_jsonl(
            self.metrics_path,
            {
                "type": "restore_batch_meta",
                "system": "origami",
                "restore_count": len(restores),
                "request_ids": [restore.request_id for restore in restores],
                "metric_request_ids": [
                    _restore_metric_id(restore) for restore in restores
                ],
                "cache_keys": [restore.cache_key for restore in restores],
                "ts": time.time(),
            },
        )
        for restore in restores:
            layer_futures: dict[str, Future[Any]] = {}
            for layer_name in self.kv_caches:
                future: Future[Any] = Future()
                layer_futures[layer_name] = future
                self._pending_layer_futures.setdefault(layer_name, []).append(future)
            self._pending_request_layer_futures[restore.request_id] = list(
                layer_futures.values()
            )
            retained = self._claim_cached_layer_events(restore)
            restore_fn = (
                self._restore_retained_request_layer_futures
                if retained is not None
                else self._restore_request_layer_futures
            )
            restore_future = self._executor.submit(
                restore_fn,
                restore,
                layer_futures,
            )
            self._restore_futures.append(restore_future)
            # vLLM scheduler waits for internal request.request_id in
            # KVConnectorOutput.finished_recving. Keep metric_request_id only
            # for log/profile rows; returning it here leaves the request stuck
            # in WAITING_FOR_REMOTE_KVS when benchmark transfer IDs differ.
            self._restore_future_to_request_id[restore_future] = restore.request_id
        write_jsonl(
            self.metrics_path,
            {
                "type": "restore_batch_submitted",
                "system": "origami",
                "restore_count": len(restores),
                "ms": _ms_since(submit_start),
                "ts": time.time(),
            },
        )

    def _retained_layer_events(
        self, request_id: str
    ) -> dict[str, torch.cuda.Event] | None:
        events: dict[str, torch.cuda.Event] = {}
        with self._fused_lock:
            request_keys = self._request_fused_keys.get(request_id, {})
            for layer_name in self.kv_caches:
                key = request_keys.get(layer_name)
                payload = self._fused_payloads.get(key) if key is not None else None
                if payload is None:
                    return None
                events[layer_name] = payload.ready_event
        return events

    def _claim_cached_layer_events(
        self, restore: OrigamiRestoreRequest
    ) -> dict[str, torch.cuda.Event] | None:
        if not self.fused_attention_enabled:
            return None
        codec = self.quantizer.quantizer_id
        with self._fused_lock:
            cached: dict[
                str, tuple[tuple[str, str, str], FusedLayerPayload]
            ] = {}
            for layer_name in self.kv_caches:
                key = (restore.cache_key, layer_name, codec)
                payload = self._fused_payloads.get(key)
                if payload is None:
                    return None
                cached[layer_name] = (key, payload)

            request_keys = self._request_fused_keys.setdefault(
                restore.request_id, {}
            )
            for layer_name, (key, payload) in cached.items():
                if request_keys.get(layer_name) != key:
                    payload.ref_count += 1
                    request_keys[layer_name] = key
            return {
                layer_name: payload.ready_event
                for layer_name, (_, payload) in cached.items()
            }

    def _restore_retained_request_layer_futures(
        self,
        restore: OrigamiRestoreRequest,
        layer_futures: dict[str, Future[Any]],
    ) -> str:
        events = self._retained_layer_events(restore.request_id)
        if events is None:
            raise RuntimeError(
                f"retained Origami payload disappeared for {restore.request_id!r}"
            )
        for layer_name, future in layer_futures.items():
            future.set_result(_LayerLoadResult(
                event=events[layer_name],
                wait_on_current_stream=False,
            ))
        write_jsonl(
            self.metrics_path,
            {
                "type": "restore_retained_gpu_payload",
                "system": "origami",
                "request_id": _restore_metric_id(restore),
                "cache_key": restore.cache_key,
                "ts": time.time(),
            },
        )
        return _restore_metric_id(restore)

    def wait_for_layer_load(self, layer_name: str) -> None:
        futures = self._pending_layer_futures.pop(layer_name, [])
        for future in futures:
            result = future.result()
            if isinstance(result, _LayerLoadResult):
                event = result.event
            else:
                event = result
            wait_on_current_stream = (
                not isinstance(result, _LayerLoadResult)
                or result.wait_on_current_stream
            )
            if (
                event is not None
                and wait_on_current_stream
                and torch.cuda.is_available()
            ):
                torch.cuda.current_stream().wait_event(event)
            if isinstance(result, _LayerLoadResult):
                self._flush_pending_cuda_metrics(result.pending_metrics)

    def _flush_pending_cuda_metrics(
        self,
        pending_metrics: list[tuple[str, OrigamiRestoreRequest, LayerPayload,
                                    dict[str, Any]]] | None,
    ) -> None:
        if not pending_metrics:
            return
        for metric_type, restore, layer_payload, fields in pending_metrics:
            profile_fields = dict(fields)
            start_event = profile_fields.pop("_cuda_start_event", None)
            end_event = profile_fields.pop("_cuda_end_event", None)
            if start_event is not None and end_event is not None:
                elapsed_ms = None
                if end_event.query():
                    elapsed_ms = float(start_event.elapsed_time(end_event))
                    profile_fields["ms"] = elapsed_ms
                else:
                    profile_fields["cuda_timing_deferred"] = True
                input_bytes = int(profile_fields.get("input_bytes", 0) or 0)
                output_bytes = int(
                    profile_fields.get(
                        "output_bytes", profile_fields.get("bytes", 0)
                    ) or 0
                )
                bytes_value = int(profile_fields.get("bytes", output_bytes) or 0)
                if elapsed_ms is not None and elapsed_ms > 0:
                    if input_bytes:
                        profile_fields["input_gbps"] = (
                            input_bytes * 8.0 / elapsed_ms / 1e6
                        )
                    if output_bytes:
                        profile_fields["output_gbps"] = (
                            output_bytes * 8.0 / elapsed_ms / 1e6
                        )
                    if bytes_value:
                        profile_fields["gbps"] = (
                            bytes_value * 8.0 / elapsed_ms / 1e6
                        )
            self._write_restore_metric(
                metric_type,
                restore,
                layer_payload,
                profile_fields,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        del attn_metadata
        if self._metadata is None or not self._metadata.reqs_to_save:
            return
        for save in self._metadata.reqs_to_save.values():
            group_index = self.layer_to_cache_group.get(layer_name, 0)
            if group_index >= len(save.block_ids_per_group):
                continue
            block_ids = save.block_ids_per_group[group_index]
            if not block_ids:
                continue
            future = self._executor.submit(
                self._compress_layer,
                save,
                layer_name,
                kv_layer,
                block_ids,
            )
            self._save_futures.append(future)

    def wait_for_save(self) -> None:
        for future in self._save_futures:
            save, layer_payload = future.result()
            with self._save_lock:
                payload = self._save_payloads.get(save.cache_key)
                if payload is None:
                    payload = OrigamiPayload(
                        cache_key=save.cache_key,
                        request_id=save.request_id,
                        quantizer=self.quantizer.quantizer_id,
                        quantizer_config_hash=_config_hash(self.config.quantizer_config),
                        token_start=save.token_start,
                        token_count=save.num_tokens,
                    )
                    self._save_payloads[save.cache_key] = payload
                payload.layer_payloads[layer_payload.layer_name] = layer_payload
        for payload in self._save_payloads.values():
            self.store.put(payload)
        self._save_futures.clear()
        self._save_payloads.clear()

    def _restore_request(
        self,
        restore: OrigamiRestoreRequest,
    ) -> None:
        write_jsonl(
            self.metrics_path,
            {
                "type": "request_window_start",
                "system": "origami",
                "request_id": _restore_metric_id(restore),
                "cache_key": restore.cache_key,
                "ts": time.time(),
            },
        )
        artifact = self.store.prepare_for_restore(
            restore.cache_key,
            lossless_backend=self.cpu_codec.backend,
            dynamic_huffman=self.config.qat_dynamic_huffman,
            qat_max_instances=self.config.qat_max_instances,
            read_bandwidth_gbps=self.config.read_bandwidth_gbps,
            artifact_ingress_mode=self.config.artifact_ingress_mode,
            metrics_path=self.metrics_path,
            request_id=_restore_metric_id(restore),
            system="origami",
        )
        try:
            restored_chunks_by_layer = None
            if (
                isinstance(artifact, PreparedOrigamiPayload)
                and artifact.request_groups
            ):
                restored_chunks_by_layer = self._restore_request_groups(
                    restore, artifact
                )
            for layer_name in self.kv_caches:
                self._restore_layer_from_artifact(
                    restore,
                    artifact,
                    layer_name,
                    restored_chunks_by_layer=restored_chunks_by_layer,
                )
        finally:
            write_jsonl(
                self.metrics_path,
                {
                    "type": "restore_end",
                    "system": "origami",
                    "request_id": _restore_metric_id(restore),
                    "cache_key": restore.cache_key,
                    "ts": time.time(),
                },
            )

    def _restore_request_layer_futures(
        self,
        restore: OrigamiRestoreRequest,
        layer_futures: dict[str, Future[Any]],
    ) -> str:
        write_jsonl(
            self.metrics_path,
            {
                "type": "request_window_start",
                "system": "origami",
                "request_id": _restore_metric_id(restore),
                "cache_key": restore.cache_key,
                "ts": time.time(),
            },
        )
        try:
            artifact = self.store.prepare_for_restore(
                restore.cache_key,
                lossless_backend=self.cpu_codec.backend,
                dynamic_huffman=self.config.qat_dynamic_huffman,
                qat_max_instances=self.config.qat_max_instances,
                read_bandwidth_gbps=self.config.read_bandwidth_gbps,
                artifact_ingress_mode=self.config.artifact_ingress_mode,
                metrics_path=self.metrics_path,
                request_id=_restore_metric_id(restore),
                system="origami",
            )
            if (
                isinstance(artifact, PreparedOrigamiPayload)
                and artifact.request_groups
            ):
                self._restore_prepared_request_groups_windowed(
                    restore,
                    artifact,
                    layer_futures,
                )
            else:
                self._restore_request_layers_fallback(
                    restore,
                    artifact,
                    layer_futures,
                )
            return _restore_metric_id(restore)
        except Exception as exc:
            for future in layer_futures.values():
                if not future.done():
                    future.set_exception(exc)
            raise
        finally:
            write_jsonl(
                self.metrics_path,
                {
                    "type": "restore_end",
                    "system": "origami",
                    "request_id": _restore_metric_id(restore),
                    "cache_key": restore.cache_key,
                    "ts": time.time(),
                },
            )

    def _restore_request_layers_fallback(
        self,
        restore: OrigamiRestoreRequest,
        artifact: OrigamiPayload | PreparedOrigamiPayload,
        layer_futures: dict[str, Future[Any]],
    ) -> None:
        for layer_name, future in layer_futures.items():
            if future.done():
                continue
            try:
                event = self._restore_layer_from_artifact(
                    restore, artifact, layer_name
                )
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(event)

    def _restore_prepared_request_groups_windowed(
        self,
        restore: OrigamiRestoreRequest,
        artifact: PreparedOrigamiPayload,
        layer_futures: dict[str, Future[Any]],
    ) -> None:
        layer_states: dict[str, dict[str, Any]] = {}
        chunk_positions: dict[str, dict[int, int]] = {}
        for layer_name, future in layer_futures.items():
            layer_payload = artifact.payload.layer_payloads.get(layer_name)
            if layer_payload is None:
                future.set_result(None)
                continue
            chunk_positions[layer_name] = {
                int(chunk.chunk_id): index
                for index, chunk in enumerate(layer_payload.chunks)
            }
            layer_states[layer_name] = {
                "future": future,
                "payload": layer_payload,
                "chunks": [None] * len(layer_payload.chunks),
                "windows": [],
                "start_perf": time.perf_counter(),
            }

        target_chunks = max(1, self.config.effective_qat_pipeline_target_chunks())
        window_id = 0
        inflight_windows: list[Future[dict[str, Any]]] = []
        max_window_slots = max(1, int(self.config.qat_pipeline_slots))

        def consume_finished_windows(block: bool) -> None:
            if not inflight_windows:
                return
            if block:
                done, _ = wait(inflight_windows, return_when=FIRST_COMPLETED)
            else:
                done = {future for future in inflight_windows if future.done()}
            for future in list(done):
                inflight_windows.remove(future)
                self._consume_prepared_window(
                    future.result(),
                    restore,
                    artifact,
                    layer_states,
                    chunk_positions,
                )

        for group_index, group in enumerate(artifact.request_groups or []):
            refs = list(group.refs)
            start = 0
            while start < len(refs):
                consume_finished_windows(block=False)
                end = min(start + target_chunks, len(refs))
                window_refs = refs[start:end]
                indices = list(range(start, end))
                output_bytes = sum(int(ref.output_bytes) for ref in window_refs)
                compressed_bytes = self._estimate_window_compressed_bytes(
                    group.compressed_bytes,
                    group.unpacked_bytes,
                    output_bytes,
                )
                underfilled_reason = (
                    "request_tail"
                    if len(indices) < target_chunks
                    else ""
                )
                current = self._qat_executor.submit(
                    self._decompress_prepared_window,
                    restore,
                    group,
                    group_index,
                    window_id,
                    indices,
                    window_refs,
                    output_bytes,
                    compressed_bytes,
                    target_chunks,
                    underfilled_reason,
                )
                inflight_windows.append(current)
                if len(inflight_windows) >= max_window_slots:
                    consume_finished_windows(block=True)
                start = end
                window_id += 1
        while inflight_windows:
            consume_finished_windows(block=True)

        for layer_name, state in layer_states.items():
            future = state["future"]
            if future.done():
                continue
            if any(chunk is None for chunk in state["chunks"]):
                future.set_exception(
                    RuntimeError(f"Origami window restore missed chunks for {layer_name}")
                )
            else:
                self._restore_ready_layer_from_chunks(
                    restore,
                    artifact,
                    layer_name,
                    state,
                )

    @staticmethod
    def _estimate_window_compressed_bytes(
        group_compressed_bytes: int,
        group_unpacked_bytes: int,
        window_unpacked_bytes: int,
    ) -> int:
        group_compressed_bytes = int(group_compressed_bytes)
        group_unpacked_bytes = int(group_unpacked_bytes)
        window_unpacked_bytes = int(window_unpacked_bytes)
        if group_unpacked_bytes <= 0:
            return group_compressed_bytes
        return int(
            round(group_compressed_bytes * window_unpacked_bytes / group_unpacked_bytes)
        )

    def _decompress_prepared_window(
        self,
        restore: OrigamiRestoreRequest,
        group: Any,
        group_index: int,
        window_id: int,
        indices: list[int],
        refs: list[Any],
        output_bytes: int,
        compressed_bytes: int,
        target_chunks: int,
        underfilled_reason: str,
    ) -> dict[str, Any]:
        queue_wait_ms = 0.0
        start = time.perf_counter()
        with nvtx_range("origami:qat_decompress_window"):
            flat, qat_profile = (
                native_cpu.decompress_prepared_raw_deflate_window_with_profile(
                    group.prepared,
                    indices,
                    int(output_bytes),
                    inflight=self.config.qat_inflight,
                    batch=self.config.qat_batch,
                    max_instances=self.config.qat_max_instances,
                )
            )
        ms = _ms_since(start)
        return {
            "group_index": group_index,
            "window_id": window_id,
            "indices": indices,
            "refs": refs,
            "flat": flat.detach().cpu().to(torch.uint8).reshape(-1),
            "compressed_bytes": int(qat_profile.get("compressed_bytes", 0))
            or int(compressed_bytes),
            "unpacked_bytes": int(qat_profile.get("unpacked_bytes", 0))
            or int(output_bytes),
            "ms": ms,
            "queue_wait_ms": queue_wait_ms,
            "target_chunks": int(target_chunks),
            "underfilled_reason": underfilled_reason,
            "qat_profile": qat_profile,
            "layers_touched": sorted({str(ref.layer_name) for ref in refs}),
        }

    def _consume_prepared_window(
        self,
        window: dict[str, Any],
        restore: OrigamiRestoreRequest,
        artifact: PreparedOrigamiPayload,
        layer_states: dict[str, dict[str, Any]],
        chunk_positions: dict[str, dict[int, int]],
    ) -> None:
        refs = window["refs"]
        flat = window["flat"]
        cursor = 0
        for ref in refs:
            layer_name = str(ref.layer_name)
            state = layer_states.get(layer_name)
            if state is None:
                cursor += int(ref.output_bytes)
                continue
            position = chunk_positions[layer_name][int(ref.chunk_id)]
            state["chunks"][position] = flat.narrow(0, cursor, int(ref.output_bytes))
            state["windows"].append(int(window["window_id"]))
            cursor += int(ref.output_bytes)
        if cursor != int(flat.numel()):
            raise RuntimeError("Origami window QAT output split mismatch")

        touched_layers = window["layers_touched"]
        layer_payload = next(
            (
                artifact.payload.layer_payloads[name]
                for name in touched_layers
                if name in artifact.payload.layer_payloads
            ),
            next(iter(artifact.payload.layer_payloads.values())),
        )
        chunks = len(refs)
        workers = int(window["qat_profile"].get("workers", 0) or 0)
        slots_per_worker = (
            min(int(self.config.qat_inflight), max(1, (chunks + workers - 1) // workers))
            if workers > 0
            else 0
        )
        self._write_restore_metric(
            "lossless_decompress",
            restore,
            layer_payload,
            {
                "layer_name": "__window__",
                "scope": "chunk_window",
                "window_id": int(window["window_id"]),
                "group_index": int(window["group_index"]),
                "device": "cpu",
                "backend": self.cpu_codec.backend,
                "qat_codec": self.cpu_codec.qat_codec,
                "prepared": True,
                "chunks": chunks,
                "target_chunks": int(window["target_chunks"]),
                "qat_underfilled": bool(window["underfilled_reason"]),
                "qat_underfilled_reason": str(window["underfilled_reason"]),
                "layers_touched": touched_layers,
                "requests_touched": 1,
                "active_workers": workers,
                "chunks_per_worker": (chunks / workers if workers > 0 else 0.0),
                "slots_per_worker": slots_per_worker,
                "queue_wait_ms": float(window["queue_wait_ms"]),
                "compressed_bytes": int(window["compressed_bytes"]),
                "unpacked_bytes": int(window["unpacked_bytes"]),
                "ms": float(window["ms"]),
                "compressed_gbps": (
                    int(window["compressed_bytes"]) * 8.0 / float(window["ms"]) / 1e6
                    if float(window["ms"]) > 0
                    else 0.0
                ),
                "unpacked_gbps": (
                    int(window["unpacked_bytes"]) * 8.0 / float(window["ms"]) / 1e6
                    if float(window["ms"]) > 0
                    else 0.0
                ),
                **_qat_profile_metric_fields(window["qat_profile"]),
            },
        )

        for layer_name in touched_layers:
            state = layer_states.get(layer_name)
            if state is None or state["future"].done():
                continue
            if all(chunk is not None for chunk in state["chunks"]):
                self._restore_ready_layer_from_chunks(
                    restore,
                    artifact,
                    layer_name,
                    state,
                )

    def _restore_ready_layer_from_chunks(
        self,
        restore: OrigamiRestoreRequest,
        artifact: PreparedOrigamiPayload,
        layer_name: str,
        state: dict[str, Any],
    ) -> None:
        future: Future[Any] = state["future"]
        try:
            restored = {
                layer_name: [
                    chunk for chunk in state["chunks"] if chunk is not None
                ]
            }
            event = self._restore_layer_from_artifact(
                restore,
                artifact,
                layer_name,
                restored_chunks_by_layer=restored,
            )
            layer_ready_ms = _ms_since(float(state["start_perf"]))
            self._write_restore_metric(
                "layer_ready",
                restore,
                state["payload"],
                {
                    "windows": sorted(set(int(value) for value in state["windows"])),
                    "ms": layer_ready_ms,
                    "chunks": len(state["chunks"]),
                },
            )
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(event)

    def _restore_layer_from_artifact(
        self,
        restore: OrigamiRestoreRequest,
        artifact: OrigamiPayload | PreparedOrigamiPayload,
        layer_name: str,
        restored_chunks_by_layer: dict[str, torch.Tensor | list[torch.Tensor]] | None = None,
    ) -> torch.cuda.Event | None:
        if layer_name not in self.kv_caches:
            return None
        if restored_chunks_by_layer is not None:
            layer_payload = artifact.payload.layer_payloads.get(layer_name)
            if layer_payload is None:
                return None
            restored = restored_chunks_by_layer.get(layer_name)
            if restored is None:
                return None
            if isinstance(restored, torch.Tensor):
                symbols = self._raw_symbols_fast_path_profiled(
                    layer_payload,
                    restored,
                    restore,
                    backend="raw_bytes_view",
                    chunks=len(layer_payload.chunks),
                )
            else:
                symbols = self._unpack_profiled(layer_payload, restored, restore)
        elif isinstance(artifact, PreparedOrigamiPayload):
            prepared_layer = artifact.prepared_layers.get(layer_name)
            if prepared_layer is None:
                return None
            layer_payload = prepared_layer.layer_payload
            symbols = self._restore_symbols_prepared(restore, prepared_layer)
        else:
            layer_payload = artifact.layer_payloads.get(layer_name)
            if layer_payload is None:
                return None
            symbols = self._restore_symbols(
                layer_payload,
                restore.lossless_path,
                restore=restore,
            )

        kv_cache = self.kv_caches[layer_name]
        group_index = self.layer_to_cache_group.get(layer_name, 0)
        if group_index >= len(restore.block_ids_per_group):
            return None
        block_ids = list(restore.block_ids_per_group[group_index])
        if not block_ids:
            return None
        tokens_per_block = int(getattr(self.kv_cache_config, "block_size", 0) or 0)
        if tokens_per_block > 0 and int(restore.num_tokens) > 0:
            needed_blocks = (int(restore.num_tokens) + tokens_per_block - 1) // tokens_per_block
            block_ids = block_ids[:needed_blocks]

        fused_event = self._store_fused_layer_payload(
            restore,
            layer_payload,
            symbols,
        )
        if fused_event is not None:
            return fused_event

        direct_event = self._try_restore_direct_to_cache(
            restore,
            layer_payload,
            symbols,
            kv_cache,
            block_ids,
        )
        if direct_event is not None:
            return direct_event

        materialize_start = time.perf_counter()
        with nvtx_range("origami:restore_materialize"):
            kv_tensor = self.quantizer.dequantize(
                symbols, layer_payload.quant_metadata
            )
        original_tokens = int(layer_payload.quant_metadata.get("token_count", 0) or 0)
        kv_tensor = _slice_kv_tensor_to_tokens(
            kv_tensor,
            layer_payload.quant_metadata,
            restore.num_tokens,
        )
        materialize_ms = _ms_since(materialize_start)
        bytes_materialized = int(kv_tensor.numel() * kv_tensor.element_size())
        if original_tokens and int(restore.num_tokens) < original_tokens:
            write_jsonl(
                self.metrics_path,
                {
                    "type": "restore_token_slice",
                    "system": "origami",
                    "request_id": _restore_metric_id(restore),
                    "cache_key": restore.cache_key,
                    "layer_name": layer_name,
                    "artifact_tokens": original_tokens,
                    "restore_tokens": int(restore.num_tokens),
                    "bytes_after_slice": bytes_materialized,
                },
            )
        for profile_row in getattr(self.quantizer, "last_profile", []) or []:
            profile_fields = dict(profile_row)
            metric_type = str(profile_fields.pop("type"))
            self._write_restore_metric(
                metric_type,
                restore,
                layer_payload,
                profile_fields,
            )
        if str(layer_payload.quant_metadata.get("format")) == "cachegen_bitpacked_blob":
            self._write_restore_metric(
                "origami_gpu_materialize_total"
                if isinstance(kv_tensor, torch.Tensor)
                and kv_tensor.device.type == "cuda"
                else "origami_cpu_materialize_total",
                restore,
                layer_payload,
                {
                    "device": str(kv_tensor.device),
                    "bytes": bytes_materialized,
                    "ms": materialize_ms,
                    "gbps": (
                        bytes_materialized * 8.0 / materialize_ms / 1e6
                        if materialize_ms > 0
                        else 0.0
                    ),
                },
            )
        write_jsonl(
            self.metrics_path,
            {
                "type": "restore_materialize",
                "system": "origami",
                "request_id": _restore_metric_id(restore),
                "cache_key": restore.cache_key,
                "layer_name": layer_name,
                "quantizer": self.quantizer.quantizer_id,
                "device": str(kv_tensor.device),
                "bytes": bytes_materialized,
                "ms": materialize_ms,
                "gbps": (
                    bytes_materialized * 8.0 / materialize_ms / 1e6
                    if materialize_ms > 0
                    else 0.0
                ),
            },
        )
        is_partial_restore = bool(
            original_tokens and int(restore.num_tokens) < int(original_tokens)
        )
        _, event = copy_kv_to_hbm(
            kv_cache=kv_cache,
            block_ids=block_ids,
            value=kv_tensor,
            request_id=_restore_metric_id(restore),
            cache_key=restore.cache_key,
            system="origami",
            layer_name=layer_name,
            metrics_path=self.metrics_path,
            synchronize=is_partial_restore,
            return_event=True,
        )
        return event

    def _try_restore_direct_to_cache(
        self,
        restore: OrigamiRestoreRequest,
        layer_payload: LayerPayload,
        symbols: torch.Tensor,
        kv_cache: torch.Tensor,
        block_ids: list[int],
    ) -> _LayerLoadResult | torch.cuda.Event | None:
        metadata = layer_payload.quant_metadata
        original_tokens = int(metadata.get("token_count", 0) or 0)
        if original_tokens <= 0 or original_tokens != int(restore.num_tokens):
            return None
        can_direct = getattr(self.quantizer, "can_dequantize_to_cache", None)
        if can_direct is None or not can_direct(
            metadata,
            kv_cache,
            block_ids,
            tokens=int(restore.num_tokens),
        ):
            return None

        materialize_start = time.perf_counter()
        with nvtx_range("origami:restore_direct_to_cache"):
            event = self.quantizer.dequantize(
                symbols,
                metadata,
                dst_cache=kv_cache,
                block_ids=block_ids,
            )
        materialize_ms = _ms_since(materialize_start)
        bytes_materialized = self._restored_block_bytes(kv_cache, block_ids)
        pending_metrics: list[
            tuple[str, OrigamiRestoreRequest, LayerPayload, dict[str, Any]]
        ] = []
        fused_timing: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        for profile_row in getattr(self.quantizer, "last_profile", []) or []:
            profile_fields = dict(profile_row)
            metric_type = str(profile_fields.pop("type"))
            start_event = profile_fields.get("_cuda_start_event")
            end_event = profile_fields.get("_cuda_end_event")
            if (
                metric_type == "origami_cachegen_fused_restore_to_kv_cuda"
                and isinstance(start_event, torch.cuda.Event)
                and isinstance(end_event, torch.cuda.Event)
            ):
                fused_timing = (start_event, end_event)
            if start_event is not None and end_event is not None:
                pending_metrics.append((
                    metric_type,
                    restore,
                    layer_payload,
                    profile_fields,
                ))
            else:
                self._write_restore_metric(
                    metric_type,
                    restore,
                    layer_payload,
                    profile_fields,
                )
        restore_materialize_fields = {
            "quantizer": self.quantizer.quantizer_id,
            "device": str(kv_cache.device),
            "bytes": bytes_materialized,
            "enqueue_ms": materialize_ms,
            "direct_to_kv_cache": True,
            "h2d_elided": True,
        }
        if fused_timing is not None:
            restore_materialize_fields.update({
                "_cuda_start_event": fused_timing[0],
                "_cuda_end_event": fused_timing[1],
                "input_bytes": bytes_materialized,
                "output_bytes": bytes_materialized,
            })
            pending_metrics.append((
                "restore_materialize",
                restore,
                layer_payload,
                restore_materialize_fields,
            ))
        else:
            restore_materialize_fields["ms"] = materialize_ms
            restore_materialize_fields["gbps"] = (
                bytes_materialized * 8.0 / materialize_ms / 1e6
                if materialize_ms > 0
                else 0.0
            )
            self._write_restore_metric(
                "restore_materialize",
                restore,
                layer_payload,
                restore_materialize_fields,
            )
        if isinstance(event, torch.cuda.Event):
            return _LayerLoadResult(event=event, pending_metrics=pending_metrics)
        return None

    @staticmethod
    def _restored_block_bytes(
        kv_cache: torch.Tensor,
        block_ids: list[int],
    ) -> int:
        if not block_ids:
            return 0
        block_count = len(block_ids)
        if kv_cache.dim() == 5 and int(kv_cache.shape[1]) == 2:
            per_block = (
                int(kv_cache.shape[1])
                * int(kv_cache.shape[2])
                * int(kv_cache.shape[3])
                * int(kv_cache.shape[4])
            )
            return block_count * per_block * int(kv_cache.element_size())
        if kv_cache.dim() == 5 and int(kv_cache.shape[0]) == 2:
            per_block = (
                int(kv_cache.shape[0])
                * int(kv_cache.shape[2])
                * int(kv_cache.shape[3])
                * int(kv_cache.shape[4])
            )
            return block_count * per_block * int(kv_cache.element_size())
        return int(kv_cache.numel() * kv_cache.element_size())

    def _restore_request_groups(
        self,
        restore: OrigamiRestoreRequest,
        artifact: PreparedOrigamiPayload,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        restored_chunks: dict[str, list[torch.Tensor | None]] = {}
        chunk_positions: dict[str, dict[int, int]] = {}
        for layer_name, layer_payload in artifact.payload.layer_payloads.items():
            restored_chunks[layer_name] = [None] * len(layer_payload.chunks)
            chunk_positions[layer_name] = {
                int(chunk.chunk_id): index
                for index, chunk in enumerate(layer_payload.chunks)
            }

        total_compressed = 0
        total_unpacked = 0
        total_chunks = 0
        total_ms = 0.0
        qat_profile: dict[str, Any] = {}
        request_groups = artifact.request_groups or []
        request_flat: torch.Tensor | None = None
        request_refs = ()
        for group in request_groups:
            start = time.perf_counter()
            with nvtx_range("origami:qat_decompress"):
                restored = native_cpu.decompress_prepared_raw_deflate_many(
                    group.prepared,
                    inflight=self.config.qat_inflight,
                    batch=self.config.qat_batch,
                    max_instances=self.config.qat_max_instances,
                )
            decompress_ms = _ms_since(start)
            qat_profile = _merge_qat_profile(
                qat_profile,
                native_cpu.last_qat_profile(),
            )
            total_ms += decompress_ms
            total_compressed += int(group.compressed_bytes)
            total_unpacked += int(group.unpacked_bytes)
            total_chunks += len(group.refs)
            flat = restored[0].detach().cpu().to(torch.uint8).reshape(-1)
            if len(request_groups) == 1:
                request_flat = flat
                request_refs = group.refs
            cursor = 0
            for ref in group.refs:
                position = chunk_positions[ref.layer_name][int(ref.chunk_id)]
                restored_chunks[ref.layer_name][position] = flat.narrow(
                    0, cursor, int(ref.output_bytes)
                )
                cursor += int(ref.output_bytes)
            if cursor != int(flat.numel()):
                raise RuntimeError("Origami request-level QAT output split mismatch")

        self._write_restore_metric(
            "lossless_decompress",
            restore,
            next(iter(artifact.payload.layer_payloads.values())),
            {
                "layer_name": "__request__",
                "device": "cpu",
                "backend": self.cpu_codec.backend,
                "qat_codec": self.cpu_codec.qat_codec,
                "prepared": True,
                "scope": "request",
                "chunks": total_chunks,
                "compressed_bytes": total_compressed,
                "unpacked_bytes": total_unpacked,
                "ms": total_ms,
                "compressed_gbps": (
                    total_compressed * 8.0 / total_ms / 1e6
                    if total_ms > 0
                    else 0.0
                ),
                "unpacked_gbps": (
                    total_unpacked * 8.0 / total_ms / 1e6
                    if total_ms > 0
                    else 0.0
                ),
                **_qat_profile_metric_fields(qat_profile),
            },
        )

        raw_layer_views: dict[str, torch.Tensor] = {}
        if request_flat is not None:
            raw_layer_views = self._raw_layer_views_from_request_flat(
                artifact.payload,
                request_flat,
                request_refs,
            )

        finalized: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for layer_name, chunks in restored_chunks.items():
            if layer_name in raw_layer_views:
                finalized[layer_name] = raw_layer_views[layer_name]
                continue
            if any(chunk is None for chunk in chunks):
                raise RuntimeError(
                    f"Origami request-level restore missed chunks for {layer_name}"
                )
            finalized[layer_name] = [chunk for chunk in chunks if chunk is not None]
        return finalized

    def _restore_layer(
        self,
        restore: OrigamiRestoreRequest,
        layer_name: str,
    ) -> Any | None:
        artifact = self.store.get(restore.cache_key)
        return self._restore_layer_from_artifact(restore, artifact, layer_name)

    def _restore_symbols(
        self,
        layer_payload: LayerPayload,
        path: str,
        *,
        restore: OrigamiRestoreRequest | None = None,
    ) -> torch.Tensor:
        if self._layer_uses_nvcomp(layer_payload):
            return self._restore_symbols_nvcomp(layer_payload, restore)
        if self._layer_uses_raw(layer_payload):
            return self._restore_symbols_raw(layer_payload, restore)

        if path == "gpu":
            codec = self._get_gpu_codec()
            restored_chunks = []
            for chunk in layer_payload.chunks:
                compressed = chunk.compressed.to("cuda", non_blocking=True)
                restored_chunks.append(
                    codec.decompress(compressed, output_bytes=chunk.unpacked_bytes).cpu()
                )
            return self._unpack_native_symbols(layer_payload, restored_chunks)

        compressed_chunks = [chunk.compressed for chunk in layer_payload.chunks]
        output_bytes = [int(chunk.unpacked_bytes) for chunk in layer_payload.chunks]
        compressed_bytes = sum(
            int(chunk.compressed_bytes) for chunk in layer_payload.chunks
        )
        unpacked_bytes = sum(output_bytes)
        start = time.perf_counter()
        with nvtx_range("origami:qat_decompress"):
            restored_chunks = self.cpu_codec.decompress_many(
                compressed_chunks, output_bytes
            )
        decompress_ms = _ms_since(start)
        self._write_restore_metric(
            "lossless_decompress",
            restore,
            layer_payload,
            {
                "device": "cpu",
                "backend": self.cpu_codec.backend,
                "qat_codec": self.cpu_codec.qat_codec,
                "chunks": len(compressed_chunks),
                "compressed_bytes": compressed_bytes,
                "unpacked_bytes": unpacked_bytes,
                "ms": decompress_ms,
                "compressed_gbps": (
                    compressed_bytes * 8.0 / decompress_ms / 1e6
                    if decompress_ms > 0
                    else 0.0
                ),
                "unpacked_gbps": (
                    unpacked_bytes * 8.0 / decompress_ms / 1e6
                    if decompress_ms > 0
                    else 0.0
                ),
            },
        )
        return self._unpack_profiled(layer_payload, restored_chunks, restore)

    def _restore_symbols_prepared(
        self,
        restore: OrigamiRestoreRequest,
        prepared_layer: PreparedLayerPayload,
    ) -> torch.Tensor:
        layer_payload = prepared_layer.layer_payload
        chunk_positions = {
            int(chunk.chunk_id): index
            for index, chunk in enumerate(layer_payload.chunks)
        }
        restored_chunks: list[torch.Tensor | None] = [None] * len(layer_payload.chunks)
        total_compressed = 0
        total_unpacked = 0
        total_ms = 0.0
        qat_profile: dict[str, Any] = {}
        for group in prepared_layer.groups:
            start = time.perf_counter()
            with nvtx_range("origami:qat_decompress"):
                restored = native_cpu.decompress_prepared_raw_deflate_many(
                    group.prepared,
                    inflight=self.config.qat_inflight,
                    batch=self.config.qat_batch,
                    max_instances=self.config.qat_max_instances,
                )
            decompress_ms = _ms_since(start)
            qat_profile = _merge_qat_profile(
                qat_profile,
                native_cpu.last_qat_profile(),
            )
            total_ms += decompress_ms
            total_compressed += int(group.compressed_bytes)
            total_unpacked += int(group.unpacked_bytes)
            flat = restored[0].detach().cpu().to(torch.uint8).reshape(-1)
            cursor = 0
            for chunk_id in group.chunk_ids:
                position = chunk_positions[int(chunk_id)]
                restored_chunks[position] = flat.narrow(
                    0, cursor, int(group.output_bytes)
                )
                cursor += int(group.output_bytes)
        self._write_restore_metric(
            "lossless_decompress",
            restore,
            layer_payload,
            {
                "device": "cpu",
                "backend": self.cpu_codec.backend,
                "qat_codec": self.cpu_codec.qat_codec,
                "prepared": True,
                "chunks": len(layer_payload.chunks),
                "compressed_bytes": total_compressed,
                "unpacked_bytes": total_unpacked,
                "ms": total_ms,
                "compressed_gbps": (
                    total_compressed * 8.0 / total_ms / 1e6
                    if total_ms > 0
                    else 0.0
                ),
                "unpacked_gbps": (
                    total_unpacked * 8.0 / total_ms / 1e6
                    if total_ms > 0
                    else 0.0
                ),
                **_qat_profile_metric_fields(qat_profile),
            },
        )
        if any(chunk is None for chunk in restored_chunks):
            raise RuntimeError("Origami prepared restore missed one or more chunks")
        return self._unpack_profiled(
            layer_payload,
            [chunk for chunk in restored_chunks if chunk is not None],
            restore,
        )

    def _unpack_profiled(
        self,
        layer_payload: LayerPayload,
        restored_chunks: list[torch.Tensor],
        restore: OrigamiRestoreRequest | None,
    ) -> torch.Tensor:
        unpacked_bytes = sum(int(chunk.unpacked_bytes) for chunk in layer_payload.chunks)
        if self._should_use_gpu_bitunpack(layer_payload):
            start = time.perf_counter()
            with nvtx_range("origami:bitunpack_cuda"):
                symbols = self._unpack_native_symbols_cuda(layer_payload, restored_chunks)
            bitunpack_ms = _ms_since(start)
            self._write_restore_metric(
                "bitunpack_cuda",
                restore,
                layer_payload,
                {
                    "backend": "cuda",
                    "chunks": len(restored_chunks),
                    "unpacked_bytes": unpacked_bytes,
                    "symbols": int(symbols.numel()),
                    "ms": bitunpack_ms,
                    "unpacked_gbps": (
                        unpacked_bytes * 8.0 / bitunpack_ms / 1e6
                        if bitunpack_ms > 0
                        else 0.0
                    ),
                },
            )
            return symbols
        start = time.perf_counter()
        with nvtx_range("origami:bitunpack_avx512"):
            symbols = self._unpack_native_symbols(layer_payload, restored_chunks)
        bitunpack_ms = _ms_since(start)
        self._write_restore_metric(
            (
                "raw_bytes_view"
                if self._can_use_raw_bytes_fast_path(layer_payload)
                else "bitunpack_avx512"
            ),
            restore,
            layer_payload,
            {
                "backend": (
                    "raw_bytes_cat"
                    if self._can_use_raw_bytes_fast_path(layer_payload)
                    else self._bitunpack_backend()
                ),
                "chunks": len(restored_chunks),
                "unpacked_bytes": unpacked_bytes,
                "symbols": int(symbols.numel()),
                "ms": bitunpack_ms,
                "unpacked_gbps": (
                    unpacked_bytes * 8.0 / bitunpack_ms / 1e6
                    if bitunpack_ms > 0
                    else 0.0
                ),
            },
        )
        return symbols

    def _should_use_gpu_bitunpack(self, layer_payload: LayerPayload) -> bool:
        policy = str(self.config.bitunpack_device).lower()
        if policy == "cpu":
            return False
        if self._can_use_raw_bytes_fast_path(layer_payload):
            return False
        if not torch.cuda.is_available():
            if policy in {"gpu", "cuda"}:
                raise RuntimeError("Origami GPU bitunpack requested but CUDA is unavailable")
            return False
        if policy in {"gpu", "cuda"}:
            return True
        if policy == "auto":
            return native_gpu.available()
        return False

    def _raw_symbols_fast_path_profiled(
        self,
        layer_payload: LayerPayload,
        raw_symbols: torch.Tensor,
        restore: OrigamiRestoreRequest | None,
        *,
        backend: str,
        chunks: int,
    ) -> torch.Tensor:
        start = time.perf_counter()
        symbols = raw_symbols.detach().cpu().to(torch.uint8).reshape(-1)
        expected = self._raw_byte_symbol_count(layer_payload)
        if expected is None:
            raise RuntimeError("Origami raw-bytes fast path used for non raw-bytes payload")
        if int(symbols.numel()) != int(expected):
            raise RuntimeError(
                "Origami raw-bytes fast path size mismatch: "
                f"got {int(symbols.numel())}, expected {int(expected)}"
            )
        if not symbols.is_contiguous():
            symbols = symbols.contiguous()
        bitunpack_ms = _ms_since(start)
        fmt = str(layer_payload.quant_metadata.get("format"))
        metric_type = (
            "raw_bytes_view"
            if fmt == "cachegen_bitpacked_blob"
            else "bitunpack_avx512"
        )
        self._write_restore_metric(
            metric_type,
            restore,
            layer_payload,
            {
                "backend": backend,
                "fast_path": True,
                "chunks": chunks,
                "unpacked_bytes": int(expected),
                "symbols": int(symbols.numel()),
                "ms": bitunpack_ms,
                "unpacked_gbps": (
                    int(expected) * 8.0 / bitunpack_ms / 1e6
                    if bitunpack_ms > 0
                    else 0.0
                ),
            },
        )
        return symbols

    @staticmethod
    def _bitunpack_backend() -> str:
        try:
            return str(native_cpu.cpu_isa())
        except Exception as exc:
            return f"unknown:{type(exc).__name__}"

    def _write_restore_metric(
        self,
        metric_type: str,
        restore: OrigamiRestoreRequest | None,
        layer_payload: LayerPayload,
        fields: dict[str, Any],
    ) -> None:
        row = {
            "type": metric_type,
            "system": "origami",
            "request_id": _restore_metric_id(restore) if restore is not None else "",
            "cache_key": restore.cache_key if restore is not None else "",
            "layer_name": layer_payload.layer_name,
        }
        row.update(fields)
        write_jsonl(self.metrics_path, row)

    def _get_gpu_codec(self) -> GpuLosslessCodec:
        if self._gpu_codec is None:
            self._gpu_codec = GpuLosslessCodec(self.config.lossless_gpu_backend)
        return self._gpu_codec

    def _get_nvcomp_codec(self) -> NvcompCodec:
        if self.config.nvcomp_backend == "disabled":
            raise RuntimeError("Origami nvCOMP backend is disabled")
        if self._nvcomp_codec is None:
            self._nvcomp_codec = NvcompCodec(self.config.nvcomp_backend)
        return self._nvcomp_codec

    @staticmethod
    def _layer_uses_nvcomp(layer_payload: LayerPayload) -> bool:
        return any(
            str(chunk.codec).lower().startswith("nvcomp")
            for chunk in layer_payload.chunks
        )

    @staticmethod
    def _layer_uses_raw(layer_payload: LayerPayload) -> bool:
        return bool(layer_payload.chunks) and all(
            str(chunk.codec).lower() in {"raw", "identity"}
            for chunk in layer_payload.chunks
        )

    def _restore_symbols_raw(
        self,
        layer_payload: LayerPayload,
        restore: OrigamiRestoreRequest | None,
    ) -> torch.Tensor:
        raw_chunks: list[torch.Tensor] = []
        for chunk in layer_payload.chunks:
            raw = chunk.compressed.detach().cpu().to(torch.uint8).reshape(-1)
            if int(raw.numel()) != int(chunk.unpacked_bytes):
                raise RuntimeError(
                    "Origami raw chunk size mismatch: "
                    f"chunk={chunk.chunk_id} got {int(raw.numel())}, "
                    f"expected {int(chunk.unpacked_bytes)}"
                )
            if int(chunk.compressed_bytes) != int(chunk.unpacked_bytes):
                raise RuntimeError(
                    "Origami raw chunk metadata mismatch: "
                    f"chunk={chunk.chunk_id} compressed_bytes="
                    f"{int(chunk.compressed_bytes)} unpacked_bytes="
                    f"{int(chunk.unpacked_bytes)}"
                )
            raw_chunks.append(raw.contiguous())
        return self._unpack_profiled(layer_payload, raw_chunks, restore)

    def _compress_nvcomp_many(
        self,
        raw_chunks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not torch.cuda.is_available():
            raise RuntimeError("Origami nvCOMP backend requires CUDA")
        codec = self._get_nvcomp_codec()
        compressed: list[torch.Tensor] = []
        for raw in raw_chunks:
            data = raw.detach().to(device="cuda", dtype=torch.uint8, non_blocking=True)
            encoded = codec.compress(data.reshape(-1).contiguous())
            compressed.append(encoded.detach().cpu().to(torch.uint8).reshape(-1).contiguous())
        return compressed

    def _restore_symbols_nvcomp(
        self,
        layer_payload: LayerPayload,
        restore: OrigamiRestoreRequest | None,
    ) -> torch.Tensor:
        codec = self._get_nvcomp_codec()
        restored_chunks = []
        total_compressed = 0
        total_unpacked = 0
        enqueue_start = time.perf_counter()
        with nvtx_range("origami:kivi_nvcomp_decompress_cuda"):
            for chunk in layer_payload.chunks:
                compressed = chunk.compressed.to(
                    "cuda", dtype=torch.uint8, non_blocking=True
                )
                total_compressed += int(chunk.compressed_bytes)
                total_unpacked += int(chunk.unpacked_bytes)
                restored_chunks.append(
                    codec.decompress(
                        compressed.reshape(-1).contiguous(),
                        output_bytes=int(chunk.unpacked_bytes),
                    )
                )
        symbols = (
            restored_chunks[0].reshape(-1).contiguous()
            if len(restored_chunks) == 1
            else torch.cat([chunk.reshape(-1) for chunk in restored_chunks],
                           dim=0).contiguous()
        )
        enqueue_ms = _ms_since(enqueue_start)
        self._write_restore_metric(
            "kivi_nvcomp_decompress_cuda",
            restore,
            layer_payload,
            {
                "device": str(symbols.device),
                "backend": self.config.nvcomp_backend,
                "chunks": len(layer_payload.chunks),
                "compressed_bytes": total_compressed,
                "unpacked_bytes": total_unpacked,
                "input_bytes": total_compressed,
                "output_bytes": total_unpacked,
                "bytes": total_unpacked,
                "enqueue_ms": enqueue_ms,
                "cuda_timing_deferred": True,
            },
        )
        return symbols

    @staticmethod
    def _block_dim(kv_layer: torch.Tensor) -> int:
        if kv_layer.dim() >= 2 and int(kv_layer.shape[0]) in (1, 2):
            return 1
        return 0

    def _select_blocks(
        self,
        kv_layer: torch.Tensor,
        block_ids: tuple[int, ...],
    ) -> torch.Tensor:
        block_ids_tensor = torch.tensor(
            list(block_ids), dtype=torch.long, device=kv_layer.device
        )
        return torch.index_select(
            kv_layer, self._block_dim(kv_layer), block_ids_tensor
        ).detach()

    def _compress_layer(
        self,
        save: OrigamiSaveRequest,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_ids: tuple[int, ...],
    ) -> tuple[OrigamiSaveRequest, LayerPayload]:
        kv_blocks = self._select_blocks(kv_layer, block_ids)
        quantized = self.quantizer.quantize(kv_blocks, layer_group=layer_name)
        raw_chunks, planned_chunks, native_metadata = self._pack_quantized_symbols(
            quantized.symbols,
            quantized.metadata,
            layer_name,
        )
        use_nvcomp = (
            self.config.nvcomp_backend != "disabled"
            and self.quantizer.quantizer_id == "kivi"
        )
        compressed_chunks = (
            self._compress_nvcomp_many(raw_chunks)
            if use_nvcomp
            else self.cpu_codec.compress_many(raw_chunks)
        )
        codec_name = (
            self.config.nvcomp_backend if use_nvcomp else self.config.lossless_cpu_backend
        )
        chunks: list[ChunkRecord] = []
        for idx, (planned, raw, compressed) in enumerate(
            zip(planned_chunks, raw_chunks, compressed_chunks)
        ):
            plan_layout = planned.layout
            layout = ChunkLayout(
                layer_index=plan_layout.layer_index,
                head_start=plan_layout.head_start,
                head_end=plan_layout.head_end,
                channel_start=plan_layout.channel_start,
                channel_end=plan_layout.channel_end,
                token_start=save.token_start,
                token_end=save.token_start + save.num_tokens,
                unpacked_bytes=int(raw.numel()),
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=idx,
                    layout=layout,
                    codec=codec_name,
                    compressed=compressed,
                    compressed_bytes=int(compressed.numel()),
                    unpacked_bytes=int(raw.numel()),
                )
            )
        quant_metadata = dict(quantized.metadata)
        quant_metadata[_NATIVE_LAYOUT_KEY] = native_metadata
        layer_payload = LayerPayload(
            layer_name=layer_name,
            quantizer=self.quantizer.quantizer_id,
            quant_metadata=quant_metadata,
            chunks=chunks,
        )
        return save, layer_payload

    def _pack_quantized_symbols(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
        layer_name: str,
    ) -> tuple[list[torch.Tensor], list[PlannedChunk], dict[str, Any]]:
        flat = symbols.reshape(-1).to(torch.uint8).cpu().contiguous()
        if str(metadata.get("format")) == "cachegen_structured_blob":
            return self._pack_cachegen_structured_symbols(flat, metadata, layer_name)
        (
            bits,
            source_shape,
            source_layout,
            token_count,
            num_heads,
            head_dim,
        ) = self._parse_symbol_layout(flat, metadata)
        if int(flat.numel()) == 0:
            return [], [], {
                "version": 2,
                "bits": bits,
                "source_shape": source_shape,
                "source_layout": source_layout,
                "storage_layout": native_cpu.STORAGE_LAYOUT,
                "token_count": token_count,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "symbol_count": 0,
                "layout_policy": self.config.layout_policy,
                "bitpack": bool(self.config.bitpack),
                "chunk_specs": [],
            }
        planned_chunks = plan_head_channel_chunks(
            layer_index=_layer_index(layer_name),
            num_heads=num_heads,
            head_dim=head_dim,
            token_count=token_count,
            bytes_per_symbol=1,
            min_bytes=self.config.chunk_min_bytes,
            target_bytes=self.config.chunk_target_bytes,
            max_bytes=self.config.chunk_max_bytes,
        )
        if not planned_chunks and int(flat.numel()) > 0:
            planned_chunks = [
                PlannedChunk(
                    chunk_id=0,
                    layout=ChunkLayout(
                        layer_index=_layer_index(layer_name),
                        head_start=0,
                        head_end=1,
                        channel_start=0,
                        channel_end=int(flat.numel()),
                        token_start=0,
                        token_end=1,
                        unpacked_bytes=int(flat.numel()),
                    ),
                )
            ]
        specs = [self._layout_to_spec(chunk.layout) for chunk in planned_chunks]
        raw_chunks = native_cpu.pack_canonical_storage_chunks(
            flat,
            bits,
            source_shape,
            source_layout,
            specs,
        )
        native_metadata = {
            "version": 2,
            "bits": bits,
            "source_shape": source_shape,
            "source_layout": source_layout,
            "storage_layout": native_cpu.STORAGE_LAYOUT,
            "token_count": token_count,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "symbol_count": int(flat.numel()),
            "layout_policy": self.config.layout_policy,
            "bitpack": bool(self.config.bitpack),
            "chunk_specs": specs,
        }
        return raw_chunks, planned_chunks, native_metadata

    def _pack_cachegen_structured_symbols(
        self,
        flat: torch.Tensor,
        metadata: dict[str, Any],
        layer_name: str,
    ) -> tuple[list[torch.Tensor], list[PlannedChunk], dict[str, Any]]:
        layer_index = _layer_index(layer_name)
        token_count = int(metadata["token_count"])
        num_heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        raw_chunks: list[torch.Tensor] = []
        planned_chunks: list[PlannedChunk] = []
        components: list[dict[str, Any]] = []

        def append_quant_component(
            *,
            name: str,
            offset_key: str,
            numel_key: str,
            bits_key: str,
        ) -> None:
            offset = int(metadata[offset_key])
            numel = int(metadata[numel_key])
            bits = int(metadata[bits_key])
            source_shape = [token_count, num_heads, head_dim]
            source_layout = ["token", "head", "head_dim"]
            if numel != token_count * num_heads * head_dim:
                raise ValueError(
                    f"CacheGen {name} symbol count mismatch: "
                    f"{numel} vs {token_count * num_heads * head_dim}"
                )
            local_plans = plan_head_channel_chunks(
                layer_index=layer_index,
                num_heads=num_heads,
                head_dim=head_dim,
                token_count=token_count,
                bytes_per_symbol=1,
                min_bytes=self.config.chunk_min_bytes,
                target_bytes=self.config.chunk_target_bytes,
                max_bytes=self.config.chunk_max_bytes,
            )
            specs = [self._layout_to_spec(chunk.layout) for chunk in local_plans]
            start = len(raw_chunks)
            packed = native_cpu.pack_canonical_storage_chunks(
                flat.narrow(0, offset, numel),
                bits,
                source_shape,
                source_layout,
                specs,
            )
            raw_chunks.extend(packed)
            planned_chunks.extend(local_plans)
            components.append(
                {
                    "name": name,
                    "kind": "quant",
                    "offset": offset,
                    "numel": numel,
                    "bits": bits,
                    "source_shape": source_shape,
                    "source_layout": source_layout,
                    "chunk_start": start,
                    "chunk_count": len(packed),
                    "chunk_specs": specs,
                }
            )

        def append_raw_component(*, name: str, offset_key: str, numel_key: str) -> None:
            offset = int(metadata[offset_key])
            byte_count = int(metadata[numel_key]) * 2
            raw = flat.narrow(0, offset, byte_count).contiguous()
            start = len(raw_chunks)
            raw_chunks.append(raw)
            planned_chunks.append(
                PlannedChunk(
                    chunk_id=len(planned_chunks),
                    layout=ChunkLayout(
                        layer_index=layer_index,
                        head_start=0,
                        head_end=1,
                        channel_start=0,
                        channel_end=max(1, byte_count),
                        token_start=0,
                        token_end=1,
                        unpacked_bytes=byte_count,
                    ),
                )
            )
            components.append(
                {
                    "name": name,
                    "kind": "raw",
                    "offset": offset,
                    "byte_count": byte_count,
                    "chunk_start": start,
                    "chunk_count": 1,
                }
            )

        append_quant_component(
            name="key_q",
            offset_key="key_q_offset",
            numel_key="key_q_numel",
            bits_key="key_bits",
        )
        append_quant_component(
            name="value_q",
            offset_key="value_q_offset",
            numel_key="value_q_numel",
            bits_key="value_bits",
        )
        append_raw_component(
            name="key_max",
            offset_key="key_max_offset",
            numel_key="key_max_numel",
        )
        append_raw_component(
            name="value_max",
            offset_key="value_max_offset",
            numel_key="value_max_numel",
        )
        native_metadata = {
            "version": 3,
            "format": "cachegen_structured_blob",
            "storage_layout": native_cpu.STORAGE_LAYOUT,
            "token_count": token_count,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "symbol_count": int(flat.numel()),
            "layout_policy": "layer>head>head_dim>token",
            "bitpack": bool(self.config.bitpack),
            "components": components,
        }
        return raw_chunks, planned_chunks, native_metadata

    def _parse_symbol_layout(
        self,
        symbols: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[int, list[int], list[str], int, int, int]:
        if "origami_symbol_shape" not in metadata:
            raise ValueError("Origami quantizer metadata must include origami_symbol_shape")
        if "origami_symbol_layout" not in metadata:
            raise ValueError("Origami quantizer metadata must include origami_symbol_layout")

        requested_bits = int(
            metadata.get("origami_bits", metadata.get("bits", metadata.get("quant_bits", 8)))
        )
        if requested_bits < 1 or requested_bits > 8:
            raise ValueError("origami_bits must be in the range 1..8")
        bits = requested_bits if self.config.bitpack else 8

        token_count, num_heads, head_dim, source_shape, source_layout = (
            native_cpu.canonical_axis_sizes(
                metadata["origami_symbol_shape"],
                metadata["origami_symbol_layout"],
            )
        )
        symbol_count = int(symbols.numel())
        if _product(source_shape) != symbol_count:
            raise ValueError(
                "Origami quantizer symbol count must equal origami_symbol_shape product"
            )
        return bits, source_shape, source_layout, token_count, num_heads, head_dim

    @staticmethod
    def _layout_to_spec(layout: ChunkLayout) -> list[int]:
        return [
            int(layout.head_start),
            int(layout.head_end),
            int(layout.channel_start),
            int(layout.channel_end),
            int(layout.token_start),
            int(layout.token_end),
        ]

    @staticmethod
    def _unpack_cachegen_structured_symbols(
        native_metadata: dict[str, Any],
        raw_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        symbol_count = int(native_metadata.get("symbol_count", 0))
        if symbol_count <= 0:
            return torch.empty((0,), dtype=torch.uint8)
        symbols = torch.empty((symbol_count,), dtype=torch.uint8)
        for component in native_metadata.get("components", []):
            start = int(component["chunk_start"])
            count = int(component["chunk_count"])
            chunks = raw_chunks[start : start + count]
            if len(chunks) != count:
                raise RuntimeError(
                    "CacheGen structured native metadata chunk count does not "
                    "match payload chunks"
                )
            offset = int(component["offset"])
            kind = str(component["kind"])
            if kind == "quant":
                unpacked = native_cpu.unpack_canonical_storage_chunks(
                    chunks,
                    int(component["bits"]),
                    component["source_shape"],
                    component["source_layout"],
                    component.get("chunk_specs", []),
                ).reshape(-1)
                numel = int(component["numel"])
                if int(unpacked.numel()) < numel:
                    raise RuntimeError("CacheGen structured quant component is short")
                symbols.narrow(0, offset, numel).copy_(unpacked[:numel])
            elif kind == "raw":
                if not chunks:
                    data = torch.empty((0,), dtype=torch.uint8)
                elif len(chunks) == 1:
                    data = chunks[0].detach().cpu().to(torch.uint8).reshape(-1)
                else:
                    data = torch.cat(
                        [
                            chunk.detach().cpu().to(torch.uint8).reshape(-1)
                            for chunk in chunks
                        ]
                    )
                byte_count = int(component["byte_count"])
                if int(data.numel()) < byte_count:
                    raise RuntimeError("CacheGen structured raw component is short")
                symbols.narrow(0, offset, byte_count).copy_(data[:byte_count])
            else:
                raise RuntimeError(f"unknown CacheGen structured component kind {kind!r}")
        return symbols.contiguous()

    def _unpack_native_symbols(
        self,
        layer_payload: LayerPayload,
        raw_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        if self._can_use_raw_bytes_fast_path(layer_payload):
            symbol_count = self._raw_byte_symbol_count(layer_payload)
            if symbol_count is None:
                raise RuntimeError("Origami raw-bytes metadata is missing symbol count")
            if not raw_chunks:
                return torch.empty((0,), dtype=torch.uint8)
            if len(raw_chunks) == 1:
                symbols = raw_chunks[0].detach().cpu().to(torch.uint8).reshape(-1)
                if int(symbols.numel()) < int(symbol_count):
                    raise RuntimeError("Origami raw-bytes chunk is shorter than metadata")
                return symbols[: int(symbol_count)].contiguous()
            symbols = torch.cat(
                [
                    chunk.detach().cpu().to(torch.uint8).reshape(-1)
                    for chunk in raw_chunks
                ]
            )
            if int(symbols.numel()) < int(symbol_count):
                raise RuntimeError("Origami raw-bytes chunks are shorter than metadata")
            return symbols[: int(symbol_count)].contiguous()

        native_metadata = layer_payload.quant_metadata.get(_NATIVE_LAYOUT_KEY)
        if not native_metadata:
            return torch.cat([chunk.detach().cpu().to(torch.uint8).reshape(-1) for chunk in raw_chunks])
        if str(native_metadata.get("format")) == "cachegen_structured_blob":
            return self._unpack_cachegen_structured_symbols(
                native_metadata, raw_chunks
            )
        specs = native_metadata.get("chunk_specs", [])
        if len(specs) != len(raw_chunks):
            raise RuntimeError(
                "Origami native layout metadata chunk count does not match payload chunks"
            )
        if not specs:
            return torch.empty((0,), dtype=torch.uint8)
        symbols = native_cpu.unpack_canonical_storage_chunks(
            raw_chunks,
            int(native_metadata["bits"]),
            native_metadata["source_shape"],
            native_metadata["source_layout"],
            specs,
        )
        symbol_count = int(native_metadata.get("symbol_count", symbols.numel()))
        return symbols.reshape(-1)[:symbol_count].contiguous()

    def _unpack_native_symbols_cuda(
        self,
        layer_payload: LayerPayload,
        raw_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        native_metadata = layer_payload.quant_metadata.get(_NATIVE_LAYOUT_KEY)
        if not native_metadata:
            return torch.cat(
                [
                    chunk.detach().to("cuda", dtype=torch.uint8,
                                      non_blocking=True).reshape(-1)
                    for chunk in raw_chunks
                ]
            ).contiguous()
        if str(native_metadata.get("format")) == "cachegen_structured_blob":
            return self._unpack_cachegen_structured_symbols(
                native_metadata, raw_chunks
            )
        specs = native_metadata.get("chunk_specs", [])
        if len(specs) != len(raw_chunks):
            raise RuntimeError(
                "Origami native layout metadata chunk count does not match payload chunks"
            )
        if not specs:
            return torch.empty((0,), dtype=torch.uint8, device="cuda")
        bits = int(native_metadata["bits"])
        symbols = native_gpu.unpack_canonical_storage_chunks(
            raw_chunks,
            bits,
            native_metadata["source_shape"],
            native_metadata["source_layout"],
            specs,
            device="cuda",
        )
        symbol_count = int(native_metadata.get("symbol_count", symbols.numel()))
        return symbols.reshape(-1)[:symbol_count].contiguous()

    def _raw_layer_views_from_request_flat(
        self,
        payload: OrigamiPayload,
        flat: torch.Tensor,
        refs: tuple[Any, ...],
    ) -> dict[str, torch.Tensor]:
        ranges: dict[str, list[int]] = {}
        invalid_layers: set[str] = set()
        last_layer: str | None = None
        cursor = 0
        for ref in refs:
            layer_name = str(ref.layer_name)
            output_bytes = int(ref.output_bytes)
            if layer_name not in ranges:
                ranges[layer_name] = [cursor, cursor + output_bytes]
            else:
                if last_layer != layer_name:
                    invalid_layers.add(layer_name)
                ranges[layer_name][1] = cursor + output_bytes
            cursor += output_bytes
            last_layer = layer_name
        if cursor != int(flat.numel()):
            return {}

        views: dict[str, torch.Tensor] = {}
        for layer_name, layer_payload in payload.layer_payloads.items():
            if layer_name in invalid_layers:
                continue
            if layer_name not in ranges:
                continue
            if not self._can_use_raw_bytes_fast_path(layer_payload):
                continue
            expected = self._raw_byte_symbol_count(layer_payload)
            if expected is None:
                continue
            start, end = ranges[layer_name]
            if end - start != int(expected):
                continue
            views[layer_name] = flat.narrow(0, start, int(expected))
        return views

    @staticmethod
    def _raw_byte_symbol_count(layer_payload: LayerPayload) -> int | None:
        metadata = layer_payload.quant_metadata
        native_metadata = metadata.get(_NATIVE_LAYOUT_KEY) or {}
        fmt = str(metadata.get("format", layer_payload.quantizer))
        bits = int(native_metadata.get("bits", metadata.get("origami_bits", 8)))
        if bits != 8:
            return None
        if fmt == "raw_bytes":
            value = native_metadata.get("symbol_count", metadata.get("byte_count"))
            if value is None:
                return None
            return int(value)
        if fmt == "kivi_structured_blob":
            value = native_metadata.get("symbol_count", metadata.get("symbol_byte_count"))
            if value is None:
                return None
            value = int(value)
            source_shape = [int(dim) for dim in native_metadata.get("source_shape", [])]
            source_layout = [str(axis) for axis in native_metadata.get("source_layout", [])]
            if source_shape != [1, 1, value]:
                return None
            if source_layout != ["token", "head", "head_dim"]:
                return None
            return value
        if fmt not in {"cachegen_quantized_blob", "cachegen_bitpacked_blob"}:
            return None
        value = native_metadata.get("symbol_count", metadata.get("symbol_byte_count"))
        if value is None:
            return None
        value = int(value)
        source_shape = [int(dim) for dim in native_metadata.get("source_shape", [])]
        source_layout = [str(axis) for axis in native_metadata.get("source_layout", [])]
        if source_shape != [1, 1, value]:
            return None
        if source_layout != ["token", "head", "head_dim"]:
            return None
        return value

    @classmethod
    def _can_use_raw_bytes_fast_path(cls, layer_payload: LayerPayload) -> bool:
        return cls._raw_byte_symbol_count(layer_payload) is not None

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Return request ids that have completed async restore send/load operations.

        Returns:
            tuple[sending/saving_ids, recving/loading_ids]
        """
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        for restore_future in list(self._restore_futures):
            if not restore_future.done():
                continue
            req_id = self._restore_future_to_request_id.pop(restore_future, "")
            if not req_id:
                # If we ever see a future without request mapping (stale/legacy),
                # remove it and continue.
                self._restore_futures.remove(restore_future)
                continue

            try:
                restore_future.result()
            except Exception:
                self._pending_request_layer_futures.pop(req_id, None)
                for layer_name, futures in self._pending_layer_futures.items():
                    self._pending_layer_futures[layer_name] = [
                        future for future in futures if future.done()
                    ]
                self._restore_futures.remove(restore_future)
                raise

            layer_futures = self._pending_request_layer_futures.get(req_id, [])
            ready = layer_futures and all(future.done() for future in layer_futures)
            if ready:
                try:
                    for future in layer_futures:
                        future.result()
                except Exception:
                    self._pending_request_layer_futures.pop(req_id, None)
                    raise
            if ready:
                finished_recving.add(req_id)
                self._pending_request_layer_futures.pop(req_id, None)
            self._restore_futures.remove(restore_future)

        # Request-level restore operations are the only async path in Origami.
        # Keep API compatibility by returning empty sending set.
        for layer_futures in self._pending_request_layer_futures.values():
            # Best effort cleanup for entries already completed but not yet popped
            # due to race between this call and wait_for_layer_load().
            if all(future.done() for future in layer_futures):
                # do not mark as finished here; wait_for_layer_load may still be
                # waiting for layer-specific completion in scheduler.
                continue

        return finished_sending, finished_recving

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
