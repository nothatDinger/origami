# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.benchmark_utils import (
    controlled_read,
    preload_memory_artifacts,
    nvtx_range,
    write_jsonl,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
    native_cpu,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    ChunkLayout,
    ChunkRecord,
    LayerPayload,
    OrigamiPayload,
)


def _safe_key(cache_key: str) -> str:
    return str(cache_key).replace("/", "_").replace(":", "_")


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _cpu_u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.uint8).reshape(-1).contiguous()


def _codec_uses_qat_bundle(codec: str) -> bool:
    normalized = str(codec).lower()
    return normalized not in {"raw", "identity"} and not normalized.startswith("nvcomp")


@dataclass(frozen=True)
class PreparedChunkGroup:
    layer_name: str
    output_bytes: int
    chunk_ids: tuple[int, ...]
    compressed_bytes: int
    unpacked_bytes: int
    prepared: native_cpu.QatPreparedPayload


@dataclass(frozen=True)
class PreparedChunkRef:
    layer_name: str
    chunk_id: int
    output_bytes: int


@dataclass(frozen=True)
class PreparedRequestGroup:
    output_bytes: int
    refs: tuple[PreparedChunkRef, ...]
    compressed_bytes: int
    unpacked_bytes: int
    prepared: native_cpu.QatPreparedPayload


@dataclass
class PreparedLayerPayload:
    layer_payload: LayerPayload
    groups: list[PreparedChunkGroup]


@dataclass
class PreparedOrigamiPayload:
    payload: OrigamiPayload
    prepared_layers: dict[str, PreparedLayerPayload]
    request_groups: list[PreparedRequestGroup] | None = None
    artifact_format: str = "bundle_v1"


class OrigamiStore(ABC):

    @abstractmethod
    def put(self, payload: OrigamiPayload) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, cache_key: str) -> OrigamiPayload:
        raise NotImplementedError

    def prepare_for_restore(
        self,
        cache_key: str,
        *,
        lossless_backend: str,
        dynamic_huffman: bool,
        qat_max_instances: int,
        read_bandwidth_gbps: float,
        artifact_ingress_mode: str,
        metrics_path: str | Path | None,
        request_id: str,
        system: str,
    ) -> OrigamiPayload | PreparedOrigamiPayload:
        del (
            lossless_backend,
            dynamic_huffman,
            qat_max_instances,
            read_bandwidth_gbps,
            artifact_ingress_mode,
            metrics_path,
            request_id,
            system,
        )
        return self.get(cache_key)

    @abstractmethod
    def contains(self, cache_key: str) -> bool:
        raise NotImplementedError

    def preload_for_restore(
        self,
        cache_keys: list[str] | tuple[str, ...],
        *,
        lossless_backend: str,
        metrics_path: str | Path | None,
        system: str,
    ) -> None:
        del cache_keys, lossless_backend, metrics_path, system
        return


_MEMORY_STORES: dict[str, dict[str, OrigamiPayload]] = {}
_MEMORY_LOCK = Lock()


class InMemoryOrigamiStore(OrigamiStore):

    def __init__(self, namespace: str = "origami"):
        self.namespace = namespace
        with _MEMORY_LOCK:
            _MEMORY_STORES.setdefault(namespace, {})

    def put(self, payload: OrigamiPayload) -> None:
        with _MEMORY_LOCK:
            _MEMORY_STORES[self.namespace][payload.cache_key] = payload

    def get(self, cache_key: str) -> OrigamiPayload:
        with _MEMORY_LOCK:
            try:
                return _MEMORY_STORES[self.namespace][cache_key]
            except KeyError as exc:
                raise KeyError(f"Origami payload not found: {cache_key}") from exc

    def contains(self, cache_key: str) -> bool:
        with _MEMORY_LOCK:
            return cache_key in _MEMORY_STORES[self.namespace]


class LocalFileOrigamiStore(OrigamiStore):

    def __init__(self, root: str | Path, artifact_format: str = "bundle_v1"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_format = str(artifact_format)

    def _legacy_path(self, cache_key: str) -> Path:
        return self.root / f"{_safe_key(cache_key)}.pt"

    def _bundle_dir(self, cache_key: str) -> Path:
        return self.root / _safe_key(cache_key)

    def _manifest_path(self, cache_key: str) -> Path:
        return self._bundle_dir(cache_key) / "manifest.json"

    def _payload_path(self, cache_key: str) -> Path:
        return self._bundle_dir(cache_key) / "payload.bin"

    def _qat_bundle_path(self, cache_key: str) -> Path:
        return self._bundle_dir(cache_key) / "qat_bundle.dz"

    def put(self, payload: OrigamiPayload) -> None:
        if self.artifact_format == "legacy_pt":
            torch.save(payload, self._legacy_path(payload.cache_key))
            return
        self._put_bundle(payload)

    def get(self, cache_key: str) -> OrigamiPayload:
        manifest_path = self._manifest_path(cache_key)
        if self.artifact_format in {"auto", "bundle_v1"} and manifest_path.exists():
            manifest = self._read_manifest(cache_key)
            data = self._payload_path(cache_key).read_bytes()
            return self._payload_from_manifest(manifest, data)

        path = self._legacy_path(cache_key)
        if not path.exists():
            raise KeyError(f"Origami payload not found: {cache_key}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def preload_for_restore(
        self,
        cache_keys: list[str] | tuple[str, ...],
        *,
        lossless_backend: str,
        metrics_path: str | Path | None,
        system: str,
    ) -> None:
        paths: list[Path] = []
        for cache_key in cache_keys:
            if self._manifest_path(cache_key).exists() and self._manifest_uses_nvcomp(
                self._read_manifest(cache_key)
            ):
                paths.append(self._payload_path(cache_key))
            elif (
                lossless_backend == "qat"
                and native_cpu.qat_codec_path() == "dpucomp_dp"
                and self._qat_bundle_path(cache_key).exists()
            ):
                paths.append(self._qat_bundle_path(cache_key))
            elif self._manifest_path(cache_key).exists():
                paths.append(self._payload_path(cache_key))
            elif self._legacy_path(cache_key).exists():
                paths.append(self._legacy_path(cache_key))
        preload_memory_artifacts(
            [path for path in paths if path.exists()],
            system=system,
            metrics_path=metrics_path,
        )

    def prepare_for_restore(
        self,
        cache_key: str,
        *,
        lossless_backend: str,
        dynamic_huffman: bool,
        qat_max_instances: int,
        read_bandwidth_gbps: float,
        artifact_ingress_mode: str,
        metrics_path: str | Path | None,
        request_id: str,
        system: str,
    ) -> OrigamiPayload | PreparedOrigamiPayload:
        if self.artifact_format == "legacy_pt" or not self._manifest_path(
            cache_key
        ).exists():
            return self.get(cache_key)

        manifest = self._read_manifest(cache_key)
        if self._manifest_uses_nvcomp(manifest):
            ingress_mode = str(artifact_ingress_mode).lower().replace("_", "-")
            if ingress_mode == "native-file":
                ingress_mode = "memory"
            data = controlled_read(
                self._payload_path(cache_key),
                request_id=request_id,
                cache_key=cache_key,
                system=system,
                bandwidth_gbps=read_bandwidth_gbps,
                metrics_path=metrics_path,
                ingress_mode=ingress_mode,
            )
            return self._payload_from_manifest(manifest, data)

        if (
            lossless_backend == "qat"
            and native_cpu.qat_codec_path() == "dpucomp_dp"
            and self._qat_bundle_path(cache_key).exists()
        ):
            bundle_path = self._qat_bundle_path(cache_key)
            payload = self._payload_from_manifest(manifest, compressed_data=None)
            refs = tuple(self._request_refs_from_manifest(manifest))
            qat_bundle = dict(manifest.get("qat_bundle") or {})
            compressed_bytes = int(qat_bundle.get("compressed_bytes", 0))
            unpacked_bytes = int(qat_bundle.get("unpacked_bytes", 0))
            if compressed_bytes <= 0:
                compressed_bytes = sum(
                    int(row.get("compressed_bytes", 0))
                    for layer in manifest.get("layers", [])
                    for row in layer.get("chunks", [])
                )
            if unpacked_bytes <= 0:
                unpacked_bytes = sum(int(ref.output_bytes) for ref in refs)

            start = time.perf_counter()
            ingress_mode = str(artifact_ingress_mode).lower().replace("_", "-")
            with nvtx_range(f"{system}:qat_prepare"):
                if ingress_mode == "memory":
                    bundle_data = controlled_read(
                        bundle_path,
                        request_id=request_id,
                        cache_key=cache_key,
                        system=system,
                        bandwidth_gbps=read_bandwidth_gbps,
                        metrics_path=metrics_path,
                        ingress_mode="memory",
                    )
                    prepared = native_cpu.prepare_raw_deflate_bundle(
                        bundle_data,
                        dynamic_huffman=dynamic_huffman,
                        max_instances=qat_max_instances,
                    )
                else:
                    read_start_ts = time.time()
                    prepared = native_cpu.prepare_raw_deflate_bundle_file(
                        bundle_path,
                        dynamic_huffman=dynamic_huffman,
                        max_instances=qat_max_instances,
                        bandwidth_gbps=read_bandwidth_gbps,
                    )
            prepare_wall_ms = _ms_since(start)
            prepare_profile = native_cpu.last_qat_prepare_profile()
            file_read_ms = float(prepare_profile.get("prepare_file_read_ms", 0.0) or 0.0)
            file_read_io_ms = float(
                prepare_profile.get("prepare_file_read_io_ms", 0.0) or 0.0
            )
            file_read_sleep_ms = float(
                prepare_profile.get("prepare_file_read_sleep_ms", 0.0) or 0.0
            )
            file_read_bytes = int(
                prepare_profile.get("prepare_file_read_bytes", 0) or 0
            )
            if file_read_bytes <= 0:
                file_read_bytes = int(bundle_path.stat().st_size)
            if ingress_mode != "memory":
                source_read_io_gbps = (
                    file_read_bytes * 8.0 / file_read_io_ms / 1e6
                    if file_read_io_ms > 0.0
                    else 0.0
                )
                effective_limited_read_gbps = (
                    file_read_bytes * 8.0 / file_read_ms / 1e6
                    if file_read_ms > 0.0
                    else 0.0
                )
                read_end_ts = read_start_ts + file_read_ms / 1000.0
                write_jsonl(
                    metrics_path,
                    {
                        "type": "ssd_read",
                        "request_id": request_id,
                        "cache_key": cache_key,
                        "system": system,
                        "path": str(bundle_path),
                        "bytes_read": file_read_bytes,
                        "start_ts": read_start_ts,
                        "end_ts": read_end_ts,
                        "ms": file_read_ms,
                        "bandwidth_gbps": float(read_bandwidth_gbps),
                        "direct_qae": True,
                        "read_io_ms": file_read_io_ms,
                        "read_sleep_ms": file_read_sleep_ms,
                        "source_read_io_gbps": source_read_io_gbps,
                        "effective_limited_read_gbps": effective_limited_read_gbps,
                        "storage_source": "dram"
                        if str(bundle_path).startswith("/dev/shm/")
                        else "ssd",
                        "limiter_scope": "qat_input"
                        if float(read_bandwidth_gbps) > 0.0
                        else "none",
                        "artifact_ingress_mode": ingress_mode,
                    },
                )
            prepare_ms = (
                prepare_wall_ms
                if ingress_mode == "memory"
                else max(0.0, prepare_wall_ms - file_read_ms)
            )
            write_jsonl(
                metrics_path,
                {
                    "type": "qat_prepare",
                    "system": system,
                    "request_id": request_id,
                    "cache_key": cache_key,
                    "layer_name": "__request__",
                    "scope": "dpucomp_dp_bundle",
                    "qat_codec_path": native_cpu.qat_codec_path(),
                    "chunks": len(refs),
                    "compressed_bytes": compressed_bytes,
                    "unpacked_bytes": unpacked_bytes,
                    "ms": prepare_ms,
                    "wall_ms": prepare_wall_ms,
                    "direct_file_to_qae": True,
                    "compressed_gbps": (
                        compressed_bytes * 8.0 / prepare_ms / 1e6
                        if prepare_ms > 0
                        else 0.0
                    ),
                    "artifact_ingress_mode": ingress_mode,
                    **prepare_profile,
                },
            )
            prepared_layers = {
                layer_payload.layer_name: PreparedLayerPayload(
                    layer_payload=layer_payload,
                    groups=[],
                )
                for layer_payload in payload.layer_payloads.values()
            }
            return PreparedOrigamiPayload(
                payload=payload,
                prepared_layers=prepared_layers,
                request_groups=[
                    PreparedRequestGroup(
                        output_bytes=0,
                        refs=refs,
                        compressed_bytes=compressed_bytes,
                        unpacked_bytes=unpacked_bytes,
                        prepared=prepared,
                    )
                ],
                artifact_format="bundle_v1_dpucomp_dp",
            )

        if lossless_backend == "qat" and not native_cpu.qat_uses_prepared_restore():
            data = controlled_read(
                self._payload_path(cache_key),
                request_id=request_id,
                cache_key=cache_key,
                system=system,
                bandwidth_gbps=read_bandwidth_gbps,
                metrics_path=metrics_path,
                ingress_mode=artifact_ingress_mode,
            )
            return self._payload_from_manifest(manifest, data)

        if lossless_backend != "qat":
            ingress_mode = str(artifact_ingress_mode).lower().replace("_", "-")
            if ingress_mode == "native-file":
                ingress_mode = "file"
            data = controlled_read(
                self._payload_path(cache_key),
                request_id=request_id,
                cache_key=cache_key,
                system=system,
                bandwidth_gbps=read_bandwidth_gbps,
                metrics_path=metrics_path,
                ingress_mode=ingress_mode,
            )
            return self._payload_from_manifest(manifest, data)

        payload_path = self._payload_path(cache_key)
        data = controlled_read(
            payload_path,
            request_id=request_id,
            cache_key=cache_key,
            system=system,
            bandwidth_gbps=read_bandwidth_gbps,
            metrics_path=metrics_path,
            ingress_mode=artifact_ingress_mode,
        )
        bytestream = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        payload = self._payload_from_manifest(manifest, compressed_data=None)
        prepared_layers: dict[str, PreparedLayerPayload] = {}
        request_group_inputs: dict[int, dict[str, Any]] = {}

        manifest_groups = manifest.get("request_groups")
        if manifest_groups:
            for group in manifest_groups:
                chunk_ids = tuple(int(value) for value in group["chunk_ids"])
                lengths = [int(value) for value in group["lengths"]]
                output_bytes = int(group["output_bytes"])
                compressed_offset = int(group["compressed_offset"])
                compressed_bytes = int(group["compressed_bytes"])
                group_stream = bytestream.narrow(
                    0, compressed_offset, compressed_bytes
                )
                unpacked_bytes = output_bytes * len(chunk_ids)
                refs = tuple(
                    PreparedChunkRef(
                        layer_name=str(ref["layer_name"]),
                        chunk_id=int(ref["chunk_id"]),
                        output_bytes=output_bytes,
                    )
                    for ref in group["refs"]
                )
                group_input = request_group_inputs.setdefault(
                    output_bytes,
                    self._empty_request_group_input(),
                )
                group_input["streams"].append(group_stream.contiguous())
                group_input["lengths"].extend(lengths)
                group_input["refs"].extend(refs)
                group_input["compressed_bytes"] += compressed_bytes
                group_input["unpacked_bytes"] += unpacked_bytes
            for layer in manifest["layers"]:
                layer_payload = payload.layer_payloads[str(layer["layer_name"])]
                prepared_layers[layer_payload.layer_name] = PreparedLayerPayload(
                    layer_payload=layer_payload,
                    groups=[],
                )
        else:
            for layer in manifest["layers"]:
                layer_payload = payload.layer_payloads[str(layer["layer_name"])]
                groups: list[PreparedChunkGroup] = []
                for group in layer.get("groups", []):
                    chunk_ids = tuple(int(value) for value in group["chunk_ids"])
                    lengths = [int(value) for value in group["lengths"]]
                    output_bytes = int(group["output_bytes"])
                    compressed_offset = int(group["compressed_offset"])
                    compressed_bytes = int(group["compressed_bytes"])
                    group_stream = bytestream.narrow(
                        0, compressed_offset, compressed_bytes
                    )
                    unpacked_bytes = output_bytes * len(chunk_ids)
                    group_input = request_group_inputs.setdefault(
                        output_bytes,
                        self._empty_request_group_input(),
                    )
                    group_input["streams"].append(group_stream.contiguous())
                    group_input["lengths"].extend(lengths)
                    group_input["refs"].extend(
                        PreparedChunkRef(
                            layer_name=layer_payload.layer_name,
                            chunk_id=chunk_id,
                            output_bytes=output_bytes,
                        )
                        for chunk_id in chunk_ids
                    )
                    group_input["compressed_bytes"] += compressed_bytes
                    group_input["unpacked_bytes"] += unpacked_bytes
                prepared_layers[layer_payload.layer_name] = PreparedLayerPayload(
                    layer_payload=layer_payload,
                    groups=groups,
                )
        request_groups: list[PreparedRequestGroup] = []
        for output_bytes, group_input in sorted(request_group_inputs.items()):
            refs = tuple(group_input["refs"])
            if not refs:
                continue
            streams = group_input["streams"]
            group_stream = (
                streams[0]
                if len(streams) == 1
                else torch.cat(streams, dim=0).contiguous()
            )
            lengths = group_input["lengths"]
            compressed_bytes = int(group_input["compressed_bytes"])
            unpacked_bytes = int(group_input["unpacked_bytes"])
            start = time.perf_counter()
            with nvtx_range(f"{system}:qat_prepare"):
                prepared = native_cpu.prepare_raw_deflate_bytestream(
                    group_stream,
                    lengths,
                    unpacked_bytes,
                    output_bytes,
                    dynamic_huffman=dynamic_huffman,
                    max_instances=qat_max_instances,
                )
            prepare_ms = _ms_since(start)
            prepare_profile = native_cpu.last_qat_prepare_profile()
            write_jsonl(
                metrics_path,
                {
                    "type": "qat_prepare",
                    "system": system,
                    "request_id": request_id,
                    "cache_key": cache_key,
                    "layer_name": "__request__",
                    "chunks": len(refs),
                    "compressed_bytes": compressed_bytes,
                    "unpacked_bytes": unpacked_bytes,
                    "ms": prepare_ms,
                    "compressed_gbps": (
                        compressed_bytes * 8.0 / prepare_ms / 1e6
                        if prepare_ms > 0
                        else 0.0
                    ),
                    **prepare_profile,
                },
            )
            request_groups.append(
                PreparedRequestGroup(
                    output_bytes=output_bytes,
                    refs=refs,
                    compressed_bytes=compressed_bytes,
                    unpacked_bytes=unpacked_bytes,
                    prepared=prepared,
                )
            )
        if len(prepared_layers) == 1 and request_groups:
            layer_name = next(iter(prepared_layers))
            compatibility_groups = []
            for group in request_groups:
                if all(ref.layer_name == layer_name for ref in group.refs):
                    compatibility_groups.append(
                        PreparedChunkGroup(
                            layer_name=layer_name,
                            output_bytes=group.output_bytes,
                            chunk_ids=tuple(ref.chunk_id for ref in group.refs),
                            compressed_bytes=group.compressed_bytes,
                            unpacked_bytes=group.unpacked_bytes,
                            prepared=group.prepared,
                        )
                    )
            prepared_layers[layer_name].groups = compatibility_groups
        return PreparedOrigamiPayload(
            payload=payload,
            prepared_layers=prepared_layers,
            request_groups=request_groups,
        )

    def contains(self, cache_key: str) -> bool:
        if self.artifact_format == "legacy_pt":
            return self._legacy_path(cache_key).exists()
        if self._manifest_path(cache_key).exists() and self._payload_path(
            cache_key
        ).exists():
            return True
        return self.artifact_format == "auto" and self._legacy_path(cache_key).exists()

    @staticmethod
    def _manifest_uses_nvcomp(manifest: dict[str, Any]) -> bool:
        return any(
            str(row.get("codec", "")).lower().startswith("nvcomp")
            for layer in manifest.get("layers", [])
            for row in layer.get("chunks", [])
        )

    @staticmethod
    def _empty_request_group_input() -> dict[str, Any]:
        return {
            "streams": [],
            "lengths": [],
            "refs": [],
            "compressed_bytes": 0,
            "unpacked_bytes": 0,
        }

    @staticmethod
    def _request_refs_from_manifest(
        manifest: dict[str, Any],
    ) -> list[PreparedChunkRef]:
        refs: list[PreparedChunkRef] = []
        qat_bundle = manifest.get("qat_bundle") or {}
        qat_refs = qat_bundle.get("refs") or []
        for ref in qat_refs:
            refs.append(
                PreparedChunkRef(
                    layer_name=str(ref["layer_name"]),
                    chunk_id=int(ref["chunk_id"]),
                    output_bytes=int(ref["output_bytes"]),
                )
            )
        if refs:
            return refs

        request_groups = manifest.get("request_groups") or []
        for group in request_groups:
            output_bytes = int(group["output_bytes"])
            for ref in group.get("refs", []):
                refs.append(
                    PreparedChunkRef(
                        layer_name=str(ref["layer_name"]),
                        chunk_id=int(ref["chunk_id"]),
                        output_bytes=output_bytes,
                    )
                )
        if refs:
            return refs

        for layer in manifest.get("layers", []):
            layer_name = str(layer["layer_name"])
            for row in layer.get("chunks", []):
                refs.append(
                    PreparedChunkRef(
                        layer_name=layer_name,
                        chunk_id=int(row["chunk_id"]),
                        output_bytes=int(row["unpacked_bytes"]),
                    )
                )
        return refs

    @staticmethod
    def _write_dpucomp_header(
        handle: Any,
        *,
        chunk_bytes: int,
        records: list[tuple[int, int]],
    ) -> None:
        raw_bytes = sum(raw_len for raw_len, _ in records)
        handle.write(b"DPUCDZ1\0")
        handle.write(struct.pack("<IIQ", int(chunk_bytes), len(records), raw_bytes))
        for raw_len, comp_len in records:
            handle.write(struct.pack("<II", int(raw_len), int(comp_len)))

    def _put_bundle(self, payload: OrigamiPayload) -> None:
        bundle_dir = self._bundle_dir(payload.cache_key)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self._payload_path(payload.cache_key)
        qat_bundle_path = self._qat_bundle_path(payload.cache_key)
        manifest_path = self._manifest_path(payload.cache_key)

        layer_chunk_rows: dict[str, dict[int, dict[str, Any]]] = {}
        layer_group_rows: dict[str, dict[int, dict[str, Any]]] = {}
        chunks_by_size: dict[int, list[tuple[str, ChunkRecord]]] = {}
        qat_entries: list[tuple[str, ChunkRecord]] = []
        for layer_name in sorted(payload.layer_payloads):
            layer_payload = payload.layer_payloads[layer_name]
            layer_chunk_rows[layer_name] = {}
            layer_group_rows[layer_name] = {}
            for chunk in sorted(layer_payload.chunks,
                                key=lambda item: int(item.chunk_id)):
                qat_entries.append((layer_name, chunk))
                output_bytes = int(chunk.unpacked_bytes)
                chunks_by_size.setdefault(output_bytes, []).append(
                    (layer_name, chunk)
                )
        create_qat_bundle = all(
            _codec_uses_qat_bundle(chunk.codec)
            for _, chunk in qat_entries
        )

        ordered_groups = [
            (output_bytes, entries)
            for output_bytes, entries in sorted(chunks_by_size.items())
        ]
        ordered_entries = [
            (output_bytes, layer_name, chunk)
            for output_bytes, entries in ordered_groups
            for layer_name, chunk in entries
        ]
        qat_records = [
            (int(chunk.unpacked_bytes), int(chunk.compressed.numel()))
            for _, chunk in qat_entries
        ]
        max_chunk_bytes = max((raw_len for raw_len, _ in qat_records), default=1)

        request_groups = []
        qat_refs = []
        with payload_path.open("wb") as handle:
            if create_qat_bundle:
                with qat_bundle_path.open("wb") as qat_handle:
                    self._write_dpucomp_header(
                        qat_handle,
                        chunk_bytes=max_chunk_bytes,
                        records=qat_records,
                    )
                    for layer_name, chunk in qat_entries:
                        compressed = _cpu_u8(chunk.compressed)
                        data = compressed.numpy().tobytes()
                        qat_handle.write(data)
                        qat_refs.append(
                            {
                                "layer_name": layer_name,
                                "chunk_id": int(chunk.chunk_id),
                                "output_bytes": int(chunk.unpacked_bytes),
                            }
                        )
            elif qat_bundle_path.exists():
                qat_bundle_path.unlink()
            for output_bytes, entries in ordered_groups:
                request_group_offset = handle.tell()
                request_lengths = []
                request_refs = []
                for layer_name, chunk in entries:
                    compressed = _cpu_u8(chunk.compressed)
                    offset = handle.tell()
                    data = compressed.numpy().tobytes()
                    handle.write(data)
                    compressed_bytes = len(data)
                    chunk_id = int(chunk.chunk_id)
                    request_lengths.append(compressed_bytes)
                    request_refs.append({
                        "layer_name": layer_name,
                        "chunk_id": chunk_id,
                    })
                    layer_chunk_rows[layer_name][chunk_id] = {
                        "chunk_id": chunk_id,
                        "layout": asdict(chunk.layout),
                        "codec": str(chunk.codec),
                        "compressed_offset": offset,
                        "compressed_bytes": compressed_bytes,
                        "unpacked_bytes": int(chunk.unpacked_bytes),
                    }
                    layer_group = layer_group_rows[layer_name].setdefault(
                        int(output_bytes),
                        {
                            "output_bytes": int(output_bytes),
                            "chunk_ids": [],
                            "lengths": [],
                            "compressed_offset": offset,
                            "compressed_bytes": 0,
                        },
                    )
                    layer_group["chunk_ids"].append(chunk_id)
                    layer_group["lengths"].append(compressed_bytes)
                    layer_group["compressed_bytes"] += compressed_bytes
                request_groups.append(
                    {
                        "output_bytes": int(output_bytes),
                        "chunk_ids": [
                            int(ref["chunk_id"]) for ref in request_refs
                        ],
                        "refs": request_refs,
                        "lengths": request_lengths,
                        "compressed_offset": request_group_offset,
                        "compressed_bytes": sum(request_lengths),
                    }
                )

        layers = []
        for layer_name in sorted(payload.layer_payloads):
            layer_payload = payload.layer_payloads[layer_name]
            chunk_rows = layer_chunk_rows[layer_name]
            groups = [
                layer_group_rows[layer_name][output_bytes]
                for output_bytes in sorted(layer_group_rows[layer_name])
            ]
            layers.append(
                {
                    "layer_name": layer_payload.layer_name,
                    "quantizer": layer_payload.quantizer,
                    "quant_metadata": layer_payload.quant_metadata,
                    "chunks": [
                        chunk_rows[idx] for idx in sorted(chunk_rows)
                    ],
                    "groups": groups,
                }
            )

        manifest = {
            "format": "bundle_v1",
            "cache_key": payload.cache_key,
            "request_id": payload.request_id,
            "quantizer": payload.quantizer,
            "quantizer_config_hash": payload.quantizer_config_hash,
            "token_start": int(payload.token_start),
            "token_count": int(payload.token_count),
            "metadata": dict(payload.metadata),
            "layers": layers,
            "request_groups": request_groups,
            "qat_bundle": (
                {
                    "format": "DPUCDZ1",
                    "path": "qat_bundle.dz",
                    "chunk_bytes": max_chunk_bytes,
                    "chunks": len(qat_records),
                    "compressed_bytes": sum(comp_len for _, comp_len in qat_records),
                    "unpacked_bytes": sum(raw_len for raw_len, _ in qat_records),
                    "refs": qat_refs,
                }
                if create_qat_bundle
                else {}
            ),
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)

    def _read_manifest(self, cache_key: str) -> dict[str, Any]:
        path = self._manifest_path(cache_key)
        if not path.exists():
            raise KeyError(f"Origami bundle manifest not found: {cache_key}")
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != "bundle_v1":
            raise ValueError(
                f"Unsupported Origami artifact format: {manifest.get('format')!r}"
            )
        return manifest

    def _payload_from_manifest(
        self,
        manifest: dict[str, Any],
        compressed_data: bytes | None,
    ) -> OrigamiPayload:
        payload = OrigamiPayload(
            cache_key=str(manifest["cache_key"]),
            request_id=str(manifest.get("request_id", "")),
            quantizer=str(manifest["quantizer"]),
            quantizer_config_hash=str(manifest.get("quantizer_config_hash", "")),
            token_start=int(manifest.get("token_start", 0)),
            token_count=int(manifest.get("token_count", 0)),
            metadata=dict(manifest.get("metadata") or {}),
        )
        payload.metadata["artifact_format"] = "bundle_v1"
        for layer in manifest["layers"]:
            chunks = []
            for row in layer.get("chunks", []):
                compressed = torch.empty((0,), dtype=torch.uint8)
                if compressed_data is not None:
                    offset = int(row["compressed_offset"])
                    length = int(row["compressed_bytes"])
                    compressed = torch.frombuffer(
                        bytearray(compressed_data[offset:offset + length]),
                        dtype=torch.uint8,
                    ).clone()
                chunks.append(
                    ChunkRecord(
                        chunk_id=int(row["chunk_id"]),
                        layout=ChunkLayout(**row["layout"]),
                        codec=str(row.get("codec", "qat")),
                        compressed=compressed,
                        compressed_bytes=int(row["compressed_bytes"]),
                        unpacked_bytes=int(row["unpacked_bytes"]),
                    )
                )
            payload.layer_payloads[str(layer["layer_name"])] = LayerPayload(
                layer_name=str(layer["layer_name"]),
                quantizer=str(layer["quantizer"]),
                quant_metadata=dict(layer.get("quant_metadata") or {}),
                chunks=sorted(chunks, key=lambda chunk: int(chunk.chunk_id)),
            )
        return payload


def create_origami_store(
    uri: str,
    artifact_format: str = "bundle_v1",
) -> OrigamiStore:
    if uri.startswith("memory://"):
        namespace = uri[len("memory://"):] or "origami"
        return InMemoryOrigamiStore(namespace)
    if uri.startswith("file://"):
        return LocalFileOrigamiStore(
            uri[len("file://"):], artifact_format=artifact_format
        )
    return LocalFileOrigamiStore(uri, artifact_format=artifact_format)
