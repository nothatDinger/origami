# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import OrigamiConfig
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.cpu_qat import (
    CpuLosslessCodec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.gpu_nvcomp import (
    GpuLosslessCodec,
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
from vllm.distributed.kv_transfer.kv_connector.v1.origami.store import OrigamiStore

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata


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


class OrigamiConnectorWorker:

    def __init__(
        self,
        config: OrigamiConfig,
        store: OrigamiStore,
        kv_cache_config: Any | None = None,
    ):
        self.config = config
        self.store = store
        self.kv_cache_config = kv_cache_config
        self.quantizer: QuantizerAdapter = create_quantizer_adapter(
            config.quantizer, config.quantizer_config
        )
        self.cpu_codec = CpuLosslessCodec(
            backend=config.lossless_cpu_backend,
            allow_zlib_fallback=config.allow_zlib_fallback,
        )
        self._gpu_codec: GpuLosslessCodec | None = None
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.layer_to_cache_group: dict[str, int] = {}
        self._executor = ThreadPoolExecutor(max_workers=max(1, config.qat_threads))
        self._pending_layer_futures: dict[str, list[Future[Any]]] = {}
        self._save_futures: list[Future[tuple[OrigamiSaveRequest, LayerPayload]]] = []
        self._save_payloads: dict[str, OrigamiPayload] = {}
        self._save_lock = Lock()
        self._metadata: OrigamiConnectorMetadata | None = None

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
        self._pending_layer_futures.clear()
        self._save_futures.clear()
        self._save_payloads.clear()
        if not metadata.reqs_to_restore:
            return
        for restore in metadata.reqs_to_restore.values():
            for layer_name in self.kv_caches:
                future = self._executor.submit(
                    self._restore_layer,
                    restore,
                    layer_name,
                )
                self._pending_layer_futures.setdefault(layer_name, []).append(future)

    def wait_for_layer_load(self, layer_name: str) -> None:
        futures = self._pending_layer_futures.pop(layer_name, [])
        for future in futures:
            event = future.result()
            if event is not None and torch.cuda.is_available():
                torch.cuda.current_stream().wait_event(event)

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

    def _restore_layer(
        self,
        restore: OrigamiRestoreRequest,
        layer_name: str,
    ) -> Any | None:
        if layer_name not in self.kv_caches:
            return None
        payload = self.store.get(restore.cache_key)
        layer_payload = payload.layer_payloads.get(layer_name)
        if layer_payload is None:
            return None
        symbols = self._restore_symbols(layer_payload, restore.lossless_path)
        kv_tensor = self.quantizer.dequantize(symbols, layer_payload.quant_metadata)
        kv_cache = self.kv_caches[layer_name]
        group_index = self.layer_to_cache_group.get(layer_name, 0)
        if group_index >= len(restore.block_ids_per_group):
            return None
        block_ids = list(restore.block_ids_per_group[group_index])
        if not block_ids:
            return None
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
        kv_cache[block_ids_tensor] = kv_tensor.to(
            device=kv_cache.device, dtype=kv_cache.dtype
        )
        if kv_cache.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            return event
        return None

    def _restore_symbols(self, layer_payload: LayerPayload, path: str) -> torch.Tensor:
        restored_chunks: list[torch.Tensor] = []
        if path == "gpu":
            codec = self._get_gpu_codec()
            for chunk in layer_payload.chunks:
                compressed = chunk.compressed.to("cuda", non_blocking=True)
                restored_chunks.append(
                    codec.decompress(compressed, output_bytes=chunk.unpacked_bytes)
                )
            return torch.cat(restored_chunks).cpu()
        for chunk in layer_payload.chunks:
            restored_chunks.append(
                self.cpu_codec.decompress(
                    chunk.compressed,
                    output_bytes=chunk.unpacked_bytes,
                )
            )
        return torch.cat(restored_chunks)

    def _get_gpu_codec(self) -> GpuLosslessCodec:
        if self._gpu_codec is None:
            self._gpu_codec = GpuLosslessCodec(self.config.lossless_gpu_backend)
        return self._gpu_codec

    def _compress_layer(
        self,
        save: OrigamiSaveRequest,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_ids: tuple[int, ...],
    ) -> tuple[OrigamiSaveRequest, LayerPayload]:
        block_ids_tensor = torch.tensor(
            list(block_ids), dtype=torch.long, device=kv_layer.device
        )
        kv_blocks = kv_layer[block_ids_tensor].detach()
        quantized = self.quantizer.quantize(kv_blocks)
        flat = quantized.symbols.reshape(-1).to(torch.uint8).cpu().contiguous()
        chunks: list[ChunkRecord] = []
        offset = 0
        chunk_id = 0
        while offset < int(flat.numel()):
            end = min(offset + self.config.chunk_target_bytes, int(flat.numel()))
            raw = flat[offset:end].contiguous()
            compressed = self.cpu_codec.compress(raw)
            layout = ChunkLayout(
                layer_index=_layer_index(layer_name),
                head_start=0,
                head_end=0,
                channel_start=offset,
                channel_end=end,
                token_start=save.token_start,
                token_end=save.token_start + save.num_tokens,
                unpacked_bytes=int(raw.numel()),
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    layout=layout,
                    codec=self.config.lossless_cpu_backend,
                    compressed=compressed,
                    compressed_bytes=int(compressed.numel()),
                    unpacked_bytes=int(raw.numel()),
                )
            )
            offset = end
            chunk_id += 1
        layer_payload = LayerPayload(
            layer_name=layer_name,
            quantizer=self.quantizer.quantizer_id,
            quant_metadata=quantized.metadata,
            chunks=chunks,
        )
        return save, layer_payload

    def get_finished(self) -> tuple[set[str], set[str]]:
        return set(), set()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

