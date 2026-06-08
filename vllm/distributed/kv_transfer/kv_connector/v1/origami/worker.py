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
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import native_cpu
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.chunking import (
    PlannedChunk,
    plan_head_channel_chunks,
)
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
            dynamic_huffman=config.qat_dynamic_huffman,
            qat_inflight=config.qat_inflight,
            qat_batch=config.qat_batch,
            qat_max_instances=config.qat_max_instances,
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
        restored_chunks = self.cpu_codec.decompress_many(compressed_chunks, output_bytes)
        return self._unpack_native_symbols(layer_payload, restored_chunks)

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
        raw_chunks, planned_chunks, native_metadata = self._pack_quantized_symbols(
            quantized.symbols,
            quantized.metadata,
            layer_name,
        )
        compressed_chunks = self.cpu_codec.compress_many(raw_chunks)
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
                    codec=self.config.lossless_cpu_backend,
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
        if requested_bits not in {2, 4, 8}:
            raise ValueError("origami_bits must be one of 2, 4, or 8")
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

    def _unpack_native_symbols(
        self,
        layer_payload: LayerPayload,
        raw_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        native_metadata = layer_payload.quant_metadata.get(_NATIVE_LAYOUT_KEY)
        if not native_metadata:
            return torch.cat([chunk.detach().cpu().to(torch.uint8).reshape(-1) for chunk in raw_chunks])
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

    def get_finished(self) -> tuple[set[str], set[str]]:
        return set(), set()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
