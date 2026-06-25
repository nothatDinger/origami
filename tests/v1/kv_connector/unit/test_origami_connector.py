# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import OrigamiConfig
from vllm.distributed.kv_transfer.kv_connector.v1.origami.benchmark_connectors import (
    ReuseConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.benchmark_utils import (
    KernelRange,
    KernelRecord,
    TokenBucket,
    classify_kernel_records,
    controlled_read,
    copy_kv_to_hbm,
    windows_overlap,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.chunking import (
    plan_head_channel_chunks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
    native_cpu,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
    native_gpu,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.cpu_qat import (
    CpuLosslessCodec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.pipeline import (
    select_gpu_lossless_requests,
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
from vllm.distributed.kv_transfer.kv_connector.v1.origami.scheduler import (
    OrigamiConnectorScheduler,
    OrigamiOffloadController,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.store import (
    InMemoryOrigamiStore,
    LocalFileOrigamiStore,
    PreparedOrigamiPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.worker import (
    OrigamiConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.raw_tensor import (
    RawBytesTensorAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.cachegen_adapter import (
    CacheGenAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kivi_adapter import (
    KiviAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.registry import (
    create_quantizer_adapter,
)


def test_factory_registers_origami_connector() -> None:
    connector_cls = KVConnectorFactory.get_connector_class_by_name(
        "OrigamiConnector"
    )

    assert connector_cls.__name__ == "OrigamiConnector"


def test_request_level_offload_ratio_selection() -> None:
    request_ids = ["r0", "r1", "r2", "r3"]

    assert select_gpu_lossless_requests(request_ids, 0) == set()
    assert select_gpu_lossless_requests(request_ids, 25) == {"r0"}
    assert select_gpu_lossless_requests(request_ids, 50) == {"r0", "r1"}
    assert select_gpu_lossless_requests(request_ids, 75) == {"r0", "r1", "r2"}
    assert select_gpu_lossless_requests(request_ids, 100) == set(request_ids)


def test_lossless_ratio_aliases_map_to_cpu_and_gpu() -> None:
    assert (
        OrigamiOffloadController(OrigamiConfig(gpu_lossless_ratio="cpu")).ratio
        == 0
    )
    assert (
        OrigamiOffloadController(OrigamiConfig(gpu_lossless_ratio="gpu")).ratio
        == 100
    )


def test_registry_registers_kivi_quantizer() -> None:
    assert create_quantizer_adapter("kivi").quantizer_id == "kivi"


def test_auto_offload_promotes_after_three_high_watermark_steps() -> None:
    controller = OrigamiOffloadController(
        OrigamiConfig(
            gpu_lossless_ratio="auto",
            pcie_high_watermark_gbps=10.0,
            auto_decrease_enabled=False,
        )
    )

    assert controller.observe_schedule_step(10.0) == 0
    assert controller.observe_schedule_step(12.0) == 0
    assert controller.observe_schedule_step(10.0) == 25
    assert controller.observe_schedule_step(11.0) == 25
    assert controller.observe_schedule_step(11.0) == 25
    assert controller.observe_schedule_step(11.0) == 50


def test_chunk_planner_keeps_head_channel_bounds() -> None:
    chunks = plan_head_channel_chunks(
        layer_index=3,
        num_heads=2,
        head_dim=512,
        token_count=8192,
        bytes_per_symbol=1,
        min_bytes=512 << 10,
        target_bytes=1 << 20,
        max_bytes=2 << 20,
    )

    assert chunks
    assert {chunk.layout.layer_index for chunk in chunks} == {3}
    for chunk in chunks:
        layout = chunk.layout
        assert layout.head_end == layout.head_start + 1
        assert 0 <= layout.channel_start < layout.channel_end <= 512
        assert 512 << 10 <= layout.unpacked_bytes <= 2 << 20


def test_chunk_planner_combines_short_adjacent_heads() -> None:
    chunks = plan_head_channel_chunks(
        layer_index=0,
        num_heads=128,
        head_dim=64,
        token_count=128,
        bytes_per_symbol=1,
        min_bytes=512 << 10,
        target_bytes=1 << 20,
        max_bytes=2 << 20,
    )

    assert chunks
    assert any(
        chunk.layout.head_end - chunk.layout.head_start > 1 for chunk in chunks
    )
    assert all(chunk.layout.unpacked_bytes <= 2 << 20 for chunk in chunks)


def test_payload_store_roundtrip() -> None:
    store = InMemoryOrigamiStore(f"test-{uuid4()}")
    payload = OrigamiPayload(
        cache_key="cache-a",
        request_id="req-a",
        quantizer="mock_int4",
        quantizer_config_hash="hash",
        token_start=16,
        token_count=4,
    )
    payload.layer_payloads["layer_0"] = LayerPayload(
        layer_name="layer_0",
        quantizer="mock_int4",
        quant_metadata={"shape": [1, 2, 3]},
        chunks=[
            ChunkRecord(
                chunk_id=0,
                layout=ChunkLayout(
                    layer_index=0,
                    head_start=0,
                    head_end=1,
                    channel_start=0,
                    channel_end=3,
                    token_start=16,
                    token_end=20,
                    unpacked_bytes=3,
                ),
                codec="zlib",
                compressed=torch.tensor([1, 2, 3], dtype=torch.uint8),
                compressed_bytes=3,
                unpacked_bytes=3,
            )
        ],
    )

    store.put(payload)
    restored = store.get("cache-a")

    assert restored.cache_key == "cache-a"
    assert restored.layer_payloads["layer_0"].chunks[0].compressed.tolist() == [
        1,
        2,
        3,
    ]


def _sample_payload(cache_key: str = "cache-a") -> OrigamiPayload:
    payload = OrigamiPayload(
        cache_key=cache_key,
        request_id="req-a",
        quantizer="raw_bytes",
        quantizer_config_hash="hash",
        token_start=0,
        token_count=16,
    )
    payload.layer_payloads["layer_0"] = LayerPayload(
        layer_name="layer_0",
        quantizer="raw_bytes",
        quant_metadata={
            "format": "raw_bytes",
            "dtype": "uint8",
            "shape": [4],
            "origami_bits": 8,
            "origami_symbol_shape": [1, 1, 4],
            "origami_symbol_layout": ["token", "head", "head_dim"],
            "origami_native_layout": {
                "bits": 8,
                "source_shape": [1, 1, 4],
                "source_layout": ["token", "head", "head_dim"],
                "storage_layout": ["layer", "head", "head_dim", "token"],
                "token_count": 1,
                "num_heads": 1,
                "head_dim": 4,
                "symbol_count": 4,
                "layout_policy": "head_first",
                "bitpack": True,
                "chunk_specs": [[0, 1, 0, 4, 0, 1]],
            },
        },
        chunks=[
            ChunkRecord(
                chunk_id=0,
                layout=ChunkLayout(
                    layer_index=0,
                    head_start=0,
                    head_end=1,
                    channel_start=0,
                    channel_end=4,
                    token_start=0,
                    token_end=1,
                    unpacked_bytes=4,
                ),
                codec="zlib",
                compressed=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
                compressed_bytes=4,
                unpacked_bytes=4,
            )
        ],
    )
    return payload


def test_local_file_store_writes_bundle_manifest_and_payload(tmp_path) -> None:
    store = LocalFileOrigamiStore(tmp_path, artifact_format="bundle_v1")
    payload = _sample_payload("cache-bundle")

    store.put(payload)

    bundle_dir = tmp_path / "cache-bundle"
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "payload.bin").read_bytes() == bytes([1, 2, 3, 4])
    assert (bundle_dir / "qat_bundle.dz").exists()
    restored = store.get("cache-bundle")
    chunk = restored.layer_payloads["layer_0"].chunks[0]
    assert restored.metadata["artifact_format"] == "bundle_v1"
    assert chunk.compressed.tolist() == [1, 2, 3, 4]


def test_local_file_store_prepares_qat_bundle_when_available(tmp_path) -> None:
    try:
        codec = CpuLosslessCodec(
            backend="qat",
            qat_inflight=2,
            qat_batch=1,
            qat_max_instances=1,
        )
    except RuntimeError:
        return

    originals = [
        (torch.arange(512 << 10, dtype=torch.int64) & 0xFF).to(torch.uint8),
        ((torch.arange(512 << 10, dtype=torch.int64) + 7) & 0xFF).to(torch.uint8),
    ]
    compressed = codec.compress_many(originals)
    payload = OrigamiPayload(
        cache_key="cache-q prepared",
        request_id="req-q",
        quantizer="raw_bytes",
        quantizer_config_hash="hash",
        token_start=0,
        token_count=16,
    )
    chunks = []
    for idx, (raw, encoded) in enumerate(zip(originals, compressed)):
        chunks.append(
            ChunkRecord(
                chunk_id=idx,
                layout=ChunkLayout(
                    layer_index=0,
                    head_start=idx,
                    head_end=idx + 1,
                    channel_start=0,
                    channel_end=int(raw.numel()),
                    token_start=0,
                    token_end=1,
                    unpacked_bytes=int(raw.numel()),
                ),
                codec="qat",
                compressed=encoded,
                compressed_bytes=int(encoded.numel()),
                unpacked_bytes=int(raw.numel()),
            )
        )
    payload.layer_payloads["layer_0"] = LayerPayload(
        layer_name="layer_0",
        quantizer="raw_bytes",
        quant_metadata={},
        chunks=chunks,
    )
    store = LocalFileOrigamiStore(tmp_path, artifact_format="bundle_v1")
    metrics_path = tmp_path / "metrics.jsonl"

    store.put(payload)
    prepared = store.prepare_for_restore(
        "cache-q prepared",
        lossless_backend="qat",
        dynamic_huffman=True,
        qat_max_instances=1,
        read_bandwidth_gbps=0.0,
        metrics_path=metrics_path,
        request_id="req-q",
        system="origami",
    )

    if native_cpu.qat_uses_prepared_restore():
        assert isinstance(prepared, PreparedOrigamiPayload)
        if prepared.request_groups:
            group = prepared.request_groups[0]
        else:
            group = prepared.prepared_layers["layer_0"].groups[0]
        restored = native_cpu.decompress_prepared_raw_deflate_many(
            group.prepared,
            inflight=2,
            batch=1,
            max_instances=1,
        )[0]
        assert torch.equal(restored[: originals[0].numel()], originals[0])
        assert torch.equal(restored[originals[0].numel():], originals[1])
    else:
        assert isinstance(prepared, OrigamiPayload)
        layer_payload = prepared.layer_payloads["layer_0"]
        restored_chunks = codec.decompress_many(
            [chunk.compressed for chunk in layer_payload.chunks],
            [int(chunk.unpacked_bytes) for chunk in layer_payload.chunks],
        )
        assert torch.equal(restored_chunks[0], originals[0])
        assert torch.equal(restored_chunks[1], originals[1])
    metric_types = {
        json.loads(line)["type"]
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    }
    assert "ssd_read" in metric_types
    if native_cpu.qat_uses_prepared_restore():
        assert "qat_prepare" in metric_types
    else:
        assert "qat_prepare" not in metric_types


def test_cpu_zlib_raw_deflate_roundtrip() -> None:
    codec = CpuLosslessCodec(backend="zlib")
    original = torch.arange(4096, dtype=torch.int32).view(torch.uint8)

    compressed = codec.compress(original)
    restored = codec.decompress(compressed, output_bytes=original.numel())

    assert torch.equal(restored, original)


def test_raw_bytes_quantizer_roundtrips_without_torch_serializer() -> None:
    adapter = RawBytesTensorAdapter()
    for dtype in (torch.float16, torch.bfloat16):
        original = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).to(dtype)

        quantized = adapter.quantize(original)
        restored = adapter.dequantize(quantized.symbols, quantized.metadata)

        assert quantized.metadata["format"] == "raw_bytes"
        assert quantized.metadata["origami_bits"] == 8
        assert restored.dtype == dtype
        assert restored.shape == original.shape
        assert torch.equal(restored, original)


def _reference_kivi_dequant(
    kv: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sink_tokens: int,
) -> torch.Tensor:
    canonical = kv.detach().cpu().to(torch.float16).contiguous()
    tokens = int(canonical.shape[1])
    heads = int(canonical.shape[2])
    head_dim = int(canonical.shape[3])
    cols = heads * head_dim
    sink = min(int(sink_tokens), tokens)
    body_tokens = tokens - sink
    key_sink = canonical[0, :sink]
    value_sink = canonical[1, :sink]
    key_body = canonical[0, sink:].reshape(body_tokens, cols).float()
    value_body = canonical[1, sink:].reshape(body_tokens, cols).float()
    levels = float((1 << int(bits)) - 1)

    key_groups = (body_tokens + group_size - 1) // group_size
    if body_tokens:
        key_pad = key_groups * group_size - body_tokens
        key_padded = key_body if key_pad == 0 else torch.cat(
            [key_body, key_body[-1:].expand(key_pad, cols)], dim=0
        )
        key_grouped = key_padded.reshape(key_groups, group_size, cols)
        key_zero = torch.amin(key_grouped, dim=1)
        key_scale = torch.clamp(
            (torch.amax(key_grouped, dim=1) - key_zero) / levels,
            min=torch.finfo(torch.float32).tiny,
        )
        key_q = torch.round(
            (key_grouped - key_zero[:, None, :]) / key_scale[:, None, :]
        ).clamp(0, levels).reshape(key_groups * group_size, cols)[:body_tokens]
        key_scale = key_scale.to(torch.float16).float()
        key_zero = key_zero.to(torch.float16).float()
        row_groups = torch.arange(body_tokens, dtype=torch.long) // group_size
        key_restored = key_q * key_scale[row_groups, :] + key_zero[row_groups, :]
    else:
        key_restored = torch.empty((0, cols), dtype=torch.float32)

    value_groups = (cols + group_size - 1) // group_size
    if body_tokens:
        value_pad = value_groups * group_size - cols
        value_padded = value_body if value_pad == 0 else torch.cat(
            [value_body, value_body[:, -1:].expand(body_tokens, value_pad)], dim=1
        )
        value_grouped = value_padded.reshape(body_tokens, value_groups, group_size)
        value_zero = torch.amin(value_grouped, dim=2)
        value_scale = torch.clamp(
            (torch.amax(value_grouped, dim=2) - value_zero) / levels,
            min=torch.finfo(torch.float32).tiny,
        )
        value_q = torch.round(
            (value_grouped - value_zero[:, :, None]) / value_scale[:, :, None]
        ).clamp(0, levels).reshape(body_tokens, value_groups * group_size)[:, :cols]
        value_scale = value_scale.to(torch.float16).float()
        value_zero = value_zero.to(torch.float16).float()
        col_groups = torch.arange(cols, dtype=torch.long) // group_size
        value_restored = value_q * value_scale[:, col_groups] + value_zero[:, col_groups]
    else:
        value_restored = torch.empty((0, cols), dtype=torch.float32)

    key = torch.cat(
        [key_sink, key_restored.reshape(body_tokens, heads, head_dim).to(torch.float16)],
        dim=0,
    )
    value = torch.cat(
        [
            value_sink,
            value_restored.reshape(body_tokens, heads, head_dim).to(torch.float16),
        ],
        dim=0,
    )
    return torch.stack([key, value], dim=0).to(dtype=kv.dtype)


def test_kivi_quantizer_cpu_fallback_matches_reference_layout() -> None:
    torch.manual_seed(20260621)
    adapter = KiviAdapter({"bits": 2, "group_size": 2, "sink_tokens": 2})
    original = torch.randn((2, 7, 2, 4), dtype=torch.float16)

    quantized = adapter.quantize(original)
    restored = adapter.dequantize(quantized.symbols, quantized.metadata)
    expected = _reference_kivi_dequant(
        original, bits=2, group_size=2, sink_tokens=2
    )

    assert quantized.metadata["format"] == "kivi_structured_blob"
    assert quantized.metadata["key_order"] == ["head_dim", "layer", "head", "token"]
    assert quantized.metadata["value_order"] == ["head", "layer", "head_dim", "token"]
    assert restored.shape == original.shape
    assert torch.equal(restored, expected)
    assert torch.equal(restored[:, :2], original[:, :2])


def test_worker_save_restore_roundtrip_with_kivi_and_zlib() -> None:
    store = InMemoryOrigamiStore(f"worker-kivi-{uuid4()}")
    cfg = {"bits": 2, "group_size": 2, "sink_tokens": 1, "dequant_device": "cpu"}
    worker = OrigamiConnectorWorker(
        OrigamiConfig(
            quantizer="kivi",
            quantizer_config=cfg,
            lossless_cpu_backend="zlib",
            qat_threads=1,
        ),
        store,
    )
    kv_cache = torch.randn((4, 2, 2, 2, 4), dtype=torch.float16)
    original = kv_cache[[1, 2]].clone()
    expected_adapter = KiviAdapter(cfg)
    expected_quantized = expected_adapter.quantize(original)
    expected = expected_adapter.dequantize(
        expected_quantized.symbols, expected_quantized.metadata
    )
    worker.register_kv_caches({"layer_0": kv_cache})

    save_metadata = OrigamiConnectorMetadata(
        reqs_to_save={
            "req-1": OrigamiSaveRequest(
                request_id="req-1",
                cache_key="cache-worker-kivi",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                token_start=0,
            )
        }
    )
    worker.start_load_kv(save_metadata)
    worker.save_kv_layer("layer_0", kv_cache, None)
    worker.wait_for_save()

    stored_payload = store.get("cache-worker-kivi")
    layer_payload = stored_payload.layer_payloads["layer_0"]
    assert stored_payload.quantizer == "kivi"
    assert layer_payload.quant_metadata["format"] == "kivi_structured_blob"
    assert layer_payload.chunks

    kv_cache[[1, 2]] = 0
    restore_metadata = OrigamiConnectorMetadata(
        reqs_to_restore={
            "req-1": OrigamiRestoreRequest(
                request_id="req-1",
                cache_key="cache-worker-kivi",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                lossless_path="cpu",
            )
        }
    )
    worker.start_load_kv(restore_metadata)
    worker.wait_for_layer_load("layer_0")
    worker.shutdown()

    assert torch.equal(kv_cache[[1, 2]], expected)


def test_kivi_cuda_direct_restore_matches_cpu_fallback() -> None:
    if not torch.cuda.is_available():
        return
    cfg = {"bits": 2, "group_size": 2, "sink_tokens": 1}
    adapter = KiviAdapter(cfg)
    original = torch.randn((2, 2, 2, 1, 4), dtype=torch.float16)
    quantized = adapter.quantize(original)
    expected = adapter.dequantize(quantized.symbols, quantized.metadata)
    kv_cache = torch.zeros((4, 2, 2, 1, 4), dtype=torch.float16, device="cuda")

    event = adapter.dequantize(
        quantized.symbols,
        quantized.metadata,
        dst_cache=kv_cache,
        block_ids=[1, 2],
    )
    if isinstance(event, torch.cuda.Event):
        event.synchronize()

    assert torch.equal(kv_cache[[1, 2]].cpu(), expected)
    assert {
        row["type"] for row in adapter.last_profile
    } >= {"origami_kivi_dequantize_to_kv_cuda"}


def test_worker_kivi_nvcomp_backend_roundtrip_when_available(tmp_path) -> None:
    if not torch.cuda.is_available():
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.gpu_nvcomp import (
            NvcompCodec,
        )

        NvcompCodec("nvcomp_deflate")
    except Exception:
        return

    metrics_dir = tmp_path / "metrics"
    store = InMemoryOrigamiStore(f"worker-kivi-nvcomp-{uuid4()}")
    cfg = {"bits": 2, "group_size": 2, "sink_tokens": 1}
    worker = OrigamiConnectorWorker(
        OrigamiConfig(
            quantizer="kivi",
            quantizer_config=cfg,
            lossless_cpu_backend="zlib",
            nvcomp_backend="nvcomp_deflate",
            metrics_dir=str(metrics_dir),
            qat_threads=1,
        ),
        store,
        SimpleNamespace(block_size=2),
    )
    kv_cache = torch.randn((4, 2, 2, 1, 4), dtype=torch.float16, device="cuda")
    original = kv_cache[[1, 2]].detach().cpu()
    expected_adapter = KiviAdapter(cfg)
    expected_quantized = expected_adapter.quantize(original)
    expected = expected_adapter.dequantize(
        expected_quantized.symbols, expected_quantized.metadata
    )
    worker.register_kv_caches({"layer_0": kv_cache})

    worker.start_load_kv(
        OrigamiConnectorMetadata(
            reqs_to_save={
                "req-1": OrigamiSaveRequest(
                    request_id="req-1",
                    cache_key="cache-worker-kivi-nvcomp",
                    block_ids_per_group=((1, 2),),
                    num_tokens=4,
                    token_start=0,
                )
            }
        )
    )
    worker.save_kv_layer("layer_0", kv_cache, None)
    worker.wait_for_save()
    layer_payload = store.get("cache-worker-kivi-nvcomp").layer_payloads["layer_0"]
    assert {chunk.codec for chunk in layer_payload.chunks} == {"nvcomp_deflate"}

    kv_cache[[1, 2]] = 0
    worker.start_load_kv(
        OrigamiConnectorMetadata(
            reqs_to_restore={
                "req-1": OrigamiRestoreRequest(
                    request_id="req-1",
                    cache_key="cache-worker-kivi-nvcomp",
                    block_ids_per_group=((1, 2),),
                    num_tokens=4,
                    lossless_path="cpu",
                )
            }
        )
    )
    worker.wait_for_layer_load("layer_0")
    worker.shutdown()

    assert torch.equal(kv_cache[[1, 2]].detach().cpu(), expected)
    rows = [
        json.loads(line)
        for line in (metrics_dir / "origami_metrics.jsonl").read_text().splitlines()
    ]
    metric_types = {row["type"] for row in rows}
    assert "kivi_nvcomp_decompress_cuda" in metric_types
    assert "origami_kivi_dequantize_to_kv_cuda" in metric_types


def test_scheduler_restore_metadata_and_incremental_save_metadata() -> None:
    cfg = OrigamiConfig(gpu_lossless_ratio=50)
    scheduler = OrigamiConnectorScheduler(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=16)), cfg
    )
    request = SimpleNamespace(
        request_id="req-1",
        num_tokens=64,
        kv_transfer_params={
            "origami_resume_prefill": True,
            "origami_cache_key": "cache-1",
            "origami_num_tokens": 32,
        },
    )
    blocks = SimpleNamespace(get_block_ids=lambda: [[10, 11, 12], [20, 21, 22]])

    matched, async_load = scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, blocks, matched)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req-1", block_ids=[[10, 11, 12], [20, 21, 22]])
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"req-1": 48},
    )

    metadata = scheduler.build_connector_meta(output)

    assert matched == 32
    assert async_load is False
    assert metadata.reqs_to_restore["req-1"].cache_key == "cache-1"
    assert metadata.reqs_to_restore["req-1"].block_ids_per_group == (
        (10, 11),
        (20, 21),
    )
    assert metadata.reqs_to_restore["req-1"].lossless_path == "gpu"
    assert metadata.reqs_to_save["req-1"].num_tokens == 16
    assert metadata.reqs_to_save["req-1"].token_start == 32


def test_scheduler_resume_subtracts_local_prefix_cache_hits() -> None:
    cfg = OrigamiConfig(gpu_lossless_ratio=0)
    scheduler = OrigamiConnectorScheduler(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=16)), cfg
    )
    request = SimpleNamespace(
        request_id="req-local-hit",
        num_tokens=96,
        kv_transfer_params={
            "origami_resume_prefill": True,
            "origami_cache_key": "cache-local-hit",
            "origami_num_tokens": 64,
        },
    )
    blocks = SimpleNamespace(get_block_ids=lambda: [[10, 11], [20, 21]])

    matched, async_load = scheduler.get_num_new_matched_tokens(request, 48)
    scheduler.update_state_after_alloc(request, blocks, matched)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req-local-hit", block_ids=[[10, 11], [20, 21]])
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"req-local-hit": 80},
    )

    metadata = scheduler.build_connector_meta(output)

    assert matched == 16
    assert async_load is False
    assert metadata.reqs_to_restore["req-local-hit"].num_tokens == 16
    assert metadata.reqs_to_restore["req-local-hit"].block_ids_per_group == (
        (10,),
        (20,),
    )
    assert metadata.reqs_to_save["req-local-hit"].token_start == 64
    assert metadata.reqs_to_save["req-local-hit"].num_tokens == 16


def test_scheduler_save_prefill_metadata_without_restore() -> None:
    cfg = OrigamiConfig(gpu_lossless_ratio=100)
    scheduler = OrigamiConnectorScheduler(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=16)), cfg
    )
    request = SimpleNamespace(
        request_id="warmup-1",
        num_tokens=64,
        kv_transfer_params={
            "origami_save_prefill": True,
            "origami_cache_key": "cache-warmup",
            "origami_num_tokens": 64,
        },
    )

    matched, async_load = scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, SimpleNamespace(), matched)
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="warmup-1",
                            block_ids=[[10, 11, 12, 13], [20, 21, 22, 23]])
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"warmup-1": 64},
    )

    metadata = scheduler.build_connector_meta(output)

    assert matched == 0
    assert async_load is False
    assert metadata.reqs_to_restore == {}
    assert metadata.reqs_to_save["warmup-1"] == OrigamiSaveRequest(
        request_id="warmup-1",
        cache_key="cache-warmup",
        block_ids_per_group=((10, 11, 12, 13), (20, 21, 22, 23)),
        num_tokens=64,
        token_start=0,
    )


def test_scheduler_save_prefill_flag_takes_precedence_over_restore_keys() -> None:
    cfg = OrigamiConfig(gpu_lossless_ratio=100)
    scheduler = OrigamiConnectorScheduler(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=16)), cfg
    )
    request = SimpleNamespace(
        request_id="warmup-2",
        num_tokens=48,
        kv_transfer_params={
            "origami_save_prefill": True,
            "origami_resume_prefill": True,
            "origami_cache_key": "cache-warmup-2",
            "origami_num_tokens": 32,
        },
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="warmup-2",
                            block_ids=[[30, 31, 32], [40, 41, 42]])
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"warmup-2": 48},
    )

    matched, async_load = scheduler.get_num_new_matched_tokens(request, 0)
    metadata = scheduler.build_connector_meta(output)

    assert matched == 0
    assert async_load is False
    assert metadata.reqs_to_restore == {}
    assert metadata.reqs_to_save["warmup-2"].cache_key == "cache-warmup-2"
    assert metadata.reqs_to_save["warmup-2"].num_tokens == 32
    assert metadata.reqs_to_save["warmup-2"].token_start == 0


def test_worker_save_restore_roundtrip_with_mock_quantizer_and_zlib() -> None:
    store = InMemoryOrigamiStore(f"worker-{uuid4()}")
    worker = OrigamiConnectorWorker(
        OrigamiConfig(
            quantizer="mock_int4",
            lossless_cpu_backend="zlib",
            qat_threads=1,
        ),
        store,
    )
    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    original = kv_cache[[1, 2]].clone()
    worker.register_kv_caches({"layer_0": kv_cache})

    save_metadata = OrigamiConnectorMetadata(
        reqs_to_save={
            "req-1": OrigamiSaveRequest(
                request_id="req-1",
                cache_key="cache-worker",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                token_start=8,
            )
        }
    )
    worker.start_load_kv(save_metadata)
    worker.save_kv_layer("layer_0", kv_cache, None)
    worker.wait_for_save()

    assert store.contains("cache-worker")
    stored_payload = store.get("cache-worker")
    assert stored_payload.token_start == 8
    assert stored_payload.token_count == 4
    layer_payload = stored_payload.layer_payloads["layer_0"]
    native_layout = layer_payload.quant_metadata["origami_native_layout"]
    assert native_layout["bits"] == 8
    assert native_layout["source_layout"] == ["head", "token", "head_dim"]
    assert native_layout["storage_layout"] == ["layer", "head", "head_dim", "token"]
    assert native_layout["chunk_specs"]
    assert layer_payload.chunks[0].unpacked_bytes > 0

    kv_cache[[1, 2]] = 0
    restore_metadata = OrigamiConnectorMetadata(
        reqs_to_restore={
            "req-1": OrigamiRestoreRequest(
                request_id="req-1",
                cache_key="cache-worker",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                lossless_path="cpu",
            )
        }
    )
    worker.start_load_kv(restore_metadata)
    worker.wait_for_layer_load("layer_0")
    worker.shutdown()

    assert torch.equal(kv_cache[[1, 2]], original)


def test_worker_save_restore_roundtrip_with_raw_bytes_and_zlib() -> None:
    store = InMemoryOrigamiStore(f"worker-raw-bytes-{uuid4()}")
    worker = OrigamiConnectorWorker(
        OrigamiConfig(
            quantizer="raw_bytes",
            lossless_cpu_backend="zlib",
            qat_threads=1,
        ),
        store,
    )
    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.float16).reshape(4, 2, 3)
    original = kv_cache[[1, 2]].clone()
    worker.register_kv_caches({"layer_0": kv_cache})

    save_metadata = OrigamiConnectorMetadata(
        reqs_to_save={
            "req-1": OrigamiSaveRequest(
                request_id="req-1",
                cache_key="cache-worker-raw-bytes",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                token_start=8,
            )
        }
    )
    worker.start_load_kv(save_metadata)
    worker.save_kv_layer("layer_0", kv_cache, None)
    worker.wait_for_save()

    stored_payload = store.get("cache-worker-raw-bytes")
    assert stored_payload.quantizer == "raw_bytes"
    layer_payload = stored_payload.layer_payloads["layer_0"]
    assert layer_payload.quant_metadata["format"] == "raw_bytes"
    assert layer_payload.quant_metadata["origami_bits"] == 8
    assert layer_payload.quant_metadata["origami_native_layout"]["chunk_specs"]

    kv_cache[[1, 2]] = 0
    restore_metadata = OrigamiConnectorMetadata(
        reqs_to_restore={
            "req-1": OrigamiRestoreRequest(
                request_id="req-1",
                cache_key="cache-worker-raw-bytes",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                lossless_path="cpu",
            )
        }
    )
    worker.start_load_kv(restore_metadata)
    worker.wait_for_layer_load("layer_0")
    worker.shutdown()

    assert torch.equal(kv_cache[[1, 2]], original)


def test_worker_save_restore_roundtrip_with_bundle_store(tmp_path) -> None:
    store = LocalFileOrigamiStore(tmp_path / "store", artifact_format="bundle_v1")
    metrics_dir = tmp_path / "metrics"
    worker = OrigamiConnectorWorker(
        OrigamiConfig(
            quantizer="raw_bytes",
            lossless_cpu_backend="zlib",
            artifact_format="bundle_v1",
            qat_threads=1,
            metrics_dir=str(metrics_dir),
        ),
        store,
    )
    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.float16).reshape(4, 2, 3)
    original = kv_cache[[1, 2]].clone()
    worker.register_kv_caches({"layer_0": kv_cache})

    save_metadata = OrigamiConnectorMetadata(
        reqs_to_save={
            "req-1": OrigamiSaveRequest(
                request_id="req-1",
                cache_key="cache-worker-bundle",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                token_start=0,
            )
        }
    )
    worker.start_load_kv(save_metadata)
    worker.save_kv_layer("layer_0", kv_cache, None)
    worker.wait_for_save()

    assert (tmp_path / "store" / "cache-worker-bundle" / "manifest.json").exists()
    assert (tmp_path / "store" / "cache-worker-bundle" / "payload.bin").exists()

    kv_cache[[1, 2]] = 0
    restore_metadata = OrigamiConnectorMetadata(
        reqs_to_restore={
            "req-1": OrigamiRestoreRequest(
                request_id="req-1",
                cache_key="cache-worker-bundle",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                lossless_path="cpu",
            )
        }
    )
    worker.start_load_kv(restore_metadata)
    worker.wait_for_layer_load("layer_0")
    worker.shutdown()

    assert torch.equal(kv_cache[[1, 2]], original)
    rows = [
        json.loads(line)
        for line in (metrics_dir / "origami_metrics.jsonl").read_text().splitlines()
    ]
    assert "lossless_decompress" in {row["type"] for row in rows}
    assert "raw_bytes_view" in {row["type"] for row in rows}
    assert "h2d_copy" in {row["type"] for row in rows}


def test_worker_restore_loads_artifact_once_per_request() -> None:
    class CountingStore(InMemoryOrigamiStore):
        def __init__(self):
            super().__init__(f"counting-{uuid4()}")
            self.prepare_calls = 0

        def prepare_for_restore(self, cache_key: str, **kwargs):
            self.prepare_calls += 1
            return super().prepare_for_restore(cache_key, **kwargs)

    store = CountingStore()
    worker = OrigamiConnectorWorker(
        OrigamiConfig(
            quantizer="raw_bytes",
            lossless_cpu_backend="zlib",
            qat_threads=1,
        ),
        store,
    )
    layer_0 = torch.arange(4 * 2 * 3, dtype=torch.float16).reshape(4, 2, 3)
    layer_1 = (torch.arange(4 * 2 * 3, dtype=torch.float16) + 100).reshape(4, 2, 3)
    original_0 = layer_0[[1, 2]].clone()
    original_1 = layer_1[[1, 2]].clone()
    worker.register_kv_caches({"layer_0": layer_0, "layer_1": layer_1})

    save_metadata = OrigamiConnectorMetadata(
        reqs_to_save={
            "req-1": OrigamiSaveRequest(
                request_id="req-1",
                cache_key="cache-once",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                token_start=0,
            )
        }
    )
    worker.start_load_kv(save_metadata)
    worker.save_kv_layer("layer_0", layer_0, None)
    worker.save_kv_layer("layer_1", layer_1, None)
    worker.wait_for_save()

    layer_0[[1, 2]] = 0
    layer_1[[1, 2]] = 0
    restore_metadata = OrigamiConnectorMetadata(
        reqs_to_restore={
            "req-1": OrigamiRestoreRequest(
                request_id="req-1",
                cache_key="cache-once",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                lossless_path="cpu",
            )
        }
    )
    worker.start_load_kv(restore_metadata)
    worker.wait_for_layer_load("layer_0")
    worker.wait_for_layer_load("layer_1")
    worker.shutdown()

    assert store.prepare_calls == 1
    assert torch.equal(layer_0[[1, 2]], original_0)
    assert torch.equal(layer_1[[1, 2]], original_1)



def test_native_bitpack_layout_roundtrip_for_1_to_8_bits() -> None:
    token_count = 5
    num_heads = 2
    head_dim = 7
    specs = torch.tensor(
        [
            [0, 1, 0, head_dim, 0, token_count],
            [1, 2, 0, head_dim, 0, token_count],
        ],
        dtype=torch.int64,
    )

    for bits in range(1, 9):
        mask = (1 << bits) - 1
        symbols = (
            torch.arange(token_count * num_heads * head_dim, dtype=torch.int64) & mask
        ).to(torch.uint8)
        packed = native_cpu.pack_head_channel_chunks(
            symbols,
            bits,
            token_count,
            num_heads,
            head_dim,
            specs,
        )
        restored = native_cpu.unpack_head_channel_chunks(
            packed,
            bits,
            token_count,
            num_heads,
            head_dim,
            specs,
        )

        assert torch.equal(restored, symbols)


def test_native_cuda_bitpack_matches_cpu_when_available() -> None:
    if not torch.cuda.is_available():
        return

    token_count = 4
    num_heads = 3
    head_dim = 5
    specs = torch.tensor(
        [
            [0, 2, 0, head_dim, 0, token_count],
            [2, 3, 0, 2, 0, token_count],
            [2, 3, 2, head_dim, 0, token_count],
        ],
        dtype=torch.int64,
    )
    base = torch.arange(
        token_count * num_heads * head_dim,
        dtype=torch.uint8,
    ).reshape(token_count, num_heads, head_dim)

    for layout in (
        ["token", "head", "head_dim"],
        ["head", "token", "head_dim"],
        ["head", "head_dim", "token"],
        ["layer", "token", "head", "head_dim"],
    ):
        if "layer" in layout:
            source = base.unsqueeze(0)
            source_shape = [1, token_count, num_heads, head_dim]
        else:
            source = base.permute(
                [{"token": 0, "head": 1, "head_dim": 2}[axis] for axis in layout]
            ).contiguous()
            source_shape = list(source.shape)
        symbols = source.reshape(-1).contiguous()
        for bits in range(1, 9):
            masked = (symbols.to(torch.int64) & ((1 << bits) - 1)).to(torch.uint8)
            packed = native_cpu.pack_canonical_storage_chunks(
                masked,
                bits,
                source_shape,
                layout,
                specs,
            )
            cpu_restored = native_cpu.unpack_canonical_storage_chunks(
                packed,
                bits,
                source_shape,
                layout,
                specs,
            )
            gpu_restored = native_gpu.unpack_canonical_storage_chunks(
                packed,
                bits,
                source_shape,
                layout,
                specs,
            )
            assert torch.equal(gpu_restored.cpu(), cpu_restored)


def test_cachegen_bitpacked_blob_dequantizes_on_cpu_and_gpu_when_available() -> None:
    kv = torch.randn(2, 6, 2, 8, dtype=torch.float16)
    cpu_adapter = CacheGenAdapter({"dequant_device": "cpu"})
    quantized = cpu_adapter.quantize(kv, layer_group="layer_0")

    assert quantized.metadata["format"] == "cachegen_bitpacked_blob"
    assert 1 <= int(quantized.metadata["key_bits"]) <= 8
    assert 1 <= int(quantized.metadata["value_bits"]) <= 8
    restored_cpu = cpu_adapter.dequantize(quantized.symbols, quantized.metadata)
    assert restored_cpu.shape == kv.shape
    assert restored_cpu.dtype == kv.dtype

    if not torch.cuda.is_available():
        return
    gpu_adapter = CacheGenAdapter({"dequant_device": "gpu"})
    restored_gpu = gpu_adapter.dequantize(quantized.symbols, quantized.metadata)
    assert restored_gpu.device.type == "cuda"
    assert torch.allclose(restored_gpu.cpu(), restored_cpu, atol=1e-3, rtol=1e-3)




def test_native_canonical_storage_order_for_source_layouts() -> None:
    token_count = 3
    num_heads = 2
    head_dim = 4
    base = torch.arange(token_count * num_heads * head_dim, dtype=torch.uint8).reshape(
        token_count,
        num_heads,
        head_dim,
    )
    specs = torch.tensor(
        [[0, num_heads, 0, head_dim, 0, token_count]],
        dtype=torch.int64,
    )

    for layout in (
        ["token", "head", "head_dim"],
        ["head", "token", "head_dim"],
        ["head", "head_dim", "token"],
    ):
        source = base.permute([{"token": 0, "head": 1, "head_dim": 2}[axis] for axis in layout])
        packed = native_cpu.pack_canonical_storage_chunks(
            source.contiguous().reshape(-1),
            8,
            list(source.shape),
            layout,
            specs,
        )
        expected = base.permute(1, 2, 0).contiguous().reshape(-1)
        assert torch.equal(packed[0], expected)
        restored = native_cpu.unpack_canonical_storage_chunks(
            packed,
            8,
            list(source.shape),
            layout,
            specs,
        )
        assert torch.equal(restored, source.contiguous().reshape(-1))


def test_native_canonical_storage_rank4_layer_axis_roundtrip() -> None:
    token_count = 3
    num_heads = 2
    head_dim = 4
    base = torch.arange(token_count * num_heads * head_dim, dtype=torch.uint8).reshape(
        1,
        token_count,
        num_heads,
        head_dim,
    )
    layout = ["layer", "token", "head", "head_dim"]
    specs = torch.tensor(
        [[0, num_heads, 0, head_dim, 0, token_count]],
        dtype=torch.int64,
    )

    packed = native_cpu.pack_canonical_storage_chunks(
        base.reshape(-1),
        8,
        list(base.shape),
        layout,
        specs,
    )
    expected = base.squeeze(0).permute(1, 2, 0).contiguous().reshape(-1)
    restored = native_cpu.unpack_canonical_storage_chunks(
        packed,
        8,
        list(base.shape),
        layout,
        specs,
    )

    assert torch.equal(packed[0], expected)
    assert torch.equal(restored, base.reshape(-1))


def test_worker_requires_quantizer_symbol_shape_and_layout_metadata() -> None:
    worker = OrigamiConnectorWorker(
        OrigamiConfig(lossless_cpu_backend="zlib", qat_threads=1),
        InMemoryOrigamiStore(f"missing-layout-{uuid4()}"),
    )

    try:
        worker._pack_quantized_symbols(
            torch.arange(12, dtype=torch.uint8),
            {"origami_symbol_shape": [3, 2, 2]},
            "layer_0",
        )
    except ValueError as exc:
        assert "origami_symbol_layout" in str(exc)
    else:
        raise AssertionError("missing origami_symbol_layout should fail")

    try:
        worker._pack_quantized_symbols(
            torch.arange(12, dtype=torch.uint8),
            {"origami_symbol_layout": ["token", "head", "head_dim"]},
            "layer_0",
        )
    except ValueError as exc:
        assert "origami_symbol_shape" in str(exc)
    else:
        raise AssertionError("missing origami_symbol_shape should fail")


def test_native_bitpack_combined_head_channel_specs_roundtrip() -> None:
    token_count = 4
    num_heads = 3
    head_dim = 6
    specs = torch.tensor(
        [
            [0, 2, 0, head_dim, 0, token_count],
            [2, 3, 0, 3, 0, token_count],
            [2, 3, 3, head_dim, 0, token_count],
        ],
        dtype=torch.int64,
    )
    symbols = (
        torch.arange(token_count * num_heads * head_dim, dtype=torch.int64) & 0x0F
    ).to(torch.uint8)

    packed = native_cpu.pack_head_channel_chunks(
        symbols,
        4,
        token_count,
        num_heads,
        head_dim,
        specs,
    )
    restored = native_cpu.unpack_head_channel_chunks(
        packed,
        4,
        token_count,
        num_heads,
        head_dim,
        specs,
    )

    assert torch.equal(restored, symbols)


def test_cpu_zlib_many_chunk_raw_deflate_roundtrip() -> None:
    codec = CpuLosslessCodec(backend="zlib")
    originals = [
        torch.arange(1027, dtype=torch.int32).view(torch.uint8),
        torch.arange(2049, dtype=torch.int16).view(torch.uint8),
    ]

    compressed = codec.compress_many(originals)
    restored = codec.decompress_many(compressed, [int(t.numel()) for t in originals])

    assert len(restored) == len(originals)
    for got, expected in zip(restored, originals):
        assert torch.equal(got, expected)


def test_qat_backend_roundtrip_when_available() -> None:
    try:
        codec = CpuLosslessCodec(
            backend="qat",
            qat_inflight=2,
            qat_batch=1,
            qat_max_instances=1,
        )
    except RuntimeError:
        return

    originals = [
        (torch.arange(512 << 10, dtype=torch.int64) & 0xFF).to(torch.uint8),
        (torch.arange(1 << 20, dtype=torch.int64) & 0xFF).to(torch.uint8),
    ]
    output_bytes = [int(t.numel()) for t in originals]
    compressed = codec.compress_many(originals)
    restored = codec.decompress_many(compressed, output_bytes)

    for got, expected in zip(restored, originals):
        assert torch.equal(got, expected)

    prepared = native_cpu.prepare_raw_deflate_many(
        compressed,
        output_bytes,
        max_instances=1,
    )
    prepared_restored = native_cpu.decompress_prepared_raw_deflate_many(
        prepared,
        inflight=2,
        batch=1,
        max_instances=1,
    )
    for got, expected in zip(prepared_restored, originals):
        assert torch.equal(got, expected)



def test_worker_save_restore_roundtrip_with_mock_quantizer_and_qat_when_available() -> None:
    try:
        config = OrigamiConfig(
            quantizer="mock_int4",
            lossless_cpu_backend="qat",
            qat_threads=1,
            qat_inflight=2,
            qat_batch=1,
            qat_max_instances=1,
        )
        store = InMemoryOrigamiStore(f"worker-qat-{uuid4()}")
        worker = OrigamiConnectorWorker(config, store)
    except RuntimeError:
        return

    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    original = kv_cache[[1, 2]].clone()
    worker.register_kv_caches({"layer_0": kv_cache})

    save_metadata = OrigamiConnectorMetadata(
        reqs_to_save={
            "req-1": OrigamiSaveRequest(
                request_id="req-1",
                cache_key="cache-worker-qat",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                token_start=8,
            )
        }
    )
    worker.start_load_kv(save_metadata)
    worker.save_kv_layer("layer_0", kv_cache, None)
    worker.wait_for_save()

    kv_cache[[1, 2]] = 0
    restore_metadata = OrigamiConnectorMetadata(
        reqs_to_restore={
            "req-1": OrigamiRestoreRequest(
                request_id="req-1",
                cache_key="cache-worker-qat",
                block_ids_per_group=((1, 2),),
                num_tokens=4,
                lossless_path="cpu",
            )
        }
    )
    worker.start_load_kv(restore_metadata)
    worker.wait_for_layer_load("layer_0")
    worker.shutdown()

    assert torch.equal(kv_cache[[1, 2]], original)


def test_factory_registers_independent_reuse_connectors() -> None:
    expected = {
        "RawKVReuseConnector": "RawKVReuseConnector",
        "CacheGenReuseConnector": "CacheGenReuseConnector",
        "OrigamiReuseConnector": "OrigamiReuseConnector",
    }

    for name, class_name in expected.items():
        connector_cls = KVConnectorFactory.get_connector_class_by_name(name)
        assert connector_cls.__name__ == class_name


def test_benchmark_controlled_read_uses_token_bucket_per_chunk(tmp_path, monkeypatch) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"0123456789")
    calls: list[int] = []

    class Recorder:
        def consume(self, num_bytes: int) -> None:
            calls.append(num_bytes)

    monkeypatch.setattr(
        TokenBucket,
        "from_gbps",
        classmethod(lambda cls, gbps: Recorder()),
    )
    metrics_path = tmp_path / "metrics.jsonl"

    data = controlled_read(
        path,
        request_id="req",
        cache_key="cache",
        system="raw_kv_reuse",
        bandwidth_gbps=10,
        metrics_path=metrics_path,
        chunk_bytes=4,
    )

    assert data == b"0123456789"
    assert calls == [4, 4, 2]
    metric = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert metric["type"] == "ssd_read"
    assert metric["bytes_read"] == 10


def test_benchmark_copy_kv_to_hbm_records_cpu_copy_metrics(tmp_path) -> None:
    kv_cache = torch.zeros(2, 4, 3, dtype=torch.float32)
    value = torch.ones(2, 2, 3, dtype=torch.float32)
    metrics_path = tmp_path / "h2d.jsonl"

    metric = copy_kv_to_hbm(
        kv_cache=kv_cache,
        block_ids=[1, 3],
        value=value,
        request_id="req",
        cache_key="cache",
        system="raw_kv_reuse",
        layer_name="layer_0",
        metrics_path=metrics_path,
    )

    assert torch.equal(kv_cache[:, [1, 3], :], value)
    assert metric.bytes_copied == value.numel() * value.element_size()
    assert metric.h2d_ms >= 0
    written = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["type"] == "h2d_copy"
    assert written["h2d_gbps"] >= 0


def test_benchmark_metadata_uses_transfer_id_independent_of_request_id() -> None:
    metadata = ReuseConnectorMetadata(system="cachegen")

    metadata.add_request(
        request_id="vllm-auto-7",
        transfer_id="dataset-row-42",
        remote_address="127.0.0.1:29581",
        token_ids=[1, 2, 3],
        block_ids=[10, 11],
    )

    request = metadata.requests[0]
    assert request.request_id == "vllm-auto-7"
    assert request.transfer_id == "dataset-row-42"
    assert request.remote_address == "127.0.0.1:29581"
    assert request.block_ids.tolist() == [10, 11]


def test_benchmark_windows_overlap_detects_pcie_contention() -> None:
    assert not windows_overlap(1.0, 2.0, 2.01, 3.0)
    assert windows_overlap(1.0, 2.5, 2.0, 3.0)


def test_benchmark_kernel_classifier_uses_nvtx_ranges() -> None:
    totals = classify_kernel_records(
        kernels=[
            KernelRecord(name="attn", start_ns=10, end_ns=30),
            KernelRecord(name="dq", start_ns=110, end_ns=150),
            KernelRecord(name="dc", start_ns=210, end_ns=260),
            KernelRecord(name="misc", start_ns=400, end_ns=450),
        ],
        ranges=[
            KernelRange(name="prefill_attention", start_ns=0, end_ns=100),
            KernelRange(name="origami:dequant", start_ns=100, end_ns=200),
            KernelRange(name="origami:decompress", start_ns=200, end_ns=300),
        ],
    )

    assert totals["attention"] == 0.00002
    assert totals["dequant"] == 0.00004
    assert totals["decompress"] == 0.00005
    assert totals["other_gpu_kernel"] == 0.00005


def test_prefill_benchmark_uses_production_origami_connector() -> None:
    script_dir = Path(__file__).resolve().parents[4] / "scripts" / "origami"
    script_path = script_dir / "run_prefill_reuse_benchmark.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib_util.spec_from_file_location(
            "run_prefill_reuse_benchmark_test", script_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(script_dir))
        except ValueError:
            pass

    assert module.PREFILL_SYSTEM_TO_CONNECTOR["origami"] == "OrigamiConnector"
    assert module.PREFILL_SYSTEM_TO_CONNECTOR["raw_kv_reuse"] == "RawKVReuseConnector"
    assert module.PREFILL_SYSTEM_TO_CONNECTOR["cachegen"] == "CacheGenReuseConnector"
