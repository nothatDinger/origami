# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import OrigamiConfig
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.chunking import (
    plan_head_channel_chunks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
    native_cpu,
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
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.worker import (
    OrigamiConnectorWorker,
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


def test_cpu_zlib_raw_deflate_roundtrip() -> None:
    codec = CpuLosslessCodec(backend="zlib")
    original = torch.arange(4096, dtype=torch.int32).view(torch.uint8)

    compressed = codec.compress(original)
    restored = codec.decompress(compressed, output_bytes=original.numel())

    assert torch.equal(restored, original)


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



def test_native_bitpack_layout_roundtrip_for_2_4_8_bits() -> None:
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

    for bits in (2, 4, 8):
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
