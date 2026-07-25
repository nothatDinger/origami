# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.fused_attention import (
    FusedLayerPayload,
    reference_fused_prefix_attention,
    run_fused_prefix_attention,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    OrigamiRestoreRequest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kivi_adapter import (
    KiviAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kvquant_adapter import (
    KVQuantAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.worker import (
    OrigamiConnectorWorker,
)

pytestmark = pytest.mark.cuda


def _run_case(
    quantizer: str,
    bits: int,
    dtype: torch.dtype,
    query_tokens: int,
) -> None:
    torch.manual_seed(1234 + bits + query_tokens)
    prefix_tokens = 19
    kv_heads = 2
    query_heads = 4
    head_dim = 128
    if quantizer == "kivi":
        adapter = KiviAdapter({
            "bits": bits,
            "group_size": 7,
            "sink_tokens": 3,
            "dequant_device": "cpu",
        })
    else:
        adapter = KVQuantAdapter({"bits": bits, "dequant_device": "cpu"})

    prefix = torch.randn(
        2, prefix_tokens, kv_heads, head_dim, dtype=dtype
    )
    quantized = adapter.quantize(prefix)
    symbols = quantized.symbols.cuda(non_blocking=True)
    ready = torch.cuda.Event()
    ready.record()
    payload = FusedLayerPayload(
        request_id="request",
        cache_key="cache",
        layer_name="model.layers.0.self_attn.attn",
        quantizer=quantizer,
        symbols=symbols,
        metadata=quantized.metadata,
        ready_event=ready,
    )

    sequence_length = prefix_tokens + query_tokens
    block_size = 16
    num_blocks = math.ceil(sequence_length / block_size)
    kv_cache = torch.zeros(
        2,
        num_blocks,
        block_size,
        kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    suffix_key = torch.randn(
        query_tokens, kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    suffix_value = torch.randn_like(suffix_key)
    for token_index in range(query_tokens):
        position = prefix_tokens + token_index
        kv_cache[0, position // block_size, position % block_size] = suffix_key[
            token_index
        ]
        kv_cache[1, position // block_size, position % block_size] = suffix_value[
            token_index
        ]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    query = torch.randn(
        query_tokens, query_heads, head_dim, dtype=dtype, device="cuda"
    )
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    run_fused_prefix_attention(
        payload,
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        query_start_position=prefix_tokens,
        sequence_length=sequence_length,
        softmax_scale=scale,
        output=output,
    )
    reference = reference_fused_prefix_attention(
        payload,
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        query_start_position=prefix_tokens,
        sequence_length=sequence_length,
        softmax_scale=scale,
    )
    tolerance = 1.5e-2 if dtype == torch.float16 else 3e-2
    torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "quantizer,bits",
    [
        ("kivi", 2),
        ("kivi", 3),
        ("kivi", 4),
        ("kivi", 8),
        ("kvquant", 2),
        ("kvquant", 3),
        ("kvquant", 4),
        ("kvquant", 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_origami_fused_attention_codecs(quantizer, bits, dtype):
    _run_case(quantizer, bits, dtype, query_tokens=1)


@pytest.mark.parametrize("query_tokens", [16, 128, 1024])
def test_origami_fused_attention_ragged_query_lengths(query_tokens):
    _run_case("kivi", 4, torch.float16, query_tokens=query_tokens)


def test_origami_fused_payload_refcounts_and_retention():
    worker = OrigamiConnectorWorker.__new__(OrigamiConnectorWorker)
    worker.fused_attention_enabled = True
    worker.quantizer = SimpleNamespace(quantizer_id="kivi")
    worker.kv_caches = {"layer": torch.empty(0, device="cuda")}
    worker._fused_lock = Lock()
    worker._fused_payloads = {}
    worker._request_fused_keys = {}

    ready = torch.cuda.Event()
    ready.record()
    payload = FusedLayerPayload(
        request_id="request-a",
        cache_key="cache",
        layer_name="layer",
        quantizer="kivi",
        symbols=torch.empty(1, dtype=torch.uint8, device="cuda"),
        metadata={"token_count": 1},
        ready_event=ready,
    )
    key = ("cache", "layer", "kivi")
    worker._fused_payloads[key] = payload
    worker._request_fused_keys["request-a"] = {"layer": key}

    restore_a = OrigamiRestoreRequest(
        request_id="request-a",
        cache_key="cache",
        block_ids_per_group=((0,),),
        num_tokens=1,
    )
    assert worker._claim_cached_layer_events(restore_a) == {"layer": ready}
    assert payload.ref_count == 1

    restore_b = OrigamiRestoreRequest(
        request_id="request-b",
        cache_key="cache",
        block_ids_per_group=((1,),),
        num_tokens=1,
    )
    assert worker._claim_cached_layer_events(restore_b) == {"layer": ready}
    assert payload.ref_count == 2

    worker.release_fused_requests({"request-b"})
    assert payload.ref_count == 1
    assert key in worker._fused_payloads
    worker.release_fused_requests({"request-a"})
    assert key not in worker._fused_payloads
