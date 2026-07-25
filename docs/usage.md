# Using Origami

This page contains the common commands for checking the environment, validating
the lossless path, and configuring the vLLM connector.

## Check QAT

```bash
python3 scripts/check_qat.py
python3 scripts/check_qat.py --json
python3 scripts/check_qat.py --try-extension
```

Use `--require` when the command should fail if QAT is not ready:

```bash
python3 scripts/check_qat.py --try-extension --require
```

## Test the Lossless Path

QAT path:

```bash
ORIGAMI_QAT_CODEC=qat_codec \
python3 scripts/test_lossless.py --backend qat --chunks 8 --chunk-bytes 1048576
```

Software path:

```bash
python3 scripts/test_lossless.py --backend zlib --chunks 8 --chunk-bytes 262144
```

Complete local test entry:

```bash
bash scripts/run_tests.sh
```

Allow the software path on machines without QAT:

```bash
ORIGAMI_REQUIRE_QAT=0 bash scripts/run_tests.sh
```

## vLLM Connector Configuration

The connector is selected through vLLM's KV-transfer config. The following
configuration writes Origami artifacts to `/tmp/origami-store` and uses the
software lossless backend:

```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --kv-transfer-config '{
    "kv_connector": "OrigamiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "origami_store_uri": "file:///tmp/origami-store",
      "origami_artifact_format": "bundle_v1",
      "origami_quantizer": "cachegen",
      "origami_lossless_cpu_backend": "zlib"
    }
  }'
```

On a QAT machine, change only the lossless backend:

```json
"origami_lossless_cpu_backend": "qat"
```

Useful QAT tuning fields:

```json
{
  "origami_qat_inflight": 32,
  "origami_qat_batch": 32,
  "origami_qat_max_instances": 32,
  "origami_qat_pipeline_target_chunks": 512,
  "origami_qat_pipeline_slots": 3,
  "origami_qat_dynamic_huffman": true
}
```

## Quantized-prefix fused attention

KIVI and KVQuant prefixes can remain compressed on the GPU and be consumed by
Origami's CUDA online-softmax attention kernel. A mixed batch runs this work on
a reusable high-priority stream while cold prefill remains on FlashAttention's
current stream:

```json
{
  "origami_quantizer": "kivi",
  "origami_quantizer_config": {
    "bits": 4,
    "group_size": 64,
    "sink_tokens": 128
  },
  "origami_fused_attention": "auto",
  "origami_fused_execution": "auto",
  "origami_fused_stream_priority": -1,
  "origami_cold_prefill_chunk_tokens": 1024,
  "origami_fused_max_requests": 8,
  "origami_batch_policy": "restored_priority_mixed"
}
```

`origami_fused_attention` defaults to `off`. `auto` falls back to explicit
dequantization when an artifact or execution mode is incompatible; `required`
fails startup or restore instead. `origami_fused_execution` accepts `auto`,
`parallel`, or `serial`. The other batch policies are `restored_only`, which
does not admit cold prefill while restored work is present, and
`pure_vllm_baseline`, which disables fused attention and Origami scheduling
priority.

The first fused implementation requires NVIDIA SM80+, TP/PP/DP/DCP 1,
FP16/BF16, head size 128, standard Llama or Mistral decoder self-attention, and
the NHD paged-KV layout. It does not support FP8 KV cache, speculative decoding,
sliding-window or cross-attention, MLA, or full CUDA Graph execution. Fused
batches use eager execution; ordinary batches retain the configured graph mode.

## Patch Workflow

Regenerate the patch after changing the vLLM fork:

```bash
bash scripts/export_vllm_patch.sh
```

Apply the patch to another vLLM 0.18.0 checkout:

```bash
bash scripts/apply_to_vllm.sh /path/to/vllm-0.18.0
```
