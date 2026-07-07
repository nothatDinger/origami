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

## Patch Workflow

Regenerate the patch after changing the vLLM fork:

```bash
bash scripts/export_vllm_patch.sh
```

Apply the patch to another vLLM 0.18.0 checkout:

```bash
bash scripts/apply_to_vllm.sh /path/to/vllm-0.18.0
```
