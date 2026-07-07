# Origami: QAT-Assisted KV-Cache Fetching for LLM Serving

Origami is a vLLM-based system for accelerating long-context LLM serving when
requests reuse KV caches stored outside the GPU. Remote KV fetching can dominate
time-to-first-token (TTFT), and common GPU-side byte-stream decompression can
interfere with prefill compute. Origami addresses this bottleneck with a
codec-friendly KV layout, Intel QuickAssist Technology (QAT) lossless
decompression, reuse-aware scheduling, and a layer-wise restore pipeline that
overlaps host-side restore with GPU work.

This repository contains the Origami implementation for vLLM 0.18.0, native QAT
and layout kernels, baseline integrations, and scripts for installing and
testing the system.

## Repository Layout

- `vllm/`: vLLM with the Origami KV-transfer connector.
- `csrc/origami/`: native layout, bitpacking, and QAT sources.
- `scripts/`: setup, QAT detection, test, and patch utilities.

## Installation

Apply Origami to a clean vLLM 0.18.0 source tree and install it:

```bash
git clone <repo-url> Origami
git clone https://github.com/vllm-project/vllm.git vllm-origami
cd vllm-origami
git checkout v0.18.0

../Origami/scripts/apply_to_vllm.sh .
python3 -m pip install -e .
```

## QAT Check

Origami's main lossless path uses Intel QAT. Check whether the machine exposes
the required device, driver, headers, libraries, and native runtime:

```bash
cd ../Origami
python3 scripts/check_qat.py
python3 scripts/check_qat.py --try-extension --require
```

## Tests

Run the QAT lossless path:

```bash
ORIGAMI_QAT_CODEC_PATH=dpucomp_dp python3 scripts/test_lossless.py --backend qat
```

Run the software path when QAT is unavailable:

```bash
python3 scripts/test_lossless.py --backend zlib
```

Run the test entry:

```bash
bash scripts/run_tests.sh
```

For machines without QAT:

```bash
ORIGAMI_REQUIRE_QAT=0 bash scripts/run_tests.sh
```
