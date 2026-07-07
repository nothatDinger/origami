#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}"
PATCH_DIR="${ROOT_DIR}/patches"
PATCH_FILE="${PATCH_DIR}/origami-vllm-0.18.0.patch"
BASE_REF="${ORIGAMI_VLLM_BASE_REF:-a804d94}"

mkdir -p "${PATCH_DIR}"

pushd "${VLLM_DIR}" >/dev/null

missing=0
for path in \
  csrc/origami/qat_deflate.cpp \
  csrc/origami/bitpack_layout_cuda.cpp \
  csrc/origami/bitpack_layout_cuda.cu \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/benchmark_utils.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/benchmark_connectors.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/lossless/native_gpu.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/quantization/kivi_adapter.py
do
  if [[ ! -f "${path}" ]]; then
    echo "missing expected Origami file: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" == "1" ]]; then
  exit 2
fi

git add -N -f \
  csrc/origami/qat_deflate.cpp \
  csrc/origami/bitpack_layout_cuda.cpp \
  csrc/origami/bitpack_layout_cuda.cu \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/benchmark_utils.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/benchmark_connectors.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/lossless/native_gpu.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/quantization/kivi_adapter.py

git diff --binary "${BASE_REF}" -- . \
  ':(exclude)README.md' \
  ':(exclude)patches/origami-vllm-0.18.0.patch' > "${PATCH_FILE}"

popd >/dev/null

echo "Wrote ${PATCH_FILE}"
