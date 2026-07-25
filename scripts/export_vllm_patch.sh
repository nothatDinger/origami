#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}"
PATCH_DIR="${ROOT_DIR}/patches"
PATCH_FILE="${PATCH_DIR}/origami-vllm-0.18.0.patch"
BASE_REF="${ORIGAMI_VLLM_BASE_REF:-a804d94}"
BASE_WHEEL="${ORIGAMI_VLLM_BASE_WHEEL:-${ROOT_DIR}/vllm-0.18.0-cp38-abi3-manylinux_2_31_x86_64.whl}"

UPSTREAM_FILES=(
  vllm/v1/attention/backend.py
  vllm/v1/attention/backends/flash_attn.py
  vllm/v1/worker/gpu_model_runner.py
  vllm/v1/worker/gpu/attn_utils.py
  vllm/v1/worker/gpu/input_batch.py
  vllm/v1/worker/gpu/model_runner.py
  vllm/v1/worker/gpu/model_states/default.py
  vllm/v1/worker/ubatch_utils.py
)

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
  vllm/distributed/kv_transfer/kv_connector/v1/origami/quantization/kivi_adapter.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/fused_attention.py \
  tests/v1/kv_connector/unit/test_origami_fused_attention.py \
  "${UPSTREAM_FILES[@]}"
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
  vllm/distributed/kv_transfer/kv_connector/v1/origami/quantization/kivi_adapter.py \
  vllm/distributed/kv_transfer/kv_connector/v1/origami/fused_attention.py \
  tests/v1/kv_connector/unit/test_origami_fused_attention.py

git diff --binary "${BASE_REF}" -- . \
  ':(exclude)README.md' \
  ':(exclude)patches/origami-vllm-0.18.0.patch' > "${PATCH_FILE}"

popd >/dev/null

if [[ ! -f "${BASE_WHEEL}" ]]; then
  echo "missing pristine vLLM wheel: ${BASE_WHEEL}" >&2
  echo "set ORIGAMI_VLLM_BASE_WHEEL to a vLLM 0.18.0 wheel" >&2
  exit 2
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
mkdir -p "${TEMP_DIR}/original" "${TEMP_DIR}/modified"
unzip -q "${BASE_WHEEL}" "${UPSTREAM_FILES[@]}" -d "${TEMP_DIR}/original"
for path in "${UPSTREAM_FILES[@]}"; do
  mkdir -p "${TEMP_DIR}/modified/$(dirname "${path}")"
  cp "${ROOT_DIR}/${path}" "${TEMP_DIR}/modified/${path}"
done

pushd "${TEMP_DIR}" >/dev/null
set +e
git diff --no-index --binary --src-prefix=a/ --dst-prefix=b/ \
  original modified > upstream.patch
diff_status=$?
set -e
popd >/dev/null
if [[ "${diff_status}" -gt 1 ]]; then
  echo "failed to diff upstream vLLM files" >&2
  exit "${diff_status}"
fi

sed \
  -e 's#^diff --git a/original/#diff --git a/#' \
  -e 's# b/modified/# b/#' \
  -e 's#^--- a/original/#--- a/#' \
  -e 's#^+++ b/modified/#+++ b/#' \
  "${TEMP_DIR}/upstream.patch" >> "${PATCH_FILE}"

echo "Wrote ${PATCH_FILE}"
