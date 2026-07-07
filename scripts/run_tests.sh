#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
REQUIRE_QAT="${ORIGAMI_REQUIRE_QAT:-1}"

cd "${ROOT_DIR}"

if [[ "${PYTHON_BIN}" == */* ]]; then
  PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi

echo "== QAT environment =="
if [[ "${REQUIRE_QAT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/check_qat.py --try-extension --require
  LOSSLESS_BACKEND="${ORIGAMI_TEST_BACKEND:-qat}"
else
  "${PYTHON_BIN}" scripts/check_qat.py --try-extension || true
  LOSSLESS_BACKEND="${ORIGAMI_TEST_BACKEND:-zlib}"
fi

echo
echo "== Lossless round trip (${LOSSLESS_BACKEND}) =="
ORIGAMI_QAT_CODEC="${ORIGAMI_QAT_CODEC:-qat_codec}" \
  "${PYTHON_BIN}" scripts/test_lossless.py \
    --backend "${LOSSLESS_BACKEND}" \
    --chunks "${ORIGAMI_TEST_CHUNKS:-4}" \
    --chunk-bytes "${ORIGAMI_TEST_CHUNK_BYTES:-1048576}"
