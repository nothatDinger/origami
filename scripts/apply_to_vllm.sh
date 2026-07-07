#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${ROOT_DIR}/patches/origami-vllm-0.18.0.patch"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <vllm-source-dir>" >&2
  exit 2
fi

TARGET_DIR="$1"

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "target is not a git checkout: ${TARGET_DIR}" >&2
  exit 2
fi

if [[ ! -f "${TARGET_DIR}/pyproject.toml" || ! -d "${TARGET_DIR}/vllm" ]]; then
  echo "target does not look like a vLLM source tree: ${TARGET_DIR}" >&2
  exit 2
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "missing patch file: ${PATCH_FILE}" >&2
  exit 2
fi

pushd "${TARGET_DIR}" >/dev/null

if git status --porcelain | grep -q .; then
  echo "target tree has local changes; commit or stash them before applying Origami" >&2
  exit 2
fi

if ! git describe --tags --exact-match HEAD 2>/dev/null | grep -Eq '^v?0\.18\.0$'; then
  echo "warning: target HEAD is not tagged v0.18.0; checking whether the patch applies" >&2
fi

git apply --check "${PATCH_FILE}"
git apply "${PATCH_FILE}"

if [[ -f README.md ]] && ! grep -q "vLLM 0.18.0 with Origami" README.md; then
  python3 - <<'PY'
from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")
banner = """# vLLM 0.18.0 with Origami

This tree is a vLLM 0.18.0 fork with Origami integrated as a KV-transfer
connector. Origami adds codec-friendly KV cache layout, Intel QAT-backed
lossless restore, reuse-aware scheduling, and a layer-wise restore pipeline for
remote KV-cache fetching in long-context LLM serving.

The upstream vLLM README is preserved below for attribution and general vLLM
background.

---

"""
if text.startswith("<!-- markdownlint-disable MD001 MD041 -->\n"):
    marker = "<!-- markdownlint-disable MD001 MD041 -->\n"
    text = marker + banner + text[len(marker):]
else:
    text = banner + text
path.write_text(text, encoding="utf-8")
PY
fi

echo "Applied Origami patch to ${TARGET_DIR}"
popd >/dev/null
