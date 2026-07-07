#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ORIGAMI_VENV_DIR:-${ROOT_DIR}/.venv-origami}"
PYTHON_BIN="${PYTHON:-python3}"
INSTALL_VLLM=1
VLLM_SOURCE="${ROOT_DIR}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vllm)
      INSTALL_VLLM=0
      shift
      ;;
    --vllm-source)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

if [[ "${INSTALL_VLLM}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -m pip install -e "${VLLM_SOURCE}"
fi

cat <<EOF
Environment ready:
  source ${VENV_DIR}/bin/activate

Useful checks:
  python3 scripts/check_qat.py --try-extension
  python3 scripts/test_lossless.py --backend qat
EOF
