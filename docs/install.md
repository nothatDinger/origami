# Installing Origami

Origami is implemented as a vLLM 0.18.0 fork plus a patch that can be applied to
a clean vLLM source tree. The recommended Python version is 3.12. The vLLM fork
tracks `torch==2.10.0`; install a PyTorch wheel that matches the CUDA toolkit
available on the target machine.

## Basic Environment

From the repository root:

```bash
bash scripts/setup_env.sh
source .venv-origami/bin/activate
```

The setup script creates `.venv-origami`, upgrades packaging tools, installs the
top-level `requirements.txt`, and installs the included vLLM fork in editable
mode unless `--no-vllm` is passed.

To install dependencies manually:

```bash
python3 -m venv .venv-origami
source .venv-origami/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch==2.10.0
python3 -m pip install -r requirements.txt
python3 -m pip install -e ./vllm-0.18.0
```

For CUDA wheels, use the PyTorch index that matches your driver/toolkit. For CPU
only testing, install a CPU wheel instead and use the `zlib` lossless backend.

## Applying Origami to vLLM 0.18.0

Origami can be applied to an external vLLM checkout:

```bash
git clone https://github.com/vllm-project/vllm.git vllm-origami
cd vllm-origami
git checkout v0.18.0

../Origami/scripts/apply_to_vllm.sh .
python3 -m pip install -e .
```

The apply script checks that the target looks like a vLLM source tree, runs
`git apply --check`, applies `patches/origami-vllm-0.18.0.patch`, and adds a
short Origami note to the target vLLM README. If the target tree has local
changes, commit or stash them before applying the patch.

## QAT Requirements

The QAT path requires:

- Intel QAT-capable hardware visible through PCIe or platform-integrated QAT.
- Loaded QAT/adf kernel drivers.
- Device nodes such as `/dev/qat*`, `/dev/usdm*`, `/dev/vfio/vfio`, or an
  equivalent VFIO/UIO configuration that the current user can access.
- QAT headers, especially `qat/cpa.h`.
- QAT user-space libraries: `libqat`, `libusdm`, and `libcrypto`.

Check the machine:

```bash
python3 scripts/check_qat.py
python3 scripts/check_qat.py --try-extension
```

If the extension check fails while PCI devices and libraries are present, inspect
device permissions and whether the QAT service/driver has started user-space
instances.

## Useful Environment Variables

- `ORIGAMI_QAT_CODEC_PATH=dpucomp_dp`: default prepared QAT data-plane path.
- `ORIGAMI_LIGHT_IMPORT=1`: import the Origami codec without importing the full
  vLLM runtime.
- `ORIGAMI_NATIVE_BUILD_DIR=/tmp/origami-native`: native extension build cache.
- `ORIGAMI_NATIVE_VERBOSE=1`: verbose PyTorch extension builds.
- `ORIGAMI_REQUIRE_QAT=0`: allow `scripts/run_tests.sh` to use the software
  lossless path when QAT is unavailable.
