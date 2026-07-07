#!/usr/bin/env python3
"""Check whether the current host can run Origami's QAT lossless path."""

from __future__ import annotations

import argparse
import ctypes.util
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_ROOT = REPO_ROOT
QAT_RE = re.compile(r"(quickassist|\bqat\b|4xxx series qat)", re.IGNORECASE)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return 127, str(exc)
    return int(proc.returncode), proc.stdout.strip()


def find_library(name: str) -> str:
    found = ctypes.util.find_library(name)
    if found:
        return found
    for root in (
        "/usr/local/lib",
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ):
        path = Path(root) / f"lib{name}.so"
        if path.exists():
            return str(path)
    return ""


def check_pci() -> Check:
    lspci = shutil.which("lspci")
    if not lspci:
        return Check("qat_pci", False, "lspci not found")
    code, out = run([lspci, "-nn"])
    if code != 0:
        return Check("qat_pci", False, f"lspci failed: {out}")
    matches = [line for line in out.splitlines() if QAT_RE.search(line)]
    if matches:
        return Check("qat_pci", True, "; ".join(matches[:8]))
    return Check("qat_pci", False, "no QAT device reported by lspci")


def check_device_nodes() -> Check:
    patterns = ("/dev/qat*", "/dev/usdm*", "/dev/vfio/vfio", "/dev/uio*")
    nodes = sorted({node for pattern in patterns for node in glob.glob(pattern)})
    if not nodes:
        return Check("qat_device_nodes", False, "no common QAT device nodes found")
    readable = [node for node in nodes if os.access(node, os.R_OK)]
    writable = [node for node in nodes if os.access(node, os.W_OK)]
    ok = bool(readable or writable)
    return Check(
        "qat_device_nodes",
        ok,
        f"found={nodes}; readable={readable}; writable={writable}",
    )


def check_modules() -> Check:
    modules = Path("/proc/modules")
    if not modules.exists():
        return Check("qat_kernel_modules", False, "/proc/modules unavailable")
    matches = []
    for line in modules.read_text(errors="ignore").splitlines():
        name = line.split()[0]
        if "qat" in name.lower() or name.startswith("adf_"):
            matches.append(name)
    if matches:
        return Check("qat_kernel_modules", True, ", ".join(matches))
    return Check("qat_kernel_modules", False, "no loaded QAT/adf modules found")


def check_headers() -> Check:
    candidates = (
        Path("/usr/local/include/qat/cpa.h"),
        Path("/usr/include/qat/cpa.h"),
        Path("/usr/include/x86_64-linux-gnu/qat/cpa.h"),
    )
    hits = [str(path) for path in candidates if path.exists()]
    if hits:
        return Check("qat_headers", True, ", ".join(hits))
    return Check("qat_headers", False, "qat/cpa.h not found")


def check_libraries() -> list[Check]:
    out = []
    for name in ("qat", "usdm", "crypto"):
        found = find_library(name)
        out.append(
            Check(
                f"lib{name}",
                bool(found),
                found or f"lib{name} not found",
            )
        )
    return out


def check_torch() -> Check:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return Check("torch", False, f"cannot import torch: {exc}")
    cuda = getattr(torch.version, "cuda", None)
    return Check(
        "torch",
        True,
        f"torch={torch.__version__}, cuda={cuda}, cuda_available={torch.cuda.is_available()}",
    )


def try_origami_extension(max_instances: int) -> Check:
    os.environ.setdefault("ORIGAMI_LIGHT_IMPORT", "1")
    sys.path.insert(0, str(VLLM_ROOT))
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import (
            native_cpu,
        )

        codec_path = native_cpu.qat_codec_path()
        count = native_cpu.qat_instance_count(max_instances=max_instances)
    except Exception as exc:
        return Check("origami_qat_extension", False, str(exc))
    return Check(
        "origami_qat_extension",
        count > 0,
        f"codec_path={codec_path}, qat_instances={count}",
    )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[Check] = [
        Check("platform", True, platform.platform()),
        Check("python", True, sys.version.replace("\n", " ")),
        Check("vllm_tree", VLLM_ROOT.exists(), str(VLLM_ROOT)),
        check_pci(),
        check_device_nodes(),
        check_modules(),
        check_headers(),
        *check_libraries(),
        check_torch(),
    ]
    if args.try_extension:
        checks.append(try_origami_extension(max_instances=int(args.max_instances)))

    by_name = {check.name: check for check in checks}
    hardware_present = by_name["qat_pci"].ok
    driver_present = by_name["qat_kernel_modules"].ok or by_name["qat_device_nodes"].ok
    torch_present = by_name["torch"].ok
    userspace_present = (
        by_name["qat_headers"].ok
        and by_name["libqat"].ok
        and by_name["libusdm"].ok
        and by_name["libcrypto"].ok
    )
    extension = by_name.get("origami_qat_extension")
    runtime_ready = extension.ok if extension is not None else False
    minimum_runtime_ready = (
        runtime_ready
        if args.try_extension
        else hardware_present and driver_present and userspace_present and torch_present
    )

    return {
        "hardware_present": bool(hardware_present),
        "driver_present": bool(driver_present),
        "userspace_present": bool(userspace_present),
        "runtime_ready": bool(runtime_ready),
        "minimum_runtime_ready": bool(minimum_runtime_ready),
        "try_extension": bool(args.try_extension),
        "checks": [asdict(check) for check in checks],
        "environment": {
            "ORIGAMI_QAT_CODEC_PATH": os.environ.get("ORIGAMI_QAT_CODEC_PATH", ""),
            "ORIGAMI_NATIVE_BUILD_DIR": os.environ.get("ORIGAMI_NATIVE_BUILD_DIR", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print("Origami QAT environment")
    for key in (
        "hardware_present",
        "driver_present",
        "userspace_present",
        "runtime_ready",
        "minimum_runtime_ready",
    ):
        print(f"{key}: {str(report[key]).lower()}")
    print()
    for check in report["checks"]:
        status = "OK" if check["ok"] else "MISS"
        print(f"[{status}] {check['name']}: {check['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--try-extension",
        action="store_true",
        help="import Origami and ask the native QAT extension for instance count",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="return non-zero when the requested QAT readiness check fails",
    )
    parser.add_argument("--max-instances", type=int, default=4)
    args = parser.parse_args()

    report = collect(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    if args.require and not report["minimum_runtime_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
