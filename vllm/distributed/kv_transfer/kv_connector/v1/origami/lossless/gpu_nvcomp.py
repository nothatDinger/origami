# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import sys
import threading
from pathlib import Path

import torch


def _extend_nvidia_namespace_for_nvcomp() -> None:
    """Allow venvs with a regular nvidia package to see nvCOMP wheels."""
    try:
        import nvidia  # type: ignore
    except Exception:
        return

    candidate_roots: list[Path] = [
        Path.home() / ".local" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    uv_root = Path.home() / ".cache/uv/archive-v0"
    if uv_root.exists():
        candidate_roots.extend(sorted(uv_root.iterdir()))
    for root in candidate_roots:
        nvidia_root = root / "nvidia"
        if not (nvidia_root / "nvcomp").is_dir():
            continue
        nvidia_path = getattr(nvidia, "__path__", None)
        if nvidia_path is not None and str(nvidia_root) not in nvidia_path:
            nvidia_path.append(str(nvidia_root))
        if str(root) not in sys.path:
            sys.path.append(str(root))


def _normalize_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized in {"disabled", "none"}:
        return "disabled"
    if normalized in {"nvcomp", "deflate", "nvcomp_deflate"}:
        return "nvcomp_deflate"
    if normalized in {"gdeflate", "nvcomp_gdeflate"}:
        return "nvcomp_gdeflate"
    raise ValueError(
        "Origami nvCOMP backend must be one of "
        "{'disabled', 'nvcomp_deflate', 'nvcomp_gdeflate'}"
    )


def _nvcomp_algorithm(backend: str) -> str:
    normalized = _normalize_backend(backend)
    if normalized == "nvcomp_deflate":
        return "Deflate"
    if normalized == "nvcomp_gdeflate":
        return "GDeflate"
    raise ValueError(f"Backend {backend!r} does not use nvCOMP")


class NvcompCodec:

    def __init__(self, backend: str = "nvcomp"):
        self.backend = _normalize_backend(backend)
        self._codec = None
        self._nvcomp = None
        self._lock = threading.Lock()
        if self.backend == "disabled":
            return
        try:
            import nvidia.nvcomp as nvcomp  # type: ignore
        except Exception as exc:
            _extend_nvidia_namespace_for_nvcomp()
            try:
                import nvidia.nvcomp as nvcomp  # type: ignore
            except Exception as retry_exc:
                raise RuntimeError(
                    "Origami nvCOMP backend requires nvidia.nvcomp"
                ) from retry_exc
        self._nvcomp = nvcomp
        self._codec = nvcomp.Codec(algorithm=_nvcomp_algorithm(self.backend))

    def available(self) -> bool:
        return self._codec is not None and self._nvcomp is not None

    def compress(self, data: torch.Tensor) -> torch.Tensor:
        if not self.available():
            raise RuntimeError("Origami nvCOMP backend is disabled")
        flat = data.reshape(-1).to(torch.uint8).contiguous()
        if flat.device.type != "cuda":
            raise ValueError("Origami nvCOMP path requires CUDA input bytes")
        with self._lock:
            encoded = self._codec.encode(self._nvcomp.as_array(flat))
        encoded_size = int(encoded.size)
        if hasattr(encoded, "to_dlpack"):
            tensor = torch.utils.dlpack.from_dlpack(encoded.to_dlpack()).to(torch.uint8)
            return tensor.reshape(-1)[:encoded_size].clone().contiguous()
        return torch.as_tensor(encoded, device=flat.device, dtype=torch.uint8).reshape(
            -1
        )[:encoded_size].contiguous()

    def decompress(self, data: torch.Tensor, *, output_bytes: int) -> torch.Tensor:
        if not self.available():
            raise RuntimeError("Origami nvCOMP backend is disabled")
        if data.device.type != "cuda":
            raise ValueError("Origami nvCOMP path requires CUDA compressed bytes")
        with self._lock:
            decoded = self._codec.decode(
                self._nvcomp.as_array(data.reshape(-1).contiguous()),
                data_type="|u1",
            )
        if isinstance(decoded, torch.Tensor):
            output = decoded.to(torch.uint8).reshape(-1).contiguous()
        elif hasattr(decoded, "to_dlpack"):
            output = torch.utils.dlpack.from_dlpack(decoded.to_dlpack()).to(
                torch.uint8
            ).reshape(-1).clone().contiguous()
        else:
            output = torch.as_tensor(
                decoded,
                device=data.device,
                dtype=torch.uint8,
            ).reshape(-1).contiguous()
        if int(output.numel()) != int(output_bytes):
            raise RuntimeError("Origami nvCOMP decoded size mismatch")
        return output


class GpuLosslessCodec(NvcompCodec):
    """Compatibility alias for the historical Origami GPU lossless wrapper."""
