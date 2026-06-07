# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import zlib

import torch


class CpuLosslessCodec:
    """CPU lossless codec wrapper.

    The QAT backend is optional and intentionally loaded lazily. Unit tests and
    non-QAT development use the raw-DEFLATE zlib backend, which preserves the
    same bytestream contract.
    """

    def __init__(self, backend: str = "qat", allow_zlib_fallback: bool = False):
        self.backend = backend
        self.allow_zlib_fallback = allow_zlib_fallback
        self._qat = None
        if backend == "qat":
            try:
                import vllm._origami_qat as qat  # type: ignore
            except Exception:
                if not allow_zlib_fallback:
                    raise RuntimeError(
                        "Origami QAT backend is not available. Set "
                        "origami_lossless_cpu_backend='zlib' for tests or enable "
                        "origami_allow_zlib_fallback for development."
                    )
                self.backend = "zlib"
            else:
                self._qat = qat
        elif backend != "zlib":
            raise ValueError(f"Unsupported Origami CPU lossless backend {backend!r}")

    def compress(self, data: torch.Tensor) -> torch.Tensor:
        flat = data.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        if self.backend == "qat" and self._qat is not None:
            return self._qat.compress_raw_deflate(flat)
        compressor = zlib.compressobj(level=1, wbits=-15, strategy=zlib.Z_DEFAULT_STRATEGY)
        encoded = compressor.compress(bytes(flat.tolist())) + compressor.flush()
        return torch.tensor(list(encoded), dtype=torch.uint8)

    def decompress(self, data: torch.Tensor, *, output_bytes: int) -> torch.Tensor:
        flat = data.detach().cpu().to(torch.uint8).reshape(-1).contiguous()
        if self.backend == "qat" and self._qat is not None:
            return self._qat.decompress_raw_deflate(flat, int(output_bytes))
        decoded = zlib.decompress(bytes(flat.tolist()), wbits=-15)
        if len(decoded) != int(output_bytes):
            raise RuntimeError(
                f"Origami CPU lossless decoded {len(decoded)} bytes, "
                f"expected {int(output_bytes)}"
            )
        return torch.tensor(list(decoded), dtype=torch.uint8)

