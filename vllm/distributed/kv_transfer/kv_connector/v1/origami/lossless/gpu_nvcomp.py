# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch


class GpuLosslessCodec:

    def __init__(self, backend: str = "nvcomp"):
        self.backend = backend
        self._codec = None
        self._nvcomp = None
        if backend == "disabled":
            return
        if backend != "nvcomp":
            raise ValueError(f"Unsupported Origami GPU lossless backend {backend!r}")
        try:
            import nvidia.nvcomp as nvcomp  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Origami nvCOMP backend requires nvidia.nvcomp"
            ) from exc
        self._nvcomp = nvcomp
        self._codec = nvcomp.Codec(algorithm="deflate")

    def available(self) -> bool:
        return self._codec is not None and self._nvcomp is not None

    def decompress(self, data: torch.Tensor, *, output_bytes: int) -> torch.Tensor:
        if not self.available():
            raise RuntimeError("Origami GPU lossless backend is disabled")
        if data.device.type != "cuda":
            raise ValueError("Origami nvCOMP path requires CUDA compressed bytes")
        output = torch.empty((int(output_bytes),), dtype=torch.uint8, device=data.device)
        decoded = self._codec.decode(
            self._nvcomp.as_array(data.reshape(-1).contiguous()),
            data_type="|u1",
            out=output,
        )
        if isinstance(decoded, torch.Tensor):
            output = decoded.to(torch.uint8).reshape(-1).contiguous()
        if int(output.numel()) != int(output_bytes):
            raise RuntimeError("Origami nvCOMP decoded size mismatch")
        return output

