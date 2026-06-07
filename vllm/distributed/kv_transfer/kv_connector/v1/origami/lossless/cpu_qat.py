# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import zlib

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import native_cpu


class CpuLosslessCodec:
    """CPU lossless codec wrapper for Origami packed KV chunks.

    The QAT backend uses the lazy native QAT DP extension. It fails explicitly
    when QAT is unavailable unless zlib fallback is opted in. The zlib backend
    emits raw DEFLATE streams with the same bytestream contract and is intended
    for tests and development only.
    """

    def __init__(
        self,
        backend: str = "qat",
        allow_zlib_fallback: bool = False,
        *,
        dynamic_huffman: bool = True,
        qat_inflight: int = 32,
        qat_batch: int = 32,
        qat_max_instances: int = 16,
    ):
        self.backend = backend
        self.allow_zlib_fallback = allow_zlib_fallback
        self.dynamic_huffman = bool(dynamic_huffman)
        self.qat_inflight = int(qat_inflight)
        self.qat_batch = int(qat_batch)
        self.qat_max_instances = int(qat_max_instances)
        self._qat_error: str | None = None
        if backend == "qat":
            try:
                native_cpu.load_qat_extension()
                if not native_cpu.qat_available():
                    raise RuntimeError("no usable offloaded QAT DC instances found")
            except Exception as exc:
                self._qat_error = str(exc)
                if not allow_zlib_fallback:
                    raise RuntimeError(
                        "Origami QAT backend is not available. Set "
                        "origami_lossless_cpu_backend='zlib' for tests or enable "
                        "origami_allow_zlib_fallback for development. Native error: "
                        f"{exc}"
                    ) from exc
                self.backend = "zlib"
        elif backend != "zlib":
            raise ValueError(f"Unsupported Origami CPU lossless backend {backend!r}")

    @staticmethod
    def _cpu_u8(data: torch.Tensor) -> torch.Tensor:
        return data.detach().cpu().to(torch.uint8).reshape(-1).contiguous()

    @staticmethod
    def _zlib_compress(data: torch.Tensor) -> torch.Tensor:
        flat = CpuLosslessCodec._cpu_u8(data)
        compressor = zlib.compressobj(
            level=1,
            wbits=-15,
            strategy=zlib.Z_DEFAULT_STRATEGY,
        )
        encoded = compressor.compress(flat.numpy().tobytes()) + compressor.flush()
        return torch.frombuffer(bytearray(encoded), dtype=torch.uint8).clone()

    @staticmethod
    def _zlib_decompress(data: torch.Tensor, output_bytes: int) -> torch.Tensor:
        flat = CpuLosslessCodec._cpu_u8(data)
        decoded = zlib.decompress(flat.numpy().tobytes(), wbits=-15)
        if len(decoded) != int(output_bytes):
            raise RuntimeError(
                f"Origami CPU lossless decoded {len(decoded)} bytes, "
                f"expected {int(output_bytes)}"
            )
        return torch.frombuffer(bytearray(decoded), dtype=torch.uint8).clone()

    def compress_many(self, chunks: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if self.backend == "qat":
            return native_cpu.compress_raw_deflate_many(
                chunks,
                dynamic_huffman=self.dynamic_huffman,
                inflight=self.qat_inflight,
                batch=self.qat_batch,
                max_instances=self.qat_max_instances,
            )
        return [self._zlib_compress(chunk) for chunk in chunks]

    def decompress_many(
        self,
        chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
        output_bytes: list[int] | tuple[int, ...],
    ) -> list[torch.Tensor]:
        if len(chunks) != len(output_bytes):
            raise ValueError("chunks and output_bytes must have the same length")
        if self.backend == "qat":
            return native_cpu.decompress_raw_deflate_many(
                chunks,
                output_bytes,
                dynamic_huffman=self.dynamic_huffman,
                inflight=self.qat_inflight,
                batch=self.qat_batch,
                max_instances=self.qat_max_instances,
            )
        return [
            self._zlib_decompress(chunk, int(out_bytes))
            for chunk, out_bytes in zip(chunks, output_bytes)
        ]

    def compress(self, data: torch.Tensor) -> torch.Tensor:
        return self.compress_many([data])[0]

    def decompress(self, data: torch.Tensor, *, output_bytes: int) -> torch.Tensor:
        return self.decompress_many([data], [int(output_bytes)])[0]
