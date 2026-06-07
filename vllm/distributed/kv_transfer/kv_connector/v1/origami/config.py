# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.config import VllmConfig

GpuLosslessRatio = Literal["auto", 0, 25, 50, 75, 100]


def _get_extra(vllm_config: VllmConfig, key: str, default: Any) -> Any:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return default
    return kv_transfer_config.get_from_extra_config(key, default)


def _parse_ratio(value: Any) -> GpuLosslessRatio:
    if value == "auto":
        return "auto"
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if value not in {0, 25, 50, 75, 100}:
        raise ValueError(
            "origami_gpu_lossless_ratio must be 'auto' or one of "
            "{0, 25, 50, 75, 100}"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class OrigamiConfig:
    quantizer: str = "cachegen"
    quantizer_config: dict[str, Any] = field(default_factory=dict)
    store_uri: str = "memory://origami"
    lossless_cpu_backend: str = "qat"
    lossless_gpu_backend: str = "nvcomp"
    layout_policy: str = "head_first"
    bitpack: bool = True
    chunk_policy: str = "dynamic_512KiB_2MiB"
    chunk_min_bytes: int = 512 << 10
    chunk_target_bytes: int = 1 << 20
    chunk_max_bytes: int = 2 << 20
    gpu_lossless_ratio: GpuLosslessRatio = "auto"
    pcie_high_watermark_gbps: float = 48.0
    pcie_low_watermark_gbps: float = 0.0
    auto_decrease_enabled: bool = True
    qat_threads: int = 16
    qat_inflight: int = 32
    qat_batch: int = 32
    qat_max_instances: int = 16
    allow_zlib_fallback: bool = False
    batch_policy: str = "restored_priority_mixed"

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "OrigamiConfig":
        quantizer_config = dict(
            _get_extra(vllm_config, "origami_quantizer_config", {})
        )
        ratio = _parse_ratio(
            _get_extra(vllm_config, "origami_gpu_lossless_ratio", "auto")
        )
        config = cls(
            quantizer=str(_get_extra(vllm_config, "origami_quantizer", "cachegen")),
            quantizer_config=quantizer_config,
            store_uri=str(_get_extra(vllm_config, "origami_store_uri",
                                     "memory://origami")),
            lossless_cpu_backend=str(
                _get_extra(vllm_config, "origami_lossless_cpu_backend", "qat")
            ),
            lossless_gpu_backend=str(
                _get_extra(vllm_config, "origami_lossless_gpu_backend", "nvcomp")
            ),
            layout_policy=str(_get_extra(vllm_config, "origami_layout_policy",
                                         "head_first")),
            bitpack=bool(_get_extra(vllm_config, "origami_bitpack", True)),
            chunk_policy=str(
                _get_extra(vllm_config, "origami_chunk_policy",
                           "dynamic_512KiB_2MiB")
            ),
            chunk_min_bytes=int(
                _get_extra(vllm_config, "origami_chunk_min_bytes", 512 << 10)
            ),
            chunk_target_bytes=int(
                _get_extra(vllm_config, "origami_chunk_target_bytes", 1 << 20)
            ),
            chunk_max_bytes=int(
                _get_extra(vllm_config, "origami_chunk_max_bytes", 2 << 20)
            ),
            gpu_lossless_ratio=ratio,
            pcie_high_watermark_gbps=float(
                _get_extra(vllm_config, "origami_pcie_high_watermark_gbps", 48.0)
            ),
            pcie_low_watermark_gbps=float(
                _get_extra(vllm_config, "origami_pcie_low_watermark_gbps", 0.0)
            ),
            auto_decrease_enabled=bool(
                _get_extra(vllm_config, "origami_auto_decrease_enabled", True)
            ),
            qat_threads=int(_get_extra(vllm_config, "origami_qat_threads", 16)),
            qat_inflight=int(_get_extra(vllm_config, "origami_qat_inflight", 32)),
            qat_batch=int(_get_extra(vllm_config, "origami_qat_batch", 32)),
            qat_max_instances=int(
                _get_extra(vllm_config, "origami_qat_max_instances", 16)
            ),
            allow_zlib_fallback=bool(
                _get_extra(vllm_config, "origami_allow_zlib_fallback", False)
            ),
            batch_policy=str(
                _get_extra(vllm_config, "origami_batch_policy",
                           "restored_priority_mixed")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.chunk_min_bytes <= 0 or self.chunk_target_bytes <= 0:
            raise ValueError("Origami chunk sizes must be positive")
        if not (self.chunk_min_bytes <= self.chunk_target_bytes <=
                self.chunk_max_bytes):
            raise ValueError(
                "Origami chunk sizes must satisfy min <= target <= max"
            )
        if self.layout_policy not in {
                "head_first",
                "layer_token_channel",
                "baseline",
        }:
            raise ValueError(f"Unsupported Origami layout policy {self.layout_policy}")
        if self.batch_policy not in {
                "restored_priority_mixed",
                "restored_only",
                "pure_vllm_baseline",
        }:
            raise ValueError(f"Unsupported Origami batch policy {self.batch_policy}")

