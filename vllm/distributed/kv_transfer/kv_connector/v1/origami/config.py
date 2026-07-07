# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING or os.environ.get("ORIGAMI_LIGHT_IMPORT") != "1":
    from vllm.config import VllmConfig
else:
    VllmConfig = Any  # type: ignore[misc, assignment]

GpuLosslessRatio = Literal["auto", "cpu", "gpu", 0, 25, 50, 75, 100]
NormalizedGpuLosslessRatio = Literal["auto", 0, 25, 50, 75, 100]
OrigamiArtifactFormat = Literal["auto", "bundle_v1", "legacy_pt"]
OrigamiDevicePolicy = Literal["auto", "cpu", "gpu", "cuda"]
OrigamiNvcompBackend = Literal["disabled", "nvcomp_deflate", "nvcomp_gdeflate"]


def _get_extra(vllm_config: VllmConfig, key: str, default: Any) -> Any:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return default
    return kv_transfer_config.get_from_extra_config(key, default)


def _parse_ratio(value: Any) -> NormalizedGpuLosslessRatio:
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "auto":
            return "auto"
        if value == "cpu":
            return 0
        if value == "gpu":
            return 100
        if value.isdigit():
            value = int(value)
    if value not in {0, 25, 50, 75, 100}:
        raise ValueError(
            "origami_gpu_lossless_ratio must be 'auto', 'cpu', 'gpu', "
            "or one of {0, 25, 50, 75, 100}"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class OrigamiConfig:
    quantizer: str = "cachegen"
    quantizer_config: dict[str, Any] = field(default_factory=dict)
    store_uri: str = "memory://origami"
    artifact_format: OrigamiArtifactFormat = "bundle_v1"
    lossless_cpu_backend: str = "qat"
    lossless_gpu_backend: str = "nvcomp"
    nvcomp_backend: OrigamiNvcompBackend = "disabled"
    layout_policy: str = "head_first"
    bitpack: bool = True
    chunk_policy: str = "fixed_1MiB"
    chunk_min_bytes: int = 1 << 20
    chunk_target_bytes: int = 1 << 20
    chunk_max_bytes: int = 1 << 20
    gpu_lossless_ratio: GpuLosslessRatio = "auto"
    pcie_high_watermark_gbps: float = 48.0
    pcie_low_watermark_gbps: float = 0.0
    auto_decrease_enabled: bool = True
    qat_threads: int = 16
    qat_inflight: int = 32
    qat_batch: int = 32
    qat_max_instances: int = 32
    qat_pipeline_target_chunks: int = 512
    qat_pipeline_slots: int = 3
    qat_dynamic_huffman: bool = True
    allow_zlib_fallback: bool = False
    batch_policy: str = "restored_priority_mixed"
    read_bandwidth_gbps: float = 0.0
    artifact_ingress_mode: str = "file"
    artifact_cache_keys: tuple[str, ...] = ()
    metrics_dir: str = ""
    dequant_device: str = "auto"
    bitunpack_device: OrigamiDevicePolicy = "auto"

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "OrigamiConfig":
        quantizer_config = dict(
            _get_extra(vllm_config, "origami_quantizer_config", {})
        )
        dequant_device = str(
            _get_extra(vllm_config, "origami_dequant_device", "auto")
        ).lower()
        bitunpack_device = str(
            _get_extra(vllm_config, "origami_bitunpack_device", "auto")
        ).lower()
        quantizer_config.setdefault("dequant_device", dequant_device)
        ratio = _parse_ratio(
            _get_extra(vllm_config, "origami_gpu_lossless_ratio", "auto")
        )
        config = cls(
            quantizer=str(_get_extra(vllm_config, "origami_quantizer", "cachegen")),
            quantizer_config=quantizer_config,
            store_uri=str(_get_extra(vllm_config, "origami_store_uri",
                                     "memory://origami")),
            artifact_format=str(
                _get_extra(vllm_config, "origami_artifact_format", "bundle_v1")
            ),
            lossless_cpu_backend=str(
                _get_extra(vllm_config, "origami_lossless_cpu_backend", "qat")
            ),
            lossless_gpu_backend=str(
                _get_extra(vllm_config, "origami_lossless_gpu_backend", "nvcomp")
            ),
            nvcomp_backend=str(
                _get_extra(vllm_config, "origami_nvcomp_backend", "disabled")
            ).lower(),
            layout_policy=str(_get_extra(vllm_config, "origami_layout_policy",
                                         "head_first")),
            bitpack=bool(_get_extra(vllm_config, "origami_bitpack", True)),
            chunk_policy=str(
                _get_extra(vllm_config, "origami_chunk_policy",
                           "fixed_1MiB")
            ),
            chunk_min_bytes=int(
                _get_extra(vllm_config, "origami_chunk_min_bytes", 1 << 20)
            ),
            chunk_target_bytes=int(
                _get_extra(vllm_config, "origami_chunk_target_bytes", 1 << 20)
            ),
            chunk_max_bytes=int(
                _get_extra(vllm_config, "origami_chunk_max_bytes", 1 << 20)
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
                _get_extra(vllm_config, "origami_qat_max_instances", 32)
            ),
            qat_pipeline_target_chunks=int(
                _get_extra(vllm_config, "origami_qat_pipeline_target_chunks", 512)
            ),
            qat_pipeline_slots=int(
                _get_extra(vllm_config, "origami_qat_pipeline_slots", 3)
            ),
            qat_dynamic_huffman=bool(
                _get_extra(vllm_config, "origami_qat_dynamic_huffman", True)
            ),
            allow_zlib_fallback=bool(
                _get_extra(vllm_config, "origami_allow_zlib_fallback", False)
            ),
            batch_policy=str(
                _get_extra(vllm_config, "origami_batch_policy",
                           "restored_priority_mixed")
            ),
            read_bandwidth_gbps=float(
                _get_extra(
                    vllm_config,
                    "origami_read_bandwidth_gbps",
                    _get_extra(vllm_config, "reuse_read_bandwidth_gbps", 0.0),
                )
            ),
            artifact_ingress_mode=str(
                _get_extra(
                    vllm_config,
                    "origami_artifact_ingress_mode",
                    _get_extra(vllm_config, "reuse_artifact_ingress_mode", "file"),
                )
            ).lower().replace("_", "-"),
            artifact_cache_keys=tuple(
                str(key)
                for key in (
                    _get_extra(vllm_config, "origami_artifact_cache_keys", None)
                    or _get_extra(vllm_config, "reuse_artifact_cache_keys", ())
                    or ()
                )
                if key
            ),
            metrics_dir=str(
                _get_extra(
                    vllm_config,
                    "origami_metrics_dir",
                    _get_extra(vllm_config, "reuse_metrics_dir", ""),
                )
            ),
            dequant_device=dequant_device,
            bitunpack_device=bitunpack_device,
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
        if self.artifact_format not in {"auto", "bundle_v1", "legacy_pt"}:
            raise ValueError(
                "origami_artifact_format must be one of "
                "{'auto', 'bundle_v1', 'legacy_pt'}"
            )
        if self.nvcomp_backend not in {
                "disabled",
                "nvcomp_deflate",
                "nvcomp_gdeflate",
        }:
            raise ValueError(
                "origami_nvcomp_backend must be one of "
                "{'disabled', 'nvcomp_deflate', 'nvcomp_gdeflate'}"
            )
        if self.read_bandwidth_gbps < 0:
            raise ValueError("origami_read_bandwidth_gbps must be non-negative")
        if self.artifact_ingress_mode not in {"file", "memory", "native-file"}:
            raise ValueError(
                "origami_artifact_ingress_mode must be one of "
                "{'file', 'memory', 'native-file'}"
            )
        if self.qat_pipeline_target_chunks < 0:
            raise ValueError(
                "origami_qat_pipeline_target_chunks must be non-negative"
            )
        if self.qat_pipeline_slots <= 0:
            raise ValueError("origami_qat_pipeline_slots must be positive")
        if self.dequant_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError(
                "origami_dequant_device must be one of {'auto', 'cpu', 'gpu', 'cuda'}"
            )
        if self.bitunpack_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError(
                "origami_bitunpack_device must be one of {'auto', 'cpu', 'gpu', 'cuda'}"
            )

    def effective_qat_pipeline_target_chunks(self) -> int:
        if self.qat_pipeline_target_chunks > 0:
            return int(self.qat_pipeline_target_chunks)
        return max(1, int(self.qat_max_instances) * 4)
