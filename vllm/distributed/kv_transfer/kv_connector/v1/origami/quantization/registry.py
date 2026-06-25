# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.base import (
    QuantizerAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.cachegen_adapter import (
    CacheGenAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kvquant_adapter import (
    KVQuantAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.kivi_adapter import (
    KiviAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.mock_int4 import (
    MockInt4Adapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.raw_tensor import (
    RawBytesTensorAdapter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.turboquant_adapter import (
    TurboQuantAdapter,
)

_REGISTRY: dict[str, type[QuantizerAdapter]] = {
    "cachegen": CacheGenAdapter,
    "kivi": KiviAdapter,
    "kvquant": KVQuantAdapter,
    "turboquant": TurboQuantAdapter,
    "mock_int4": MockInt4Adapter,
    "raw_bytes": RawBytesTensorAdapter,
}


def create_quantizer_adapter(
    name: str,
    config: dict | None = None,
) -> QuantizerAdapter:
    normalized = name.lower()
    if normalized not in _REGISTRY:
        raise ValueError(
            f"Unsupported Origami quantizer {name!r}; supported values are "
            f"{sorted(_REGISTRY)}"
        )
    return _REGISTRY[normalized](config)
