# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.raw_tensor import (
    TorchSerializedTensorAdapter,
)


class CacheGenAdapter(TorchSerializedTensorAdapter):
    """CacheGen adapter entry point.

    The initial implementation uses a reversible serialized tensor payload so
    the Origami connector can be tested without CacheGen calibration/runtime
    state. The adapter boundary is intentionally stable for replacing this with
    the real third-party CacheGen quantizer.
    """

    quantizer_id = "cachegen"

