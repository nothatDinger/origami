# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.origami.quantization.raw_tensor import (
    TorchSerializedTensorAdapter,
)


class MockInt4Adapter(TorchSerializedTensorAdapter):
    quantizer_id = "mock_int4"

