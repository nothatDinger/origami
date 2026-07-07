# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

if os.environ.get("ORIGAMI_LIGHT_IMPORT") != "1":
    from vllm.distributed.kv_transfer.kv_connector.v1.origami.connector import (
        OrigamiConnector,
    )

    __all__ = ["OrigamiConnector"]
else:
    __all__: list[str] = []
