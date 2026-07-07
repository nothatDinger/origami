# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence


def select_gpu_lossless_requests(
    request_ids: Sequence[str],
    ratio: int,
) -> set[str]:
    if ratio <= 0 or not request_ids:
        return set()
    if ratio >= 100:
        return set(request_ids)
    count = int(round(len(request_ids) * ratio / 100.0))
    count = max(1, min(len(request_ids), count))
    return set(sorted(request_ids)[:count])
