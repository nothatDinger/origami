# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import OrigamiConfig
from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    OrigamiConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.scheduler import (
    OrigamiConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.store import (
    create_origami_store,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.worker import (
    OrigamiConnectorWorker,
)
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class OrigamiConnector(KVConnectorBase_V1, SupportsHMA):

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.config = OrigamiConfig.from_vllm_config(vllm_config)
        self.store = create_origami_store(self.config.store_uri)
        self.connector_scheduler: OrigamiConnectorScheduler | None = None
        self.connector_worker: OrigamiConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OrigamiConnectorScheduler(
                vllm_config, self.config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OrigamiConnectorWorker(
                self.config, self.store, kv_cache_config
            )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OrigamiConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self.connector_worker is not None
        self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del kwargs
        assert self.connector_worker is not None
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata)

    def wait_for_save(self):
        assert self.connector_worker is not None
        self.connector_worker.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        del finished_req_ids
        if self.connector_worker is None:
            return set(), set()
        return self.connector_worker.get_finished()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any) -> None:
        if self.connector_scheduler is None:
            return
        observed = None
        for source in (
            getattr(connector_output, "kv_connector_stats", None),
            getattr(connector_output, "kv_connector_worker_meta", None),
        ):
            if source is None:
                continue
            for attr in ("origami_pcie_gbps", "pcie_gbps"):
                value = getattr(source, attr, None)
                if value is not None:
                    observed = value
                    break
            if observed is not None:
                break
        if observed is not None:
            self.connector_scheduler.observe_pcie_bandwidth_gbps(float(observed))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

