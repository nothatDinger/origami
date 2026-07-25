# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import (
    OrigamiConfig,
    _parse_ratio,
    fused_attention_compatibility,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.pipeline import (
    select_gpu_lossless_requests,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    OrigamiConnectorMetadata,
    OrigamiRestoreRequest,
    OrigamiSaveRequest,
)
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class OrigamiOffloadController:
    LEVELS = (0, 25, 50, 75, 100)

    def __init__(self, config: OrigamiConfig):
        self.config = config
        ratio = _parse_ratio(config.gpu_lossless_ratio)
        if ratio == "auto":
            self._level = 0
            self._auto = True
        else:
            self._level = self.LEVELS.index(int(ratio))
            self._auto = False
        self._high_steps = 0
        self._low_steps = 0

    @property
    def ratio(self) -> int:
        return self.LEVELS[self._level]

    def observe_schedule_step(self, pcie_gbps: float) -> int:
        if not self._auto:
            return self.ratio
        if pcie_gbps >= self.config.pcie_high_watermark_gbps:
            self._high_steps += 1
            self._low_steps = 0
            if self._high_steps >= 3 and self._level < len(self.LEVELS) - 1:
                self._level += 1
                self._high_steps = 0
        elif (
            self.config.auto_decrease_enabled
            and self.config.pcie_low_watermark_gbps > 0
            and pcie_gbps <= self.config.pcie_low_watermark_gbps
        ):
            self._low_steps += 1
            self._high_steps = 0
            if self._low_steps >= 3 and self._level > 0:
                self._level -= 1
                self._low_steps = 0
        else:
            self._high_steps = 0
            self._low_steps = 0
        return self.ratio


def _request_params(request: "Request") -> dict[str, Any]:
    params = getattr(request, "kv_transfer_params", None)
    return params if isinstance(params, dict) else {}


def _is_save_prefill_request(request: "Request") -> bool:
    params = _request_params(request)
    return bool(params.get("origami_save_prefill"))


def _is_resume_prefill_request(request: "Request") -> bool:
    if _is_save_prefill_request(request):
        return False
    params = _request_params(request)
    return bool(
        params.get("origami_resume_prefill")
        or params.get("origami_cache_key")
        or params.get("origami_payload_id")
    )


def _cache_key(request: "Request") -> str:
    params = _request_params(request)
    return str(
        params.get("origami_cache_key")
        or params.get("origami_payload_id")
        or request.request_id
    )


def _metric_request_id(request: "Request") -> str:
    params = _request_params(request)
    return str(
        params.get("origami_request_id")
        or params.get("transfer_id")
        or request.request_id
    )


@dataclass(frozen=True)
class _SavePlan:
    cache_key: str
    token_start: int = 0
    token_limit: int | None = None


class OrigamiConnectorScheduler:

    def __init__(self, vllm_config: "VllmConfig", config: OrigamiConfig):
        self.vllm_config = vllm_config
        self.config = config
        self.block_size = vllm_config.cache_config.block_size
        self.controller = OrigamiOffloadController(config)
        self.fused_enabled, self.fused_disabled_reason = (
            fused_attention_compatibility(vllm_config, config)
        )
        self._restored_requests: set[str] = set()
        self._restored_prefix_tokens: dict[str, int] = {}
        self._pending_restores: dict[str, OrigamiRestoreRequest] = {}
        self._save_candidates: dict[str, _SavePlan] = {}
        self._last_observed_pcie_gbps = 0.0

    def observe_pcie_bandwidth_gbps(self, value: float) -> None:
        self._last_observed_pcie_gbps = max(0.0, float(value))

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        req_id = request.request_id
        params = _request_params(request)
        configured_tokens = params.get("origami_num_tokens")

        if _is_save_prefill_request(request):
            if configured_tokens is None:
                tokens_to_save = int(request.num_tokens)
            else:
                tokens_to_save = max(0, min(int(configured_tokens), request.num_tokens))
            if tokens_to_save > 0:
                self._save_candidates[req_id] = _SavePlan(
                    cache_key=_cache_key(request),
                    token_start=0,
                    token_limit=tokens_to_save,
                )
            return 0, False

        if req_id in self._restored_requests:
            return 0, False
        if not _is_resume_prefill_request(request):
            return 0, False
        if configured_tokens is None:
            remaining = max(0, request.num_tokens - num_computed_tokens)
            matched = max(0, remaining - 1)
        else:
            target = max(0, min(int(configured_tokens), request.num_tokens))
            matched = max(0, target - int(num_computed_tokens))
        if matched <= 0:
            return 0, False
        return matched, True

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        req_id = request.request_id
        if num_external_tokens <= 0:
            return
        if not _is_resume_prefill_request(request):
            return
        block_groups = blocks.get_block_ids()
        num_blocks = cdiv(num_external_tokens, self.block_size)
        block_ids_per_group = tuple(
            tuple(int(block_id) for block_id in group[:num_blocks])
            for group in block_groups
        )
        self._pending_restores[req_id] = OrigamiRestoreRequest(
            request_id=req_id,
            cache_key=_cache_key(request),
            block_ids_per_group=block_ids_per_group,
            num_tokens=int(num_external_tokens),
            lossless_path="cpu",
            metric_request_id=_metric_request_id(request),
        )
        self._restored_requests.add(req_id)
        self._restored_prefix_tokens[req_id] = int(num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> OrigamiConnectorMetadata:
        # Preemption frees the paged blocks but not the worker's compressed
        # payload. Permit the request to match the external prefix again so
        # the allocator reserves replacement logical blocks.
        self._restored_requests.difference_update(
            scheduler_output.preempted_req_ids or set()
        )
        ratio = self.controller.observe_schedule_step(self._last_observed_pcie_gbps)
        request_ids = sorted(self._pending_restores)
        gpu_request_ids = select_gpu_lossless_requests(request_ids, ratio)

        restores: dict[str, OrigamiRestoreRequest] = {}
        for req_id, restore in self._pending_restores.items():
            restores[req_id] = OrigamiRestoreRequest(
                request_id=restore.request_id,
                cache_key=restore.cache_key,
                block_ids_per_group=restore.block_ids_per_group,
                num_tokens=restore.num_tokens,
                lossless_path="gpu" if req_id in gpu_request_ids else "cpu",
                metric_request_id=restore.metric_request_id,
            )

        saves = self._build_save_metadata(scheduler_output)
        fused_request_ids: tuple[str, ...] = ()
        fused_query_start_positions: dict[str, int] = {}
        fused_query_token_counts: dict[str, int] = {}
        fused_request_indices: dict[str, int] = {}
        fused_codecs: dict[str, str] = {}
        fused_prefix_lengths: dict[str, int] = {}
        if self.fused_enabled:
            fused_request_ids = tuple(
                req_id
                for req_id in scheduler_output.num_scheduled_tokens
                if req_id in self._restored_requests
            )
            fused_set = set(fused_request_ids)
            for request in scheduler_output.scheduled_new_reqs:
                if request.req_id in fused_set:
                    fused_query_start_positions[request.req_id] = int(
                        request.num_computed_tokens
                    )
            cached = scheduler_output.scheduled_cached_reqs
            for req_id, num_computed_tokens in zip(
                cached.req_ids, cached.num_computed_tokens
            ):
                if req_id in fused_set:
                    fused_query_start_positions[req_id] = int(num_computed_tokens)
            fused_query_token_counts = {
                req_id: int(scheduler_output.num_scheduled_tokens[req_id])
                for req_id in fused_request_ids
            }
            fused_request_indices = {
                req_id: index
                for index, req_id in enumerate(
                    scheduler_output.num_scheduled_tokens
                )
                if req_id in fused_set
            }
            fused_codecs = {
                req_id: self.config.quantizer for req_id in fused_request_ids
            }
            fused_prefix_lengths = {
                req_id: int(self._restored_prefix_tokens[req_id])
                for req_id in fused_request_ids
            }
        self._pending_restores.clear()
        return OrigamiConnectorMetadata(
            reqs_to_restore=restores,
            reqs_to_save=saves,
            gpu_lossless_ratio=ratio,
            fused_request_ids=fused_request_ids,
            fused_query_start_positions=fused_query_start_positions,
            fused_query_token_counts=fused_query_token_counts,
            fused_request_indices=fused_request_indices,
            fused_codecs=fused_codecs,
            fused_prefix_lengths=fused_prefix_lengths,
        )

    def _build_save_metadata(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, OrigamiSaveRequest]:
        saves: dict[str, OrigamiSaveRequest] = {}

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            if req_id in self._save_candidates:
                plan = self._save_candidates[req_id]
                scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                incremental_tokens = max(0, scheduled - plan.token_start)
                if plan.token_limit is not None:
                    incremental_tokens = min(
                        incremental_tokens,
                        max(0, plan.token_limit - plan.token_start),
                    )
                if incremental_tokens <= 0:
                    continue
                saves[req_id] = OrigamiSaveRequest(
                    request_id=req_id,
                    cache_key=plan.cache_key,
                    block_ids_per_group=tuple(tuple(g) for g in new_req.block_ids),
                    num_tokens=incremental_tokens,
                    token_start=plan.token_start,
                )

        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            new_blocks = cached.new_block_ids[index]
            if new_blocks is None:
                continue
            token_count = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if token_count <= 0:
                continue
            plan = self._save_candidates.get(req_id)
            if plan is None:
                continue
            token_start = (
                int(cached.num_computed_tokens[index])
                if index < len(cached.num_computed_tokens)
                else 0
            )
            if plan is not None and plan.token_limit is not None:
                token_count = min(token_count, max(0, plan.token_limit - token_start))
                if token_count <= 0:
                    continue
            saves[req_id] = OrigamiSaveRequest(
                request_id=req_id,
                cache_key=plan.cache_key,
                block_ids_per_group=tuple(tuple(g) for g in new_blocks),
                num_tokens=int(token_count),
                token_start=token_start,
            )
        return saves

    def request_finished(self, request: "Request") -> None:
        req_id = request.request_id
        self._restored_requests.discard(req_id)
        self._restored_prefix_tokens.pop(req_id, None)
        self._save_candidates.pop(req_id, None)
