# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.benchmark_utils import (
    controlled_read,
    copy_kv_to_hbm,
    nvtx_range,
    preload_memory_artifacts,
    safe_cache_key,
    safe_layer_name,
    torch_load_from_bytes,
    write_jsonl,
)
from vllm.distributed.kv_transfer.kv_connector.v1.origami.config import OrigamiConfig
from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless import native_cpu
from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import OrigamiPayload
from vllm.distributed.kv_transfer.kv_connector.v1.origami.worker import OrigamiConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReuseReqMeta:
    request_id: str
    transfer_id: str
    remote_address: str
    block_ids: torch.Tensor
    num_tokens: int

    @classmethod
    def make(
        cls,
        *,
        request_id: str,
        transfer_id: str,
        remote_address: str,
        token_ids: list[int] | None,
        block_ids: list[int],
    ) -> "ReuseReqMeta":
        return cls(
            request_id=request_id,
            transfer_id=transfer_id,
            remote_address=remote_address,
            block_ids=torch.tensor([int(b) for b in block_ids], dtype=torch.long),
            num_tokens=len(token_ids or []),
        )


@dataclass(frozen=True)
class ReuseRestoreMeta:
    request_id: str
    transfer_id: str
    cache_key: str
    block_ids_per_group: tuple[tuple[int, ...], ...]
    num_tokens: int


@dataclass
class ReuseConnectorMetadata(KVConnectorMetadata):
    requests: list[ReuseReqMeta] = field(default_factory=list)
    restores: dict[str, ReuseRestoreMeta] = field(default_factory=dict)
    system: str = "unknown"

    def add_request(
        self,
        *,
        request_id: str,
        transfer_id: str,
        remote_address: str,
        token_ids: list[int] | None,
        block_ids: list[int],
    ) -> None:
        self.requests.append(
            ReuseReqMeta.make(
                request_id=request_id,
                transfer_id=transfer_id,
                remote_address=remote_address,
                token_ids=token_ids,
                block_ids=block_ids,
            )
        )


@dataclass
class _ChunkedPrefillState:
    block_ids: list[int]
    prompt_token_ids: list[int] | None
    kv_transfer_params: dict[str, Any]


class _NullStore:
    def put(self, payload: OrigamiPayload) -> None:  # pragma: no cover - worker helper
        del payload

    def get(self, cache_key: str) -> OrigamiPayload:  # pragma: no cover
        raise KeyError(cache_key)

    def contains(self, cache_key: str) -> bool:  # pragma: no cover
        del cache_key
        return False


def _request_params(request: "Request") -> dict[str, Any]:
    params = getattr(request, "kv_transfer_params", None)
    return params if isinstance(params, dict) else {}


def _sampling_kv_params(sampling_params: Any) -> dict[str, Any]:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return {}
    params = extra_args.get("kv_transfer_params")
    return params if isinstance(params, dict) else {}


def _cache_key_from_params(params: dict[str, Any]) -> str | None:
    value = params.get("cache_key") or params.get("origami_cache_key") or params.get(
        "origami_payload_id"
    )
    return str(value) if value is not None else None


def _cache_key(request: "Request") -> str | None:
    return _cache_key_from_params(_request_params(request))


def _cache_tokens_from_params(params: dict[str, Any], num_tokens: int) -> int:
    value = params.get("cached_prefix_tokens", params.get("origami_num_tokens"))
    if value is None:
        return 0
    return max(0, min(int(value), int(num_tokens)))


def _cache_tokens(request: "Request") -> int:
    return _cache_tokens_from_params(_request_params(request), int(request.num_tokens))


def _transfer_id_from_params(default: str, params: dict[str, Any]) -> str:
    value = params.get("transfer_id") or params.get("remote_request_id")
    return str(value) if value is not None else str(default)


def _remote_decode_enabled(params: dict[str, Any]) -> bool:
    return bool(params.get("do_remote_decode", True))


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _address_from_request_id(request_id: str, is_prefill: bool) -> str | None:
    pattern = r"___decode_addr_(.*):(\d+)" if is_prefill else r"___prefill_addr_(.*):(\d+)___"
    match = re.search(pattern, request_id)
    if not match:
        return None
    return f"{match.group(1)}:{int(match.group(2))}"


def _remote_address_from_params(
    *,
    request_id: str,
    params: dict[str, Any],
    is_prefill: bool,
) -> str:
    keys = ("decode_addr", "remote_decode_addr", "remote_address") if is_prefill else (
        "prefill_addr",
        "remote_prefill_addr",
        "remote_address",
    )
    for key in keys:
        value = params.get(key)
        if value:
            return str(value)
    parsed = _address_from_request_id(request_id, is_prefill=is_prefill)
    if parsed is not None:
        return parsed
    side = "decode" if is_prefill else "prefill"
    raise ValueError(
        f"{request_id} missing {side} address; pass kv_transfer_params.{side}_addr"
    )


def _ranked_address(base_address: str, rank: int) -> str:
    host, sep, port = base_address.rpartition(":")
    if not sep:
        raise ValueError(f"Invalid P/D address {base_address!r}; expected host:port")
    return f"{host}:{int(port) + int(rank)}"


def _layer_index(layer_name: str) -> int:
    digits = ""
    for ch in reversed(str(layer_name)):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else 0


def _select_artifact_layer(tensor: torch.Tensor, layer_name: str) -> torch.Tensor:
    if tensor.dim() == 5 and tensor.shape[0] == 1 and tensor.shape[1] == 2:
        return tensor[0]
    if tensor.dim() == 5 and tensor.shape[0] > 2 and tensor.shape[1] == 2:
        index = _layer_index(layer_name)
        if index >= tensor.shape[0]:
            raise IndexError(
                f"Layer index {index} out of range for artifact shape {tuple(tensor.shape)}"
            )
        return tensor[index]
    return tensor


def _setup_cachegen_runtime(repo_root: Path) -> None:
    import sys
    import types

    cachegen_root = repo_root / "thrid_party" / "CacheGen"
    lmcache_root = cachegen_root / "LMCache"
    torchac_root = lmcache_root / "third_party" / "torchac_cuda"
    cachegen_deps = cachegen_root / ".cache" / "cachegen_deps"
    for import_path in (cachegen_root, lmcache_root, torchac_root, cachegen_deps):
        value = str(import_path)
        if value not in sys.path:
            sys.path.insert(0, value)

    try:
        import nvtx  # noqa: F401
    except ModuleNotFoundError:
        nvtx_module = types.ModuleType("nvtx")

        def annotate(*args: Any, **kwargs: Any):
            del kwargs
            if args and callable(args[0]) and len(args) == 1:
                return args[0]

            def decorator(func: Any) -> Any:
                return func

            return decorator

        nvtx_module.annotate = annotate
        nvtx_module.start_range = lambda *args, **kwargs: None
        nvtx_module.end_range = lambda *args, **kwargs: None
        sys.modules["nvtx"] = nvtx_module


def _extract_layer_blocks(
    layer: torch.Tensor,
    block_ids: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor | None:
    if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
        return layer[block_ids.to(layer.device), ...]
    if layer.shape[0] == 2:
        return layer[:, block_ids.to(layer.device), ...]
    return None


def _inject_layer_blocks(
    layer: torch.Tensor,
    block_ids: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
) -> None:
    block_ids = block_ids.to(layer.device)
    kv_cache = kv_cache.to(device=layer.device, dtype=layer.dtype, non_blocking=False)
    if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
        num_blocks = kv_cache.shape[0]
        layer[block_ids[:num_blocks], ...] = kv_cache
        return
    if layer.shape[0] == 2:
        num_blocks = kv_cache.shape[1]
        layer[:, block_ids[:num_blocks], ...] = kv_cache
        return
    raise ValueError(f"Unsupported KV cache layout for layer shape {tuple(layer.shape)}")


class _BenchmarkReuseConnectorBase(KVConnectorBase_V1, SupportsHMA):
    system_name = "base"

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self.is_producer = bool(self._kv_transfer_config.is_kv_producer)
        self.is_consumer = bool(self._kv_transfer_config.is_kv_consumer)
        self._requests_need_load: dict[str, tuple[Any, list[int]]] = {}
        self._pending_restores: dict[str, ReuseRestoreMeta] = {}
        self._chunked_prefill: dict[str, _ChunkedPrefillState] = {}
        self._request_id_to_transfer_id: dict[str, str] = {}
        self._artifact_payload_cache: dict[tuple[str, str], Any] = {}
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.layer_to_cache_group: dict[str, int] = {}

        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self.bandwidth_gbps = float(extra.get("reuse_read_bandwidth_gbps", 0.0))
        self.artifact_ingress_mode = str(
            extra.get("reuse_artifact_ingress_mode", "file")
        ).lower().replace("_", "-")
        if self.artifact_ingress_mode == "native-file":
            self.artifact_ingress_mode = "memory"
        self.artifact_cache_keys = tuple(
            str(key)
            for key in (extra.get("reuse_artifact_cache_keys", ()) or ())
            if key
        )
        self.metrics_dir = str(extra.get("reuse_metrics_dir", "")) or None
        self.raw_root = Path(str(extra.get("raw_kv_root", extra.get("backend_raw_uri", ""))))
        self.cachegen_root = Path(str(extra.get("cachegen_store_uri", "")))
        self.origami_root = Path(str(extra.get("origami_store_uri", "")).removeprefix("file://"))
        self.cachegen_model_name = str(extra.get("cachegen_model_name", "Mistral-7B"))
        self.cachegen_quant_level = str(extra.get("cachegen_quant_level", "2"))
        self.enable_p2p = bool(extra.get("reuse_enable_p2p", True))

        self.metrics_path = None
        if self.metrics_dir is not None:
            self.metrics_path = str(Path(self.metrics_dir) / f"{self.system_name}_metrics.jsonl")
        if self.artifact_ingress_mode not in {"file", "memory"}:
            raise ValueError(
                "reuse_artifact_ingress_mode must be one of {'file', 'memory'}, "
                f"got {self.artifact_ingress_mode!r}"
            )
        if (
            role == KVConnectorRole.WORKER
            and self.artifact_ingress_mode == "memory"
            and self.artifact_cache_keys
        ):
            self._preload_memory_artifacts()

        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = (
            get_world_group().local_rank if role == KVConnectorRole.WORKER else 0
        )
        self.p2p_nccl_engine = (
            P2pNcclEngine(
                local_rank=self._local_rank,
                config=self._kv_transfer_config,
                hostname="",
                port_offset=self._rank,
            )
            if role == KVConnectorRole.WORKER and self.enable_p2p
            else None
        )
        self._origami_worker: OrigamiConnectorWorker | None = None
        if self.system_name == "origami" and role == KVConnectorRole.WORKER:
            self._origami_worker = OrigamiConnectorWorker(
                OrigamiConfig.from_vllm_config(vllm_config), _NullStore(), kv_cache_config
            )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = dict(kv_caches)
        self.layer_to_cache_group = self._build_layer_to_cache_group()
        if self._origami_worker is not None:
            self._origami_worker.register_kv_caches(kv_caches)

    def _build_layer_to_cache_group(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        groups = getattr(self._kv_cache_config, "kv_cache_groups", None)
        if groups:
            for idx, group in enumerate(groups):
                for layer_name in getattr(group, "layer_names", ()):  # pragma: no branch
                    mapping[str(layer_name)] = idx
        for layer_name in self.kv_caches:
            mapping.setdefault(layer_name, 0)
        return mapping

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            cache_key = _cache_key(request)
            matched = _cache_tokens(request)
            if cache_key is None or matched <= 0:
                return 0, False
            remaining = int(request.num_tokens) - int(num_computed_tokens)
            return max(0, min(matched, max(0, remaining - 1))), False

        params = _request_params(request)
        if params and not params.get("do_remote_prefill", True):
            return 0, False
        prompt_token_ids = request.prompt_token_ids or []
        num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens
        return max(0, num_external_tokens), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens <= 0:
            return
        params = _request_params(request)
        transfer_id = _transfer_id_from_params(request.request_id, params)
        if self.is_producer:
            cache_key = _cache_key_from_params(params)
            if cache_key is None:
                return
            block_groups = blocks.get_block_ids()
            restore_tokens = _cache_tokens_from_params(params, int(request.num_tokens))
            if restore_tokens <= 0:
                restore_tokens = int(num_external_tokens)
            num_blocks = (restore_tokens + self._block_size - 1) // self._block_size
            for group in block_groups:
                if len(group) < num_blocks:
                    raise ValueError(
                        "Insufficient KV blocks for SSD restore: "
                        f"request_id={request.request_id}, cache_key={cache_key}, "
                        f"needed={num_blocks}, allocated={len(group)}"
                    )
            self._pending_restores[request.request_id] = ReuseRestoreMeta(
                request_id=request.request_id,
                transfer_id=transfer_id,
                cache_key=cache_key,
                block_ids_per_group=tuple(
                    tuple(int(block_id) for block_id in group[:num_blocks])
                    for group in block_groups
                ),
                num_tokens=restore_tokens,
            )
            return

        self._requests_need_load[request.request_id] = (
            request,
            [int(block_id) for block_id in blocks.get_block_ids()[0]],
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ReuseConnectorMetadata:
        meta = ReuseConnectorMetadata(system=self.system_name)
        if self.is_producer:
            meta.restores.update(self._pending_restores)
            self._pending_restores.clear()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                params = _sampling_kv_params(new_req.sampling_params)
                if not _remote_decode_enabled(params):
                    continue
                transfer_id = _transfer_id_from_params(new_req.req_id, params)
                remote_address = _remote_address_from_params(
                    request_id=new_req.req_id,
                    params=params,
                    is_prefill=True,
                )
                self._request_id_to_transfer_id[new_req.req_id] = transfer_id
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                if num_tokens < len(new_req.prompt_token_ids or []):
                    self._chunked_prefill[new_req.req_id] = _ChunkedPrefillState(
                        block_ids=[int(b) for b in new_req.block_ids[0]],
                        prompt_token_ids=new_req.prompt_token_ids,
                        kv_transfer_params=dict(params),
                    )
                    continue
                meta.add_request(
                    request_id=new_req.req_id,
                    transfer_id=transfer_id,
                    remote_address=remote_address,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=[int(b) for b in new_req.block_ids[0]],
                )
                continue
            if new_req.req_id in self._requests_need_load:
                request, block_ids = self._requests_need_load.pop(new_req.req_id)
                params = _request_params(request)
                transfer_id = _transfer_id_from_params(new_req.req_id, params)
                remote_address = _remote_address_from_params(
                    request_id=new_req.req_id,
                    params=params,
                    is_prefill=False,
                )
                self._request_id_to_transfer_id[new_req.req_id] = transfer_id
                meta.add_request(
                    request_id=new_req.req_id,
                    transfer_id=transfer_id,
                    remote_address=remote_address,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=block_ids,
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed = req_id in cached_reqs.resumed_req_ids
            if self.is_producer:
                if req_id not in self._chunked_prefill or new_block_ids is None:
                    continue
                state = self._chunked_prefill[req_id]
                params = state.kv_transfer_params
                if not _remote_decode_enabled(params):
                    self._chunked_prefill.pop(req_id, None)
                    continue
                transfer_id = _transfer_id_from_params(req_id, params)
                remote_address = _remote_address_from_params(
                    request_id=req_id,
                    params=params,
                    is_prefill=True,
                )
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                block_ids = [int(b) for b in new_block_ids[0]]
                if not resumed:
                    block_ids = state.block_ids + block_ids
                prompt_token_ids = state.prompt_token_ids or []
                if num_tokens < len(prompt_token_ids):
                    self._chunked_prefill[req_id] = _ChunkedPrefillState(
                        block_ids=block_ids,
                        prompt_token_ids=prompt_token_ids,
                        kv_transfer_params=params,
                    )
                    continue
                meta.add_request(
                    request_id=req_id,
                    transfer_id=transfer_id,
                    remote_address=remote_address,
                    token_ids=prompt_token_ids,
                    block_ids=block_ids,
                )
                self._chunked_prefill.pop(req_id, None)
                continue
            if not resumed:
                break
            if req_id in self._requests_need_load and new_block_ids is not None:
                request, _ = self._requests_need_load.pop(req_id)
                params = _request_params(request)
                transfer_id = _transfer_id_from_params(req_id, params)
                remote_address = _remote_address_from_params(
                    request_id=req_id,
                    params=params,
                    is_prefill=False,
                )
                token_ids = request.all_token_ids[: int(num_computed_tokens) + 1]
                meta.add_request(
                    request_id=req_id,
                    transfer_id=transfer_id,
                    remote_address=remote_address,
                    token_ids=token_ids,
                    block_ids=[int(b) for b in new_block_ids[0]],
                )

        if not self.is_producer:
            self._requests_need_load.clear()
        return meta

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ReuseConnectorMetadata)
        if self.is_producer:
            self._restore_from_ssd(metadata)
            return
        self._recv_from_prefill(metadata, forward_context)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not self.is_producer:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ReuseConnectorMetadata)
        if self.p2p_nccl_engine is None:
            return
        for request in metadata.requests:
            self._request_id_to_transfer_id[request.request_id] = request.transfer_id
            remote_address = _ranked_address(request.remote_address, self._rank)
            kv_cache = _extract_layer_blocks(kv_layer, request.block_ids, attn_metadata)
            if kv_cache is None:
                continue
            self.p2p_nccl_engine.send_tensor(
                request.transfer_id + "#" + layer_name,
                kv_cache,
                remote_address,
            )

    def wait_for_save(self):
        if self.is_producer and self.p2p_nccl_engine is not None:
            self.p2p_nccl_engine.wait_for_sent()

    def get_finished(
        self, finished_req_ids: set[str], **kwargs: Any
    ) -> tuple[set[str] | None, set[str] | None]:
        del kwargs
        if self.p2p_nccl_engine is None:
            return None, None
        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        finished_transfer_ids = set(finished_req_ids)
        for request_id in finished_req_ids:
            transfer_id = self._request_id_to_transfer_id.get(request_id)
            if transfer_id:
                finished_transfer_ids.add(transfer_id)
        return self.p2p_nccl_engine.get_finished(finished_transfer_ids, no_compile_layers)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        self._chunked_prefill.pop(request.request_id, None)
        self._pending_restores.pop(request.request_id, None)
        self._requests_need_load.pop(request.request_id, None)
        self._request_id_to_transfer_id.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        self._chunked_prefill.pop(request.request_id, None)
        self._pending_restores.pop(request.request_id, None)
        self._requests_need_load.pop(request.request_id, None)
        self._request_id_to_transfer_id.pop(request.request_id, None)
        return False, None

    def _restore_from_ssd(self, metadata: ReuseConnectorMetadata) -> None:
        for restore in metadata.restores.values():
            start_ts = time.time()
            write_jsonl(
                self.metrics_path,
                {
                    "type": "request_window_start",
                    "system": self.system_name,
                    "request_id": restore.request_id,
                    "transfer_id": restore.transfer_id,
                    "cache_key": restore.cache_key,
                    "ts": start_ts,
                },
            )
            try:
                for layer_name, kv_cache in self.kv_caches.items():
                    group_index = self.layer_to_cache_group.get(layer_name, 0)
                    if group_index >= len(restore.block_ids_per_group):
                        continue
                    block_ids = restore.block_ids_per_group[group_index]
                    if not block_ids:
                        continue
                    value = self._load_artifact_layer(restore, layer_name)
                    copy_kv_to_hbm(
                        kv_cache=kv_cache,
                        block_ids=block_ids,
                        value=value,
                        request_id=restore.request_id,
                        cache_key=restore.cache_key,
                        system=self.system_name,
                        layer_name=layer_name,
                        metrics_path=self.metrics_path,
                    )
            finally:
                self._artifact_payload_cache.pop((restore.request_id, restore.cache_key), None)
            write_jsonl(
                self.metrics_path,
                {
                    "type": "restore_end",
                    "system": self.system_name,
                    "request_id": restore.request_id,
                    "transfer_id": restore.transfer_id,
                    "cache_key": restore.cache_key,
                    "ts": time.time(),
                },
            )

    def _recv_from_prefill(
        self,
        metadata: ReuseConnectorMetadata,
        forward_context: "ForwardContext",
    ) -> None:
        if not metadata.requests or forward_context.attn_metadata is None:
            return
        assert self.p2p_nccl_engine is not None
        attn_metadata = forward_context.attn_metadata
        for request in metadata.requests:
            self._request_id_to_transfer_id[request.request_id] = request.transfer_id
            remote_address = _ranked_address(request.remote_address, self._rank)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue
                layer_tensor = kv_cache[forward_context.virtual_engine]
                received = self.p2p_nccl_engine.recv_tensor(
                    request.transfer_id + "#" + layer_name,
                    remote_address,
                )
                if received is None:
                    logger.warning("%s recv missing KV for %s", self.system_name, request.request_id)
                    continue
                _inject_layer_blocks(layer_tensor, request.block_ids, received, attn_metadata)

    def _read_artifact(self, path: Path, restore: ReuseRestoreMeta) -> bytes:
        return controlled_read(
            path,
            request_id=restore.request_id,
            cache_key=restore.cache_key,
            system=self.system_name,
            bandwidth_gbps=self.bandwidth_gbps,
            metrics_path=self.metrics_path,
            ingress_mode=self.artifact_ingress_mode,
        )

    def _artifact_paths_for_cache_key(self, cache_key: str) -> list[Path]:
        safe_key = safe_cache_key(cache_key)
        if self.system_name == "raw_kv_reuse":
            root = self.raw_root / safe_key
            full = root / "kv.pt"
            if full.exists():
                return [full]
            return sorted(root.glob("**/*.pt"))
        if self.system_name == "cachegen":
            root = self.cachegen_root / safe_key
            full = root / "payload.bin"
            if full.exists():
                return [full]
            manifest_path = root / "layer_payloads.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                paths: list[Path] = []
                for row in manifest.get("layers", []):
                    payload = row.get("payload")
                    if payload:
                        paths.append(root / str(payload))
                if paths:
                    return paths
            return sorted(root.glob("**/payload.bin"))
        return []

    def _preload_memory_artifacts(self) -> None:
        paths: list[Path] = []
        for cache_key in self.artifact_cache_keys:
            paths.extend(self._artifact_paths_for_cache_key(cache_key))
        existing = [path for path in paths if path.exists()]
        if not existing:
            return
        preload_memory_artifacts(
            existing,
            system=self.system_name,
            metrics_path=self.metrics_path,
        )

    def _load_artifact_layer(
        self,
        restore: ReuseRestoreMeta,
        layer_name: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def parse_request_id(request_id: str, is_prefill: bool = True) -> tuple[str, int]:
        address = _address_from_request_id(request_id, is_prefill=is_prefill)
        if address is None:
            raise ValueError(f"Request id {request_id} does not contain P/D address metadata")
        host, _, port = address.rpartition(":")
        return host, int(port)


class RawKVReuseConnector(_BenchmarkReuseConnectorBase):
    system_name = "raw_kv_reuse"

    def _raw_payload_for_restore(self, restore: ReuseRestoreMeta) -> Any:
        key = (restore.request_id, restore.cache_key)
        cached = self._artifact_payload_cache.get(key)
        if cached is not None:
            return cached
        safe_key = safe_cache_key(restore.cache_key)
        path = self.raw_root / safe_key / "kv.pt"
        if not path.exists():
            raise FileNotFoundError(f"raw KV artifact not found for {restore.cache_key}: {path}")
        with nvtx_range("raw_kv_reuse:deserialize"):
            payload = torch_load_from_bytes(self._read_artifact(path, restore))
        self._artifact_payload_cache[key] = payload
        return payload

    def _load_artifact_layer(self, restore: ReuseRestoreMeta, layer_name: str) -> torch.Tensor:
        key = safe_cache_key(restore.cache_key)
        layer = safe_layer_name(layer_name)
        full_path = self.raw_root / key / "kv.pt"
        if full_path.exists():
            payload = self._raw_payload_for_restore(restore)
        else:
            candidates = [
                self.raw_root / key / f"{layer}.pt",
                self.raw_root / key / layer_name / "kv.pt",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise FileNotFoundError(
                    f"raw KV artifact not found for {restore.cache_key} {layer_name}"
                )
            with nvtx_range("raw_kv_reuse:deserialize"):
                payload = torch_load_from_bytes(self._read_artifact(path, restore))
        if isinstance(payload, torch.Tensor):
            return _select_artifact_layer(payload.detach().cpu(), layer_name)
        if isinstance(payload, dict):
            if "layers" in payload and layer_name in payload["layers"]:
                return _select_artifact_layer(
                    payload["layers"][layer_name].detach().cpu(), layer_name
                )
            if "kv" in payload:
                kv = payload["kv"]
                if isinstance(kv, dict):
                    return _select_artifact_layer(kv[layer_name].detach().cpu(), layer_name)
                if isinstance(kv, torch.Tensor):
                    return _select_artifact_layer(kv.detach().cpu(), layer_name)
        raise ValueError(f"Unsupported raw KV artifact format for {restore.cache_key}")


class CacheGenReuseConnector(_BenchmarkReuseConnectorBase):
    system_name = "cachegen"

    def _record_cachegen_tensor_ready(
        self,
        restore: ReuseRestoreMeta,
        tensor: torch.Tensor,
    ) -> None:
        write_jsonl(
            self.metrics_path,
            {
                "type": "cachegen_restore_tensor_ready",
                "stage": "cachegen_restore_tensor_ready",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "device": str(tensor.device),
                "is_cuda": bool(tensor.is_cuda),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": int(tensor.numel() * tensor.element_size()),
            },
        )

    def _profile_cachegen_gpu_to_cpu(
        self,
        restore: ReuseRestoreMeta,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self.metrics_path is None:
            return tensor.detach().cpu()
        input_bytes = int(tensor.numel() * tensor.element_size())
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_ts = time.time()
            start_perf = time.perf_counter()
            start_event.record()
            result = tensor.detach().cpu()
            end_event.record()
            torch.cuda.synchronize(tensor.device)
            wall_ms = _ms_since(start_perf)
            cuda_ms = float(start_event.elapsed_time(end_event))
            end_ts = time.time()
        else:
            start_ts = time.time()
            start_perf = time.perf_counter()
            result = tensor.detach().cpu()
            wall_ms = _ms_since(start_perf)
            cuda_ms = 0.0
            end_ts = time.time()
        write_jsonl(
            self.metrics_path,
            {
                "type": "cachegen_gpu_to_cpu",
                "stage": "cachegen_gpu_to_cpu",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "ms": wall_ms,
                "cuda_ms": cuda_ms,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "input_bytes": input_bytes,
                "output_bytes": input_bytes,
                "input_gbps": input_bytes * 8.0 / wall_ms / 1e6 if wall_ms > 0 else 0.0,
                "output_gbps": input_bytes * 8.0 / wall_ms / 1e6 if wall_ms > 0 else 0.0,
            },
        )
        return result

    def _with_cachegen_profile_env(
        self,
        restore: ReuseRestoreMeta,
        data_bytes: int,
    ) -> dict[str, str | None]:
        values = {
            "CACHEGEN_PROFILE_METRICS_PATH": str(self.metrics_path or ""),
            "CACHEGEN_PROFILE_SYSTEM": self.system_name,
            "CACHEGEN_PROFILE_REQUEST_ID": restore.request_id,
            "CACHEGEN_PROFILE_TRANSFER_ID": restore.transfer_id,
            "CACHEGEN_PROFILE_CACHE_KEY": restore.cache_key,
            "CACHEGEN_PROFILE_PAYLOAD_BYTES": str(data_bytes),
        }
        previous = {key: os.environ.get(key) for key in values}
        for key, value in values.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        return previous

    @staticmethod
    def _restore_env(previous: dict[str, str | None]) -> None:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _cachegen_payload_for_restore(self, restore: ReuseRestoreMeta) -> torch.Tensor:
        key = (restore.request_id, restore.cache_key)
        cached = self._artifact_payload_cache.get(key)
        if cached is not None:
            assert isinstance(cached, torch.Tensor)
            return cached
        safe_key = safe_cache_key(restore.cache_key)
        payload_path = self.cachegen_root / safe_key / "payload.bin"
        meta_path = self.cachegen_root / safe_key / "meta.json"
        if not payload_path.exists():
            raise FileNotFoundError(
                f"CacheGen payload not found for {restore.cache_key}: {payload_path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        data = self._read_artifact(payload_path, restore)
        if meta.get("format") == "torch.save":
            with nvtx_range("cachegen:decode"):
                payload = torch_load_from_bytes(data)
            tensor = payload["kv"] if isinstance(payload, dict) else payload
        else:
            with nvtx_range("cachegen:decode"):
                repo_root = Path(__file__).resolve().parents[7]
                _setup_cachegen_runtime(repo_root)
                os.environ.setdefault("QUANT_LEVEL", self.cachegen_quant_level)
                from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
                from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

                chunk_size = int(meta.get("chunk_size", meta.get("cached_prefix_tokens", restore.num_tokens)))
                lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
                metadata = LMCacheEngineMetadata(
                    model_name=str(meta.get("model", self.cachegen_model_name)),
                    fmt=str(meta.get("fmt", "huggingface")),
                    world_size=1,
                    worker_id=0,
                )
                previous_env = self._with_cachegen_profile_env(restore, len(data))
                try:
                    tensor = CacheGenDeserializer(lmcache_config, metadata).from_bytes(data)
                finally:
                    self._restore_env(previous_env)
        self._record_cachegen_tensor_ready(restore, tensor)
        self._artifact_payload_cache[key] = tensor
        return tensor

    def _load_artifact_layer(self, restore: ReuseRestoreMeta, layer_name: str) -> torch.Tensor:
        key = safe_cache_key(restore.cache_key)
        layer = safe_layer_name(layer_name)
        full_payload_path = self.cachegen_root / key / "payload.bin"
        layer_manifest_path = self.cachegen_root / key / "layer_payloads.json"
        if full_payload_path.exists() and not layer_manifest_path.exists():
            tensor = self._cachegen_payload_for_restore(restore)
            return _select_artifact_layer(tensor, layer_name)
        layer_dir = self.cachegen_root / key / layer
        payload_path = layer_dir / "payload.bin"
        meta_path = layer_dir / "meta.json"
        if not payload_path.exists():
            tensor_path = self.cachegen_root / key / f"{layer}.pt"
            if tensor_path.exists():
                with nvtx_range("cachegen:decode"):
                    payload = torch_load_from_bytes(self._read_artifact(tensor_path, restore))
                tensor = payload["kv"] if isinstance(payload, dict) else payload
                return _select_artifact_layer(tensor, layer_name)
            raise FileNotFoundError(f"CacheGen payload not found for {restore.cache_key} {layer_name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        data = self._read_artifact(payload_path, restore)
        if meta.get("format") == "torch.save":
            with nvtx_range("cachegen:decode"):
                payload = torch_load_from_bytes(data)
            tensor = payload["kv"] if isinstance(payload, dict) else payload
            return _select_artifact_layer(tensor, layer_name)
        with nvtx_range("cachegen:decode"):
            repo_root = Path(__file__).resolve().parents[7]
            _setup_cachegen_runtime(repo_root)
            os.environ.setdefault("QUANT_LEVEL", self.cachegen_quant_level)
            from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
            from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

            chunk_size = int(meta.get("chunk_size", meta.get("cached_prefix_tokens", restore.num_tokens)))
            lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
            metadata = LMCacheEngineMetadata(
                model_name=str(meta.get("model", self.cachegen_model_name)),
                fmt=str(meta.get("fmt", "huggingface")),
                world_size=1,
                worker_id=0,
            )
            tensor = CacheGenDeserializer(lmcache_config, metadata).from_bytes(data)
        return _select_artifact_layer(tensor, layer_name)


class OrigamiReuseConnector(_BenchmarkReuseConnectorBase):
    system_name = "origami"

    def _origami_payload_for_restore(self, restore: ReuseRestoreMeta) -> OrigamiPayload:
        key = (restore.request_id, restore.cache_key)
        cached = self._artifact_payload_cache.get(key)
        if cached is not None:
            assert isinstance(cached, OrigamiPayload)
            return cached
        safe_key = safe_cache_key(restore.cache_key)
        path = self.origami_root / f"{safe_key}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Origami payload not found for {restore.cache_key}: {path}")
        data = self._read_artifact(path, restore)
        start = time.perf_counter()
        with nvtx_range("origami:artifact_deserialize"):
            payload = torch_load_from_bytes(data)
        write_jsonl(
            self.metrics_path,
            {
                "type": "artifact_deserialize",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "bytes": len(data),
                "ms": _ms_since(start),
            },
        )
        if not isinstance(payload, OrigamiPayload):
            raise ValueError(f"Unsupported Origami payload object: {type(payload)!r}")
        expected_quantizer = (
            getattr(self._origami_worker.quantizer, "quantizer_id", None)
            if self._origami_worker is not None
            else None
        )
        if expected_quantizer and payload.quantizer != expected_quantizer:
            raise ValueError(
                "Origami artifact quantizer mismatch for "
                f"{restore.cache_key}: artifact={payload.quantizer!r}, "
                f"benchmark={expected_quantizer!r}. Regenerate Origami artifacts "
                "with matching --origami-quantizer or pass the legacy quantizer "
                "explicitly for compatibility-only runs."
            )
        self._artifact_payload_cache[key] = payload
        return payload

    @staticmethod
    def _bitunpack_backend() -> str:
        try:
            return str(native_cpu.cpu_isa())
        except Exception as exc:
            return f"unknown:{type(exc).__name__}"

    def _restore_symbols_cpu_profiled(
        self,
        restore: ReuseRestoreMeta,
        layer_payload: Any,
    ) -> torch.Tensor:
        assert self._origami_worker is not None
        compressed_chunks = [chunk.compressed for chunk in layer_payload.chunks]
        output_bytes = [int(chunk.unpacked_bytes) for chunk in layer_payload.chunks]
        compressed_bytes = sum(int(chunk.compressed_bytes) for chunk in layer_payload.chunks)
        unpacked_bytes = sum(output_bytes)
        artifact_codecs = sorted({str(chunk.codec) for chunk in layer_payload.chunks})
        backend = self._origami_worker.cpu_codec.backend

        start = time.perf_counter()
        with nvtx_range("origami:qat_decompress" if backend == "qat" else "origami:lossless_decompress"):
            restored_chunks = self._origami_worker.cpu_codec.decompress_many(
                compressed_chunks,
                output_bytes,
            )
        decompress_ms = _ms_since(start)
        write_jsonl(
            self.metrics_path,
            {
                "type": "lossless_decompress",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "layer_name": layer_payload.layer_name,
                "device": "cpu",
                "backend": backend,
                "artifact_codecs": artifact_codecs,
                "chunks": len(compressed_chunks),
                "compressed_bytes": compressed_bytes,
                "unpacked_bytes": unpacked_bytes,
                "ms": decompress_ms,
                "compressed_gbps": (
                    compressed_bytes * 8.0 / decompress_ms / 1e6
                    if decompress_ms > 0
                    else 0.0
                ),
                "unpacked_gbps": (
                    unpacked_bytes * 8.0 / decompress_ms / 1e6
                    if decompress_ms > 0
                    else 0.0
                ),
            },
        )

        start = time.perf_counter()
        with nvtx_range("origami:bitunpack_avx512"):
            symbols = self._origami_worker._unpack_native_symbols(
                layer_payload,
                restored_chunks,
            )
        bitunpack_ms = _ms_since(start)
        write_jsonl(
            self.metrics_path,
            {
                "type": "bitunpack_avx512",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "layer_name": layer_payload.layer_name,
                "backend": self._bitunpack_backend(),
                "chunks": len(restored_chunks),
                "unpacked_bytes": unpacked_bytes,
                "symbols": int(symbols.numel()),
                "ms": bitunpack_ms,
                "unpacked_gbps": (
                    unpacked_bytes * 8.0 / bitunpack_ms / 1e6
                    if bitunpack_ms > 0
                    else 0.0
                ),
            },
        )
        return symbols

    def _load_artifact_layer(self, restore: ReuseRestoreMeta, layer_name: str) -> torch.Tensor:
        assert self._origami_worker is not None
        payload = self._origami_payload_for_restore(restore)
        layer_payload = payload.layer_payloads.get(layer_name)
        if layer_payload is None:
            raise KeyError(f"Origami payload {restore.cache_key} missing layer {layer_name}")
        symbols = self._restore_symbols_cpu_profiled(restore, layer_payload)
        start = time.perf_counter()
        with nvtx_range("origami:restore_materialize"):
            tensor = self._origami_worker.quantizer.dequantize(
                symbols, layer_payload.quant_metadata
            )
        materialize_ms = _ms_since(start)
        tensor = tensor.detach().cpu()
        bytes_materialized = int(tensor.numel() * tensor.element_size())
        write_jsonl(
            self.metrics_path,
            {
                "type": "restore_materialize",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "layer_name": layer_name,
                "quantizer": getattr(self._origami_worker.quantizer, "quantizer_id", "unknown"),
                "bytes": bytes_materialized,
                "ms": materialize_ms,
                "gbps": (
                    bytes_materialized * 8.0 / materialize_ms / 1e6
                    if materialize_ms > 0
                    else 0.0
                ),
            },
        )
        write_jsonl(
            self.metrics_path,
            {
                "type": "decompress_device",
                "system": self.system_name,
                "request_id": restore.request_id,
                "transfer_id": restore.transfer_id,
                "cache_key": restore.cache_key,
                "layer_name": layer_name,
                "decompress_device": "cpu",
                "backend": self._origami_worker.cpu_codec.backend,
            },
        )
        return tensor
