# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.origami.metadata import (
    OrigamiPayload,
)


class OrigamiStore(ABC):

    @abstractmethod
    def put(self, payload: OrigamiPayload) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, cache_key: str) -> OrigamiPayload:
        raise NotImplementedError

    @abstractmethod
    def contains(self, cache_key: str) -> bool:
        raise NotImplementedError


_MEMORY_STORES: dict[str, dict[str, OrigamiPayload]] = {}
_MEMORY_LOCK = Lock()


class InMemoryOrigamiStore(OrigamiStore):

    def __init__(self, namespace: str = "origami"):
        self.namespace = namespace
        with _MEMORY_LOCK:
            _MEMORY_STORES.setdefault(namespace, {})

    def put(self, payload: OrigamiPayload) -> None:
        with _MEMORY_LOCK:
            _MEMORY_STORES[self.namespace][payload.cache_key] = payload

    def get(self, cache_key: str) -> OrigamiPayload:
        with _MEMORY_LOCK:
            try:
                return _MEMORY_STORES[self.namespace][cache_key]
            except KeyError as exc:
                raise KeyError(f"Origami payload not found: {cache_key}") from exc

    def contains(self, cache_key: str) -> bool:
        with _MEMORY_LOCK:
            return cache_key in _MEMORY_STORES[self.namespace]


class LocalFileOrigamiStore(OrigamiStore):

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str) -> Path:
        safe_key = cache_key.replace("/", "_").replace(":", "_")
        return self.root / f"{safe_key}.pt"

    def put(self, payload: OrigamiPayload) -> None:
        torch.save(payload, self._path(payload.cache_key))

    def get(self, cache_key: str) -> OrigamiPayload:
        path = self._path(cache_key)
        if not path.exists():
            raise KeyError(f"Origami payload not found: {cache_key}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def contains(self, cache_key: str) -> bool:
        return self._path(cache_key).exists()


def create_origami_store(uri: str) -> OrigamiStore:
    if uri.startswith("memory://"):
        namespace = uri[len("memory://"):] or "origami"
        return InMemoryOrigamiStore(namespace)
    if uri.startswith("file://"):
        return LocalFileOrigamiStore(uri[len("file://"):])
    return LocalFileOrigamiStore(uri)

