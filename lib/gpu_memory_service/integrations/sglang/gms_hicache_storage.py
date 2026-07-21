# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang HiCache storage backend backed by the GMS daemon.

SGLang invokes storage backends from its prefetch and backup threads.  The
backend therefore keeps the interface synchronous and lets HiCache own all
scheduling, prefix matching, and host-pool allocation.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)


class GMSHiCacheStorage(HiCacheStorage):
    """Persist native SGLang HiCache pages in GMS-owned host memory."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        config = storage_config.extra_config or {}
        backend_kwargs = backend_kwargs or {}
        self._socket = str(
            config.get("daemon_socket")
            or os.environ.get("GMS_KV_DAEMON_SOCKET")
            or os.environ.get("GMS_KVR_DAEMON_SOCKET")
            or ""
        )
        client_factory = backend_kwargs.get("client_factory")
        if not self._socket and client_factory is None:
            raise ValueError("GMS daemon socket is required")
        if client_factory is None:
            from gms_kv_ring.daemon.client import DaemonClient

            def client_factory():
                return DaemonClient(self._socket)

        self._client_factory: Callable[[], Any] = client_factory
        self._client = None
        self._client_lock = threading.Lock()
        self._namespace = str(
            config.get("gms_namespace") or self._default_namespace(storage_config)
        )
        self.registered_pools: dict[Any, Any] = {}

    @staticmethod
    def _default_namespace(config: HiCacheStorageConfig) -> str:
        model = config.model_name or "unknown-model"
        tp = (
            "replicated"
            if config.is_mla_model
            else f"{config.tp_rank}/{config.tp_size}"
        )
        return (
            "sglang-hicache-v1"
            f"|model={model}"
            f"|tp={tp}"
            f"|pp={config.pp_rank}/{config.pp_size}"
            f"|cp={config.attn_cp_rank}/{config.attn_cp_size}"
            f"|page_first={int(config.is_page_first_layout)}"
        )

    def _call(self, operation):
        with self._client_lock:
            for attempt in range(2):
                try:
                    if self._client is None:
                        self._client = self._client_factory()
                    return operation(self._client)
                except Exception:
                    if self._client is not None:
                        try:
                            self._client.close()
                        except Exception:
                            pass
                        self._client = None
                    if attempt:
                        raise
        raise AssertionError("unreachable")

    @staticmethod
    def _key(key: str, pool_name: Any = PoolName.KV) -> bytes:
        if pool_name not in (PoolName.KV, "kv", "__default__"):
            key = f"{key}.{pool_name}"
        return key.encode("utf-8")

    @staticmethod
    def _to_bytes(value: torch.Tensor) -> bytes:
        return value.detach().contiguous().view(torch.uint8).numpy().tobytes()

    @staticmethod
    def _copy_bytes(value: bytes, target: torch.Tensor) -> torch.Tensor:
        target_bytes = target.view(torch.uint8).reshape(-1)
        if len(value) != target_bytes.numel():
            raise ValueError(
                f"GMS page size mismatch: expected {target_bytes.numel()}, got {len(value)}"
            )
        target_bytes.copy_(torch.frombuffer(bytearray(value), dtype=torch.uint8))
        return target

    def _lookup(self, keys: list[bytes]) -> list[bool]:
        if not keys:
            return []
        return self._call(
            lambda client: client.persistent_kv_lookup(self._namespace, keys)
        )

    def _load(self, keys: list[bytes]) -> list[bytes | None]:
        if not keys:
            return []
        return self._call(
            lambda client: client.persistent_kv_load(self._namespace, keys)
        )

    def _store(self, items: list[tuple[bytes, bytes]]) -> list[bool]:
        if not items:
            return []
        return self._call(
            lambda client: client.persistent_kv_store(self._namespace, items)
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor | None = None,
        target_sizes: Any = None,
    ) -> torch.Tensor | None:
        if target_location is None:
            raise ValueError("target_location is required")
        value = self._load([self._key(key)])[0]
        return None if value is None else self._copy_bytes(value, target_location)

    def batch_get(
        self,
        keys: list[str],
        target_locations: list[torch.Tensor] | None = None,
        target_sizes: Any = None,
    ) -> list[torch.Tensor | None]:
        if target_locations is None or len(target_locations) != len(keys):
            raise ValueError("one target location is required for each key")
        values = self._load([self._key(key) for key in keys])
        if len(values) != len(keys):
            raise ValueError("GMS returned the wrong number of pages")
        return [
            None if value is None else self._copy_bytes(value, target)
            for value, target in zip(values, target_locations)
        ]

    def set(
        self,
        key: str,
        value: torch.Tensor | None = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        if value is None:
            raise ValueError("value is required")
        return self._store([(self._key(key), self._to_bytes(value))]) == [True]

    def batch_set(
        self,
        keys: list[str],
        values: list[torch.Tensor] | None = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        if values is None or len(values) != len(keys):
            raise ValueError("one value is required for each key")
        stored = self._store(
            [
                (self._key(key), self._to_bytes(value))
                for key, value in zip(keys, values)
            ]
        )
        return len(stored) == len(keys) and all(stored)

    def exists(self, key: str) -> bool:
        return self._lookup([self._key(key)]) == [True]

    def batch_exists(self, keys: list[str], extra_info=None) -> int:
        hits = self._lookup([self._key(key) for key in keys])
        return next((index for index, hit in enumerate(hits) if not hit), len(hits))

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self._pool_io(PoolName.KV, keys, host_indices, self._load_pages)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self._pool_io(PoolName.KV, keys, host_indices, self._store_pages)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        query = [(PoolName.KV, key) for key in keys]
        query.extend(
            (transfer.name, key) for transfer in pool_transfers or [] for key in keys
        )
        hits = self._lookup([self._key(key, pool) for pool, key in query])
        present = dict(zip(query, hits))
        kv_pages = next(
            (index for index, key in enumerate(keys) if not present[PoolName.KV, key]),
            len(keys),
        )
        counts: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        for transfer in pool_transfers or []:
            if not final_pages:
                break
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (
                        index
                        for index in range(kv_pages)
                        if not present[transfer.name, keys[index]]
                    ),
                    kv_pages,
                )
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = next(
                    (
                        prefix
                        for prefix in range(kv_pages, 0, -1)
                        if all(
                            present[transfer.name, keys[index]]
                            for index in range(max(0, prefix - trailing), prefix)
                        )
                    ),
                    0,
                )
            if boundary:
                counts[transfer.name] = boundary
            final_pages = min(final_pages, boundary)
        return PoolTransferResult(final_pages, counts)

    def _pool_io(self, pool_name, keys, host_indices, operation):
        host_pool = (
            self.mem_pool_host
            if pool_name == PoolName.KV
            else self.registered_pools[pool_name]
        )
        page_size = getattr(host_pool, "page_size", 1) or 1
        if host_indices is None or host_indices.numel() != len(keys) * page_size:
            return [False] * len(keys)
        slots = [
            int(host_indices[index * page_size].item()) for index in range(len(keys))
        ]
        return operation(pool_name, keys, host_pool, slots)

    def _load_pages(self, pool_name, keys, host_pool, slots):
        values = self._load([self._key(key, pool_name) for key in keys])
        results = []
        for value, slot in zip(values, slots):
            if value is None:
                results.append(False)
                continue
            target = host_pool.get_dummy_flat_data_page()
            self._copy_bytes(value, target)
            host_pool.set_from_flat_data_page(slot, target)
            results.append(True)
        return results

    def _store_pages(self, pool_name, keys, host_pool, slots):
        return self._store(
            [
                (
                    self._key(key, pool_name),
                    self._to_bytes(host_pool.get_data_page(slot, flat=True)),
                )
                for key, slot in zip(keys, slots)
            ]
        )

    def _batch_io_v2(self, transfers, operation):
        return {
            str(transfer.name): self._pool_io(
                transfer.name, transfer.keys or [], transfer.host_indices, operation
            )
            for transfer in transfers
        }

    def batch_get_v2(self, transfers, extra_info=None):
        return self._batch_io_v2(transfers, self._load_pages)

    def batch_set_v2(self, transfers, extra_info=None):
        return self._batch_io_v2(transfers, self._store_pages)

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None
