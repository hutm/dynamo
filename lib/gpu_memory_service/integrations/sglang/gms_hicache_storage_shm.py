# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory transport optimization for the GMS SGLang backend."""

from __future__ import annotations

import logging
import mmap
import os
import threading
import uuid
from dataclasses import dataclass

from gpu_memory_service.integrations.sglang.gms_hicache_storage import (
    GMSHiCacheStorage,
)

logger = logging.getLogger(__name__)


@dataclass
class _StagingPool:
    path: str
    mapped: mmap.mmap
    view: memoryview
    rows: int
    row_size: int
    daemon_id: str | None = None


class GMSSharedMemoryHiCacheStorage(GMSHiCacheStorage):
    """Move HiCache pages through reusable mmap staging instead of JSON."""

    def __init__(self, storage_config, backend_kwargs=None) -> None:
        super().__init__(storage_config, backend_kwargs)
        config = storage_config.extra_config or {}
        self._staging_rows = max(1, int(config.get("gms_staging_rows", 128)))
        self._staging_dir = str(config.get("gms_staging_dir", "/dev/shm"))
        self._staging_pools: dict[int, _StagingPool] = {}
        self._transfer_lock = threading.Lock()

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
                    for pool in self._staging_pools.values():
                        pool.daemon_id = None
                    if attempt:
                        raise
        raise AssertionError("unreachable")

    def _pool(self, row_size: int) -> _StagingPool:
        pool = self._staging_pools.get(row_size)
        if pool is None:
            path = os.path.join(
                self._staging_dir,
                f"gms-hicache-{os.getpid()}-{uuid.uuid4().hex}.mmap",
            )
            size = self._staging_rows * row_size
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, size)
                mapped = mmap.mmap(
                    fd,
                    size,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            finally:
                os.close(fd)
            pool = _StagingPool(
                path,
                mapped,
                memoryview(mapped).cast("B"),
                self._staging_rows,
                row_size,
            )
            self._staging_pools[row_size] = pool
        if pool.daemon_id is None:
            pool.daemon_id = self._call(
                lambda client: client.persistent_kv_attach_pool(
                    pool.path, pool.rows, pool.row_size
                )
            )
        return pool

    @staticmethod
    def _byte_view(tensor) -> memoryview:
        import torch

        return memoryview(tensor.detach().contiguous().view(torch.uint8).numpy()).cast(
            "B"
        )

    def _store_pages(self, pool_name, keys, host_pool, slots):
        results = []
        with self._transfer_lock:
            for start in range(0, len(keys), self._staging_rows):
                batch_keys = keys[start : start + self._staging_rows]
                batch_slots = slots[start : start + self._staging_rows]
                pages = [
                    self._byte_view(host_pool.get_data_page(slot, flat=True))
                    for slot in batch_slots
                ]
                if not pages:
                    continue
                row_size = pages[0].nbytes
                if any(page.nbytes != row_size for page in pages):
                    raise ValueError("SGLang HiCache pages have inconsistent sizes")
                staging = self._pool(row_size)
                for index, page in enumerate(pages):
                    offset = index * row_size
                    staging.view[offset : offset + row_size] = page
                results.extend(
                    self._call(
                        lambda client: client.persistent_kv_store_pool(
                            self._namespace,
                            staging.daemon_id,
                            [
                                (self._key(key, pool_name), index)
                                for index, key in enumerate(batch_keys)
                            ],
                        )
                    )
                )
        return results

    def _load_pages(self, pool_name, keys, host_pool, slots):
        results = []
        with self._transfer_lock:
            for start in range(0, len(keys), self._staging_rows):
                batch_keys = keys[start : start + self._staging_rows]
                batch_slots = slots[start : start + self._staging_rows]
                if not batch_keys:
                    continue
                target = host_pool.get_dummy_flat_data_page()
                row_size = self._byte_view(target).nbytes
                staging = self._pool(row_size)
                loaded = self._call(
                    lambda client: client.persistent_kv_load_pool(
                        self._namespace,
                        staging.daemon_id,
                        [self._key(key, pool_name) for key in batch_keys],
                        list(range(len(batch_keys))),
                    )
                )
                if not loaded:
                    results.extend([False] * len(batch_keys))
                    continue
                for index, slot in enumerate(batch_slots):
                    page = host_pool.get_dummy_flat_data_page()
                    offset = index * row_size
                    self._byte_view(page)[:] = staging.view[offset : offset + row_size]
                    host_pool.set_from_flat_data_page(slot, page)
                    results.append(True)
        return results

    def close(self) -> None:
        for pool in self._staging_pools.values():
            if pool.daemon_id is not None:
                try:
                    self._call(
                        lambda client, pool_id=pool.daemon_id: (
                            client.persistent_kv_detach_pool(pool_id)
                        )
                    )
                except Exception:
                    logger.debug("GMS staging detach failed", exc_info=True)
            pool.view.release()
            pool.mapped.close()
            try:
                os.unlink(pool.path)
            except FileNotFoundError:
                pass
        self._staging_pools.clear()
        super().close()
