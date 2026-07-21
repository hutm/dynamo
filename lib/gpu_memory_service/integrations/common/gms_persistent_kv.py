# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple asynchronous PersistentKVTier backed by a GMS daemon.

This reference implementation intentionally uses the daemon's JSON control
socket for bytes. It establishes correctness before shared-memory or NIXL
transport optimizations are introduced independently.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Callable, TypeVar

from gpu_memory_service.integrations.common.persistent_kv import (
    KVKey,
    KVLookup,
    KVTransfer,
    KVTransferResult,
    PersistentKVTier,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class GMSPersistentKVTier(PersistentKVTier):
    def __init__(
        self,
        primary_kv_view: memoryview,
        daemon_socket: str | None = None,
        namespace: str | None = None,
        row_size: int | None = None,
        client_factory=None,
        max_workers: int = 1,
        transport: str = "json",
        pool_path: str | None = None,
        lookup_batch_delay_us: int = 50,
        miss_ttl_ms: int = 100,
    ) -> None:
        self._view = primary_kv_view.cast("B")
        shape = primary_kv_view.shape
        if primary_kv_view.ndim == 2:
            self._num_rows = int(shape[0])
            self._row_size = int(shape[1])
        elif row_size:
            self._row_size = int(row_size)
            self._num_rows = self._view.nbytes // self._row_size
        else:
            raise ValueError("row_size is required for a one-dimensional pool")
        if self._num_rows * self._row_size != self._view.nbytes:
            raise ValueError("attached pool size is not an exact row multiple")

        self._socket = str(
            daemon_socket
            or os.environ.get("GMS_KV_DAEMON_SOCKET")
            or os.environ.get("GMS_KVR_DAEMON_SOCKET")
            or ""
        )
        if not self._socket and client_factory is None:
            raise ValueError("daemon_socket is required")
        self._namespace = str(
            namespace
            or os.environ.get("GMS_KV_DIRECTORY_MANIFEST")
            or "native-engine-kv-v1"
        )
        self._transport = str(transport).strip().lower()
        if self._transport not in ("json", "shm"):
            raise ValueError("transport must be 'json' or 'shm'")
        self._pool_path = None if pool_path is None else str(pool_path)
        if self._transport == "shm" and not self._pool_path:
            raise ValueError("pool_path is required for shm transport")
        self._pool_id = None
        if client_factory is None:
            from gms_kv_ring.daemon.client import DaemonClient

            def client_factory():
                return DaemonClient(self._socket)

        self._client_factory = client_factory
        self._client = None
        self._client_lock = threading.Lock()
        self._lock = threading.Lock()
        self._states: dict[KVKey, KVLookup] = {}
        self._miss_observed: dict[KVKey, float] = {}
        self._miss_ttl = max(0, int(miss_ttl_ms)) / 1_000
        self._lookup_pending: set[KVKey] = set()
        self._lookup_scheduled = False
        self._lookup_batch_delay = max(0, int(lookup_batch_delay_us)) / 1_000_000
        self._futures: set[Future] = set()
        self._completed = deque()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="gms-persistent-kv",
        )
        self._closed = False

    def _call(self, operation: Callable[[object], _T]) -> _T:
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
                    self._pool_id = None
                    if attempt:
                        raise
            raise AssertionError("unreachable")

    def _ensure_pool(self, client) -> str:
        if self._pool_id is None:
            self._pool_id = client.persistent_kv_attach_pool(
                self._pool_path,
                self._num_rows,
                self._row_size,
            )
        return self._pool_id

    def _submit(self, function, *args) -> None:
        if self._closed:
            raise RuntimeError("persistent KV tier is closed")
        future = self._executor.submit(function, *args)
        with self._lock:
            self._futures.add(future)

        def done(completed: Future) -> None:
            with self._lock:
                self._futures.discard(completed)

        future.add_done_callback(done)

    def _row_bounds(self, slot: int) -> tuple[int, int]:
        if slot < 0 or slot >= self._num_rows:
            raise IndexError(f"primary KV slot {slot} is out of range")
        start = slot * self._row_size
        return start, start + self._row_size

    def _read_rows(self, slots: tuple[int, ...]) -> list[bytes]:
        return [bytes(self._view[slice(*self._row_bounds(slot))]) for slot in slots]

    def _write_rows(self, slots: tuple[int, ...], values: list[bytes]) -> None:
        if len(slots) != len(values):
            raise ValueError("daemon returned the wrong number of KV rows")
        for value in values:
            if len(value) != self._row_size:
                raise ValueError("daemon returned a KV row with the wrong size")
        for slot, value in zip(slots, values):
            self._view[slice(*self._row_bounds(slot))] = value

    def lookup(self, key: KVKey) -> KVLookup:
        now = time.monotonic()
        schedule = False
        with self._lock:
            state = self._states.get(key)
            if state is KVLookup.MISS:
                observed = self._miss_observed.setdefault(key, now)
                if now - observed < self._miss_ttl:
                    return state
                self._states.pop(key, None)
                self._miss_observed.pop(key, None)
            elif state is not None:
                return state
            self._states[key] = KVLookup.PENDING
            self._lookup_pending.add(key)
            if not self._lookup_scheduled:
                self._lookup_scheduled = True
                schedule = True
        if schedule:
            self._submit(self._lookup_batches)
        return KVLookup.PENDING

    def _lookup_batches(self) -> None:
        if self._lookup_batch_delay:
            time.sleep(self._lookup_batch_delay)
        while True:
            with self._lock:
                keys = tuple(self._lookup_pending)
                self._lookup_pending.clear()
                if not keys:
                    self._lookup_scheduled = False
                    return
            try:
                hits = self._call(
                    lambda client: client.persistent_kv_lookup(
                        self._namespace,
                        list(map(bytes, keys)),
                    )
                )
                if len(hits) != len(keys):
                    raise ValueError("daemon returned the wrong lookup result count")
            except Exception:
                logger.warning("GMS persistent KV lookup failed", exc_info=True)
                hits = [False] * len(keys)
            with self._lock:
                for key, hit in zip(keys, hits):
                    self._states[key] = KVLookup.READY if hit else KVLookup.MISS
                    if hit:
                        self._miss_observed.pop(key, None)

    def submit_store(self, transfer: KVTransfer) -> None:
        with self._lock:
            for key in transfer.keys:
                self._states[key] = KVLookup.PENDING
        self._submit(self._store, transfer)

    def _store(self, transfer: KVTransfer) -> None:
        success = False
        try:
            if self._transport == "shm":
                stored = self._call(
                    lambda client: client.persistent_kv_store_pool(
                        self._namespace,
                        self._ensure_pool(client),
                        list(zip(map(bytes, transfer.keys), transfer.slots)),
                    )
                )
            else:
                values = self._read_rows(transfer.slots)
                stored = self._call(
                    lambda client: client.persistent_kv_store(
                        self._namespace,
                        list(zip(map(bytes, transfer.keys), values)),
                    )
                )
            success = len(stored) == len(transfer.keys) and all(stored)
        except Exception:
            logger.warning("GMS persistent KV store failed", exc_info=True)
        with self._lock:
            state = KVLookup.READY if success else KVLookup.MISS
            for key in transfer.keys:
                self._states[key] = state
            self._completed.append(KVTransferResult(transfer.operation_id, success))

    def submit_load(self, transfer: KVTransfer) -> None:
        self._submit(self._load, transfer)

    def _load(self, transfer: KVTransfer) -> None:
        success = False
        try:
            if self._transport == "shm":
                success = self._call(
                    lambda client: client.persistent_kv_load_pool(
                        self._namespace,
                        self._ensure_pool(client),
                        list(map(bytes, transfer.keys)),
                        list(transfer.slots),
                    )
                )
            else:
                values = self._call(
                    lambda client: client.persistent_kv_load(
                        self._namespace,
                        list(map(bytes, transfer.keys)),
                    )
                )
                if len(values) == len(transfer.keys) and all(
                    value is not None for value in values
                ):
                    self._write_rows(transfer.slots, values)
                    success = True
        except Exception:
            logger.warning("GMS persistent KV load failed", exc_info=True)
        with self._lock:
            if not success:
                for key in transfer.keys:
                    self._states[key] = KVLookup.MISS
            self._completed.append(KVTransferResult(transfer.operation_id, success))

    def poll_completed(self):
        with self._lock:
            completed = list(self._completed)
            self._completed.clear()
        return completed

    def has_pending_work(self) -> bool:
        with self._lock:
            return bool(self._futures or self._completed)

    def drain(self) -> None:
        while True:
            with self._lock:
                pending = tuple(self._futures)
            if not pending:
                return
            wait(pending)

    def close(self) -> None:
        if self._closed:
            return
        self.drain()
        self._closed = True
        self._executor.shutdown(wait=True)
        with self._client_lock:
            if self._client is not None:
                if self._pool_id is not None:
                    try:
                        self._client.persistent_kv_detach_pool(self._pool_id)
                    except Exception:
                        logger.debug("GMS persistent pool detach failed", exc_info=True)
                    self._pool_id = None
                self._client.close()
                self._client = None
        self._view.release()
