# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_memory_service.integrations.common.gms_persistent_kv import GMSPersistentKVTier
from gpu_memory_service.integrations.common.persistent_kv import (
    KVKey,
    KVLookup,
    KVTransfer,
    OperationId,
)


class _Client:
    values = {}
    lookup_calls = []

    def close(self):
        return None

    def persistent_kv_lookup(self, namespace, keys):
        self.lookup_calls.append(tuple(keys))
        return [(namespace, key) in self.values for key in keys]

    def persistent_kv_store(self, namespace, items):
        for key, value in items:
            self.values[namespace, key] = value
        return [True] * len(items)

    def persistent_kv_load(self, namespace, keys):
        return [self.values.get((namespace, key)) for key in keys]


def test_async_store_lookup_and_load_round_trip():
    _Client.values.clear()
    _Client.lookup_calls.clear()
    pool = bytearray(b"abcdefgh" + b"........")
    tier = GMSPersistentKVTier(
        memoryview(pool),
        namespace="test",
        row_size=8,
        client_factory=_Client,
    )
    key = KVKey(b"native-key")
    try:
        store = KVTransfer(OperationId(1), (key,), (0,))
        tier.submit_store(store)
        assert tier.lookup(key) in (KVLookup.PENDING, KVLookup.READY)
        tier.drain()
        assert tier.lookup(key) is KVLookup.READY
        assert [
            (int(item.operation_id), item.success) for item in tier.poll_completed()
        ] == [(1, True)]

        load = KVTransfer(OperationId(2), (key,), (1,))
        tier.submit_load(load)
        tier.drain()
        assert pool == bytearray(b"abcdefghabcdefgh")
        assert [
            (int(item.operation_id), item.success) for item in tier.poll_completed()
        ] == [(2, True)]
    finally:
        tier.close()


def test_cold_prefix_lookup_is_batched_off_the_caller_thread():
    _Client.values.clear()
    _Client.lookup_calls.clear()
    keys = tuple(KVKey(f"key-{index}".encode()) for index in range(128))
    tier = GMSPersistentKVTier(
        memoryview(bytearray(8)),
        namespace="test",
        row_size=8,
        client_factory=_Client,
        lookup_batch_delay_us=500,
    )
    try:
        assert all(tier.lookup(key) is KVLookup.PENDING for key in keys)
        tier.drain()
        assert all(tier.lookup(key) is KVLookup.MISS for key in keys)
        assert len(_Client.lookup_calls) == 1
        assert set(_Client.lookup_calls[0]) == set(map(bytes, keys))
    finally:
        tier.close()
