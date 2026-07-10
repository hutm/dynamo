# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_memory_service.integrations.common.gms_persistent_kv import (
    GMSPersistentKVTier,
)
from gpu_memory_service.integrations.common.persistent_kv import KVKey, KVLookup


class _Client:
    values = set()

    def close(self):
        return None

    def persistent_kv_lookup(self, namespace, keys):
        return [(namespace, key) in self.values for key in keys]


def test_cached_miss_refreshes_asynchronously_after_primary_store():
    _Client.values.clear()
    key = KVKey(b"late-primary-key")
    tier = GMSPersistentKVTier(
        memoryview(bytearray(8)),
        namespace="model",
        row_size=8,
        client_factory=_Client,
        miss_ttl_ms=0,
    )
    try:
        assert tier.lookup(key) is KVLookup.PENDING
        tier.drain()
        _Client.values.add(("model", bytes(key)))
        assert tier.lookup(key) is KVLookup.PENDING
        tier.drain()
        assert tier.lookup(key) is KVLookup.READY
    finally:
        tier.close()
