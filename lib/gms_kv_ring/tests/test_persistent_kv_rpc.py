# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import ctypes

from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager
from gms_kv_ring.daemon.rpc_dispatch import dispatch


def test_persistent_kv_rpc_round_trip_uses_gms_host_tier(monkeypatch, tmp_path):
    import gms_kv_ring.daemon.host_tier as host_mod

    allocations = {}

    def alloc(size):
        buffer = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(buffer)
        allocations[ptr] = buffer
        return ptr

    monkeypatch.setattr(host_mod, "_alloc_host", alloc)
    monkeypatch.setattr(host_mod, "_free_host", allocations.pop)
    manager = GmsKvCacheManager(storage_dir=str(tmp_path))
    key = b"vllm-native-key\x00\x03"
    payload = bytes(range(64))

    assert dispatch(
        manager,
        {
            "op": "persistent_kv_lookup",
            "namespace": "model-layout",
            "keys": [key.hex()],
        },
    )["hits"] == [False]

    stored = dispatch(
        manager,
        {
            "op": "persistent_kv_store",
            "namespace": "model-layout",
            "items": [
                {
                    "key": key.hex(),
                    "data": base64.b64encode(payload).decode("ascii"),
                }
            ],
        },
    )
    assert stored["stored"] == [True]
    assert manager.host_tier.n_slots() == 1

    assert dispatch(
        manager,
        {
            "op": "persistent_kv_lookup",
            "namespace": "model-layout",
            "keys": [key.hex()],
        },
    )["hits"] == [True]
    loaded = dispatch(
        manager,
        {
            "op": "persistent_kv_load",
            "namespace": "model-layout",
            "keys": [key.hex()],
        },
    )
    assert base64.b64decode(loaded["values"][0]) == payload
