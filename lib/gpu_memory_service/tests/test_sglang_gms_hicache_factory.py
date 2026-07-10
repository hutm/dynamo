# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage import StorageBackendFactory

from gpu_memory_service.integrations.sglang.gms_hicache_storage_shm import (
    GMSSharedMemoryHiCacheStorage,
)


def test_latest_sglang_factory_loads_gms_without_registration(monkeypatch):
    class Client:
        def close(self):
            return None

    monkeypatch.setattr(
        "gms_kv_ring.daemon.client.DaemonClient", lambda socket: Client()
    )
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="model",
        extra_config={
            "backend_name": "gms",
            "module_path": (
                "gpu_memory_service.integrations.sglang.gms_hicache_storage_shm"
            ),
            "class_name": "GMSSharedMemoryHiCacheStorage",
            "daemon_socket": "/tmp/gms.sock",
            "interface_v1": 1,
        },
    )
    backend = StorageBackendFactory.create_backend("dynamic", config, object())
    try:
        assert isinstance(backend, GMSSharedMemoryHiCacheStorage)
    finally:
        backend.close()
