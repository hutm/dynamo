# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
import time

import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

from gpu_memory_service.integrations.sglang.gms_hicache_storage import (
    GMSHiCacheStorage,
)


class _HostPool:
    page_size = 2

    def __init__(self, pages):
        self.pages = [torch.tensor(page, dtype=torch.uint8) for page in pages]

    def get_data_page(self, index, flat=True):
        return self.pages[index // self.page_size]

    def get_dummy_flat_data_page(self):
        return torch.zeros_like(self.pages[0])

    def set_from_flat_data_page(self, index, data_page):
        self.pages[index // self.page_size].copy_(data_page)


def _spawn_daemon(socket_path):
    from gms_kv_ring.daemon.server import Daemon

    daemon = Daemon(socket_path)
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(daemon.serve())
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not os.path.exists(socket_path) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert os.path.exists(socket_path)
    return daemon, loop, thread


def _config(socket_path):
    return HiCacheStorageConfig(
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
        extra_config={"daemon_socket": socket_path},
    )


def test_shadow_hydrates_native_sglang_hashes_from_daemon_owned_ram(tmp_path):
    socket_path = str(tmp_path / "gms.sock")
    daemon, loop, thread = _spawn_daemon(socket_path)
    primary_pool = _HostPool([[1, 2, 3, 4], [5, 6, 7, 8]])
    indices = torch.tensor([0, 1, 2, 3])
    primary = GMSHiCacheStorage(_config(socket_path))
    primary.register_mem_pool_host(primary_pool)
    try:
        assert primary.batch_set_v1(["native-a", "native-b"], indices) == [True, True]
        primary.close()

        shadow_pool = _HostPool([[0, 0, 0, 0], [0, 0, 0, 0]])
        shadow = GMSHiCacheStorage(_config(socket_path))
        shadow.register_mem_pool_host(shadow_pool)
        try:
            assert shadow.batch_exists(["native-a", "native-b"]) == 2
            assert shadow.batch_get_v1(["native-a", "native-b"], indices) == [
                True,
                True,
            ]
            assert torch.equal(shadow_pool.pages[0], primary_pool.pages[0])
            assert torch.equal(shadow_pool.pages[1], primary_pool.pages[1])
        finally:
            shadow.close()
    finally:
        loop.call_soon_threadsafe(daemon.stop)
        thread.join(timeout=5)
