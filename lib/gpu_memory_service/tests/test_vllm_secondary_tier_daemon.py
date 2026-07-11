# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
import time

import numpy as np
from gpu_memory_service.integrations.vllm.gms_secondary_tier import (
    GMSSecondaryTierManager,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata


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


def _job(job_id, key, slot, *, promotion):
    return JobMetadata(
        job_id=job_id,
        keys=[key],
        block_ids=np.array([slot]),
        is_promotion=promotion,
        req_context=ReqContext("request"),
    )


def test_shadow_adapter_loads_native_key_from_daemon_owned_ram(tmp_path):
    socket_path = str(tmp_path / "gms.sock")
    daemon, loop, thread = _spawn_daemon(socket_path)
    key = make_offload_key(b"native-vllm-hash", 2)
    primary_pool = np.zeros((2, 4096), dtype=np.uint8)
    primary_pool[0] = np.arange(4096, dtype=np.uint8)
    primary = GMSSecondaryTierManager(
        offloading_spec=object(),
        primary_kv_view=memoryview(primary_pool),
        daemon_socket=socket_path,
        namespace="model-layout",
    )
    try:
        primary.submit_store(_job(1, key, 0, promotion=False))
        primary.drain_jobs()
        assert [
            (result.job_id, result.success) for result in primary.get_finished_jobs()
        ] == [(1, True)]
        primary.shutdown()

        # A new adapter has no local key metadata. Its first lookup hydrates
        # asynchronously from GMS; a later lookup observes the persistent hit.
        shadow = GMSSecondaryTierManager(
            offloading_spec=object(),
            primary_kv_view=memoryview(primary_pool),
            daemon_socket=socket_path,
            namespace="model-layout",
        )
        try:
            assert shadow.lookup(key, ReqContext("shadow")) is LookupResult.RETRY
            shadow.drain_jobs()
            assert shadow.lookup(key, ReqContext("shadow")) is LookupResult.HIT

            primary_pool[1].fill(0)
            shadow.submit_load(_job(2, key, 1, promotion=True))
            shadow.drain_jobs()
            assert [
                (result.job_id, result.success) for result in shadow.get_finished_jobs()
            ] == [(2, True)]
            np.testing.assert_array_equal(primary_pool[1], primary_pool[0])
        finally:
            shadow.shutdown()
    finally:
        loop.call_soon_threadsafe(daemon.stop)
        thread.join(timeout=5)
