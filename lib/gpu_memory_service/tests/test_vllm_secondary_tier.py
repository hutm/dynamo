# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

import numpy as np
import pytest
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

from gpu_memory_service.integrations.common.persistent_kv import (
    KVLookup,
    KVTransferResult,
)
from gpu_memory_service.integrations.vllm.gms_secondary_tier import (
    GMSSecondaryTierManager,
    register_gms_secondary_tier,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]


class _Backend:
    def __init__(self):
        self.states = {}
        self.stores = []
        self.loads = []
        self.completed = deque()

    def lookup(self, key):
        return self.states.get(key, KVLookup.MISS)

    def submit_store(self, transfer):
        self.stores.append(transfer)

    def submit_load(self, transfer):
        self.loads.append(transfer)

    def poll_completed(self):
        while self.completed:
            yield self.completed.popleft()

    def has_pending_work(self):
        return bool(self.stores or self.loads or self.completed)

    def drain(self):
        return None

    def close(self):
        return None


def _manager(backend):
    return GMSSecondaryTierManager(
        offloading_spec=object(),
        primary_kv_view=memoryview(bytearray(64)),
        backend=backend,
    )


def _job(job_id=4):
    return JobMetadata(
        job_id=job_id,
        keys=[make_offload_key(b"native-vllm-hash", 3)],
        block_ids=np.array([7]),
        is_promotion=False,
        req_context=ReqContext("request"),
    )


def test_lookup_maps_native_vllm_key_without_rehashing():
    backend = _Backend()
    manager = _manager(backend)
    key = make_offload_key(b"native-vllm-hash", 3)

    assert manager.lookup(key, ReqContext("r")) is LookupResult.MISS
    backend.states[bytes(key)] = KVLookup.PENDING
    assert manager.lookup(key, ReqContext("r")) is LookupResult.RETRY
    backend.states[bytes(key)] = KVLookup.READY
    assert manager.lookup(key, ReqContext("r")) is LookupResult.HIT


def test_jobs_preserve_batch_identity_keys_and_primary_slots():
    backend = _Backend()
    manager = _manager(backend)
    job = _job()

    manager.submit_store(job)
    manager.submit_load(job)

    for transfer in (backend.stores[0], backend.loads[0]):
        assert int(transfer.operation_id) == 4
        assert transfer.keys == (bytes(job.keys[0]),)
        assert transfer.slots == (7,)


def test_completion_is_forwarded_once():
    backend = _Backend()
    manager = _manager(backend)
    backend.completed.append(KVTransferResult(4, True))

    assert [
        (result.job_id, result.success) for result in manager.get_finished_jobs()
    ] == [(4, True)]
    assert manager.get_finished_jobs() == []


def test_registration_is_idempotent_and_uses_vllm_factory():
    register_gms_secondary_tier()
    register_gms_secondary_tier()
    assert (
        SecondaryTierFactory.get_tier_class({"type": "gms"}) is GMSSecondaryTierManager
    )
