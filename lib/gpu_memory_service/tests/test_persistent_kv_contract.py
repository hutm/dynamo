# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

import pytest
from gpu_memory_service.integrations.common.persistent_kv import (
    KVKey,
    KVLookup,
    KVTransfer,
    KVTransferResult,
    OperationId,
    PersistentKVTier,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _transfer(keys=(b"a",), slots=(0,), operation_id=1):
    return KVTransfer(
        operation_id=OperationId(operation_id),
        keys=tuple(KVKey(key) for key in keys),
        slots=slots,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"operation_id": -1}, "operation_id"),
        ({"keys": ()}, "at least one"),
        ({"keys": (b"a", b"b"), "slots": (0,)}, "equal lengths"),
        ({"keys": (b"",)}, "must not be empty"),
        ({"slots": (-1,)}, "non-negative"),
    ],
)
def test_transfer_rejects_ambiguous_batches(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _transfer(**kwargs)


class _FakeTier(PersistentKVTier):
    def __init__(self):
        self.ready = set()
        self.pending = {}
        self.completed = deque()

    def lookup(self, key):
        if key in self.ready:
            return KVLookup.READY
        if key in self.pending:
            return KVLookup.PENDING
        return KVLookup.MISS

    def submit_store(self, transfer):
        for key in transfer.keys:
            self.pending[key] = transfer.operation_id

    def submit_load(self, transfer):
        assert all(key in self.ready for key in transfer.keys)
        self.completed.append(KVTransferResult(transfer.operation_id, True))

    def finish_store(self, transfer):
        for key in transfer.keys:
            self.pending.pop(key)
            self.ready.add(key)
        self.completed.append(KVTransferResult(transfer.operation_id, True))

    def poll_completed(self):
        while self.completed:
            yield self.completed.popleft()

    def has_pending_work(self):
        return bool(self.pending or self.completed)

    def drain(self):
        assert not self.pending


def test_contract_distinguishes_pending_from_ready_and_polls_once():
    tier = _FakeTier()
    store = _transfer(operation_id=7)

    assert tier.lookup(store.keys[0]) is KVLookup.MISS
    tier.submit_store(store)
    assert tier.lookup(store.keys[0]) is KVLookup.PENDING
    assert tier.has_pending_work()

    tier.finish_store(store)
    assert tier.lookup(store.keys[0]) is KVLookup.READY
    assert list(tier.poll_completed()) == [KVTransferResult(OperationId(7), True)]
    assert list(tier.poll_completed()) == []
    assert not tier.has_pending_work()
    tier.drain()


def test_load_uses_the_same_content_key_without_engine_metadata():
    tier = _FakeTier()
    store = _transfer(keys=(b"native-engine-hash",), slots=(3,), operation_id=8)
    tier.submit_store(store)
    tier.finish_store(store)
    list(tier.poll_completed())

    load = _transfer(
        keys=(b"native-engine-hash",),
        slots=(11,),
        operation_id=9,
    )
    tier.submit_load(load)

    assert list(tier.poll_completed()) == [KVTransferResult(OperationId(9), True)]
