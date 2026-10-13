# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover on SGLang's weight-cache daemon (upstream memory) instead of GMS."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.sglang import upstream_memory

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_enabled_only_for_client_mode_with_kv_restore(monkeypatch):
    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_KV_RESTORE", "1")
    assert upstream_memory.upstream_memory_enabled(
        SimpleNamespace(weight_cache_mode="client")
    )
    assert not upstream_memory.upstream_memory_enabled(
        SimpleNamespace(weight_cache_mode="off")
    )
    assert not upstream_memory.upstream_memory_enabled(None)
    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_KV_RESTORE", "0")
    assert not upstream_memory.upstream_memory_enabled(
        SimpleNamespace(weight_cache_mode="client")
    )


@pytest.mark.parametrize(
    "engine_id,role", [("0", "writer"), ("1", "standby"), ("2", "standby")]
)
def test_role_and_mapped_standby(monkeypatch, engine_id, role):
    monkeypatch.setenv("ENGINE_ID", engine_id)
    monkeypatch.delenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", raising=False)
    assert upstream_memory.configure_kv_restore_role() == role
    import os

    assert os.environ["SGLANG_WEIGHT_CACHE_KV_RESTORE_ROLE"] == role
    assert os.environ["DYN_SGLANG_GMS_LOCK_BEFORE_INIT"] == (
        "1" if role == "writer" else "0"
    )


@pytest.mark.asyncio
async def test_resume_promotes_once_before_resuming(monkeypatch):
    monkeypatch.setenv("DYN_SGLANG_KV_RESTORE_PROMOTE_TIMEOUT_SECS", "9")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "1.5")
    order = []
    promote = AsyncMock(side_effect=lambda *a: order.append("promote") or 7)
    delegate = SimpleNamespace(
        quiesce=AsyncMock(return_value=True),
        resume=AsyncMock(side_effect=lambda tags: order.append("resume") or True),
    )
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(kv_restore_promote=promote)
    )
    controller = upstream_memory.PromotingController(delegate, engine)
    assert await controller.resume(["kv_cache"])
    assert await controller.resume(["kv_cache"])
    promote.assert_awaited_once_with(9.0, 1.5)
    assert order == ["promote", "resume", "resume"]


@pytest.mark.asyncio
async def test_prepare_failover_uses_the_injected_fence(monkeypatch):
    from dynamo.common import gms_failover

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    fences = []

    async def fence(role):
        fences.append(role)

    class Lock:
        async def acquire(self, engine_id=None):
            return True

    gms_fence = AsyncMock()
    monkeypatch.setattr(gms_failover, "run_gms_failover_post_lock_fence", gms_fence)
    monkeypatch.setattr(
        gms_failover,
        "_failover_lock_inputs",
        lambda factory: ("/tmp/lock", "engine-0", False, Lock()),
    )
    monkeypatch.setattr(
        gms_failover, "_try_acquire_active_lock", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        gms_failover, "wait_for_armed_standby_before_serving", AsyncMock()
    )
    monkeypatch.setattr(
        gms_failover, "claim_discovery_for_active_engine", lambda owner: None
    )
    activation = await gms_failover.prepare_gms_failover(
        SimpleNamespace(),
        SimpleNamespace(),
        backend_name="sglang",
        post_lock_fence=fence,
        preconnect_gpu_proof=False,
    )
    assert activation.enabled
    assert fences == ["active"]
    gms_fence.assert_not_awaited()
