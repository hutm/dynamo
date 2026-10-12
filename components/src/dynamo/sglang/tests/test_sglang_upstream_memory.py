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
    # configure_kv_restore_role marks upstream mode in os.environ; register
    # the keys so monkeypatch restores them and later GMS-mode tests stay GMS.
    for key in ("DYN_FAILOVER_UPSTREAM_MEMORY", "GMS_VLLM_WRITER_COHORT_PATH",
                "GMS_SGLANG_WRITER_COHORT_PATH", "SGLANG_WEIGHT_CACHE_KV_RESTORE_ROLE",
                "DYN_SGLANG_GMS_LOCK_BEFORE_INIT",
                "DYN_FAILOVER_UPSTREAM_BOOT_COHORT"):
        # setenv first: delenv records nothing for an absent key.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
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


def test_fence_kills_children_without_gms_proof_in_upstream_mode(monkeypatch):
    # The cluster keeps the gms-mps provider for MPS recycling, but with
    # upstream memory there is no GMS server to certify quiescence: waiting
    # for that proof retained the active lock forever and blocked takeover.
    import gpu_memory_service.integrations.common.gpu_quiescence as quiescence

    from dynamo.common import upstream_failover
    from dynamo.sglang import failover_watchdog

    monkeypatch.setenv(upstream_failover.UPSTREAM_MEMORY_ENV, "1")
    monkeypatch.setattr(quiescence, "gms_mps_provider_enabled", lambda _b: True)
    monkeypatch.setattr(
        failover_watchdog,
        "quiesce_local_gpu_cohort_after_rank_loss",
        lambda *a, **k: pytest.fail("upstream mode must not ask GMS for GPU proof"),
    )
    monkeypatch.setattr(failover_watchdog, "_child_pids", lambda _e: [11, 12])
    monkeypatch.setattr(failover_watchdog, "_watchdog_processes", lambda _e: ([], []))
    monkeypatch.setattr(failover_watchdog, "_pid_running", lambda _pid: True)
    killed = []
    monkeypatch.setattr(
        failover_watchdog, "_terminate_pid", lambda pid: killed.append(pid) or True
    )
    assert failover_watchdog._fence_children(object())
    assert sorted(killed) == [11, 12]


def _clear(monkeypatch, *keys):
    for key in keys:
        # setenv first: delenv records nothing for an absent key.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


_ROLE_KEYS = (
    "DYN_FAILOVER_UPSTREAM_MEMORY",
    "DYN_FAILOVER_UPSTREAM_BOOT_COHORT",
    "GMS_VLLM_WRITER_COHORT_PATH",
    "GMS_SGLANG_WRITER_COHORT_PATH",
    "SGLANG_WEIGHT_CACHE_KV_RESTORE_ROLE",
    "DYN_SGLANG_GMS_LOCK_BEFORE_INIT",
)


@pytest.mark.asyncio
async def test_post_lock_fence_reports_the_previous_lock_holder(monkeypatch):
    from dynamo.common import upstream_failover

    calls = []
    monkeypatch.setattr(
        upstream_failover, "hint_predecessor_lost", lambda backend: calls.append(backend)
    )
    await upstream_memory.upstream_post_lock_fence("shadow")
    assert calls == ["sglang"]


@pytest.mark.asyncio
async def test_worker_rank_liveness_advertises_its_crash_hint_target(
    monkeypatch, tmp_path
):
    from dynamo.common import rank_liveness, upstream_failover
    from dynamo.sglang import failover_watchdog

    _clear(monkeypatch, *_ROLE_KEYS)
    lock = tmp_path / "failover.lock.rank-2"
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(lock))
    monkeypatch.setenv("ENGINE_ID", "0")
    monkeypatch.delenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", raising=False)
    upstream_memory.configure_kv_restore_role()

    class Client:
        def __init__(self, leader_host, rank, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setenv("DYN_GMS_RANK_LIVENESS", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_GMS_RANK_LIVENESS_ISOLATED", "0")
    monkeypatch.setenv(
        "DYN_GMS_RANK_LIVENESS_CONNECT_ADDR", "tcp://{leader_host}:29555"
    )
    monkeypatch.setattr(rank_liveness, "RankLivenessClient", Client)
    failover_watchdog.maybe_start_rank_liveness(
        SimpleNamespace(), object(), node_rank=2, leader_host="leader.example"
    )
    record = upstream_failover.take_crash_hint_target("sglang", str(lock))
    import os

    assert record is not None
    assert record["rank"] == 2
    assert record["address"] == "tcp://leader.example:29555"
    assert record["cohort"] == os.environ["GMS_SGLANG_WRITER_COHORT_PATH"]
