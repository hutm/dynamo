# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover on vLLM's weight-cache daemon (upstream memory) instead of GMS."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm import upstream_memory

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _args(load_format="ipc_cache", extra='{"kv_restore": true}'):
    return SimpleNamespace(load_format=load_format, model_loader_extra_config=extra)


def test_enabled_only_for_ipc_cache_with_kv_restore():
    assert upstream_memory.upstream_memory_enabled(_args())
    assert upstream_memory.upstream_memory_enabled(
        _args(extra={"kv_restore": True, "socket_dir": "/s"})
    )
    assert not upstream_memory.upstream_memory_enabled(_args(extra="{}"))
    assert not upstream_memory.upstream_memory_enabled(_args(load_format="gms"))


@pytest.mark.parametrize(
    "engine_id,primary_id,role",
    [
        ("0", None, "writer"),
        ("1", None, "standby"),
        ("2", "0", "standby"),
        ("1", "1", "writer"),
    ],
)
def test_role_follows_the_primary_engine_id(monkeypatch, engine_id, primary_id, role):
    monkeypatch.setenv("ENGINE_ID", engine_id)
    if primary_id is None:
        monkeypatch.delenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", raising=False)
    else:
        monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", primary_id)
    args = _args(extra='{"kv_restore": true, "socket_dir": "/s"}')
    assert upstream_memory.configure_kv_restore_role(args) == role
    assert args.model_loader_extra_config == {
        "kv_restore": True,
        "socket_dir": "/s",
        "kv_restore_role": role,
    }


@pytest.mark.asyncio
async def test_promote_calls_the_engine_core_utility(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_KV_RESTORE_PROMOTE_TIMEOUT_SECS", "12")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "3")
    call = AsyncMock(return_value=41)
    client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=call))
    assert await upstream_memory.promote_standby(client) == 41
    call.assert_awaited_once_with("kv_restore_promote", 12.0, 3.0)


@pytest.mark.asyncio
async def test_standby_warmup_runs_dummy_batches_on_every_worker(monkeypatch):
    monkeypatch.delenv("DYN_VLLM_UPSTREAM_STANDBY_WARMUP", raising=False)
    monkeypatch.setenv("DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS", "64,1,8,8")
    rpc = AsyncMock(return_value=[3, 3])
    assert await upstream_memory.warm_up_standby(SimpleNamespace(collective_rpc=rpc)) == 3
    rpc.assert_awaited_once_with("kv_restore_warmup", args=([1, 8, 64],))


@pytest.mark.asyncio
async def test_standby_warmup_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_UPSTREAM_STANDBY_WARMUP", "0")
    rpc = AsyncMock()
    assert await upstream_memory.warm_up_standby(SimpleNamespace(collective_rpc=rpc)) == 0
    rpc.assert_not_awaited()
