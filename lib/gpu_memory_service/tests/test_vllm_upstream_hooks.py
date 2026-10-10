# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS through vLLM's extension points: sleep backend and --block-pool-cls."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.device_allocator.sleep_mode_backend")

from gpu_memory_service.integrations.vllm import (  # noqa: E402
    install_kv_leases,
    sleep_backend,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]

hooks = pytest.mark.skipif(
    not sleep_backend.vllm_supports_sleep_backend_hooks(),
    reason="vLLM without sleep-backend allocation hooks",
)


@hooks
def test_gms_backend_registers_once_with_gms_capabilities():
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    sleep_backend.register()
    sleep_backend.register()
    backend_cls = SleepModeBackendFactory.get_backend_class("gms")
    assert backend_cls is sleep_backend.GMSSleepModeBackend
    assert backend_cls.preserves_kv_cache(1) and backend_cls.preserves_kv_cache(2)
    assert backend_cls.requires_cumem_allocator() is False
    assert SleepModeBackendFactory.requires_cumem_allocator("gms") is False


@hooks
def test_configure_engine_args_selects_the_gms_backend():
    engine_args = SimpleNamespace(enable_sleep_mode=False, sleep_mode_backend="cumem")
    assert sleep_backend.configure_engine_args(engine_args) is True
    assert engine_args.enable_sleep_mode is True
    assert engine_args.sleep_mode_backend == "gms"
    # Engine args without the field (older vLLM) are left alone.
    assert sleep_backend.configure_engine_args(SimpleNamespace()) is False


@hooks
def test_backend_allocation_context_delegates_to_the_bound_worker():
    backend = sleep_backend.GMSSleepModeBackend()
    assert backend.allocation_context("kv_cache") is None

    seen = []
    placed = nullcontext()

    class Worker:
        def gms_allocation_context(self, tag, kv_cache_config):
            seen.append((tag, kv_cache_config))
            return placed

    backend.bind_worker(Worker())
    config = object()
    assert backend.allocation_context("kv_cache", kv_cache_config=config) is placed
    assert seen == [("kv_cache", config)]
    with pytest.raises(RuntimeError, match="GMSWorker"):
        backend.suspend(1)


@hooks
def test_worker_routes_allocations_through_vllm_when_backend_is_active(monkeypatch):
    from gpu_memory_service.integrations.vllm.worker import GMSWorker

    placed = nullcontext()
    calls = []
    worker = object.__new__(GMSWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            sleep_mode_backend="gms",
            enable_cumem_allocator=False,
        )
    )
    backend = sleep_backend.GMSSleepModeBackend()
    backend.bind_worker(worker)
    worker._sleep_mode_backend = backend
    monkeypatch.setattr(
        GMSWorker,
        "gms_allocation_context",
        lambda self, tag, kv_cache_config=None: calls.append((tag, kv_cache_config))
        or placed,
    )

    config = object()
    assert worker._gms_uses_sleep_backend()
    assert (
        worker._maybe_get_memory_pool_context("kv_cache", kv_cache_config=config)
        is placed
    )
    assert calls == [("kv_cache", config)]


@hooks
def test_reattached_kv_uses_vllm_contents_preservation():
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv
    from vllm.v1.worker.utils import kv_cache_contents_preserved

    with install_vmm_ipc_kv._preserve_reattached_kv(True):
        assert kv_cache_contents_preserved()
    with install_vmm_ipc_kv._preserve_reattached_kv(False):
        assert not kv_cache_contents_preserved()


@hooks
def test_pause_flushes_the_gms_directory():
    from vllm.v1.core.block_pool import BlockPool

    pool_cls = install_kv_leases._build_gms_block_pool_class(BlockPool)
    flushed = []
    pool = SimpleNamespace(
        _gms_kv_directory=SimpleNamespace(
            flush_deferred=lambda timeout: flushed.append(timeout) or True
        )
    )
    pool_cls.on_scheduler_paused(pool)
    assert flushed == [2.0]

    stuck = SimpleNamespace(
        _gms_kv_directory=SimpleNamespace(flush_deferred=lambda timeout: False)
    )
    with pytest.raises(TimeoutError):
        pool_cls.on_scheduler_paused(stuck)
