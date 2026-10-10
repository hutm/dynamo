# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS as an SGLang memory-saver backend (--memory-saver-backend gms)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang.srt.utils.torch_memory_saver_adapter")

from gpu_memory_service.integrations.sglang import (  # noqa: E402
    memory_saver,
    memory_saver_backend,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]

hooks = pytest.mark.skipif(
    not memory_saver_backend.sglang_supports_memory_saver_backends(),
    reason="SGLang without memory saver backend registry",
)


class _Impl:
    def __init__(self):
        self.calls = []

    @contextmanager
    def region(self, tag, enable_cpu_backup):
        self.calls.append(("region", tag, enable_cpu_backup))
        yield

    def pause(self, tag):
        self.calls.append(("pause", tag))

    def resume(self, tag):
        self.calls.append(("resume", tag))


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(memory_saver_backend, "_adapter", None)
    instance = memory_saver_backend._get_adapter()
    instance._impl = _Impl()
    return instance


@hooks
def test_backend_registers_and_is_selected(monkeypatch, adapter):
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    monkeypatch.setattr(tms, "_backends", dict(tms._backends))
    assert memory_saver_backend.register() is True
    assert memory_saver_backend.register() is True
    assert "gms" in tms._backends
    monkeypatch.setattr(tms, "_configured_backend", lambda: "gms")
    assert tms.TorchMemorySaverAdapter.create(enable=True) is adapter


@hooks
def test_adapter_delegates_to_the_gms_impl(adapter):
    with adapter.region("kv_cache"):
        pass
    adapter.pause("kv_cache")
    adapter.resume("weights")
    assert adapter.impl_if_created().calls == [
        ("region", "kv_cache", False),
        ("pause", "kv_cache"),
        ("resume", "weights"),
    ]
    # GMS does not use torch_memory_saver's LD_PRELOAD hooks.
    with adapter.configure_subprocess():
        pass
    assert adapter.enabled


@hooks
def test_adapter_reports_preserved_contents(adapter):
    assert adapter.preserves_contents("weights")
    # Released KV may be written by another cohort engine: flush the radix cache.
    assert not adapter.preserves_contents("kv_cache")
    assert not adapter.allocation_preserves_contents("kv_cache")
    with memory_saver.persistent_kv_pool_scope(reattaching=True):
        assert adapter.allocation_preserves_contents("kv_cache")
        assert not adapter.allocation_preserves_contents("weights")
    with memory_saver.persistent_kv_pool_scope(reattaching=False):
        assert not adapter.allocation_preserves_contents("kv_cache")


@hooks
def test_impl_lookup_prefers_the_backend(adapter):
    assert memory_saver.get_gms_memory_saver_impl() is adapter.impl_if_created()


def test_native_zero_skip_leaves_torch_zeros_alone(monkeypatch):
    import torch

    entered = []

    @contextmanager
    def preserve(enabled):
        entered.append(enabled)
        yield

    monkeypatch.setattr(memory_saver, "_preserve_persistent_kv_zeros", preserve)

    @contextmanager
    def pool(*_args):
        yield

    monkeypatch.setattr(memory_saver, "gms_use_persistent_pool", pool)
    impl = object.__new__(memory_saver.GMSMemorySaverImpl)
    impl._native_kv_zero_skip = True
    impl._active_region_depth = 0
    impl._pending_write_model = None
    impl._kv_tag = "kv_pool"
    impl._device = torch.device("cpu")
    impl._finalize_pending_write = lambda: None
    from gpu_memory_service.common.locks import GrantedLockType

    impl.allocators = {
        "kv_pool": SimpleNamespace(granted_lock_type=GrantedLockType.RW_PERSISTENT)
    }
    with memory_saver.persistent_kv_pool_scope(reattaching=True):
        with impl.region("kv_cache", False):
            pass
    assert entered == [False]

    impl._native_kv_zero_skip = False
    with memory_saver.persistent_kv_pool_scope(reattaching=True):
        with impl.region("kv_cache", False):
            pass
    assert entered == [False, True]


@hooks
def test_setup_gms_selects_the_backend_on_raw_server_args(monkeypatch):
    # SGLang main has neither declare_late_resolution nor ServerArgs.override:
    # setup_gms must assign the backend itself, or the launcher keeps
    # torch_memory_saver (and its LD_PRELOAD subprocess hooks).
    import sys
    from types import ModuleType

    import gpu_memory_service.integrations.sglang as gms_sglang

    model_loader = ModuleType("gpu_memory_service.integrations.sglang.model_loader")
    model_loader.GMSModelLoader = object
    monkeypatch.setitem(
        sys.modules, "gpu_memory_service.integrations.sglang.model_loader", model_loader
    )
    monkeypatch.setattr(gms_sglang, "_gms_initialized", False)
    monkeypatch.setattr(gms_sglang, "_gms_lock_mode", None)
    monkeypatch.setattr(gms_sglang, "_gms_ro_connect_timeout_ms", None)
    monkeypatch.setattr(gms_sglang, "declare_late_resolution", None)
    server_args = SimpleNamespace(
        enable_memory_saver=False, memory_saver_backend="torch_memory_saver"
    )

    gms_sglang.setup_gms(server_args)

    assert server_args.enable_memory_saver is True
    assert (
        server_args.memory_saver_backend
        == memory_saver_backend.GMS_MEMORY_SAVER_BACKEND
    )


@hooks
def test_adapter_creates_the_impl_on_first_use(monkeypatch):
    import gpu_memory_service.integrations.sglang as gms_sglang
    from gpu_memory_service.common.locks import GrantedLockType
    from gpu_memory_service.integrations.common import utils

    created = []

    class Impl:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self._device = "cuda:3"
            self.allocators = {
                "weights": SimpleNamespace(granted_lock_type=GrantedLockType.RO)
            }

    monkeypatch.setattr(memory_saver, "GMSMemorySaverImpl", Impl)
    monkeypatch.setattr(
        utils, "torch_device", lambda: SimpleNamespace(current_device=lambda: 3)
    )
    monkeypatch.setattr(gms_sglang, "_gms_lock_mode", "ro", raising=False)
    monkeypatch.setattr(gms_sglang, "_gms_ro_connect_timeout_ms", 5, raising=False)
    # Dynamo's log handler raises on a bad format string; render it here.
    logged = []
    monkeypatch.setattr(
        memory_saver_backend.logger,
        "info",
        lambda msg, *args: logged.append(msg % args),
    )
    monkeypatch.setattr(memory_saver_backend, "_adapter", None)
    adapter = memory_saver_backend._get_adapter()

    impl = adapter.impl
    assert adapter.impl is impl
    assert created == [
        {
            "device_index": 3,
            "mode": "ro",
            "ro_connect_timeout_ms": 5,
            "native_kv_zero_skip": True,
        }
    ]
    assert logged == ["[GMS] memory saver backend gms (device=cuda:3, weights=RO)"]
