# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("gpu_memory_service", reason="gpu_memory_service is required")
torch = pytest.importorskip("torch", reason="torch is required")
from gpu_memory_service.integrations.sglang import kv_identity  # noqa: E402

import gpu_memory_service.integrations.sglang.memory_saver as gms_memory_saver  # noqa: E402
from gpu_memory_service.common.locks import (  # noqa: E402
    GrantedLockType,
    RequestedLockType,
)
from gpu_memory_service.integrations.sglang.memory_saver import (  # noqa: E402
    GMSMemorySaverImpl,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.sglang,
    pytest.mark.core,
]


class _FakeManager:
    def __init__(
        self,
        *,
        is_unmapped: bool = False,
        granted_lock_type: GrantedLockType | None = None,
    ):
        self.is_unmapped = is_unmapped
        self.granted_lock_type = granted_lock_type
        self.calls: list[object] = []

    def unmap_all_vas(self) -> None:
        self.calls.append("unmap_all_vas")
        self.is_unmapped = True

    def abort(self) -> None:
        self.calls.append("abort")
        self.granted_lock_type = None

    def connect(self, lock_type, timeout_ms=None) -> None:
        self.calls.append(("connect", lock_type, timeout_ms))
        self.granted_lock_type = GrantedLockType(lock_type.value)
        self.is_unmapped = False

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")
        self.is_unmapped = False

    def remap_persistent_vas(self, engine_id: str, *, shared: bool) -> None:
        self.calls.append(("remap_persistent_vas", engine_id, shared))
        self.is_unmapped = False


@pytest.fixture
def build_impl(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gms_memory_saver,
        "get_socket_path",
        lambda device_index, tag: str(tmp_path / f"gms-test-{device_index}-{tag}.sock"),
    )

    def build(
        *,
        weights_lock: GrantedLockType = GrantedLockType.RW,
        kv_cache_lock: GrantedLockType = GrantedLockType.RW,
        allocation_engine_id: str = "sglang-test",
        allocation_shared: bool = True,
    ):
        weights = _FakeManager(granted_lock_type=weights_lock)
        kv_cache = _FakeManager(granted_lock_type=kv_cache_lock)
        pool_calls: list[tuple[str, torch.device]] = []

        @contextmanager
        def fake_use_mem_pool(tag: str, device: torch.device):
            pool_calls.append((tag, device))
            yield

        @contextmanager
        def fake_use_persistent_pool(tag: str, device: torch.device):
            pool_calls.append((tag, device))
            yield

        monkeypatch.setattr(
            gms_memory_saver,
            "allocation_engine_id",
            lambda device: allocation_engine_id,
        )
        monkeypatch.setattr(
            gms_memory_saver, "allocation_shared", lambda: allocation_shared
        )
        monkeypatch.setattr(
            gms_memory_saver,
            "get_or_create_gms_client_memory_manager",
            lambda socket_path, device, mode, tag: weights,
        )

        def fake_get_or_create_persistent_allocator(
            socket_path, device, engine_id, tag, *, shared
        ):
            return kv_cache

        monkeypatch.setattr(
            gms_memory_saver,
            "get_or_create_persistent_allocator",
            fake_get_or_create_persistent_allocator,
        )
        monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", fake_use_mem_pool)
        monkeypatch.setattr(
            gms_memory_saver, "gms_use_persistent_pool", fake_use_persistent_pool
        )
        return (
            GMSMemorySaverImpl(device_index=0, mode=None),
            weights,
            kv_cache,
            pool_calls,
        )

    return build


@pytest.mark.parametrize(
    ("tag", "weights_lock", "expected_pool_calls"),
    [
        ("weights", GrantedLockType.RW, [("weights", torch.device("cuda", 0))]),
        ("weights", GrantedLockType.RO, []),
        (
            "kv_cache",
            GrantedLockType.RW,
            [("kv_pool:cuda0", torch.device("cuda", 0))],
        ),
        ("cuda_graph", GrantedLockType.RW, []),
    ],
)
def test_region_uses_gms_pool_only_for_rw_managed_tags(
    build_impl,
    tag,
    weights_lock,
    expected_pool_calls,
):
    impl, _, _, pool_calls = build_impl(
        weights_lock=weights_lock,
        kv_cache_lock=GrantedLockType.RW,
    )

    with impl.region(tag, enable_cpu_backup=False):
        pass

    assert pool_calls == expected_pool_calls


def test_pause_resume_routes_only_managed_tags(build_impl):
    impl, weights, kv_cache, _ = build_impl(
        weights_lock=GrantedLockType.RO,
        kv_cache_lock=GrantedLockType.RW,
    )

    impl.pause("model_weights")
    impl.resume("anything_else")

    impl.pause()
    impl.resume()

    assert weights.calls == [
        "unmap_all_vas",
        "abort",
        ("connect", RequestedLockType.RO, None),
        "remap_all_vas",
    ]
    assert kv_cache.calls == [
        "unmap_all_vas",
        "abort",
        ("connect", RequestedLockType.RW_PERSISTENT, None),
        ("remap_persistent_vas", "sglang-test", True),
    ]


def test_region_requires_rw_allocator(build_impl):
    impl, _, _, _ = build_impl()
    tag = "weights"
    impl.allocators[tag].abort()

    with pytest.raises(RuntimeError, match=rf"requires {tag!r} to be RW"):
        with impl.region(tag, enable_cpu_backup=False):
            pass


def test_write_publication_follows_outermost_region_and_switches_weights_ro(
    build_impl, monkeypatch
):
    impl, weights, _, _ = build_impl()
    model = torch.nn.Module()
    impl.preloaded_weights_bytes = 456
    finalized_models = []
    events = []

    @contextmanager
    def traced_pool(tag, device):
        events.append(f"pool_enter:{tag}")
        try:
            yield
        finally:
            events.append(f"pool_exit:{tag}")

    def finalize(allocator, pending_model):
        events.append("finalize")
        finalized_models.append(pending_model)
        allocator.granted_lock_type = GrantedLockType.RO
        return SimpleNamespace(committed_bytes=123)

    monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", traced_pool)
    monkeypatch.setattr(gms_memory_saver, "gms_use_persistent_pool", traced_pool)
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    with impl.region("weights", enable_cpu_backup=False):
        with impl.region("kv_cache", enable_cpu_backup=False):
            impl.finalize_write_mode(model)
            events.append("defer")

    assert events == [
        "pool_enter:weights",
        "pool_enter:kv_pool:cuda0",
        "defer",
        "pool_exit:kv_pool:cuda0",
        "pool_exit:weights",
        "finalize",
    ]
    assert finalized_models == [model]
    assert weights.granted_lock_type == GrantedLockType.RO
    assert impl.imported_weights_bytes == 123
    assert impl.preloaded_weights_bytes == 0

    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(torch.nn.Module())

    assert events[-1] == "finalize"
    assert finalized_models == [model]


@pytest.mark.parametrize("failure_source", ["body", "pool_exit"])
def test_nested_region_failure_discards_pending_publication(
    build_impl, monkeypatch, failure_source
):
    impl, weights, _, _ = build_impl()
    stale_model = torch.nn.Module()
    fresh_model = torch.nn.Module()
    finalize = Mock(return_value=SimpleNamespace(committed_bytes=1))
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    @contextmanager
    def maybe_failing_pool(tag, device):
        yield
        if failure_source == "pool_exit" and tag == "kv_cache":
            raise ValueError("pool exit failed")

    monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", maybe_failing_pool)

    failure_message = failure_source.replace("_", " ")
    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(stale_model)
        with pytest.raises(ValueError, match=f"{failure_message} failed"):
            with impl.region("kv_cache", enable_cpu_backup=False):
                if failure_source == "body":
                    raise ValueError("body failed")

    finalize.assert_not_called()

    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(fresh_model)
    finalize.assert_called_once_with(weights, fresh_model)


def test_failed_finalization_is_cleared_and_not_retried(build_impl, monkeypatch):
    impl, weights, _, _ = build_impl()
    model = torch.nn.Module()
    finalize = Mock(side_effect=ValueError("finalization failed"))
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    with pytest.raises(ValueError, match="finalization failed"):
        with impl.region("weights", enable_cpu_backup=False):
            impl.finalize_write_mode(model)

    with impl.region("weights", enable_cpu_backup=False):
        pass
    finalize.assert_called_once_with(weights, model)


def test_invalid_or_duplicate_publication_preserves_first_model(
    build_impl, monkeypatch
):
    impl, weights, _, _ = build_impl()
    first_model = torch.nn.Module()
    finalize = Mock(return_value=SimpleNamespace(committed_bytes=1))
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(first_model)
        with pytest.raises(TypeError, match="must not be None"):
            impl.finalize_write_mode(None)
        with pytest.raises(RuntimeError, match="publication is already pending"):
            impl.finalize_write_mode(torch.nn.Module())

    finalize.assert_called_once_with(weights, first_model)




@pytest.mark.parametrize(
    "name",
    ("DYN_SGLANG_GMS_PRIVATE_BOOTSTRAP_KV", "GMS_SGLANG_PRIVATE_BOOTSTRAP_KV"),
)
def test_removed_private_bootstrap_options_fail_closed(monkeypatch, name):
    monkeypatch.delenv("DYN_SGLANG_GMS_PRIVATE_BOOTSTRAP_KV", raising=False)
    monkeypatch.delenv("GMS_SGLANG_PRIVATE_BOOTSTRAP_KV", raising=False)
    monkeypatch.setenv(name, "1")

    with pytest.raises(RuntimeError, match="private-bootstrap KV is no longer supported"):
        kv_identity.private_bootstrap_kv_enabled()
