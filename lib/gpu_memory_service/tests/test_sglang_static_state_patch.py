# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types

import gpu_memory_service.integrations.sglang as sglang_gms
import pytest
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.sglang import patches
from gpu_memory_service.integrations.sglang.memory_saver import (
    GMSMemorySaverImpl,
)

pytestmark = pytest.mark.pre_merge


def test_patch_static_state_for_gms_targets_current_weight_updater(monkeypatch):
    module_name = "sglang.srt.managers.scheduler_components.weight_updater"
    module = types.ModuleType(module_name)
    module._export_static_state = lambda model: {"buffers": [("x", object())]}

    def _import_static_state(model, static_params):
        raise AssertionError("original import should be replaced")

    module._import_static_state = _import_static_state
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(patches, "_static_state_patched", False)

    patches.patch_static_state_for_gms()

    assert module._export_static_state(object()) == {"buffers": []}
    assert module._import_static_state(object(), {"buffers": [("x", object())]}) is None
    assert patches._static_state_patched is True


def test_configure_shared_failover_env_disables_sglang_tp_memory_check(monkeypatch):
    monkeypatch.delenv("GMS_SGLANG_SHARED_KV", raising=False)
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.delenv("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", raising=False)

    sglang_gms.configure_shared_failover_env()

    assert os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] == "0"


def test_configure_shared_failover_env_preserves_explicit_sglang_tp_memory_check(
    monkeypatch,
):
    monkeypatch.setenv("GMS_SGLANG_SHARED_KV", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "1")

    sglang_gms.configure_shared_failover_env()

    assert os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] == "1"


def test_sglang_weight_publication_is_deferred_until_model_runner_returns(monkeypatch):
    events = []

    class FakeModelRunner:
        def load_model(self):
            events.append("load-inside-region")

        def alloc_memory_pool(self, memory_pool_config=None):
            return memory_pool_config

    module_name = "sglang.srt.model_executor.model_runner"
    module = types.ModuleType(module_name)
    module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(patches, "_model_runner_patched", False)
    impl = types.SimpleNamespace(
        imported_weights_bytes=0,
        preloaded_weights_bytes=0,
        finalize_pending_write_mode=lambda: events.append("finalize-after-region"),
    )
    monkeypatch.setattr(patches, "get_gms_memory_saver_impl", lambda: impl)

    patches.patch_model_runner()
    FakeModelRunner().load_model()

    assert events == ["load-inside-region", "finalize-after-region"]


def test_pending_sglang_weight_publication_is_one_shot():
    events = []
    impl = object.__new__(GMSMemorySaverImpl)
    impl.allocators = {
        "weights": types.SimpleNamespace(granted_lock_type=GrantedLockType.RW)
    }
    impl._pending_write_model = None
    impl.finalize_write_mode = lambda model: events.append(model)
    model = object()

    impl.defer_finalize_write_mode(model)
    assert impl._pending_write_model is model
    with pytest.raises(RuntimeError, match="already pending"):
        impl.defer_finalize_write_mode(object())

    impl.finalize_pending_write_mode()
    impl.finalize_pending_write_mode()

    assert events == [model]
    assert impl._pending_write_model is None


def test_sglang_hbm_page_adoption_and_retention_are_exact():
    import torch

    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    from gpu_memory_service.integrations.sglang import install_kv_leases

    class Client:
        def __init__(self):
            self.current = {2: KVLease(2, 5)}
            self.sealed = []

        def adopt(self, leases):
            if leases != [self.current[2]]:
                return []
            adopted = KVLease(2, 6)
            self.current[2] = adopted
            return [adopted]

        def seal(self, leases):
            self.sealed.extend(leases)

        def release(self, leases):
            for lease in leases:
                if self.current.get(lease.block_id) == lease:
                    self.current.pop(lease.block_id)

    allocator = types.SimpleNamespace(
        page_size=2,
        need_sort=False,
        free_pages=torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    client = Client()
    state = {
        "client": client,
        "leases_by_page": {},
        "retained_pages": set(),
    }
    install_kv_leases._STATE[id(allocator)] = state
    try:
        indices, leases = install_kv_leases.adopt_hbm_pages(allocator, [2], [5])
        assert indices.tolist() == [4, 5]
        assert leases == [KVLease(2, 6)]
        assert allocator.free_pages.tolist() == [1, 3]
        assert state["leases_by_page"] == {2: KVLease(2, 6)}

        retained = install_kv_leases.retain_hbm_indices(allocator, indices)
        assert retained == leases
        assert state["retained_pages"] == {2}
        assert client.sealed == leases

        install_kv_leases.rollback_adopted_hbm_pages(allocator, leases)
        assert allocator.free_pages.tolist() == [2, 1, 3]
        assert state["leases_by_page"] == {}
        assert state["retained_pages"] == set()
    finally:
        install_kv_leases._STATE.pop(id(allocator), None)
