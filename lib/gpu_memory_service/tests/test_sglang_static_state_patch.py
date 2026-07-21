# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types

import gpu_memory_service.integrations.sglang as sglang_gms
import pytest
import torch
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.sglang import patches
from gpu_memory_service.integrations.sglang.memory_saver import GMSMemorySaverImpl

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

        def init_memory_pool(self, total_gpu_memory):
            return total_gpu_memory

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


def test_sglang_paged_alloc_reserves_before_mutation_and_skips_preserved_pages(
    monkeypatch,
):
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    from gpu_memory_service.integrations.sglang import install_kv_leases

    events = []

    class Base:
        def __init__(self):
            self.page_size = 64
            self.size = 20 * self.page_size
            self.free_pages = torch.arange(1, 21, dtype=torch.int64)
            self.need_sort = False

        def available_size(self):
            return len(self.free_pages) * self.page_size

    class Token(Base):
        def alloc(self, _need_size):
            return None

        def free(self, _indices):
            return None

        def clear(self):
            return None

        def available_size(self):
            return super().available_size()

    class Paged(Base):
        def alloc(self, _need_size):
            return None

        def alloc_extend(self, *_args, num_new_pages=None, **_kwargs):
            events.append("allocator-mutated")
            assert int(self.free_pages[0]) == 13
            out = self.free_pages[:1].clone() * self.page_size
            self.free_pages = self.free_pages[int(num_new_pages) :]
            return out

        def alloc_decode(self, *_args, **_kwargs):
            return None

        def free(self, _indices):
            return None

        def clear(self):
            return None

    allocator_module = types.ModuleType("sglang.srt.mem_cache.allocator")
    allocator_module.BaseTokenToKVPoolAllocator = Base
    allocator_module.TokenToKVPoolAllocator = Token
    allocator_module.PagedTokenToKVPoolAllocator = Paged
    utils_module = types.ModuleType("sglang.srt.utils")
    utils_module.get_num_new_pages = lambda **_kwargs: 1
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.allocator", allocator_module
    )
    from sglang.srt import mem_cache

    monkeypatch.setattr(mem_cache, "allocator", allocator_module, raising=False)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)

    class Client:
        namespace = "test"
        owner_id = "shadow"

        def acquire(
            self, count, *, preferred_blocks, allow_partial=False, strict_preferred
        ):
            events.append("lease-reserved")
            assert count == 1
            assert preferred_blocks == [1]
            assert allow_partial is False
            assert strict_preferred is False
            return [KVLease(13, 7)]

        def free_count(self):
            return 8

        def release(self, _leases):
            events.append("lease-released")

    monkeypatch.setattr(install_kv_leases, "_patched", False)
    monkeypatch.setattr(install_kv_leases, "_factory", None)
    assert install_kv_leases.install(lambda _allocator, _total: Client()) is True

    allocator = Paged()
    out = allocator.alloc_extend(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([-1]),
        1,
        num_new_pages=1,
    )

    assert events == ["lease-reserved", "allocator-mutated"]
    assert out.tolist() == [13 * 64]
    assert 13 not in allocator.free_pages.tolist()
    assert allocator._gms_kv_leases_by_page[13] == KVLease(13, 7)
    install_kv_leases._STATE.pop(id(allocator), None)


def test_sglang_rejected_lease_pages_return_at_tail():
    from gpu_memory_service.integrations.sglang import install_kv_leases

    allocator = types.SimpleNamespace(
        free_pages=torch.tensor([3, 4], dtype=torch.int64),
    )
    install_kv_leases._append_free_pages(allocator, [1, 2])
    assert allocator.free_pages.tolist() == [3, 4, 1, 2]
