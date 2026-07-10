# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)

from gpu_memory_service.integrations.sglang.gms_hicache_storage import (
    GMSHiCacheStorage,
)


class _Client:
    values = {}
    lookup_calls = []

    def close(self):
        return None

    def persistent_kv_lookup(self, namespace, keys):
        self.lookup_calls.append(tuple(keys))
        return [(namespace, key) in self.values for key in keys]

    def persistent_kv_store(self, namespace, items):
        for key, value in items:
            self.values[namespace, key] = value
        return [True] * len(items)

    def persistent_kv_load(self, namespace, keys):
        return [self.values.get((namespace, key)) for key in keys]


class _HostPool:
    page_size = 2

    def __init__(self, pages):
        self.pages = [torch.tensor(page, dtype=torch.uint8) for page in pages]

    def get_data_page(self, index, flat=True):
        return self.pages[index // self.page_size]

    def get_dummy_flat_data_page(self):
        return torch.zeros_like(self.pages[0])

    def set_from_flat_data_page(self, index, data_page):
        self.pages[index // self.page_size].copy_(data_page)


def _config(**extra):
    return HiCacheStorageConfig(
        tp_rank=1,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="model",
        extra_config=extra,
    )


def _backend():
    return GMSHiCacheStorage(_config(), {"client_factory": _Client})


def test_native_sglang_keys_round_trip_between_primary_and_shadow():
    _Client.values.clear()
    primary_pool = _HostPool([[1, 2, 3, 4], [5, 6, 7, 8]])
    primary = _backend()
    primary.register_mem_pool_host(primary_pool)
    try:
        indices = torch.tensor([0, 1, 2, 3])
        assert primary.batch_set_v1(["hash-a", "hash-b"], indices) == [True, True]
    finally:
        primary.close()

    shadow_pool = _HostPool([[0, 0, 0, 0], [0, 0, 0, 0]])
    shadow = _backend()
    shadow.register_mem_pool_host(shadow_pool)
    try:
        assert shadow.batch_exists(["hash-a", "hash-b", "missing"]) == 2
        assert shadow.batch_get_v1(["hash-a", "hash-b"], indices) == [True, True]
        assert torch.equal(shadow_pool.pages[0], primary_pool.pages[0])
        assert torch.equal(shadow_pool.pages[1], primary_pool.pages[1])
    finally:
        shadow.close()


def test_hybrid_pool_prefix_policies_and_page_io():
    _Client.values.clear()
    backend = _backend()
    kv_pool = _HostPool([[1, 1], [2, 2], [3, 3]])
    swa_pool = _HostPool([[4, 4], [5, 5], [6, 6]])
    backend.register_mem_pool_host(kv_pool)
    backend.register_mem_host_pool_v2(swa_pool, PoolName.SWA)
    indices = torch.tensor([0, 1, 2, 3, 4, 5])
    try:
        assert backend.batch_set_v1(["a", "b", "c"], indices) == [True] * 3
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=indices[2:],
            keys=["b", "c"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        assert backend.batch_set_v2([transfer]) == {"swa": [True, True]}

        result = backend.batch_exists_v2(["a", "b", "c"], [transfer])
        assert result.kv_hit_pages == 3
        assert result.extra_pool_hit_pages == {PoolName.KV: 3, PoolName.SWA: 3}

        swa_pool.pages[1].zero_()
        swa_pool.pages[2].zero_()
        assert backend.batch_get_v2([transfer]) == {"swa": [True, True]}
        assert swa_pool.pages[1].tolist() == [5, 5]
        assert swa_pool.pages[2].tolist() == [6, 6]
    finally:
        backend.close()


def test_lookup_is_one_batched_daemon_call():
    _Client.values.clear()
    _Client.lookup_calls.clear()
    backend = _backend()
    try:
        assert backend.batch_exists([f"key-{index}" for index in range(128)]) == 0
        assert len(_Client.lookup_calls) == 1
        assert len(_Client.lookup_calls[0]) == 128
    finally:
        backend.close()


def test_default_namespace_is_stable_and_rank_specific():
    primary = _backend()
    shadow = _backend()
    other_rank = GMSHiCacheStorage(
        HiCacheStorageConfig(**{**_config().__dict__, "tp_rank": 0}),
        {"client_factory": _Client},
    )
    try:
        assert primary._namespace == shadow._namespace
        assert primary._namespace != other_rank._namespace
    finally:
        primary.close()
        shadow.close()
        other_rank.close()
