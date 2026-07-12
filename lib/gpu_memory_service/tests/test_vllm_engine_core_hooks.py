# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install_gms_engine_core_sleep,
)


def test_sleep_utility_is_visible_on_spawned_engine_core_proc():
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    original = EngineCore.__dict__.get("gms_sleep_no_clear")
    if original is not None:
        delattr(EngineCore, "gms_sleep_no_clear")
    try:
        assert install_gms_engine_core_sleep()
        assert "gms_sleep_no_clear" in EngineCore.__dict__
        assert hasattr(EngineCoreProc, "gms_sleep_no_clear")
        assert not install_gms_engine_core_sleep()
    finally:
        if hasattr(EngineCore, "gms_sleep_no_clear"):
            delattr(EngineCore, "gms_sleep_no_clear")
        if original is not None:
            EngineCore.gms_sleep_no_clear = original


def test_block_pool_hbm_directory_survives_engine_replacement(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace

    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    methods = (
        "__init__",
        "get_new_blocks",
        "free_blocks",
        "get_num_free_blocks",
        "cache_full_blocks",
        "get_cached_block",
    )
    originals = {name: getattr(BlockPool, name) for name in methods}
    original_allocate_slots = KVCacheManager.allocate_slots
    original_directory = leases_mod.ContentDirectory
    original_factory = leases_mod._factory
    original_patched = leases_mod._patched

    class LeaseState:
        free = set(range(1, 8))
        generations = defaultdict(int)
        held = {}

    state = LeaseState()
    owners = iter(("primary", "shadow"))

    class Client:
        namespace = "test"

        def __init__(self):
            self.owner_id = next(owners)

        def free_count(self):
            return len(state.free)

        def acquire(self, count, preferred_blocks=None, **_kwargs):
            available = [
                block
                for block in (preferred_blocks or sorted(state.free))
                if block in state.free
            ]
            if len(available) < count:
                raise RuntimeError("no lease")
            result = []
            for block in available[:count]:
                state.free.remove(block)
                state.generations[block] += 1
                generation = state.generations[block]
                state.held[block] = (generation, self.owner_id)
                result.append(KVLease(block, generation))
            return result

        def seal(self, _leases):
            return None

        def release(self, released):
            for lease in released:
                current = state.held.get(lease.block_id)
                if current is not None and current[0] == lease.generation:
                    state.held.pop(lease.block_id)
                    state.free.add(lease.block_id)

        def adopt(self, old):
            if any(
                state.held.get(lease.block_id, (None,))[0] != lease.generation
                for lease in old
            ):
                return []
            result = []
            for lease in old:
                state.generations[lease.block_id] += 1
                generation = state.generations[lease.block_id]
                state.held[lease.block_id] = (generation, self.owner_id)
                result.append(KVLease(lease.block_id, generation))
            return result

    class Directory:
        enabled = True
        authoritative = True
        mode = "authoritative"
        read_view_is_current_writer = True

        def __init__(self):
            self.entries = {}

        def publish(self, items):
            for item in items:
                content_hash = item["content_hash"]
                if not item.get("sealed", True):
                    self.entries.pop(content_hash, None)
                    continue
                slots = item.get("slot_ids")
                if slots is None:
                    slots = [item["slot_id"]]
                generations = item.get("generations")
                if generations is None:
                    generations = [item.get("generation", 0)]
                self.entries[content_hash] = {
                    **item,
                    "slot_ids": slots,
                    "generations": generations,
                    "state": "active" if item.get("active") else "ready",
                }
            return len(items)

        def read_view_items(self, *, tier=None, state="ready", limit=None):
            items = [
                (key, entry)
                for key, entry in self.entries.items()
                if (not state or entry.get("state") == state)
                and (tier is None or entry.get("tier") == tier)
            ]
            return items if limit is None else items[:limit]

        def lookup_and_claim(self, hashes):
            return [
                self.entries.get(value)
                if self.entries.get(value, {}).get("state") == "ready"
                else None
                for value in hashes
            ], "claim"

        def release_claim(self, _token):
            return True

        def adopt_claim(self, _token, items):
            for item in items:
                entry = self.entries[item["content_hash"]]
                entry["generations"] = item["generations"]
                entry["state"] = "active"
            return len(items)

        def mark_hbm_dormant(self, hashes):
            for value in hashes:
                self.entries[value]["state"] = "ready"
            return len(hashes)

        def ensure_hbm_capacity(self, _required):
            return []

        def close(self):
            return None

    directory = Directory()
    monkeypatch.setenv("GMS_VLLM_HYDRATE_HBM", "1")
    try:
        leases_mod._patched = False
        leases_mod._factory = None
        leases_mod.ContentDirectory = lambda *_args, **_kwargs: directory
        assert leases_mod.install(factory=lambda _total: Client())

        primary = BlockPool(8, True, 4)
        blocks = primary.get_new_blocks(2)
        primary.cache_full_blocks(
            SimpleNamespace(block_hashes=[b"a" * 32, b"b" * 32]),
            blocks,
            0,
            2,
            4,
            0,
        )
        content_hashes = [block.block_hash for block in blocks]
        primary.free_blocks(blocks)
        assert all(directory.entries[key]["state"] == "ready" for key in content_hashes)
        assert all(block.block_id not in state.free for block in blocks)

        # A stale snapshot member must not block recovery of the valid sibling.
        stale_hash = b"stale-directory-entry"
        state.free.remove(7)
        state.held[7] = (99, "dead")
        directory.entries[stale_hash] = {
            "tier": "hbm",
            "state": "ready",
            "slot_ids": [7],
            "generations": [1],
            "engine_id": "0",
        }

        shadow = BlockPool(8, True, 4)
        recovered = shadow.get_cached_block(b"a" * 32, [0])
        assert recovered is not None
        assert recovered[0].block_id == blocks[0].block_id
        assert directory.entries[content_hashes[0]]["state"] == "active"
        assert state.held[blocks[0].block_id][1] == "shadow"

        # The non-requested sibling was hydrated into vLLM native state, so a
        # subsequent lookup is local; the stale record was invalidated only.
        hydrated = shadow.get_cached_block(b"b" * 32, [0])
        assert hydrated is not None
        assert hydrated[0].block_id == blocks[1].block_id
        assert state.held[blocks[1].block_id][1] == "shadow"
        assert stale_hash not in directory.entries
        assert shadow._gms_hydrate_hbm is False

        lookup = directory.lookup_and_claim
        directory.lookup_and_claim = lambda _keys: pytest.fail(
            "current writer queried the recovery directory after hydration"
        )
        try:
            assert shadow.get_cached_block(b"new-miss" * 4, [0]) is None
        finally:
            directory.lookup_and_claim = lookup
    finally:
        for name, original in originals.items():
            setattr(BlockPool, name, original)
        KVCacheManager.allocate_slots = original_allocate_slots
        leases_mod.ContentDirectory = original_directory
        leases_mod._factory = original_factory
        leases_mod._patched = original_patched
