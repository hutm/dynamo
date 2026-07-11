# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the engine-independent GMS KV directory POC."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from collections import deque

import pytest
from gms_kv_ring.common.content_directory import ContentDirectory
from gms_kv_ring.daemon.client import DaemonClient
from gms_kv_ring.daemon.server import Daemon
from gms_kv_ring.daemon.staging_tier import _BytearrayAllocator


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


@pytest.fixture
def directory_daemon(tmp_path):
    socket_path = str(tmp_path / "daemon.sock")
    daemon = Daemon(
        listen_socket=socket_path,
        storage_dir=str(tmp_path),
        supervise_backend=False,
        staging_capacity_bytes=1 << 20,
        staging_allocator=_BytearrayAllocator(),
    )
    loop_holder = {}

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        try:
            loop.run_until_complete(daemon.serve())
        finally:
            loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not os.path.exists(socket_path):
        time.sleep(0.02)
    assert os.path.exists(socket_path)
    try:
        yield daemon, socket_path
    finally:
        loop_holder["loop"].call_soon_threadsafe(daemon.stop)
        thread.join(timeout=3)


@pytest.fixture(autouse=True)
def synchronous_read_for_facade_tests(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_READ", "0")


def _promote(client, writer_id="engine-primary"):
    _entries, epoch, _writer = client.directory_lookup("manifest-a", [])
    promoted, new_epoch, active = client.directory_promote(epoch, writer_id)
    assert promoted is True
    assert active == writer_id
    return new_epoch


def _item(content, slot_id, generation=1, engine_id="0"):
    return {
        "content_hash": _hash(content),
        "engine_id": engine_id,
        "slot_id": slot_id,
        "generation": generation,
        "ranges": [(0, slot_id * 16, 16)],
    }


def test_directory_works_without_cross_node_transport(directory_daemon):
    daemon, socket_path = directory_daemon
    item = _item(b"alpha", 7, generation=3)
    with DaemonClient(socket_path) as client:
        _promote(client)
        result = client.directory_publish_batch("manifest-a", "engine-primary", [item])
        entries, epoch, writer = client.directory_lookup(
            "manifest-a", [item["content_hash"], _hash(b"missing")]
        )
    assert result["published"] == 1
    assert result["rejected_stale_writer"] is False
    assert writer == "engine-primary"
    assert epoch == result["directory_epoch"]
    assert entries == [
        {
            "engine_id": "0",
            "slot_ids": [7],
            "generations": [3],
            "ranges": [[0, 112, 16]],
            "state": "ready",
        },
        None,
    ]
    assert len(daemon._content_hash_index) == 0


def test_directory_is_manifest_scoped(directory_daemon):
    _daemon, socket_path = directory_daemon
    item = _item(b"same-hash", 4)
    with DaemonClient(socket_path) as client:
        _promote(client)
        client.directory_publish_batch("manifest-a", "engine-primary", [item])
        hits_a, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        hits_b, _epoch, _writer = client.directory_lookup(
            "manifest-b", [item["content_hash"]]
        )
    assert hits_a[0] is not None
    assert hits_b == [None]


def test_directory_promotion_fences_old_writer(directory_daemon):
    _daemon, socket_path = directory_daemon
    old_item = _item(b"old", 1)
    stale_item = _item(b"stale", 2)
    fresh_item = _item(b"fresh", 3, engine_id="1")
    with DaemonClient(socket_path) as client:
        old_epoch = _promote(client, "engine-primary")
        client.directory_publish_batch("manifest-a", "engine-primary", [old_item])
        promoted, new_epoch, active = client.directory_promote(
            old_epoch, "engine-shadow"
        )
        stale = client.directory_publish_batch(
            "manifest-a", "engine-primary", [stale_item]
        )
        fresh = client.directory_publish_batch(
            "manifest-a", "engine-shadow", [fresh_item]
        )
        hits, _epoch, _writer = client.directory_lookup(
            "manifest-a",
            [
                old_item["content_hash"],
                stale_item["content_hash"],
                fresh_item["content_hash"],
            ],
        )
    assert promoted is True
    assert new_epoch == old_epoch + 1
    assert active == "engine-shadow"
    assert stale["rejected_stale_writer"] is True
    assert fresh["published"] == 1
    assert hits[0] is not None
    assert hits[1] is None
    assert hits[2] is not None


def test_directory_slot_reuse_retires_old_hash(directory_daemon):
    _daemon, socket_path = directory_daemon
    old_item = _item(b"old-slot", 9, generation=1)
    new_item = _item(b"new-slot", 9, generation=2)
    with DaemonClient(socket_path) as client:
        _promote(client)
        client.directory_publish_batch("manifest-a", "engine-primary", [old_item])
        result = client.directory_publish_batch(
            "manifest-a", "engine-primary", [new_item]
        )
        hits, _epoch, _writer = client.directory_lookup(
            "manifest-a", [old_item["content_hash"], new_item["content_hash"]]
        )
    assert result["removed"] == 1
    assert hits[0] is None
    assert hits[1]["generations"] == [2]


def test_stale_generation_cannot_remove_newer_same_hash(directory_daemon):
    _daemon, socket_path = directory_daemon
    current = _item(b"same-content", 9, generation=2)
    stale_remove = {
        **current,
        "generations": [1],
        "sealed": False,
    }
    current_remove = {**current, "sealed": False}
    with DaemonClient(socket_path) as client:
        _promote(client)
        client.directory_publish_batch("manifest-a", "engine-primary", [current])
        stale = client.directory_publish_batch(
            "manifest-a", "engine-primary", [stale_remove]
        )
        retained, _epoch, _writer = client.directory_lookup(
            "manifest-a", [current["content_hash"]]
        )
        removed = client.directory_publish_batch(
            "manifest-a", "engine-primary", [current_remove]
        )
        absent, _epoch, _writer = client.directory_lookup(
            "manifest-a", [current["content_hash"]]
        )
    assert stale["removed"] == 0
    assert retained[0]["generations"] == [2]
    assert removed["removed"] == 1
    assert absent == [None]


def test_directory_promotion_is_idempotent_for_tp_ranks(directory_daemon):
    _daemon, socket_path = directory_daemon
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        promoted, same_epoch, active = client.directory_promote(
            epoch - 1, "engine-primary"
        )
        rejected, observed_epoch, observed_writer = client.directory_promote(
            epoch - 1, "engine-shadow"
        )
    assert promoted is True
    assert same_epoch == epoch
    assert active == "engine-primary"
    assert rejected is False
    assert observed_epoch == epoch
    assert observed_writer == "engine-primary"


def test_facade_claims_unowned_standalone_directory(directory_daemon, monkeypatch):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    monkeypatch.setenv("ENGINE_ID", "7")
    directory = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="different-pool-id",
    )
    item = _item(b"facade", 12)
    try:
        assert directory.writer_id == "engine-7"
        assert directory.publish([item]) == 1
        assert directory.lookup([item["content_hash"]])[0]["slot_ids"] == [12]
    finally:
        directory.close()


def test_standby_waits_for_promotion_and_refreshes_cached_epoch(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.delenv("ENGINE_ID", raising=False)
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    primary = ContentDirectory(
        socket_path, engine="vllm", block_size=16, engine_id="0", mode="authoritative"
    )
    standby = ContentDirectory(
        socket_path,
        engine="vllm",
        block_size=16,
        engine_id="1",
        mode="authoritative",
        standby=True,
    )
    promoter = ContentDirectory(
        socket_path, engine="vllm", block_size=16, engine_id="1", mode="authoritative"
    )
    first = _item(b"primary-hbm", 41)
    first.update(tier="hbm", active=True)
    second = _item(b"shadow-hbm", 42, engine_id="1")
    second.update(tier="hbm", active=True)
    try:
        assert standby.publish([second]) == 0
        assert standby.status()[1] is None
        assert primary.publish([first]) == 1
        standby.status()  # Cache the primary writer epoch before promotion.
        promoter.promote()
        assert standby.publish([second]) == 1
        assert standby.lookup([second["content_hash"]]) == [None]
        assert standby.mark_hbm_dormant([second["content_hash"]]) == 1
        assert standby.lookup([second["content_hash"]])[0]["slot_ids"] == [42]
    finally:
        primary.close()
        standby.close()
        promoter.close()


def test_lookup_retires_entry_when_published_host_bytes_are_missing(
    directory_daemon,
):
    daemon, socket_path = directory_daemon
    item = _item(b"missing-host", 13)
    item["tier"] = "host"
    with DaemonClient(socket_path) as client:
        _promote(client)
        assert (
            client.directory_publish_batch("manifest-a", "engine-primary", [item])[
                "published"
            ]
            == 1
        )
        hits, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        second, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
    assert hits == [None]
    assert second == [None]
    assert ("manifest-a", item["content_hash"]) not in daemon._content_directory


def test_lookup_retires_entry_when_host_slot_generation_was_overwritten(
    directory_daemon,
):
    daemon, socket_path = directory_daemon
    item = _item(b"overwritten-host", 14, generation=4)
    item["tier"] = "host"
    write = daemon.host_tier.reserve("0", 0, 14 * 16, 16)
    assert daemon.host_tier.commit(write, generation=4)
    with DaemonClient(socket_path) as client:
        _promote(client)
        assert (
            client.directory_publish_batch("manifest-a", "engine-primary", [item])[
                "published"
            ]
            == 1
        )
        initial, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        replacement = daemon.host_tier.reserve("0", 0, 14 * 16, 16)
        assert daemon.host_tier.commit(replacement, generation=5)
        overwritten, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
    assert initial[0]["generations"] == [4]
    assert overwritten == [None]
    assert ("manifest-a", item["content_hash"]) not in daemon._content_directory


def test_hbm_lookup_claim_blocks_capacity_eviction(directory_daemon):
    _daemon, socket_path = directory_daemon
    first = _item(b"hbm-first", 21, generation=3)
    first["tier"] = "hbm"
    second = _item(b"hbm-second", 22, generation=4)
    second["tier"] = "hbm"
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a",
            "engine-primary",
            [first, second],
            expected_epoch=epoch,
        )
        entries, token, rejected, _epoch = client.directory_lookup_claim(
            "manifest-a",
            "engine-primary",
            epoch,
            [first["content_hash"]],
        )
        assert rejected is False
        assert token
        assert entries[0]["tier"] == "hbm"

        victims, rejected = client.directory_ensure_hbm_capacity(
            "manifest-a", "engine-primary", epoch, 2
        )
        assert rejected is False
        assert [victim["slot_ids"] for victim in victims] == [[22]]

        assert client.directory_release_claim(token)
        victims, rejected = client.directory_ensure_hbm_capacity(
            "manifest-a", "engine-primary", epoch, 1
        )
        assert rejected is False
        assert [victim["slot_ids"] for victim in victims] == [[21]]


def test_hbm_claim_adoption_updates_generation(directory_daemon):
    _daemon, socket_path = directory_daemon
    item = _item(b"hbm-adopt", 23, generation=7)
    item["tier"] = "hbm"
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a",
            "engine-primary",
            [item],
            expected_epoch=epoch,
        )
        _entries, token, rejected, _epoch = client.directory_lookup_claim(
            "manifest-a",
            "engine-primary",
            epoch,
            [item["content_hash"]],
        )
        assert token and not rejected
        adopted, rejected = client.directory_adopt_claim(
            "manifest-a",
            "engine-primary",
            epoch,
            token,
            [{"content_hash": item["content_hash"], "generations": [8]}],
        )
        assert (adopted, rejected) == (1, False)
        hits, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        assert hits == [None]
        updated, rejected = client.directory_mark_hbm_dormant(
            "manifest-a",
            "engine-primary",
            epoch,
            [item["content_hash"]],
        )
        assert (updated, rejected) == (1, False)
        hits, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        assert hits[0]["generations"] == [8]
        victims, rejected = client.directory_ensure_hbm_capacity(
            "manifest-a", "engine-primary", epoch, 1
        )
        assert not rejected
        assert [victim["slot_ids"] for victim in victims] == [[23]]


def test_promotion_makes_crashed_writer_active_hbm_recoverable(directory_daemon):
    _daemon, socket_path = directory_daemon
    item = _item(b"active-before-crash", 27, generation=4)
    item.update(tier="hbm", active=True)
    with DaemonClient(socket_path) as client:
        primary_epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a",
            "engine-primary",
            [item],
            expected_epoch=primary_epoch,
        )
        before, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        assert before == [None]
        promoted, shadow_epoch, active = client.directory_promote(
            primary_epoch, "engine-shadow"
        )
        assert promoted and active == "engine-shadow"
        after, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
        assert after[0]["state"] == "ready"
        protected, rejected = client.directory_hbm_inventory(
            "engine-shadow", shadow_epoch
        )
        assert not rejected
        assert protected == {"0": [27]}


def test_promotion_releases_crashed_writer_claims(directory_daemon):
    daemon, socket_path = directory_daemon
    item = _item(b"claimed-before-crash", 24, generation=2)
    item["tier"] = "hbm"
    with DaemonClient(socket_path) as client:
        primary_epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a",
            "engine-primary",
            [item],
            expected_epoch=primary_epoch,
        )
        _entries, token, rejected, _epoch = client.directory_lookup_claim(
            "manifest-a",
            "engine-primary",
            primary_epoch,
            [item["content_hash"]],
        )
        assert token and not rejected
        promoted, shadow_epoch, active = client.directory_promote(
            primary_epoch, "engine-shadow"
        )
        assert promoted and active == "engine-shadow"
        assert token not in daemon._content_directory_claims
        victims, rejected = client.directory_ensure_hbm_capacity(
            "manifest-a", "engine-shadow", shadow_epoch, 1
        )
        assert not rejected
        assert victims[0]["slot_ids"] == [24]


def test_hbm_inventory_uses_explicit_engine_scope(directory_daemon):
    _daemon, socket_path = directory_daemon
    vllm = _item(b"vllm-hbm", 31)
    sglang = _item(b"sglang-hbm", 32)
    vllm["tier"] = sglang["tier"] = "hbm"
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        client.directory_publish_batch(
            "shared-explicit-manifest",
            "engine-primary",
            [vllm],
            expected_epoch=epoch,
            scope="vllm",
        )
        client.directory_publish_batch(
            "shared-explicit-manifest",
            "engine-primary",
            [sglang],
            expected_epoch=epoch,
            scope="sglang",
        )
        protected_vllm, rejected = client.directory_hbm_inventory(
            "engine-primary", epoch, scope="vllm"
        )
        assert not rejected
        protected_sglang, rejected = client.directory_hbm_inventory(
            "engine-primary", epoch, scope="sglang"
        )
        assert not rejected
    assert protected_vllm == {"0": [31]}
    assert protected_sglang == {"0": [32]}


def test_publication_requires_exact_writer_epoch(directory_daemon):
    _daemon, socket_path = directory_daemon
    first = _item(b"epoch-first", 25)
    stale = _item(b"epoch-stale", 26)
    with DaemonClient(socket_path) as client:
        primary_epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a",
            "engine-primary",
            [first],
            expected_epoch=primary_epoch,
        )
        promoted, shadow_epoch, _active = client.directory_promote(
            primary_epoch, "engine-shadow"
        )
        assert promoted and shadow_epoch == primary_epoch + 1
        result = client.directory_publish_batch(
            "manifest-a",
            "engine-shadow",
            [stale],
            expected_epoch=primary_epoch,
        )
        assert result["rejected_stale_writer"] is True
        hits, _epoch, _writer = client.directory_lookup(
            "manifest-a", [stale["content_hash"]]
        )
        assert hits == [None]


def test_directory_snapshot_then_delta_has_no_lost_update(directory_daemon):
    _daemon, socket_path = directory_daemon
    first = _item(b"snapshot-first", 61)
    second = _item(b"snapshot-second", 62)
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a", "engine-primary", [first], expected_epoch=epoch
        )
        snapshot, revision, observed_epoch, writer = client.directory_snapshot(
            "manifest-a"
        )
        assert snapshot[first["content_hash"]]["slot_ids"] == [61]
        assert observed_epoch == epoch
        assert writer == "engine-primary"

        client.directory_publish_batch(
            "manifest-a", "engine-primary", [second], expected_epoch=epoch
        )
        delta = client.directory_changes("manifest-a", revision)

    assert delta["reset_required"] is False
    assert delta["has_more"] is False
    assert delta["next_revision"] == delta["directory_revision"]
    assert [(change["content_hash"], change["entry"]["slot_ids"]) for change in delta["changes"]] == [
        (second["content_hash"], [62])
    ]


def test_directory_delta_reports_slot_reuse_delete_before_upsert(directory_daemon):
    _daemon, socket_path = directory_daemon
    old = _item(b"delta-old", 63, generation=1)
    new = _item(b"delta-new", 63, generation=2)
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a", "engine-primary", [old], expected_epoch=epoch
        )
        _snapshot, revision, _epoch, _writer = client.directory_snapshot("manifest-a")
        client.directory_publish_batch(
            "manifest-a", "engine-primary", [new], expected_epoch=epoch
        )
        delta = client.directory_changes("manifest-a", revision)

    assert [(change["content_hash"], change["entry"]) for change in delta["changes"]] == [
        (old["content_hash"], None),
        (
            new["content_hash"],
            {
                "engine_id": "0",
                "slot_ids": [63],
                "generations": [2],
                "ranges": [[0, 1008, 16]],
                "state": "ready",
            },
        ),
    ]


def test_directory_delta_gap_requires_atomic_resnapshot(directory_daemon):
    daemon, socket_path = directory_daemon
    daemon.kv_cache_manager._content_directory_changes = deque(maxlen=2)
    items = [_item(f"gap-{index}".encode(), 70 + index) for index in range(3)]
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        for item in items:
            client.directory_publish_batch(
                "manifest-a", "engine-primary", [item], expected_epoch=epoch
            )
        delta = client.directory_changes("manifest-a", 0)
        snapshot, revision, _epoch, _writer = client.directory_snapshot("manifest-a")

    assert delta["reset_required"] is True
    assert set(snapshot) == {item["content_hash"] for item in items}
    assert revision == delta["directory_revision"]


def test_directory_promotion_emits_recoverable_hbm_delta(directory_daemon):
    _daemon, socket_path = directory_daemon
    item = _item(b"promotion-delta", 81, generation=5)
    item.update(tier="hbm", active=True)
    with DaemonClient(socket_path) as client:
        primary_epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a",
            "engine-primary",
            [item],
            expected_epoch=primary_epoch,
        )
        _snapshot, revision, _epoch, _writer = client.directory_snapshot("manifest-a")
        promoted, shadow_epoch, _writer = client.directory_promote(
            primary_epoch, "engine-shadow"
        )
        assert promoted
        delta = client.directory_changes("manifest-a", revision)

    assert delta["directory_epoch"] == shadow_epoch
    assert len(delta["changes"]) == 1
    assert delta["changes"][0]["content_hash"] == item["content_hash"]
    assert delta["changes"][0]["entry"]["state"] == "ready"


def test_directory_snapshot_and_changes_are_read_only(directory_daemon):
    daemon, socket_path = directory_daemon
    item = _item(b"read-only-snapshot", 82)
    with DaemonClient(socket_path) as client:
        epoch = _promote(client)
        client.directory_publish_batch(
            "manifest-a", "engine-primary", [item], expected_epoch=epoch
        )
        before = daemon._content_directory_revision
        _snapshot, revision, _epoch, _writer = client.directory_snapshot("manifest-a")
        delta = client.directory_changes("manifest-a", revision)

    assert before == revision == daemon._content_directory_revision
    assert delta["changes"] == []
    assert delta["next_revision"] == revision


def test_async_read_view_serves_host_lookup_without_hot_path_rpc(
    directory_daemon, monkeypatch
):
    daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_READ", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_POLL_MS", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    item = _item(b"async-host", 91, generation=3)
    item["tier"] = "host"
    write = daemon.host_tier.reserve("0", 0, 91 * 16, 16)
    assert daemon.host_tier.commit(write, generation=3)
    with DaemonClient(socket_path) as client:
        epoch = _promote(client, "engine-test")
        client.directory_publish_batch(
            "manifest-a",
            "engine-test",
            [item],
            expected_epoch=epoch,
            scope="test",
        )

    directory = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="test",
        mode="authoritative",
    )
    try:
        assert directory.wait_until_synced(2.0)
        directory._call = lambda _operation: pytest.fail("hot-path RPC was used")
        assert directory.lookup([item["content_hash"]])[0]["tier"] == "host"
        entries, token = directory.lookup_and_claim([item["content_hash"]])
        assert entries[0]["slot_ids"] == [91]
        assert token is None
    finally:
        directory.close()


def test_async_read_view_applies_upsert_and_delete_deltas(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_READ", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_POLL_MS", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    first = _item(b"async-first", 92)
    replacement = _item(b"async-replacement", 92, generation=2)
    with DaemonClient(socket_path) as client:
        epoch = _promote(client, "engine-test")
        client.directory_publish_batch(
            "manifest-a",
            "engine-test",
            [first],
            expected_epoch=epoch,
            scope="test",
        )
        directory = ContentDirectory(
            socket_path,
            engine="test",
            block_size=16,
            engine_id="test",
            mode="authoritative",
        )
        try:
            assert directory.wait_until_synced(2.0)
            assert directory.read_view_is_current_writer
            assert directory.lookup([first["content_hash"]])[0] is not None
            client.directory_publish_batch(
                "manifest-a",
                "engine-test",
                [replacement],
                expected_epoch=epoch,
                scope="test",
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                values = directory.lookup(
                    [first["content_hash"], replacement["content_hash"]]
                )
                if values == [None, values[1]] and values[1] is not None:
                    break
                time.sleep(0.01)
            assert values[0] is None
            assert values[1]["generations"] == [2]
        finally:
            directory.close()


def test_async_read_view_uses_transactional_rpc_only_for_hbm_hits(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_READ", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_POLL_MS", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    item = _item(b"async-hbm", 93, generation=4)
    item["tier"] = "hbm"
    with DaemonClient(socket_path) as client:
        epoch = _promote(client, "engine-test")
        client.directory_publish_batch(
            "manifest-a",
            "engine-test",
            [item],
            expected_epoch=epoch,
            scope="test",
        )

    directory = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="test",
        mode="authoritative",
    )
    calls = 0
    original = directory._writer_call

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    directory._writer_call = counted
    try:
        assert directory.wait_until_synced(2.0)
        entries, token = directory.lookup_and_claim(
            [_hash(b"definite-miss"), item["content_hash"]]
        )
        assert entries[0] is None
        assert entries[1]["tier"] == "hbm"
        assert token is not None
        assert calls == 1
        assert directory.release_claim(token)
    finally:
        directory.close()


def test_async_read_view_is_fail_closed_before_first_snapshot(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_READ", "1")
    directory = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="test",
        mode="authoritative",
    )
    directory.start_async_read = lambda: True
    try:
        assert directory.read_view_ready is False
        assert directory.lookup([_hash(b"unknown")]) == [None]
        assert directory.lookup_and_claim([_hash(b"unknown")]) == ([None], None)
    finally:
        directory.close()


def test_directory_changes_long_poll_wakes_on_commit(directory_daemon):
    _daemon, socket_path = directory_daemon
    first = _item(b"long-poll-first", 94)
    second = _item(b"long-poll-second", 95)
    with DaemonClient(socket_path) as writer:
        epoch = _promote(writer)
        writer.directory_publish_batch(
            "manifest-a", "engine-primary", [first], expected_epoch=epoch
        )
        _snapshot, revision, _epoch, _writer = writer.directory_snapshot("manifest-a")
        result = {}
        ready = threading.Event()

        def wait_for_change():
            with DaemonClient(socket_path) as reader:
                ready.set()
                result.update(
                    reader.directory_changes(
                        "manifest-a", revision, wait_ms=1_000
                    )
                )

        thread = threading.Thread(target=wait_for_change)
        thread.start()
        assert ready.wait(1.0)
        time.sleep(0.02)
        started = time.monotonic()
        writer.directory_publish_batch(
            "manifest-a", "engine-primary", [second], expected_epoch=epoch
        )
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert time.monotonic() - started < 0.5
    assert [change["content_hash"] for change in result["changes"]] == [
        second["content_hash"]
    ]


def test_deferred_hbm_publish_and_dormancy_commit_in_order(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_PUBLISH", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    item = _item(b"deferred-hbm", 96, generation=8)
    item.update(tier="hbm", active=True)
    directory = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="test",
        mode="authoritative",
    )
    try:
        assert directory.publish_deferred([item]) == 1
        assert directory.mark_hbm_dormant_deferred([item["content_hash"]]) == 1
        assert directory.flush_deferred(2.0)
        with DaemonClient(socket_path) as client:
            entries, _epoch, writer = client.directory_lookup(
                "manifest-a", [item["content_hash"]]
            )
        assert writer == "engine-test"
        assert entries[0]["state"] == "ready"
        assert entries[0]["generations"] == [8]
    finally:
        directory.close()


def test_async_directory_defaults_on_and_publication_can_be_disabled(monkeypatch):
    monkeypatch.delenv("GMS_KV_DIRECTORY_ASYNC_READ", raising=False)
    monkeypatch.delenv("GMS_KV_DIRECTORY_ASYNC_PUBLISH", raising=False)
    directory = ContentDirectory(
        "/tmp/unused-gms-directory.sock",
        engine="test",
        block_size=16,
        mode="authoritative",
    )
    try:
        assert directory.async_read_enabled
        assert directory.async_publish_enabled
    finally:
        directory.close()

    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_PUBLISH", "0")
    synchronous = ContentDirectory(
        "/tmp/unused-gms-directory.sock",
        engine="test",
        block_size=16,
        mode="authoritative",
    )
    try:
        assert not synchronous.async_publish_enabled
    finally:
        synchronous.close()


def test_deferred_publication_close_is_a_durability_boundary(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_PUBLISH", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    item = _item(b"deferred-close", 97)
    directory = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="test",
        mode="authoritative",
    )
    directory.publish_deferred([item])
    directory.close()
    with DaemonClient(socket_path) as client:
        entries, _epoch, _writer = client.directory_lookup(
            "manifest-a", [item["content_hash"]]
        )
    assert entries[0]["slot_ids"] == [97]


def test_deferred_publication_surfaces_stale_writer_failure(
    directory_daemon, monkeypatch
):
    _daemon, socket_path = directory_daemon
    monkeypatch.setenv("GMS_KV_DIRECTORY_ASYNC_PUBLISH", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest-a")
    primary = ContentDirectory(
        socket_path,
        engine="test",
        block_size=16,
        engine_id="primary",
        mode="authoritative",
    )
    try:
        assert primary.publish([_item(b"deferred-owner", 98)]) == 1
        with DaemonClient(socket_path) as client:
            _entries, epoch, _writer = client.directory_lookup("manifest-a", [])
            promoted, _epoch, writer = client.directory_promote(
                epoch, "engine-shadow"
            )
            assert promoted and writer == "engine-shadow"
        primary.publish_deferred([_item(b"deferred-stale", 99)])
        with pytest.raises(RuntimeError, match="mutation worker failed"):
            primary.flush_deferred(2.0)
    finally:
        primary.close()
