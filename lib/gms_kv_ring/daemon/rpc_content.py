# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-address, staging, and restore RPC handlers."""
from __future__ import annotations

import base64
import binascii
import ctypes
import logging
import mmap
import os
import stat
import uuid
import zlib
from typing import TYPE_CHECKING, Optional

from gms_kv_ring.daemon.rpc_types import Handler, Message, Response, required_int

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from gms_kv_ring.daemon.server import Daemon
    from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager


def _persistent_kv_host_key(
    namespace: str,
    key: bytes,
) -> tuple[str, int, int]:
    """Injectively map an opaque key onto the existing HostTier key shape."""
    if not key:
        raise ValueError("persistent KV key must not be empty")
    return f"persistent-kv:{namespace}", len(key), int.from_bytes(key, "big")


def handle_persistent_kv_lookup(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    namespace = str(msg["namespace"])
    keys = [bytes.fromhex(str(value)) for value in (msg.get("keys") or [])]
    hits = []
    for key in keys:
        engine_id, layer, offset = _persistent_kv_host_key(namespace, key)
        hits.append(daemon.host_tier.get(engine_id, layer, offset) is not None)
    return {"ok": True, "hits": hits}


def _persistent_kv_store_payloads(
    daemon: "GmsKvCacheManager",
    namespace: str,
    items: list[tuple[bytes, bytes]],
) -> list[bool]:
    results = []
    protected = set()
    for key, payload in items:
        write = None
        try:
            engine_id, layer, offset = _persistent_kv_host_key(namespace, key)
            write = daemon.host_tier.reserve(
                engine_id,
                layer,
                offset,
                len(payload),
            )
            ctypes.memmove(write.host_ptr, payload, len(payload))
            crc = zlib.crc32(payload) & 0xFFFF_FFFF
            committed = daemon.host_tier.commit(write, crc)
            results.append(bool(committed))
            if committed:
                protected.add((engine_id, layer, offset))
        except (ValueError, TypeError):
            if write is not None:
                daemon.host_tier.abort(write)
            results.append(False)
    daemon._enforce_host_tier_quota(protected)
    return results


def handle_persistent_kv_store(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Reference transport: JSON/base64 bytes into GMS-owned pinned RAM."""
    try:
        items = [
            (
                bytes.fromhex(str(item["key"])),
                base64.b64decode(str(item["data"]), validate=True),
            )
            for item in (msg.get("items") or [])
        ]
    except (KeyError, ValueError, TypeError, binascii.Error):
        return {"ok": True, "stored": [False] * len(msg.get("items") or [])}
    return {
        "ok": True,
        "stored": _persistent_kv_store_payloads(
            daemon,
            str(msg["namespace"]),
            items,
        ),
    }


def handle_persistent_kv_attach_pool(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    path = os.path.realpath(str(msg["path"]))
    num_rows = required_int(msg, "num_rows")
    row_size = required_int(msg, "row_size")
    if num_rows <= 0 or row_size <= 0:
        raise ValueError("persistent KV pool geometry must be positive")
    expected_size = num_rows * row_size
    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size < expected_size:
            raise ValueError("persistent KV pool path has incompatible size or type")
        mapped = mmap.mmap(
            fd,
            expected_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(fd)
    pool_id = uuid.uuid4().hex
    with daemon._persistent_kv_pools_lock:
        daemon._persistent_kv_pools[pool_id] = {
            "mmap": mapped,
            "view": memoryview(mapped).cast("B"),
            "num_rows": num_rows,
            "row_size": row_size,
        }
    return {"ok": True, "pool_id": pool_id}


def handle_persistent_kv_detach_pool(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    with daemon._persistent_kv_pools_lock:
        pool = daemon._persistent_kv_pools.pop(str(msg["pool_id"]), None)
    if pool is None:
        return {"ok": True, "detached": False}
    pool["view"].release()
    pool["mmap"].close()
    return {"ok": True, "detached": True}


def _persistent_kv_read_pool_rows(
    daemon: "GmsKvCacheManager",
    pool_id: str,
    slots: list[int],
) -> list[bytes]:
    with daemon._persistent_kv_pools_lock:
        pool = daemon._persistent_kv_pools.get(pool_id)
        if pool is None:
            raise ValueError("unknown persistent KV pool")
        row_size = int(pool["row_size"])
        num_rows = int(pool["num_rows"])
        if any(slot < 0 or slot >= num_rows for slot in slots):
            raise IndexError("persistent KV pool slot is out of range")
        view = pool["view"]
        return [bytes(view[slot * row_size : (slot + 1) * row_size]) for slot in slots]


def handle_persistent_kv_store_pool(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    items = msg.get("items") or []
    keys = [bytes.fromhex(str(item["key"])) for item in items]
    slots = [int(item["slot"]) for item in items]
    payloads = _persistent_kv_read_pool_rows(
        daemon,
        str(msg["pool_id"]),
        slots,
    )
    stored = _persistent_kv_store_payloads(
        daemon,
        str(msg["namespace"]),
        list(zip(keys, payloads)),
    )
    return {"ok": True, "stored": stored}


def _persistent_kv_load_payloads(
    daemon: "GmsKvCacheManager",
    namespace: str,
    keys: list[bytes],
) -> list[bytes | None]:
    values = []
    for key in keys:
        engine_id, layer, offset = _persistent_kv_host_key(namespace, key)
        lease = daemon.host_tier.pin(engine_id, layer, offset)
        if lease is None:
            values.append(None)
            continue
        with lease as slot:
            payload = ctypes.string_at(slot.host_ptr, slot.size)
            if slot.crc and zlib.crc32(payload) & 0xFFFF_FFFF != slot.crc:
                values.append(None)
                continue
            values.append(payload)
    return values


def handle_persistent_kv_load(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    keys = [bytes.fromhex(str(value)) for value in (msg.get("keys") or [])]
    values = _persistent_kv_load_payloads(daemon, str(msg["namespace"]), keys)
    return {
        "ok": True,
        "values": [
            None if value is None else base64.b64encode(value).decode("ascii")
            for value in values
        ],
    }


def handle_persistent_kv_load_pool(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    keys = [bytes.fromhex(str(value)) for value in (msg.get("keys") or [])]
    slots = [int(value) for value in (msg.get("slots") or [])]
    if len(keys) != len(slots):
        raise ValueError("persistent KV keys and slots must have equal lengths")
    payloads = _persistent_kv_load_payloads(daemon, str(msg["namespace"]), keys)
    if len(payloads) != len(keys) or any(value is None for value in payloads):
        return {"ok": True, "loaded": False}
    with daemon._persistent_kv_pools_lock:
        pool = daemon._persistent_kv_pools.get(str(msg["pool_id"]))
        if pool is None:
            raise ValueError("unknown persistent KV pool")
        row_size = int(pool["row_size"])
        num_rows = int(pool["num_rows"])
        if any(slot < 0 or slot >= num_rows for slot in slots):
            raise IndexError("persistent KV pool slot is out of range")
        if any(len(payload) != row_size for payload in payloads if payload is not None):
            raise ValueError("persistent KV payload has incompatible row size")
        view = pool["view"]
        for slot, payload in zip(slots, payloads):
            assert payload is not None
            view[slot * row_size : (slot + 1) * row_size] = payload
    return {"ok": True, "loaded": True}


def handle_staging_reserve(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Receiver-side RPC for cross-node transfer (Phase 4b).
    # Router calls this on the DESTINATION daemon to
    # pre-allocate a slot + register a write target before
    # the SOURCE daemon issues the NIXL WRITE.
    #
    # Returns (reservation_id, remote_ptr). Sender uses
    # remote_ptr in `initialize_xfer(..., remote_descs)`
    # and reservation_id+content_hash in the notif payload.
    if daemon.staging_tier is None or daemon.staging_receive_buffer is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    content_hash = bytes.fromhex(str(msg["content_hash"]))
    size = int(msg["size"])
    source_daemon = str(msg.get("source_daemon", "unknown"))
    # Step 1: reserve the StagingTier slot. May coalesce
    # if another peer is already delivering this hash.
    from gms_kv_ring.daemon.staging_tier import (
        AlreadyReady,
        Rejected,
        Reservation,
        Waiter,
    )

    result = daemon.staging_tier.reserve_or_wait(
        content_hash,
        source_daemon,
    )
    if isinstance(result, AlreadyReady):
        return {
            "ok": True,
            "outcome": "already_ready",
            "bytes_size": result.hit.bytes_size,
            "generation": result.hit.generation,
        }
    if isinstance(result, Waiter):
        # Another transfer is in flight. Sender skips.
        # (Real waiter mechanics require a follow-up
        # event channel for cross-process notification;
        # current contract: router retries with backoff.)
        return {
            "ok": True,
            "outcome": "coalesced",
        }
    if isinstance(result, Rejected):
        return {
            "ok": False,
            "error": f"rejected: {result.reason}",
        }
    # Reservation — allocate receive buffer offset
    assert isinstance(result, Reservation)
    offset = daemon.staging_receive_buffer.alloc(size)
    if offset is None:
        # Out of receive-buffer capacity; release the
        # StagingTier reservation we just made.
        daemon.staging_tier.fail_reservation(
            result.reservation_id,
            "receive buffer full",
        )
        return {
            "ok": False,
            "error": "receive buffer out of capacity",
        }
    with daemon._xfers_lock:
        daemon._active_xfers[result.reservation_id] = (
            offset,
            size,
            content_hash,
        )
    remote_ptr = daemon.staging_receive_buffer.ptr_at(offset)
    return {
        "ok": True,
        "outcome": "reserved",
        "reservation_id": result.reservation_id,
        "remote_ptr": remote_ptr,
    }


def handle_staging_fail(daemon: "Daemon", msg: Message) -> Response:
    # Cleanup on transfer failure. Frees the receive
    # buffer offset and the StagingTier reservation.
    rid = str(msg["reservation_id"])
    reason = str(msg.get("reason", "external"))
    with daemon._xfers_lock:
        entry = daemon._active_xfers.pop(rid, None)
    if entry is not None and daemon.staging_receive_buffer:
        offset, size, _hash = entry
        daemon.staging_receive_buffer.free(offset, size)
    if daemon.staging_tier is not None:
        daemon.staging_tier.fail_reservation(rid, reason)
    return {"ok": True}


def handle_register_content_addresses_batch(daemon: "Daemon", msg: Message) -> Response:
    # Batched form of register_content_address — connector
    # emits one RPC per request_finished instead of one
    # per block. Each item: {content_hash, engine_id, ranges}.
    # Optional fields: {generation, sealed, metadata}. A
    # sealed=False item is deliberately not advertised.
    items = msg.get("items", []) or []
    if daemon.transport is None:
        return {"ok": True, "total_bytes": 0, "skipped": True}
    total_bytes = 0
    unsealed = 0
    registered: list[tuple[bytes, int, Optional[dict], dict]] = []
    with daemon._content_hash_lock:
        for item in items:
            try:
                ch = bytes.fromhex(str(item["content_hash"]))
                eng = str(item["engine_id"])
                ranges = [
                    (int(r["layer"]), int(r["offset"]), int(r["size"]))
                    for r in (item.get("ranges") or [])
                ]
                sealed_raw = item.get("sealed", True)
                if isinstance(sealed_raw, str):
                    sealed = sealed_raw.lower() not in (
                        "0",
                        "false",
                        "no",
                        "off",
                        "",
                    )
                else:
                    sealed = bool(sealed_raw)
                generation_raw = item.get("generation")
                generation = None if generation_raw is None else int(generation_raw)
                metadata = item.get("metadata")
                if metadata is not None and not isinstance(metadata, dict):
                    metadata = None
            except (KeyError, ValueError, TypeError) as exc:
                return {
                    "ok": False,
                    "error": f"malformed item: {exc}",
                }
            if not sealed:
                daemon._content_hash_index.pop(ch, None)
                unsealed += 1
                continue
            entry = {"engine_id": eng, "ranges": ranges}
            if generation is not None:
                entry["generation"] = generation
            if metadata is not None:
                entry["metadata"] = metadata
            daemon._content_hash_index[ch] = entry
            sz = sum(sz for _, _, sz in ranges)
            total_bytes += sz
            registered.append((ch, sz, metadata, dict(entry)))
    # Publish Stored events outside the lock to avoid
    # blocking other content-hash lookups.
    if daemon.placement_publisher is not None:
        for ch, sz, metadata, entry in registered:
            placement_metadata = daemon._content_address_placement_metadata(
                ch,
                entry,
                metadata,
            )
            try:
                daemon.placement_publisher.publish_stored(
                    content_hash=ch,
                    tier="host_pinned",
                    bytes_size=sz,
                    metadata=placement_metadata,
                )
            except TypeError:
                daemon.placement_publisher.publish_stored(
                    content_hash=ch,
                    tier="host_pinned",
                    bytes_size=sz,
                )
            except Exception:
                logger.exception(
                    "[Daemon] publish_stored failed for batch item",
                )
    return {
        "ok": True,
        "total_bytes": total_bytes,
        "skipped": False,
        "unsealed": unsealed,
    }


def handle_register_content_address(daemon: "Daemon", msg: Message) -> Response:
    # Connector calls this after a successful spill to
    # advertise the content_hash → host_tier address
    # mapping. The router uses this index to drive
    # cross-node transfers (P4c). Multi-range payload
    # because one logical block spans N layers.
    content_hash = bytes.fromhex(str(msg["content_hash"]))
    engine_id = str(msg["engine_id"])
    ranges_raw = msg.get("ranges", []) or []
    ranges = [(int(r["layer"]), int(r["offset"]), int(r["size"])) for r in ranges_raw]
    sealed_raw = msg.get("sealed", True)
    if isinstance(sealed_raw, str):
        sealed = sealed_raw.lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    else:
        sealed = bool(sealed_raw)
    if not sealed:
        with daemon._content_hash_lock:
            daemon._content_hash_index.pop(content_hash, None)
        return {"ok": True, "total_size": 0, "unsealed": 1}
    generation_raw = msg.get("generation")
    generation = None if generation_raw is None else int(generation_raw)
    metadata = msg.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        metadata = None
    entry = {"engine_id": engine_id, "ranges": ranges}
    if generation is not None:
        entry["generation"] = generation
    if metadata is not None:
        entry["metadata"] = metadata
    with daemon._content_hash_lock:
        daemon._content_hash_index[content_hash] = entry
    total_size = sum(sz for _, _, sz in ranges)
    if daemon.placement_publisher is not None:
        placement_metadata = daemon._content_address_placement_metadata(
            content_hash,
            entry,
            metadata,
        )
        try:
            daemon.placement_publisher.publish_stored(
                content_hash=content_hash,
                tier="host_pinned",
                bytes_size=total_size,
                metadata=placement_metadata,
            )
        except TypeError:
            daemon.placement_publisher.publish_stored(
                content_hash=content_hash,
                tier="host_pinned",
                bytes_size=total_size,
            )
        except Exception:
            logger.exception(
                "[Daemon] publish_stored failed",
            )
    return {"ok": True, "total_size": total_size}


def _directory_record_change_locked(
    daemon: "GmsKvCacheManager",
    key: tuple[str, bytes],
    entry: Optional[dict],
    *,
    scope: str = "",
) -> int:
    """Append one public upsert/delete after its mutation is committed.

    The caller holds the content-hash lock. Revisions are global to the
    daemon, while consumers filter by manifest and optional engine scope.
    """
    daemon._content_directory_revision += 1
    revision = int(daemon._content_directory_revision)
    daemon._content_directory_changes.append(
        {
            "revision": revision,
            "manifest_id": key[0],
            "content_hash": key[1].hex(),
            "scope": str(scope or (entry or {}).get("_scope", "")),
            "entry": None if entry is None else _directory_public_entry(entry),
        }
    )
    daemon._content_hash_lock.notify_all()
    return revision


def _directory_remove_locked(
    daemon: "GmsKvCacheManager",
    key: tuple[str, bytes],
) -> bool:
    """Remove one directory entry and its reverse slot mappings."""
    entry = daemon._content_directory.pop(key, None)
    if entry is None:
        return False
    manifest_id, content_hash = key
    engine_id = str(entry["engine_id"])
    for slot_id in entry["slot_ids"]:
        slot_key = (manifest_id, engine_id, int(slot_id))
        if daemon._content_directory_by_slot.get(slot_key) == content_hash:
            daemon._content_directory_by_slot.pop(slot_key, None)
    _directory_record_change_locked(
        daemon,
        key,
        None,
        scope=str(entry.get("_scope", "")),
    )
    return True


def _directory_public_entry(entry: dict) -> dict:
    """Return a wire-safe residency without daemon bookkeeping fields."""
    return {name: value for name, value in entry.items() if not name.startswith("_")}


def _directory_touch_locked(daemon: "GmsKvCacheManager", entry: dict) -> None:
    daemon._content_directory_access_seq += 1
    entry["_last_access_seq"] = int(daemon._content_directory_access_seq)


def _directory_writer_matches(
    daemon: "GmsKvCacheManager",
    writer_id: str,
    expected_epoch: int,
) -> bool:
    return daemon._content_directory_writer_id == writer_id and int(
        daemon._content_directory_epoch
    ) == int(expected_epoch)


def _directory_release_claim_locked(
    daemon: "GmsKvCacheManager",
    claim_token: str,
) -> bool:
    claim = daemon._content_directory_claims.pop(claim_token, None)
    if claim is None:
        return False
    for key, generations in claim["entries"]:
        entry = daemon._content_directory.get(key)
        if entry is None:
            continue
        if tuple(entry.get("generations") or ()) != generations:
            continue
        entry["_claim_count"] = max(0, int(entry.get("_claim_count", 0)) - 1)
    return True


def _directory_release_writer_claims_locked(
    daemon: "GmsKvCacheManager",
    writer_id: Optional[str],
) -> int:
    if writer_id is None:
        return 0
    tokens = [
        token
        for token, claim in daemon._content_directory_claims.items()
        if claim.get("writer_id") == writer_id
    ]
    for token in tokens:
        _directory_release_claim_locked(daemon, token)
    return len(tokens)


def _directory_entry_ready(daemon: "GmsKvCacheManager", entry: dict) -> bool:
    """Validate physical readiness for entries published by engine adapters."""
    tier = str(entry.get("tier", ""))
    if tier not in ("host", "storage"):
        return True
    engine_id = str(entry["engine_id"])
    ranges = entry.get("ranges") or []
    slot_ids = entry.get("slot_ids") or []
    generations = entry.get("generations") or []
    if (
        not ranges
        or not slot_ids
        or len(slot_ids) != len(generations)
        or len(ranges) % len(slot_ids) != 0
    ):
        return False
    getter = daemon.host_tier.get if tier == "host" else daemon.storage_tier.get
    ranges_per_slot = len(ranges) // len(slot_ids)
    for index, (slot_id, generation) in enumerate(zip(slot_ids, generations)):
        begin = index * ranges_per_slot
        end = begin + ranges_per_slot
        for layer, offset, _size in ranges[begin:end]:
            physical = getter(engine_id, int(layer), int(offset))
            if physical is None:
                return False
            if (
                tier == "host"
                and int(generation)
                and int(physical.generation) != int(generation)
            ):
                return False
        if tier == "storage" and int(generation):
            with daemon._lock:
                current = daemon._storage_slot_generations.get(engine_id, {}).get(
                    int(slot_id), 0
                )
            if int(current) != int(generation):
                return False
    return True


def handle_directory_promote(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """CAS-promote a writer after the external failover fence is held.

    Repeating the call for the active writer is idempotent so TP ranks can
    safely execute the same post-lock hook. A different writer must present
    the current epoch; a successful promotion increments it and immediately
    fences publications from the former writer.
    """
    writer_id = str(msg.get("writer_id", "")).strip()
    if not writer_id:
        return {"ok": False, "error": "writer_id is required"}
    try:
        expected_epoch = int(msg["expected_epoch"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "expected_epoch is required"}
    with daemon._content_hash_lock:
        current = int(daemon._content_directory_epoch)
        active = daemon._content_directory_writer_id
        if active == writer_id:
            return {
                "ok": True,
                "promoted": True,
                "directory_epoch": current,
                "writer_id": active,
            }
        if expected_epoch != current:
            return {
                "ok": True,
                "promoted": False,
                "directory_epoch": current,
                "writer_id": active,
            }
        # The external failover lock already fenced the former writer. Its
        # active HBM entries are now dormant recovery candidates: the bytes
        # and leases remain in GMS, but no live engine may mutate them.
        for key, entry in daemon._content_directory.items():
            if (
                entry.get("tier") == "hbm"
                and entry.get("state") == "active"
                and entry.get("_owner_writer") == active
            ):
                entry["state"] = "ready"
                entry.pop("_owner_writer", None)
                _directory_touch_locked(daemon, entry)
                _directory_record_change_locked(daemon, key, entry)

        # Drop abandoned lookup claims so a crashed engine cannot pin HBM
        # forever.
        _directory_release_writer_claims_locked(daemon, active)
        daemon._content_directory_epoch = current + 1
        daemon._content_directory_writer_id = writer_id
        return {
            "ok": True,
            "promoted": True,
            "directory_epoch": current + 1,
            "writer_id": writer_id,
        }


def handle_directory_snapshot(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Return one atomic public inventory snapshot and its delta cursor."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    scope = str(msg.get("scope", ""))
    if not manifest_id:
        return {"ok": False, "error": "manifest_id is required"}
    with daemon._content_hash_lock:
        items = []
        for key, entry in list(daemon._content_directory.items()):
            if key[0] != manifest_id:
                continue
            if scope and entry.get("_scope") != scope:
                continue
            if not _directory_entry_ready(daemon, entry):
                _directory_remove_locked(daemon, key)
                continue
            items.append(
                {
                    "content_hash": key[1].hex(),
                    "entry": _directory_public_entry(entry),
                }
            )
        return {
            "ok": True,
            "items": items,
            "directory_epoch": int(daemon._content_directory_epoch),
            "directory_revision": int(daemon._content_directory_revision),
            "writer_id": daemon._content_directory_writer_id,
        }


def handle_directory_changes(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Return committed changes after a cursor or request a fresh snapshot."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    scope = str(msg.get("scope", ""))
    try:
        after = max(0, int(msg.get("after_revision", 0)))
        limit = min(16_384, max(1, int(msg.get("limit", 4096))))
        wait_ms = min(1_000, max(0, int(msg.get("wait_ms", 0))))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed change cursor: {exc}"}
    if not manifest_id:
        return {"ok": False, "error": "manifest_id is required"}
    with daemon._content_hash_lock:
        if wait_ms and int(daemon._content_directory_revision) <= after:
            daemon._content_hash_lock.wait(wait_ms / 1_000)
        current = int(daemon._content_directory_revision)
        changes_log = daemon._content_directory_changes
        oldest = int(changes_log[0]["revision"]) if changes_log else current + 1
        reset_required = after > current or after < oldest - 1
        if reset_required:
            return {
                "ok": True,
                "changes": [],
                "next_revision": current,
                "directory_revision": current,
                "directory_epoch": int(daemon._content_directory_epoch),
                "writer_id": daemon._content_directory_writer_id,
                "has_more": False,
                "reset_required": True,
            }

        selected = []
        next_revision = after
        exhausted = True
        for change in changes_log:
            revision = int(change["revision"])
            if revision <= after:
                continue
            next_revision = revision
            if change["manifest_id"] == manifest_id and (
                not scope or change.get("scope") == scope
            ):
                selected.append(
                    {
                        "revision": revision,
                        "content_hash": change["content_hash"],
                        "entry": change["entry"],
                    }
                )
                if len(selected) >= limit:
                    exhausted = False
                    break
        if exhausted:
            # Writer promotion can change the epoch without an entry delta.
            next_revision = current
        return {
            "ok": True,
            "changes": selected,
            "next_revision": int(next_revision),
            "directory_revision": current,
            "directory_epoch": int(daemon._content_directory_epoch),
            "writer_id": daemon._content_directory_writer_id,
            "has_more": int(next_revision) < current,
            "reset_required": False,
        }


def handle_directory_lookup(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Return manifest-scoped READY entries aligned with requested hashes."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    if not manifest_id:
        return {"ok": False, "error": "manifest_id is required"}
    try:
        content_hashes = [bytes.fromhex(str(h)) for h in msg.get("hashes", [])]
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed hash: {exc}"}
    with daemon._content_hash_lock:
        entries = []
        for content_hash in content_hashes:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if entry is not None and not _directory_entry_ready(daemon, entry):
                _directory_remove_locked(daemon, key)
                entry = None
            if entry is not None and entry.get("state") != "ready":
                entry = None
            if entry is not None:
                _directory_touch_locked(daemon, entry)
            entries.append(None if entry is None else _directory_public_entry(entry))
        epoch = int(daemon._content_directory_epoch)
        writer_id = daemon._content_directory_writer_id
    return {
        "ok": True,
        "entries": entries,
        "directory_epoch": epoch,
        "writer_id": writer_id,
    }


def handle_directory_lookup_claim(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Lookup READY entries and pin every hit under one opaque claim."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    writer_id = str(msg.get("writer_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        content_hashes = [bytes.fromhex(str(h)) for h in msg.get("hashes", [])]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed lookup claim: {exc}"}
    if not manifest_id or not writer_id:
        return {"ok": False, "error": "manifest_id and writer_id are required"}

    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "entries": [None] * len(content_hashes),
                "claim_token": None,
                "rejected_stale_writer": True,
                "directory_epoch": int(daemon._content_directory_epoch),
                "writer_id": daemon._content_directory_writer_id,
            }
        entries: list[Optional[dict]] = []
        claimed = []
        for content_hash in content_hashes:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if entry is not None and not _directory_entry_ready(daemon, entry):
                _directory_remove_locked(daemon, key)
                entry = None
            if entry is None or entry.get("state") != "ready":
                entries.append(None)
                continue
            entry["_claim_count"] = int(entry.get("_claim_count", 0)) + 1
            _directory_touch_locked(daemon, entry)
            claimed.append((key, tuple(entry.get("generations") or ())))
            entries.append(_directory_public_entry(entry))
        claim_token = uuid.uuid4().hex if claimed else None
        if claim_token is not None:
            daemon._content_directory_claims[claim_token] = {
                "writer_id": writer_id,
                "epoch": expected_epoch,
                "entries": claimed,
            }
        if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS"):
            count = int(getattr(daemon, "_content_directory_diag_claims", 0))
            if count < 16:
                daemon._content_directory_diag_claims = count + 1
                logger.warning(
                    "[GMS-KVDiag] claim writer=%s manifest=%s keys=%s matched=%s",
                    writer_id,
                    manifest_id,
                    [value.hex()[:16] for value in content_hashes],
                    [entry is not None for entry in entries],
                )
        return {
            "ok": True,
            "entries": entries,
            "claim_token": claim_token,
            "rejected_stale_writer": False,
            "directory_epoch": int(daemon._content_directory_epoch),
            "writer_id": daemon._content_directory_writer_id,
        }


def handle_directory_release_claim(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    token = str(msg.get("claim_token", "")).strip()
    if not token:
        return {"ok": False, "error": "claim_token is required"}
    with daemon._content_hash_lock:
        released = _directory_release_claim_locked(daemon, token)
    return {"ok": True, "released": released}


def handle_directory_adopt_claim(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Commit new lease generations for HBM entries held by a claim."""
    token = str(msg.get("claim_token", "")).strip()
    writer_id = str(msg.get("writer_id", "")).strip()
    manifest_id = str(msg.get("manifest_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        updates = [
            (
                bytes.fromhex(str(item["content_hash"])),
                [int(value) for value in item["generations"]],
            )
            for item in (msg.get("items") or [])
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed adopt: {exc}"}
    with daemon._content_hash_lock:
        claim = daemon._content_directory_claims.get(token)
        if (
            claim is None
            or claim.get("writer_id") != writer_id
            or not _directory_writer_matches(daemon, writer_id, expected_epoch)
        ):
            return {
                "ok": True,
                "adopted": 0,
                "rejected_stale_writer": True,
            }
        claim_keys = {key for key, _generations in claim["entries"]}
        parsed = []
        for content_hash, generations in updates:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if (
                key not in claim_keys
                or entry is None
                or entry.get("tier") != "hbm"
                or len(generations) != len(entry.get("slot_ids") or [])
            ):
                return {"ok": False, "error": "adopt entry is not a claimed HBM hit"}
            parsed.append((key, entry, generations))
        _directory_release_claim_locked(daemon, token)
        for key, entry, generations in parsed:
            entry["generations"] = generations
            entry["state"] = "active"
            entry["_owner_writer"] = writer_id
            _directory_touch_locked(daemon, entry)
            _directory_record_change_locked(daemon, key, entry)
        return {
            "ok": True,
            "adopted": len(parsed),
            "rejected_stale_writer": False,
        }


def handle_directory_mark_hbm_dormant(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Move writer-owned active HBM records into the evictable READY set."""
    writer_id = str(msg.get("writer_id", "")).strip()
    manifest_id = str(msg.get("manifest_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        hashes = [bytes.fromhex(str(value)) for value in msg.get("hashes", [])]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed dormant update: {exc}"}
    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "updated": 0,
                "rejected_stale_writer": True,
            }
        updated = 0
        for content_hash in hashes:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if (
                entry is None
                or entry.get("tier") != "hbm"
                or entry.get("state") != "active"
                or entry.get("_owner_writer") != writer_id
            ):
                continue
            entry["state"] = "ready"
            entry.pop("_owner_writer", None)
            _directory_touch_locked(daemon, entry)
            _directory_record_change_locked(daemon, key, entry)
            updated += 1
        return {
            "ok": True,
            "updated": updated,
            "rejected_stale_writer": False,
        }


def handle_directory_ensure_hbm_capacity(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Retire the coldest unclaimed dormant HBM records.

    Exact lease generations are returned to the active engine, which releases
    the corresponding shared-memory leases only after the directory stops
    advertising them. This is a generic LRU over dormant records, not a prefix
    or request policy.
    """
    manifest_id = str(msg.get("manifest_id", "")).strip()
    writer_id = str(msg.get("writer_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        required = max(0, int(msg.get("required_blocks", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed capacity request: {exc}"}
    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "victims": [],
                "rejected_stale_writer": True,
            }
        candidates = [
            (key, entry)
            for key, entry in daemon._content_directory.items()
            if key[0] == manifest_id
            and entry.get("tier") == "hbm"
            and entry.get("state") == "ready"
            and int(entry.get("_claim_count", 0)) == 0
        ]
        candidates.sort(key=lambda pair: int(pair[1].get("_last_access_seq", 0)))
        victims = []
        freed = 0
        for key, entry in candidates:
            victims.append(
                {
                    "content_hash": key[1].hex(),
                    "engine_id": str(entry["engine_id"]),
                    "slot_ids": [int(value) for value in entry["slot_ids"]],
                    "generations": [
                        int(value) for value in entry.get("generations") or []
                    ],
                }
            )
            freed += len(entry["slot_ids"])
            _directory_remove_locked(daemon, key)
            if freed >= required:
                break
        return {
            "ok": True,
            "victims": victims,
            "freed_blocks": freed,
            "rejected_stale_writer": False,
        }


def handle_directory_hbm_inventory(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Return HBM slot ids that selective post-fence reclaim must preserve."""
    writer_id = str(msg.get("writer_id", "")).strip()
    scope = str(msg.get("scope", ""))
    try:
        expected_epoch = int(msg["expected_epoch"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "expected_epoch is required"}
    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "protected": {},
                "rejected_stale_writer": True,
            }
        protected: dict[str, set[int]] = {}
        for entry in daemon._content_directory.values():
            if scope and entry.get("_scope") != scope:
                continue
            if entry.get("tier") != "hbm" or entry.get("state") not in (
                "ready",
                "active",
            ):
                continue
            protected.setdefault(str(entry["engine_id"]), set()).update(
                int(slot_id) for slot_id in entry["slot_ids"]
            )
        return {
            "ok": True,
            "protected": {
                engine_id: sorted(slot_ids) for engine_id, slot_ids in protected.items()
            },
            "rejected_stale_writer": False,
        }


def handle_directory_publish_batch(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Atomically publish sealed GMS residencies for one active writer."""
    writer_id = str(msg.get("writer_id", "")).strip()
    manifest_id = str(msg.get("manifest_id", "")).strip()
    scope = str(msg.get("scope", ""))
    try:
        expected_epoch = int(msg["expected_epoch"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "expected_epoch is required"}
    if not writer_id or not manifest_id:
        return {
            "ok": False,
            "error": "writer_id and manifest_id are required",
        }
    parsed = []
    for item in msg.get("items", []) or []:
        try:
            content_hash = bytes.fromhex(str(item["content_hash"]))
            engine_id = str(item["engine_id"])
            raw_slots = item.get("slot_ids")
            if raw_slots is None:
                raw_slots = [item["slot_id"]]
            slot_ids = [int(slot) for slot in raw_slots]
            if not slot_ids:
                raise ValueError("slot_ids is empty")
            raw_generations = item.get("generations")
            if raw_generations is None:
                raw_generations = [item.get("generation", 0)] * len(slot_ids)
            generations = [int(generation) for generation in raw_generations]
            if len(generations) != len(slot_ids):
                raise ValueError("generations length differs from slot_ids")
            ranges = [
                (int(r["layer"]), int(r["offset"]), int(r["size"]))
                for r in (item.get("ranges") or [])
            ]
            tier = str(item.get("tier", ""))
            sealed = bool(item.get("sealed", True))
            active_hbm = bool(item.get("active", False)) and tier == "hbm"
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"malformed item: {exc}"}
        parsed.append(
            (
                content_hash,
                engine_id,
                slot_ids,
                generations,
                ranges,
                tier,
                sealed,
                active_hbm,
            )
        )
    if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS"):
        count = int(getattr(daemon, "_content_directory_diag_publishes", 0))
        if count < 16:
            daemon._content_directory_diag_publishes = count + 1
            logger.warning(
                "[GMS-KVDiag] publish writer=%s manifest=%s keys=%s active=%s",
                writer_id,
                manifest_id,
                [value[0].hex()[:16] for value in parsed[:4]],
                [value[7] for value in parsed[:4]],
            )

    with daemon._content_hash_lock:
        active = daemon._content_directory_writer_id
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "published": 0,
                "rejected_stale_writer": True,
                "directory_epoch": int(daemon._content_directory_epoch),
                "writer_id": active,
            }
        # Validate every conflict before mutating any record so the batch is
        # actually atomic on malformed reuse or concurrent claims.
        seen_keys = set()
        seen_slots = {}
        for (
            content_hash,
            engine_id,
            slot_ids,
            generations,
            _ranges,
            _tier,
            sealed,
            _active_hbm,
        ) in parsed:
            key = (manifest_id, content_hash)
            if key in seen_keys:
                return {"ok": False, "error": "duplicate directory content hash"}
            seen_keys.add(key)
            existing = daemon._content_directory.get(key)
            if (
                not sealed
                and existing is not None
                and any(generations)
                and tuple(existing.get("generations") or ()) != tuple(generations)
            ):
                # A delayed invalidation for an older lease generation must
                # not retire a newer publication for the same content hash.
                continue
            if existing is not None and int(existing.get("_claim_count", 0)):
                action = "remove" if not sealed else "replace"
                return {
                    "ok": False,
                    "error": f"cannot {action} a claimed directory entry",
                }
            if not sealed:
                continue
            for slot_id in slot_ids:
                slot_key = (manifest_id, engine_id, slot_id)
                pending_hash = seen_slots.get(slot_key)
                if pending_hash is not None and pending_hash != content_hash:
                    return {"ok": False, "error": "duplicate directory slot reuse"}
                seen_slots[slot_key] = content_hash
                old_hash = daemon._content_directory_by_slot.get(slot_key)
                if old_hash is None or old_hash == content_hash:
                    continue
                old_entry = daemon._content_directory.get((manifest_id, old_hash))
                if old_entry is not None and int(old_entry.get("_claim_count", 0)):
                    return {
                        "ok": False,
                        "error": "cannot reuse a claimed directory slot",
                    }

        published = 0
        removed = 0
        for (
            content_hash,
            engine_id,
            slot_ids,
            generations,
            ranges,
            tier,
            sealed,
            active_hbm,
        ) in parsed:
            key = (manifest_id, content_hash)
            if not sealed:
                existing = daemon._content_directory.get(key)
                if (
                    existing is not None
                    and any(generations)
                    and tuple(existing.get("generations") or ()) != tuple(generations)
                ):
                    continue
                removed += int(_directory_remove_locked(daemon, key))
                continue
            # Reusing any physical slot retires the entire previous logical
            # entry before the new entry becomes discoverable.
            for slot_id in slot_ids:
                old_hash = daemon._content_directory_by_slot.get(
                    (manifest_id, engine_id, slot_id)
                )
                if old_hash is not None and old_hash != content_hash:
                    removed += int(
                        _directory_remove_locked(daemon, (manifest_id, old_hash))
                    )
            entry = {
                "engine_id": engine_id,
                "slot_ids": slot_ids,
                "generations": generations,
                "ranges": ranges,
                "state": "active" if active_hbm else "ready",
                "_claim_count": 0,
                "_scope": scope,
            }
            if active_hbm:
                entry["_owner_writer"] = writer_id
            if tier:
                entry["tier"] = tier
            daemon._content_directory[key] = entry
            _directory_touch_locked(daemon, entry)
            for slot_id in slot_ids:
                daemon._content_directory_by_slot[(manifest_id, engine_id, slot_id)] = (
                    content_hash
                )
            _directory_record_change_locked(daemon, key, entry)
            published += 1
        epoch = int(daemon._content_directory_epoch)
    return {
        "ok": True,
        "published": published,
        "removed": removed,
        "rejected_stale_writer": False,
        "directory_epoch": epoch,
        "writer_id": writer_id,
    }


def handle_notify_kv_arrived(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Engine on the decode side tells its local daemon
    # "I just NIXL-read these hashes; please publish
    # PlacementEvent::Stored". Off the critical path —
    # this is for the indexer, not for the data flow.
    if daemon.placement_publisher is None:
        return {"ok": True, "published": 0}
    items = msg.get("items") or []
    published = 0
    for it in items:
        try:
            content_hash = bytes.fromhex(str(it["content_hash"]))
            size = int(it.get("size", 0))
        except (KeyError, ValueError):
            continue
        metadata = it.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            metadata = None
        try:
            daemon.placement_publisher.publish_stored(
                content_hash=content_hash,
                tier="external",
                bytes_size=size,
                metadata=metadata,
            )
            published += 1
        except Exception:  # noqa: BLE001
            logger.exception("[Daemon] notify_kv_arrived: publish_stored failed")
    return {"ok": True, "published": published}


def handle_restore_staging_ranges(daemon: "Daemon", msg: Message) -> Response:
    # Blocking control-plane restore from StagingTier into
    # explicit destination ranges. Unlike restore_staging_blocks,
    # this does not assume one content hash maps to one daemon
    # block stride. SGLang uses it for page hashes made from
    # several per-token KV slots.
    engine_id = str(msg["engine_id"])
    with daemon._lock:
        pool = daemon._pools.get(engine_id)
    if pool is None:
        return {
            "ok": False,
            "error": "engine pool not attached",
        }
    if daemon.staging_tier is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    raw_items = msg.get("items") or []
    from cuda.bindings import driver as drv
    from gms_kv_ring.common import metrics

    dsts: list[int] = []
    srcs: list[int] = []
    sizes: list[int] = []
    consume_handles: list = []
    valid_items = 0
    success = True
    try:
        for it in raw_items:
            try:
                content_hash = bytes.fromhex(
                    str(it["content_hash"]),
                )
                generation = int(it["generation"])
                raw_ranges = it.get("ranges") or []
                ranges = [
                    (
                        int(r["layer"]),
                        int(r["offset"]),
                        int(r["size"]),
                    )
                    for r in raw_ranges
                ]
            except (KeyError, ValueError, TypeError):
                success = False
                break
            if not ranges:
                success = False
                break
            consume = daemon.staging_tier.begin_consume(
                content_hash,
                generation,
            )
            if consume is None:
                logger.warning(
                    "restore-staging-ranges: hash=%s " "generation=%d not READY",
                    content_hash.hex()[:16],
                    generation,
                )
                success = False
                break
            consume_handles.append(consume)
            ptr_info = daemon.staging_tier.consume_pointer(
                consume,
            )
            if ptr_info is None:
                logger.warning(
                    "restore-staging-ranges: hash=%s disappeared after pin",
                    content_hash.hex()[:16],
                )
                success = False
                break
            src_ptr, bytes_size, _crc32 = ptr_info
            cursor = 0
            for layer_idx, offset, size in ranges:
                ld = pool.layers.get(int(layer_idx))
                if ld is None:
                    logger.warning(
                        "restore-staging-ranges: unknown layer=%d",
                        int(layer_idx),
                    )
                    success = False
                    break
                offset_i = int(offset)
                size_i = int(size)
                if offset_i < 0 or size_i <= 0 or offset_i + size_i > int(ld.size):
                    logger.warning(
                        "restore-staging-ranges: invalid "
                        "range layer=%d offset=%d size=%d "
                        "layer_size=%d",
                        int(layer_idx),
                        offset_i,
                        size_i,
                        int(ld.size),
                    )
                    success = False
                    break
                if cursor + size_i > int(bytes_size):
                    logger.warning(
                        "restore-staging-ranges: payload too "
                        "small for layer=%d offset=%d "
                        "cursor=%d size=%d payload=%d",
                        int(layer_idx),
                        offset_i,
                        cursor,
                        size_i,
                        int(bytes_size),
                    )
                    success = False
                    break
                dsts.append(int(ld.va) + offset_i)
                srcs.append(int(src_ptr) + cursor)
                sizes.append(size_i)
                cursor += size_i
            if not success:
                break
            if cursor != int(bytes_size):
                logger.warning(
                    "restore-staging-ranges: payload size "
                    "mismatch for hash=%s consumed=%d "
                    "payload=%d",
                    content_hash.hex()[:16],
                    cursor,
                    int(bytes_size),
                )
                success = False
                break
            valid_items += 1
        if success and dsts:
            for d, s, sz in zip(dsts, srcs, sizes):
                drv.cuMemcpyAsync(
                    drv.CUdeviceptr(d),
                    drv.CUdeviceptr(s),
                    int(sz),
                    int(pool.stream),
                )
            metrics.restore_h2d_bytes.inc(
                engine_id=engine_id,
                n=sum(sizes),
            )
            drv.cuStreamSynchronize(int(pool.stream))
        else:
            success = False
    except Exception:  # noqa: BLE001
        logger.warning(
            "restore-staging-ranges: copy failed",
            exc_info=True,
        )
        success = False
    finally:
        for consume in consume_handles:
            try:
                daemon.staging_tier.end_consume(consume)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "restore-staging-ranges: end_consume failed",
                    exc_info=True,
                )
    return {
        "ok": True,
        "success": bool(success),
        "requested": len(raw_items),
        "restored": valid_items if success else 0,
    }


def handle_restore_host_blocks(daemon: "Daemon", msg: Message) -> Response:
    engine_id = str(msg["engine_id"])
    src_engine_id = str(msg["src_engine_id"])
    with daemon._lock:
        dest_pool = daemon._pools.get(engine_id)
        src_pool = daemon._pools.get(src_engine_id)
    if dest_pool is None or src_pool is None or dest_pool.restore_consumer is None:
        return {
            "ok": False,
            "error": "engine restore consumer or source pool not attached",
        }
    raw_items = msg.get("items") or []
    block_pairs: list[tuple[int, int]] = []
    expected_generations: dict[int, int] = {}
    for it in raw_items:
        try:
            src_block = int(it["src_block"])
            dest_block = int(it["dest_block"])
            generation = int(it.get("generation", 0))
        except (KeyError, ValueError, TypeError):
            continue
        block_pairs.append((src_block, dest_block))
        if generation:
            expected_generations[src_block] = generation
    if not block_pairs:
        return {
            "ok": False,
            "error": "no valid host restore items",
        }
    success = dest_pool.restore_consumer._process_host_tier(
        {
            "src_engine_id": src_engine_id,
            "block_pairs": block_pairs,
            "expected_generations": expected_generations,
        },
        dest_pool,
        src_pool,
    )
    return {
        "ok": True,
        "success": bool(success),
        "requested": len(block_pairs),
        "restored": len(block_pairs) if success else 0,
    }


def handle_restore_staging_blocks(daemon: "Daemon", msg: Message) -> Response:
    # Blocking control-plane restore from StagingTier. Used
    # by connectors whose load hook is synchronous (for
    # example TRT-LLM): the caller supplies content hashes,
    # destination block ids, and staging generations. The
    # daemon copies bytes into HBM and returns only after
    # its stream is synchronized.
    engine_id = str(msg["engine_id"])
    with daemon._lock:
        pool = daemon._pools.get(engine_id)
    if pool is None or pool.restore_consumer is None:
        return {
            "ok": False,
            "error": "engine restore consumer not attached",
        }
    if daemon.staging_tier is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    raw_items = msg.get("items") or []
    block_pairs: list = []
    allocated: list[int] = []
    with daemon._staging_restore_lock:
        for it in raw_items:
            try:
                content_hash = bytes.fromhex(
                    str(it["content_hash"]),
                )
                dest_block = int(it["dest_block"])
                generation = int(it["generation"])
            except (KeyError, ValueError, TypeError):
                continue
            for _ in range(0xFFFF_FFFE):
                hid = daemon._next_staging_restore_handle
                daemon._next_staging_restore_handle += 1
                if daemon._next_staging_restore_handle > 0xFFFF_FFFF:
                    daemon._next_staging_restore_handle = 1
                if hid not in daemon._staging_restore_handles:
                    break
            else:
                continue
            daemon._staging_restore_handles[int(hid)] = (
                content_hash,
                generation,
            )
            allocated.append(int(hid))
            block_pairs.append((int(hid), dest_block))
    if not block_pairs:
        return {
            "ok": False,
            "error": "no valid staging restore items",
        }
    try:
        success = pool.restore_consumer._process_staging(
            {"block_pairs": block_pairs},
            pool,
        )
    finally:
        # _process_staging consumes handles as it reaches
        # them. Release any handles after a partial failure.
        with daemon._staging_restore_lock:
            for hid in allocated:
                daemon._staging_restore_handles.pop(hid, None)
    return {
        "ok": True,
        "success": bool(success),
        "requested": len(block_pairs),
        "restored": len(block_pairs) if success else 0,
    }


def handle_register_staging_restore_handles(daemon: "Daemon", msg: Message) -> Response:
    # Worker-side connector is about to push a
    # FLAG_SOURCE_STAGING restore ring record. The ring's
    # src field is only u32, so we create one-shot local
    # handles that resolve to (content_hash, generation).
    if daemon.staging_tier is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    items = msg.get("items") or []
    handles: list = []
    with daemon._staging_restore_lock:
        for it in items:
            try:
                content_hash = bytes.fromhex(
                    str(it["content_hash"]),
                )
                generation = int(it["generation"])
            except (KeyError, ValueError, TypeError):
                handles.append(None)
                continue
            # Monotonic u32 handle allocation. Skip 0 so
            # tests and logs can treat it as invalid.
            for _ in range(0xFFFF_FFFE):
                hid = daemon._next_staging_restore_handle
                daemon._next_staging_restore_handle += 1
                if daemon._next_staging_restore_handle > 0xFFFF_FFFF:
                    daemon._next_staging_restore_handle = 1
                if hid not in daemon._staging_restore_handles:
                    break
            else:
                handles.append(None)
                continue
            daemon._staging_restore_handles[int(hid)] = (
                content_hash,
                generation,
            )
            handles.append(int(hid))
    return {"ok": True, "handles": handles}


def handle_release_staging_restore_handles(daemon: "Daemon", msg: Message) -> Response:
    handles = msg.get("handles") or []
    released = 0
    with daemon._staging_restore_lock:
        for hid in handles:
            try:
                hid_i = int(hid)
            except (ValueError, TypeError):
                continue
            if daemon._staging_restore_handles.pop(hid_i, None) is not None:
                released += 1
    return {"ok": True, "released": released}


def handle_staging_scan(daemon: "Daemon", msg: Message) -> Response:
    # Phase 2 of cross-node design (see docs/CROSS_NODE_DESIGN.md
    # §4.3 — batch RPC chosen over SHM ring). Hashes are
    # hex-encoded for JSON transport. Disabled (returns empty)
    # if staging_tier wasn't enabled at daemon construction.
    if daemon.staging_tier is None:
        return {"ok": True, "hits": {}}
    req_hashes = msg.get("hashes", []) or []
    hashes_bytes = [bytes.fromhex(h) for h in req_hashes]
    raw_hits = daemon.staging_tier.scan(hashes_bytes)
    hits = {
        h.hex(): {
            "bytes_size": hit.bytes_size,
            "crc32": hit.crc32,
            "generation": hit.generation,
        }
        for h, hit in raw_hits.items()
    }
    return {"ok": True, "hits": hits}


HANDLERS: dict[str, Handler] = {
    "persistent_kv_attach_pool": handle_persistent_kv_attach_pool,
    "persistent_kv_detach_pool": handle_persistent_kv_detach_pool,
    "persistent_kv_lookup": handle_persistent_kv_lookup,
    "persistent_kv_store": handle_persistent_kv_store,
    "persistent_kv_load": handle_persistent_kv_load,
    "persistent_kv_store_pool": handle_persistent_kv_store_pool,
    "persistent_kv_load_pool": handle_persistent_kv_load_pool,
    "directory_snapshot": handle_directory_snapshot,
    "directory_changes": handle_directory_changes,
    "directory_lookup": handle_directory_lookup,
    "directory_lookup_claim": handle_directory_lookup_claim,
    "directory_promote": handle_directory_promote,
    "directory_publish_batch": handle_directory_publish_batch,
    "directory_release_claim": handle_directory_release_claim,
    "directory_adopt_claim": handle_directory_adopt_claim,
    "directory_mark_hbm_dormant": handle_directory_mark_hbm_dormant,
    "directory_ensure_hbm_capacity": handle_directory_ensure_hbm_capacity,
    "directory_hbm_inventory": handle_directory_hbm_inventory,
    "notify_kv_arrived": handle_notify_kv_arrived,
    "register_content_address": handle_register_content_address,
    "register_content_addresses_batch": handle_register_content_addresses_batch,
    "register_staging_restore_handles": handle_register_staging_restore_handles,
    "release_staging_restore_handles": handle_release_staging_restore_handles,
    "restore_host_blocks": handle_restore_host_blocks,
    "restore_staging_blocks": handle_restore_staging_blocks,
    "restore_staging_ranges": handle_restore_staging_ranges,
    "staging_fail": handle_staging_fail,
    "staging_reserve": handle_staging_reserve,
    "staging_scan": handle_staging_scan,
}
