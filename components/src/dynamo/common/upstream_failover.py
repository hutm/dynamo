# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover plumbing for engines whose memory lives in their own weight-cache
daemon (upstream memory) rather than GMS.

The engine's KV restore fences the old writer itself: promotion adopts only
committed blocks and quarantines the old writer's in-flight ones until its
processes are gone plus a grace period. So a broken cohort may fail-stop at
once (no GMS/MPS proof), and a rank whose engine worker dies reports it to the
TP leader directly instead of waiting for heartbeat silence.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

UPSTREAM_MEMORY_ENV = "DYN_FAILOVER_UPSTREAM_MEMORY"
# The crash-hint cohort path this process generated for its own boot. Only a
# path generated here is known to be unique to this engine boot, which is
# what makes a takeover hint safe to send (see advertise_crash_hint_target).
_BOOT_COHORT_ENV = "DYN_FAILOVER_UPSTREAM_BOOT_COHORT"
CRASH_HINT_TARGET_SUFFIX = ".crash-hint-target"
TAKEOVER_SOURCE = "failover-lock-takeover"
_COHORT_ENVS = {
    "vllm": "GMS_VLLM_WRITER_COHORT_PATH",
    "sglang": "GMS_SGLANG_WRITER_COHORT_PATH",
}


def mark_upstream_memory_active(backend_name: str) -> None:
    """Record upstream-memory mode and give this boot a crash-hint identity.

    The rank-liveness channel matches a crash notification against the
    marker path a rank advertises in its heartbeats; without GMS writer
    cohorts there is none, so use a unique per-boot path for that purpose.
    """
    os.environ[UPSTREAM_MEMORY_ENV] = "1"
    env = _COHORT_ENVS[backend_name]
    if os.environ.get(env):
        return
    lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/tmp/dynamo-failover.lock")
    directory = Path(lock_path).resolve().parent / ".upstream-crash-hints"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[env] = str(directory / uuid.uuid4().hex)
    os.environ[_BOOT_COHORT_ENV] = os.environ[env]


def upstream_memory_active() -> bool:
    return os.environ.get(UPSTREAM_MEMORY_ENV) == "1"


def fail_stop(reason: str) -> None:
    logger.warning("[upstream memory] %s; fail-stopping this engine process", reason)
    os.kill(os.getpid(), signal.SIGKILL)


def _report_crash(backend_name: str, rank: int, pid: int, source: str) -> None:
    cohort = os.environ.get(_COHORT_ENVS[backend_name])
    if not cohort:
        return
    try:
        from gpu_memory_service.common.gpu_failure_marker import (
            publish_gpu_failure_marker,
        )

        publish_gpu_failure_marker(cohort, rank=rank, pid=pid, source=source)
    except Exception:  # noqa: BLE001 - best effort latency hint
        logger.warning("[upstream memory] could not publish the crash marker")
    address = os.environ.get("DYN_GMS_RANK_LIVENESS_CONNECT_ADDR", "")
    if not address:
        return
    _send_crash_hint(address, cohort, rank, pid, source)


def _send_crash_hint(
    address: str, cohort: str, rank: int, pid: int, source: str
) -> None:
    try:
        import zmq

        # A private context: close() only hands the message to ZMQ's I/O
        # thread, and the caller SIGKILLs this process right after. term()
        # blocks until the hint is flushed (a fresh TCP connect to the leader
        # in another pod) or LINGER expires.
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 500)
        sock.connect(address)
        sock.send_multipart(
            [
                b"gpu-failed-v1",
                cohort.encode(),
                str(rank).encode(),
                str(pid).encode(),
                source.encode(),
            ]
        )
        sock.close()
        ctx.term()
    except Exception:  # noqa: BLE001 - best effort latency hint
        logger.warning("[upstream memory] could not send the crash hint")


def _is_engine_worker(proc) -> bool:
    """Engine workers are multiprocessing-spawned children (vLLM renames them
    VLLM::...). Short-lived helpers, such as vLLM's model-inspection
    subprocess, exit normally during startup and must not count as crashes."""
    try:
        if proc.name().startswith("VLLM::"):
            return True
        cmdline = " ".join(proc.cmdline())
    except Exception:  # noqa: BLE001 - gone or inaccessible: not watchable
        return False
    return "multiprocessing" in cmdline and "resource_tracker" not in cmdline


_PF_EXITING = 0x4


def _exiting(pid: int) -> bool:
    """Whether the process has started to exit (or is gone). The kernel sets
    PF_EXITING when do_exit begins; tearing down a worker's GPU mappings
    can take another ~0.5 s before the process is reaped, and the crash hint
    should not wait for that."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return True
    # Fields after the parenthesised comm: state ppid pgrp session tty_nr
    # tpgid flags ...
    fields = stat[stat.rindex(b")") + 2 :].split()
    try:
        return bool(int(fields[6]) & _PF_EXITING)
    except (IndexError, ValueError):
        return False


def start_child_crash_watch(
    backend_name: str,
    rank: int,
    poll_s: float = 0.02,
    stop: threading.Event | None = None,
) -> threading.Thread:
    """Watch the engine worker processes this process spawns from now on; when
    one exits, tell the TP leader and fail-stop (the cohort cannot make
    progress).

    Children that already exist (e.g. the liveness helper) are not watched.
    """
    import psutil

    me = psutil.Process()
    preexisting = {child.pid for child in me.children()}

    def run() -> None:
        watched: dict[int, psutil.Process] = {}
        while stop is None or not stop.is_set():
            for child in me.children():
                if (
                    child.pid not in preexisting
                    and child.pid not in watched
                    and _is_engine_worker(child)
                ):
                    watched[child.pid] = child
            for pid, child in list(watched.items()):
                try:
                    dead = (
                        _exiting(pid)
                        or not child.is_running()
                        or child.status() == psutil.STATUS_ZOMBIE
                    )
                except psutil.NoSuchProcess:
                    dead = True
                if dead:
                    del watched[pid]
                    _report_crash(backend_name, rank, pid, "engine-worker-exit")
                    fail_stop(f"engine worker {pid} exited")
            time.sleep(poll_s)

    thread = threading.Thread(target=run, name="upstream-crash-watch", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Takeover hint: a standby that wins a rank's failover lock reports that rank's
# previous active engine as lost to that engine's TP leader.
#
# When the fault kills the active engine's whole process tree on a rank, the
# crash watch dies with it and the leader would notice only at the liveness
# timeout. The rank's standby, however, wins the per-rank lock as soon as the
# kernel releases the dead holder's flock. The active engine therefore leaves
# its crash-hint identity (its per-boot cohort path and its leader's liveness
# address) next to that lock, and whoever acquires the lock next consumes it
# and sends the hint.
#
# Safety: the leader's monitor accepts a hint only when its cohort path equals
# the marker the reported rank advertised in its own heartbeats, and that path
# is a fresh UUID per engine boot. A stale record (from a holder whose cohort
# is long gone, or an address now bound by another cohort's leader) can only
# name a boot that no live monitor still tracks for that rank, so it is
# ignored. Losing the lock means the named engine no longer owns its rank, so
# the hint is true whenever it is accepted.
# ---------------------------------------------------------------------------


def crash_hint_target_path(lock_path: str | None = None) -> Path:
    lock_path = lock_path or os.environ.get(
        "FAILOVER_LOCK_PATH", "/tmp/dynamo-failover.lock"
    )
    return Path(lock_path + CRASH_HINT_TARGET_SUFFIX)


def advertise_crash_hint_target(
    backend_name: str, rank: int, address: str, lock_path: str | None = None
) -> bool:
    """Record this rank's crash-hint identity next to its failover lock.

    Call only while holding the rank's failover lock, after the liveness
    client (which advertises the same cohort path in its heartbeats) is
    configured for ``address``. Returns whether a record was written.
    """
    if not upstream_memory_active() or rank < 1 or not address:
        return False
    cohort = os.environ.get(_COHORT_ENVS[backend_name])
    if not cohort or cohort != os.environ.get(_BOOT_COHORT_ENV):
        # A configured (not per-boot) cohort path could be shared by a later
        # boot; a hint naming it might match a healthy cohort.
        return False
    target = crash_hint_target_path(lock_path)
    record = {
        "version": 1,
        "backend": backend_name,
        "cohort": cohort,
        "rank": int(rank),
        "pid": os.getpid(),
        "address": address,
    }
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(record))
        os.replace(tmp, target)
    except OSError:
        logger.warning(
            "[upstream memory] could not record the crash-hint target at %s",
            target,
            exc_info=True,
        )
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    logger.info(
        "[upstream memory] rank %d crash-hint target recorded at %s", rank, target
    )
    return True


def take_crash_hint_target(
    backend_name: str, lock_path: str | None = None
) -> dict | None:
    """Consume the previous lock holder's crash-hint record, if valid."""
    target = crash_hint_target_path(lock_path)
    try:
        raw = target.read_text()
    except OSError:
        return None
    try:
        target.unlink()
    except OSError:
        pass
    try:
        record = json.loads(raw)
        if (
            record.get("version") != 1
            or record.get("backend") != backend_name
            or not isinstance(record.get("cohort"), str)
            or not record["cohort"]
            or not isinstance(record.get("address"), str)
            or not record["address"]
            or int(record.get("rank", -1)) < 1
            or int(record.get("pid", 0)) <= 0
        ):
            raise ValueError("invalid record")
    except (ValueError, TypeError, AttributeError):
        logger.warning("[upstream memory] ignoring a malformed crash-hint target")
        return None
    return record


def hint_predecessor_lost(
    backend_name: str, lock_path: str | None = None
) -> threading.Thread | None:
    """Right after this engine won its rank's failover lock, tell the previous
    holder's TP leader that the holder's rank is lost.

    Sends in the background: the hint is a latency optimisation and must not
    delay this engine's own takeover.
    """
    if not upstream_memory_active():
        return None
    record = take_crash_hint_target(backend_name, lock_path)
    if record is None:
        return None
    if record["cohort"] == os.environ.get(_COHORT_ENVS[backend_name]):
        return None  # our own record: nothing to report
    rank, pid = int(record["rank"]), int(record["pid"])
    logger.warning(
        "[upstream memory] took over rank %d from engine pid %d; sending its "
        "leader a crash hint at %s",
        rank,
        pid,
        record["address"],
    )
    thread = threading.Thread(
        target=_send_crash_hint,
        args=(record["address"], record["cohort"], rank, pid, TAKEOVER_SOURCE),
        name="upstream-takeover-hint",
        daemon=True,
    )
    thread.start()
    return thread
