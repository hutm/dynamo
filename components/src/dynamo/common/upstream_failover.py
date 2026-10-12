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

import logging
import os
import signal
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

UPSTREAM_MEMORY_ENV = "DYN_FAILOVER_UPSTREAM_MEMORY"
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
    backend_name: str, rank: int, poll_s: float = 0.02
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
        while True:
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
