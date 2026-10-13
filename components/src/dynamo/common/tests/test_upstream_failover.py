# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-stop and crash hints for failover on engine weight-cache daemons."""

import multiprocessing
import subprocess
import sys
import threading

import pytest

from dynamo.common import upstream_failover

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch):
    # Each test models a fresh engine process for mark_upstream_memory_active.
    monkeypatch.setattr(upstream_failover, "_marked_pid", None)


def test_mark_sets_mode_and_a_per_boot_crash_hint_cohort(monkeypatch, tmp_path):
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(tmp_path / "locks" / "failover.lock"))
    for key in (
        upstream_failover.UPSTREAM_MEMORY_ENV,
        "GMS_VLLM_WRITER_COHORT_PATH",
        upstream_failover._CONFIGURED_COHORT_ENV,
    ):
        # setenv first: delenv records nothing for an absent key, and mark
        # sets these in os.environ.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    upstream_failover.mark_upstream_memory_active("vllm")
    assert upstream_failover.upstream_memory_active()
    cohort = upstream_failover.os.environ["GMS_VLLM_WRITER_COHORT_PATH"]
    assert cohort.startswith(str(tmp_path / "locks" / ".upstream-crash-hints"))
    upstream_failover.mark_upstream_memory_active("vllm")  # idempotent
    assert upstream_failover.os.environ["GMS_VLLM_WRITER_COHORT_PATH"] == cohort


def _sleep(seconds):
    import time

    time.sleep(seconds)


def _watch(monkeypatch, request):
    stopped = threading.Event()
    reports = []
    monkeypatch.setattr(
        upstream_failover,
        "_report_crash",
        lambda backend, rank, pid, source: reports.append((backend, rank, source)),
    )
    monkeypatch.setattr(upstream_failover, "fail_stop", lambda reason: stopped.set())
    # Stop the watch with the test: once monkeypatch restores the real
    # fail_stop, a leftover watch would SIGKILL the test runner on the next
    # multiprocessing child exit.
    halt = threading.Event()
    request.addfinalizer(halt.set)
    thread = upstream_failover.start_child_crash_watch(
        "vllm", 3, poll_s=0.01, stop=halt
    )
    request.addfinalizer(lambda: thread.join(timeout=1.0))
    return stopped, reports


def test_worker_exit_reports_and_fail_stops(monkeypatch, request):
    stopped, reports = _watch(monkeypatch, request)
    worker = multiprocessing.get_context("spawn").Process(target=_sleep, args=(1.0,))
    worker.start()
    try:
        assert stopped.wait(20)
    finally:
        worker.join()
    assert reports and reports[0] == ("vllm", 3, "engine-worker-exit")


def test_short_lived_helper_subprocess_is_not_a_crash(monkeypatch, request):
    # e.g. vLLM's model-inspection subprocess during config resolution
    stopped, reports = _watch(monkeypatch, request)
    subprocess.run([sys.executable, "-c", "import time; time.sleep(0.3)"], check=True)
    assert not stopped.wait(0.5)
    assert reports == []


def test_crash_hint_is_flushed_before_report_returns(monkeypatch, tmp_path):
    # The caller SIGKILLs itself right after _report_crash, so the hint must
    # already be on the wire when it returns.
    zmq = pytest.importorskip("zmq")
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    monkeypatch.setenv("DYN_GMS_RANK_LIVENESS_CONNECT_ADDR", f"tcp://127.0.0.1:{port}")
    cohort = str(tmp_path / "cohort")
    monkeypatch.setenv("GMS_VLLM_WRITER_COHORT_PATH", cohort)
    try:
        upstream_failover._report_crash("vllm", 5, 4242, "engine-worker-exit")
        assert router.poll(2000)
        frames = router.recv_multipart()
        assert frames[1:] == [
            b"gpu-failed-v1",
            cohort.encode(),
            b"5",
            b"4242",
            b"engine-worker-exit",
        ]
    finally:
        router.close(0)
        ctx.term()


def test_exiting_is_false_for_a_live_process_and_true_once_gone():
    assert not upstream_failover._exiting(upstream_failover.os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert upstream_failover._exiting(child.pid)


def _boot(monkeypatch, tmp_path, backend="sglang"):
    """An upstream-memory engine boot with a per-boot crash-hint cohort."""
    lock = tmp_path / "locks" / "failover.lock.rank-3"
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(lock))
    env = upstream_failover._COHORT_ENVS[backend]
    # Clear every backend's cohort: the liveness client reports the first one
    # set, and other test modules may leave one behind in os.environ.
    for key in (upstream_failover.UPSTREAM_MEMORY_ENV,
                *upstream_failover._COHORT_ENVS.values(),
                upstream_failover._CONFIGURED_COHORT_ENV):
        # setenv first: delenv records nothing for an absent key.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    upstream_failover.mark_upstream_memory_active(backend)
    return str(lock), upstream_failover.os.environ[env]


def _new_boot(monkeypatch, backend="sglang"):
    """Start the next engine boot on the same lock (a new process)."""
    env = upstream_failover._COHORT_ENVS[backend]
    monkeypatch.delenv(env)
    monkeypatch.setattr(upstream_failover, "_marked_pid", None)
    upstream_failover.mark_upstream_memory_active(backend)
    return upstream_failover.os.environ[env]


def test_active_advertises_and_successor_consumes_crash_hint_target(
    monkeypatch, tmp_path
):
    lock, cohort = _boot(monkeypatch, tmp_path)
    assert upstream_failover.advertise_crash_hint_target(
        "sglang", 3, "tcp://leader:29555"
    )
    target = upstream_failover.crash_hint_target_path(lock)
    assert target.exists()
    record = upstream_failover.take_crash_hint_target("sglang")
    assert record["cohort"] == cohort
    assert record["rank"] == 3
    assert record["address"] == "tcp://leader:29555"
    assert record["pid"] == upstream_failover.os.getpid()
    # Consumed: a later holder cannot send the same hint again.
    assert not target.exists()
    assert upstream_failover.take_crash_hint_target("sglang") is None


def test_no_crash_hint_target_without_a_per_boot_cohort(monkeypatch, tmp_path):
    lock, _ = _boot(monkeypatch, tmp_path)
    # The leader rank has no leader to report to.
    assert not upstream_failover.advertise_crash_hint_target("sglang", 0, "tcp://x:1")
    # A configured (shared across boots) cohort path could name a healthy
    # later boot: never advertise it.
    monkeypatch.setenv("GMS_SGLANG_WRITER_COHORT_PATH", str(tmp_path / "configured"))
    monkeypatch.setattr(upstream_failover, "_marked_pid", None)
    upstream_failover.mark_upstream_memory_active("sglang")
    assert upstream_failover.os.environ["GMS_SGLANG_WRITER_COHORT_PATH"] == str(
        tmp_path / "configured"
    )
    assert not upstream_failover.advertise_crash_hint_target("sglang", 3, "tcp://x:1")
    monkeypatch.delenv(upstream_failover.UPSTREAM_MEMORY_ENV)
    assert not upstream_failover.advertise_crash_hint_target("sglang", 3, "tcp://x:1")
    assert not upstream_failover.crash_hint_target_path(lock).exists()


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"version": 1, "backend": "vllm", "cohort": "/c", "rank": 3, "pid": 9, '
        '"address": "tcp://x:1"}',
        '{"version": 1, "backend": "sglang", "cohort": "/c", "rank": 0, "pid": 9, '
        '"address": "tcp://x:1"}',
        '{"version": 1, "backend": "sglang", "cohort": "", "rank": 3, "pid": 9, '
        '"address": "tcp://x:1"}',
    ],
)
def test_malformed_crash_hint_target_is_consumed_and_ignored(
    monkeypatch, tmp_path, content
):
    lock, _ = _boot(monkeypatch, tmp_path)
    target = upstream_failover.crash_hint_target_path(lock)
    target.write_text(content)
    assert upstream_failover.hint_predecessor_lost("sglang") is None
    assert not target.exists()


def _monitor_tracking(endpoint, rank, cohort, calls, fired):
    """The active cohort's leader monitor, with `rank` heartbeating `cohort`."""
    import time

    import zmq
    from gpu_memory_service.common.gpu_failure_marker import gpu_failure_marker_path

    from dynamo.common import rank_liveness as rl

    monitor = rl.RankLivenessMonitor(
        lambda r, reason: (calls.append((r, reason)), fired.set()),
        bind_addr=endpoint,
        timeout_ms_override=5_000,
        expected_ranks={rank},
        failure_marker_path=str(gpu_failure_marker_path(cohort + "-leader")),
        broadcast_fence=True,
    )
    heartbeat = zmq.Context.instance().socket(zmq.DEALER)
    heartbeat.setsockopt(zmq.IDENTITY, f"rank-{rank}".encode())
    heartbeat.setsockopt(zmq.LINGER, 0)
    heartbeat.connect(endpoint)
    monitor.start()
    marker = str(gpu_failure_marker_path(cohort))
    heartbeat.send_multipart([marker.encode(), b"hb"])
    deadline = time.monotonic() + 2.0
    while (
        monitor._rank_failure_marker_paths.get(rank) != marker
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert monitor._rank_failure_marker_paths.get(rank) == marker
    return monitor, heartbeat


def _endpoint():
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{probe.getsockname()[1]}"


def test_standby_takeover_hint_fail_stops_the_active_cohort_at_once(
    monkeypatch, tmp_path
):
    pytest.importorskip("zmq")
    pytest.importorskip("gpu_memory_service")
    import threading as _threading

    endpoint = _endpoint()
    _, active_cohort = _boot(monkeypatch, tmp_path)
    calls, fired = [], _threading.Event()
    monitor, heartbeat = _monitor_tracking(endpoint, 3, active_cohort, calls, fired)
    try:
        # The active engine of rank 3 holds the lock and advertises.
        assert upstream_failover.advertise_crash_hint_target("sglang", 3, endpoint)
        # Its whole tree dies; the standby (a new boot) wins the lock.
        _new_boot(monkeypatch)
        sender = upstream_failover.hint_predecessor_lost("sglang")
        assert sender is not None
        sender.join(2.0)
        # Far below the 5 s heartbeat deadline, and without waiting for the
        # dead rank's fence ack.
        assert fired.wait(1.0)
        assert calls == [(3, "gpu-crash-interlock-zmq")]
    finally:
        heartbeat.close(0)
        monitor.stop()


def test_stale_takeover_hint_never_fail_stops_a_healthy_cohort(
    monkeypatch, tmp_path
):
    pytest.importorskip("zmq")
    pytest.importorskip("gpu_memory_service")
    import threading as _threading

    endpoint = _endpoint()
    _boot(monkeypatch, tmp_path)
    # An old holder of rank 3's lock left a record, then its cohort died.
    assert upstream_failover.advertise_crash_hint_target("sglang", 3, endpoint)
    # A healthy cohort now listens on the same address, and its rank 3 is a
    # different boot.
    healthy_cohort = _new_boot(monkeypatch)
    calls, fired = [], _threading.Event()
    monitor, heartbeat = _monitor_tracking(endpoint, 3, healthy_cohort, calls, fired)
    try:
        _new_boot(monkeypatch)
        sender = upstream_failover.hint_predecessor_lost("sglang")
        assert sender is not None
        sender.join(2.0)
        assert not fired.wait(0.5)
        assert calls == []
    finally:
        heartbeat.close(0)
        monitor.stop()


def test_takeover_hint_needs_upstream_mode(monkeypatch, tmp_path):
    lock, _ = _boot(monkeypatch, tmp_path)
    assert upstream_failover.advertise_crash_hint_target("sglang", 3, "tcp://x:1")
    monkeypatch.delenv(upstream_failover.UPSTREAM_MEMORY_ENV)
    assert upstream_failover.hint_predecessor_lost("sglang") is None
    assert upstream_failover.crash_hint_target_path(lock).exists()


@pytest.mark.asyncio
async def test_deployed_sglang_path_hints_through_the_isolated_liveness_client(
    monkeypatch, tmp_path
):
    """The deployed SGLang order: mark upstream mode, then the guarded engine
    replaces the cohort with a GMS writer cohort (prepare_writer_cohort), then
    the non-leader starts its process-isolated liveness client and records
    the target. The standby that wins the lock must fail-stop the leader.

    Regression: the target used to be skipped silently because the cohort
    no longer equalled the path generated at mark time.
    """
    pytest.importorskip("zmq")
    pytest.importorskip("gpu_memory_service")
    import asyncio
    import threading as _threading
    import time
    from types import SimpleNamespace

    from gpu_memory_service.integrations.sglang import writer_lifecycle

    from dynamo.common import rank_liveness as rl
    from dynamo.sglang import failover_watchdog

    endpoint = _endpoint()
    lock, _ = _boot(monkeypatch, tmp_path)
    monkeypatch.setattr(writer_lifecycle, "_boot", None)
    monkeypatch.setattr(writer_lifecycle, "_writer_fds", [])
    writers = writer_lifecycle.prepare_writer_cohort()  # create_guarded_engine
    cohort = upstream_failover.os.environ["GMS_SGLANG_WRITER_COHORT_PATH"]
    assert cohort == str(writers)

    calls, fired = [], _threading.Event()
    monitor = rl.RankLivenessMonitor(
        lambda r, reason: (calls.append((r, reason)), fired.set()),
        bind_addr=endpoint,
        timeout_ms_override=10_000,
        expected_ranks={3},
        broadcast_fence=True,
    )
    monitor.start()
    monkeypatch.setenv("DYN_GMS_RANK_LIVENESS", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_GMS_RANK_LIVENESS_ISOLATED", "1")
    monkeypatch.setenv("DYN_GMS_RANK_LIVENESS_CONNECT_ADDR", endpoint)
    client = failover_watchdog.maybe_start_rank_liveness(
        SimpleNamespace(), object(), node_rank=3, leader_host="127.0.0.1"
    )
    try:
        assert isinstance(client, rl.IsolatedRankLivenessClient)
        assert upstream_failover.crash_hint_target_path(lock).exists()
        from gpu_memory_service.common.gpu_failure_marker import (
            gpu_failure_marker_path,
        )

        marker = str(gpu_failure_marker_path(cohort))
        deadline = time.monotonic() + 20.0
        while (
            monitor._rank_failure_marker_paths.get(3) != marker
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)
        assert monitor._rank_failure_marker_paths.get(3) == marker
        # The engine-0 tree dies; the standby (a new process) wins the lock.
        _new_boot(monkeypatch)
        sender = upstream_failover.hint_predecessor_lost("sglang")
        assert sender is not None
        sender.join(2.0)
        assert fired.wait(1.0)
        assert calls == [(3, "gpu-crash-interlock-zmq")]
    finally:
        client.stop()
        monitor.stop()
        for fd in writer_lifecycle._writer_fds:
            upstream_failover.os.close(fd)
