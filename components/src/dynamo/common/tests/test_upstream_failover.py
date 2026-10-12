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


def test_mark_sets_mode_and_a_per_boot_crash_hint_cohort(monkeypatch, tmp_path):
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(tmp_path / "locks" / "failover.lock"))
    monkeypatch.delenv(upstream_failover.UPSTREAM_MEMORY_ENV, raising=False)
    monkeypatch.delenv("GMS_VLLM_WRITER_COHORT_PATH", raising=False)
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
