# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-stop and crash hints for failover on engine weight-cache daemons."""

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


def test_child_exit_reports_and_fail_stops(monkeypatch):
    stopped = threading.Event()
    reports = []
    monkeypatch.setattr(
        upstream_failover,
        "_report_crash",
        lambda backend, rank, pid, source: reports.append((backend, rank, source)),
    )
    monkeypatch.setattr(upstream_failover, "fail_stop", lambda reason: stopped.set())
    upstream_failover.start_child_crash_watch("vllm", 3, poll_s=0.01)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    try:
        assert stopped.wait(10)
    finally:
        child.wait()
    assert reports and reports[0] == ("vllm", 3, "engine-worker-exit")
