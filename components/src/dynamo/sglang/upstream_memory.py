# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover on SGLang's own weight-cache daemon instead of GMS.

With ``--weight-cache-mode client`` and ``SGLANG_WEIGHT_CACHE_KV_RESTORE=1``,
the per-node SGLang weight-cache daemon holds the weights and the active
engine's KV pool. Dynamo keeps the orchestration (per-rank failover locks,
liveness, discovery, request migration): the shadow maps both as a
``standby`` and stays paused; after every rank holds its lock, the leader
promotes the engine's KV sessions (adopting the committed radix directory and
quarantining the dead writer's in-flight pages) and resumes it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROLE_ENV = "SGLANG_WEIGHT_CACHE_KV_RESTORE_ROLE"
WRITER = "writer"
STANDBY = "standby"


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def upstream_memory_enabled(server_args: Any) -> bool:
    return getattr(server_args, "weight_cache_mode", "off") == "client" and _truthy(
        os.environ.get("SGLANG_WEIGHT_CACHE_KV_RESTORE")
    )


def is_primary_engine() -> bool:
    return os.environ.get("ENGINE_ID", "0") == os.environ.get(
        "DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0"
    )


def configure_kv_restore_role() -> str:
    """Primary attaches as the KV writer, every other engine as a standby.
    The primary takes its failover lock before engine init (as in GMS mode);
    the standby pre-initializes with the pool mapped and paused."""
    from dynamo.common.upstream_failover import mark_upstream_memory_active

    mark_upstream_memory_active("sglang")
    role = WRITER if is_primary_engine() else STANDBY
    os.environ[ROLE_ENV] = role
    os.environ["DYN_SGLANG_GMS_LOCK_BEFORE_INIT"] = "1" if role == WRITER else "0"
    logger.info(
        "[upstream memory] engine %s attaches to the weight-cache daemon as %s",
        os.environ.get("ENGINE_ID", "0"),
        role,
    )
    return role


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, os.environ.get(name))
        return default


async def no_post_lock_fence(role: str) -> None:
    """The daemon's promotion fences KV ownership; nothing GMS-side to do."""


class PromotingController:
    """Wraps the leader's pause controller: resuming a standby first promotes
    the engine's KV sessions on every TP rank."""

    def __init__(self, delegate: Any, engine: Any) -> None:
        self._delegate = delegate
        self._engine = engine
        self._promoted = False

    async def quiesce(self, tags: Optional[list[str]] = None) -> bool:
        return await self._delegate.quiesce(tags)

    async def resume(self, tags: Optional[list[str]] = None) -> bool:
        if not self._promoted:
            import time

            started = time.monotonic()
            adopted = await self._engine.tokenizer_manager.kv_restore_promote(
                _float_env("DYN_SGLANG_KV_RESTORE_PROMOTE_TIMEOUT_SECS", 60.0),
                _float_env("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", 2.0),
            )
            self._promoted = True
            logger.info(
                "[upstream memory] KV sessions promoted: adopted %s pages in %.1f ms",
                adopted,
                (time.monotonic() - started) * 1000.0,
            )
        return await self._delegate.resume(tags)

    def mark_resumed(self) -> None:
        mark = getattr(self._delegate, "mark_resumed", None)
        if mark is not None:
            mark()
