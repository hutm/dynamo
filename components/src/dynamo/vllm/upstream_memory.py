# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover on vLLM's own weight-cache daemon instead of GMS.

With ``--load-format ipc_cache`` and the loader's ``kv_restore`` option, the
per-node ``vllm preload`` daemon holds the weights and the active engine's KV
pool. Dynamo keeps the orchestration: the shadow engine maps both as a
``standby`` (no second copy), stays paused and undiscoverable, takes the
failover lock when the active engine dies, promotes its KV session (adopting
the committed prefix directory and quarantining the dead writer's in-flight
blocks), then resumes and registers with discovery.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

KV_RESTORE_ROLE_KEY = "kv_restore_role"
WRITER = "writer"
STANDBY = "standby"


def _extra_config(engine_args: Any) -> dict:
    extra = getattr(engine_args, "model_loader_extra_config", None) or {}
    if isinstance(extra, str):
        extra = json.loads(extra) if extra.strip() else {}
    return dict(extra)


def upstream_memory_enabled(engine_args: Any) -> bool:
    """Whether this engine uses the weight-cache daemon for weights and KV."""
    return str(getattr(engine_args, "load_format", "")) == "ipc_cache" and bool(
        _extra_config(engine_args).get("kv_restore")
    )


def is_primary_engine() -> bool:
    return os.environ.get("ENGINE_ID", "0") == os.environ.get(
        "DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0"
    )


def configure_kv_restore_role(engine_args: Any) -> str:
    """Attach the primary engine as the KV writer and every other engine
    (the shadow, or a restarted former primary) as a standby."""
    from dynamo.common.upstream_failover import mark_upstream_memory_active

    mark_upstream_memory_active("vllm")
    extra = _extra_config(engine_args)
    role = WRITER if is_primary_engine() else STANDBY
    extra[KV_RESTORE_ROLE_KEY] = role
    engine_args.model_loader_extra_config = extra
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


def promote_timeout_s() -> float:
    return _float_env("DYN_VLLM_KV_RESTORE_PROMOTE_TIMEOUT_SECS", 60.0)


def quarantine_grace_s() -> float:
    # Same knob as GMS's process-death reclaim: how long after the old
    # writer's processes are gone its in-flight blocks stay unallocatable.
    return _float_env("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", 2.0)


_DEFAULT_WARMUP_TOKEN_COUNTS = "1,2,3,5,6,9,10,17,18,65,66,704,705"


def standby_warmup_token_counts() -> list[int]:
    """Batch sizes the standby compiles before it waits for the lock; empty
    when DYN_VLLM_UPSTREAM_STANDBY_WARMUP is off. Defaults to the GMS
    promotion warmup's token counts."""
    if os.environ.get("DYN_VLLM_UPSTREAM_STANDBY_WARMUP", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return []
    raw = os.environ.get(
        "DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS",
        _DEFAULT_WARMUP_TOKEN_COUNTS,
    )
    try:
        return sorted({int(x) for x in raw.split(",") if x.strip()})
    except ValueError:
        logger.warning("Ignoring invalid warmup token counts %r", raw)
        return sorted({int(x) for x in _DEFAULT_WARMUP_TOKEN_COUNTS.split(",")})


async def warm_up_standby(engine_client: Any) -> int:
    """Compile the standby's forward path on every rank with dummy batches,
    which never write KV, so the writer's pool stays untouched while the
    standby gets warm. Returns the number of batch sizes run."""
    counts = standby_warmup_token_counts()
    if not counts:
        return 0
    results = await engine_client.collective_rpc("kv_restore_warmup", args=(counts,))
    return max((int(r or 0) for r in results or []), default=0)


async def promote_standby(engine_client: Any) -> int:
    """Promote this engine's KV sessions on every rank; returns the number of
    adopted prefix-cache blocks."""
    engine_core = getattr(engine_client, "engine_core", None)
    call_utility = getattr(engine_core, "call_utility_async", None)
    if not callable(call_utility):
        raise RuntimeError("upstream-memory promotion needs the engine core client")
    adopted = await call_utility(
        "kv_restore_promote", promote_timeout_s(), quarantine_grace_s()
    )
    return int(adopted or 0)
