# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS as a vLLM sleep-mode backend.

vLLM versions with ``SleepModeBackend.allocation_context`` and
``preserves_kv_cache`` let a backend place weight and KV allocations and keep
the prefix cache across sleep. GMS uses them instead of overriding worker
internals: the backend routes the ``kv_cache`` tag to the GMS persistent pool
and declares that KV survives sleep, so ``EngineCore.sleep`` keeps the prefix
cache. Suspend and resume stay in ``GMSWorker``, which owns the GMS sessions.

Registered from the ``vllm.general_plugins`` entry point (every vLLM process)
and explicitly by Dynamo before it builds the engine config.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

GMS_SLEEP_MODE_BACKEND = "gms"


def vllm_supports_sleep_backend_hooks() -> bool:
    """Whether vLLM lets a sleep-mode backend place allocations and keep KV."""
    try:
        from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
    except ImportError:
        return False
    return hasattr(SleepModeBackend, "allocation_context") and hasattr(
        SleepModeBackend, "preserves_kv_cache"
    )


def register() -> None:
    """``vllm.general_plugins`` entry point: register the ``gms`` backend."""
    if not vllm_supports_sleep_backend_hooks():
        return
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    if GMS_SLEEP_MODE_BACKEND in SleepModeBackendFactory._registry:
        return
    SleepModeBackendFactory.register_backend(
        GMS_SLEEP_MODE_BACKEND, __name__, "GMSSleepModeBackend"
    )


def configure_engine_args(engine_args: Any) -> bool:
    """Select the ``gms`` sleep backend on ``engine_args`` when vLLM has the
    hooks; returns whether it did. Older vLLM keeps the worker overrides."""
    if not (
        vllm_supports_sleep_backend_hooks()
        and hasattr(engine_args, "sleep_mode_backend")
    ):
        return False
    register()
    engine_args.enable_sleep_mode = True
    engine_args.sleep_mode_backend = GMS_SLEEP_MODE_BACKEND
    logger.info(
        "[GMS] vLLM sleep_mode_backend=%s: GMS places weight and KV "
        "allocations through vLLM's sleep-backend hooks",
        GMS_SLEEP_MODE_BACKEND,
    )
    return True


def __getattr__(name: str) -> Any:
    # Defined lazily so importing this module never requires vLLM.
    if name != "GMSSleepModeBackend":
        raise AttributeError(name)
    backend_cls = _build_backend_class()
    globals()[name] = backend_cls
    return backend_cls


def _build_backend_class():
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackend

    class GMSSleepModeBackend(SleepModeBackend):
        """Place weights and KV through GMS; KV survives every sleep level."""

        def __init__(self) -> None:
            super().__init__()
            self._worker = None

        def bind_worker(self, worker) -> None:
            self._worker = worker

        def allocation_context(self, tag: str, *, kv_cache_config=None):
            if self._worker is None:
                return None
            return self._worker.gms_allocation_context(tag, kv_cache_config)

        def suspend(self, level: int = 1) -> None:
            raise RuntimeError("GMSWorker.sleep owns GMS suspend")

        def resume(self, tags=None) -> None:
            raise RuntimeError("GMSWorker.wake_up owns GMS resume")

        @classmethod
        def preserves_kv_cache(cls, level: int = 1) -> bool:
            # The GMS daemon owns KV pages; sleep only unmaps them.
            return True

        @classmethod
        def requires_cumem_allocator(cls) -> bool:
            return False

        @classmethod
        def preserves_communicators(cls) -> bool:
            return True

    GMSSleepModeBackend.__module__ = __name__
    return GMSSleepModeBackend
