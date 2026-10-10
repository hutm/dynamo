# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS as an SGLang memory-saver backend.

SGLang versions with ``register_memory_saver_backend`` select the backend with
``--memory-saver-backend``, so GMS no longer replaces torch_memory_saver's
initializer and ``configure_subprocess``. The adapter wraps one
``GMSMemorySaverImpl`` per process; SGLang's KV pools skip zero-filling when
the adapter reports that a region re-attached existing pages.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

logger = logging.getLogger(__name__)

GMS_MEMORY_SAVER_BACKEND = "gms"

_adapter = None


def sglang_supports_memory_saver_backends() -> bool:
    try:
        from sglang.srt.utils.torch_memory_saver_adapter import (  # noqa: F401
            register_memory_saver_backend,
        )
    except ImportError:
        return False
    return True


def register() -> bool:
    """Register the ``gms`` backend; returns whether this SGLang supports it."""
    if not sglang_supports_memory_saver_backends():
        return False
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    if GMS_MEMORY_SAVER_BACKEND not in tms._backends:
        tms.register_memory_saver_backend(GMS_MEMORY_SAVER_BACKEND, _get_adapter)
    return True


def current_impl():
    """The process's GMS implementation if the backend has created one."""
    return None if _adapter is None else _adapter.impl_if_created()


def _get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = _build_adapter_class()()
    return _adapter


def _build_adapter_class():
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

    class GMSMemorySaverAdapter(TorchMemorySaverAdapter):
        """Weights and the physical KV pool live in GMS and survive release."""

        def __init__(self) -> None:
            self._impl = None

        @property
        def impl(self):
            # Created on first use: the CUDA device and the GMS lock mode chosen
            # by setup_gms() are only known inside the scheduler process.
            if self._impl is None:
                import gpu_memory_service.integrations.sglang as gms_sglang
                from gpu_memory_service.integrations.common.utils import torch_device
                from gpu_memory_service.integrations.sglang.memory_saver import (
                    GMSMemorySaverImpl,
                )

                self._impl = GMSMemorySaverImpl(
                    device_index=torch_device().current_device(),
                    mode=gms_sglang._gms_lock_mode,
                    ro_connect_timeout_ms=gms_sglang._gms_ro_connect_timeout_ms,
                    native_kv_zero_skip=True,
                )
                logger.info(
                    "[GMS] memory saver backend %s (device=%s, weights=%s)",
                    GMS_MEMORY_SAVER_BACKEND,
                    self._impl._device,
                    self._impl.allocators["weights"].granted_lock_type.name,
                )
            return self._impl

        def impl_if_created(self):
            return self._impl

        def configure_subprocess(self):
            # GMS does not use torch_memory_saver's LD_PRELOAD allocator hooks.
            return nullcontext()

        def region(self, tag: str, enable_cpu_backup: bool = False):
            return self.impl.region(tag, enable_cpu_backup)

        def cuda_graph(self, **kwargs: Any):
            return self.impl.cuda_graph(**kwargs)

        def disable(self):
            return nullcontext()

        def pause(self, tag: str):
            return self.impl.pause(tag)

        def resume(self, tag: str):
            return self.impl.resume(tag)

        def preserves_contents(self, tag: str) -> bool:
            # Weights are published and survive release. KV pages survive too,
            # but while released another cohort engine may write them, so the
            # radix cache must still be flushed on KV release.
            return tag == "weights"

        def allocation_preserves_contents(self, tag: str) -> bool:
            from gpu_memory_service.integrations.sglang.memory_saver import (
                persistent_kv_reattaching,
            )

            return tag == "kv_cache" and persistent_kv_reattaching()

        @property
        def enabled(self):
            return True

    return GMSMemorySaverAdapter
