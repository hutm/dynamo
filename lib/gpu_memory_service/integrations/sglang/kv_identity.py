# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent KV identity helpers for the SGLang GMS integration."""

from __future__ import annotations

from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_engine_id,
)


def truthy_env(name: str, *, default: bool = False) -> bool:
    return env_enabled_by_default(name, default=default)


def shared_kv_enabled() -> bool:
    return truthy_env(
        "GMS_SGLANG_SHARED_KV",
        default=truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE", default=False),
    )


def private_bootstrap_kv_enabled() -> bool:
    if truthy_env(
        "DYN_SGLANG_GMS_PRIVATE_BOOTSTRAP_KV",
        default=truthy_env("GMS_SGLANG_PRIVATE_BOOTSTRAP_KV", default=False),
    ):
        raise RuntimeError(
            "SGLang GMS private-bootstrap KV is no longer supported: its "
            "client-local scratch isolation was removed. Use the regular "
            "sleeping-shadow failover path instead."
        )
    return False


def stable_engine_id(device: int) -> str:
    return get_gms_persistent_kv_engine_id(
        "sglang", device, "GMS_SGLANG_VMM_IPC_ENGINE_ID"
    )


def allocator_tag(device: int) -> str:
    """Process-local Torch allocator tag for this device's KV pool."""
    return f"kv_pool:cuda{int(device)}"


def allocation_engine_id(device: int) -> str:
    private_bootstrap_kv_enabled()
    return stable_engine_id(device)


def allocation_shared() -> bool:
    private_bootstrap_kv_enabled()
    return shared_kv_enabled()
