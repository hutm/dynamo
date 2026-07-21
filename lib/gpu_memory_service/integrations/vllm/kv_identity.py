# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent KV identity helpers for the vLLM GMS integration."""

from __future__ import annotations

from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_persistent_kv_engine_id,
)


def truthy_env(name: str, *, default: bool = False) -> bool:
    return env_enabled_by_default(name, default=default)


def shared_kv_enabled() -> bool:
    return truthy_env(
        "GMS_VLLM_SHARED_KV",
        default=(
            truthy_env("DYN_VLLM_GMS_SHADOW_MODE", default=False)
            or truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE", default=False)
        ),
    )


def stable_engine_id(device: int) -> str:
    return get_gms_persistent_kv_engine_id("vllm", device, "GMS_VLLM_VMM_IPC_ENGINE_ID")


def allocation_engine_id(device: int) -> str:
    return stable_engine_id(device)


def allocation_shared() -> bool:
    return shared_kv_enabled()


def use_existing_shared_geometry() -> bool:
    return shared_kv_enabled()
