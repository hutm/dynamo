# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-native spec entry point that registers GMS inside EngineCore."""

from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from gpu_memory_service.integrations.vllm.gms_secondary_tier import (
    register_gms_secondary_tier,
)

register_gms_secondary_tier()


class GMSTieringOffloadingSpec(TieringOffloadingSpec):
    """TieringOffloadingSpec with the GMS secondary tier installed."""
