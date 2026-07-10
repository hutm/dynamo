# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

from gpu_memory_service.integrations.vllm.gms_offloading_spec import (
    GMSTieringOffloadingSpec,
)
from gpu_memory_service.integrations.vllm.gms_secondary_tier import (
    GMSSecondaryTierManager,
)


def test_custom_spec_import_registers_gms_in_current_process():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "spec_name": "GMSTieringOffloadingSpec",
                "spec_module_path": (
                    "gpu_memory_service.integrations.vllm.gms_offloading_spec"
                ),
            }
        )
    )

    assert OffloadingSpecFactory.get_spec_cls(config) is GMSTieringOffloadingSpec
    assert (
        SecondaryTierFactory.get_tier_class({"type": "gms"}) is GMSSecondaryTierManager
    )
