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


def test_custom_spec_warms_native_promotion_before_serving(monkeypatch):
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    calls = []

    class FakeWorker:
        def submit_load(self, job_id, source, destination):
            calls.append((job_id, source, destination))
            return True

        def wait(self, job_ids):
            calls.append(("wait", job_ids))

        def get_finished(self):
            return [SimpleNamespace(success=True)]

    worker = FakeWorker()
    monkeypatch.setattr(
        TieringOffloadingSpec, "create_worker", lambda self, caches: worker
    )
    spec = object.__new__(GMSTieringOffloadingSpec)
    spec.block_size_factor = 1
    spec.num_blocks = 32
    caches = SimpleNamespace(
        group_data_refs=[(), ()],
        tensors=[SimpleNamespace(tensor=SimpleNamespace(shape=(32,)))],
    )

    assert spec.create_worker(caches) is worker
    job_id, source, destination = calls[0]
    assert job_id == -1
    assert source.block_ids.tolist() == list(range(16)) * 2
    assert destination.block_ids.tolist() == list(range(16)) * 2
    assert list(destination.group_sizes) == [16, 16]
    assert list(destination.block_indices) == [0, 0]
    assert calls[1] == ("wait", {-1})
