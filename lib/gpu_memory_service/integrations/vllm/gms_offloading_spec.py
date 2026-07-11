# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-native spec entry point that registers GMS inside EngineCore."""

from gpu_memory_service.integrations.vllm.gms_secondary_tier import (
    register_gms_secondary_tier,
)
from typing_extensions import override
from vllm.v1.kv_offload.base import CanonicalKVCaches, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

register_gms_secondary_tier()


class GMSTieringOffloadingSpec(TieringOffloadingSpec):
    """TieringOffloadingSpec with the GMS secondary tier installed."""

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        worker = super().create_worker(kv_caches)
        groups = len(kv_caches.group_data_refs)
        warm_blocks = min(
            16,
            self.num_blocks,
            *(int(cache.tensor.shape[0]) for cache in kv_caches.tensors),
        )
        cpu_blocks = (
            warm_blocks + self.block_size_factor - 1
        ) // self.block_size_factor
        job_id = -1
        worker.submit_load(
            job_id,
            CPULoadStoreSpec(list(range(cpu_blocks)) * groups),
            GPULoadStoreSpec(
                list(range(warm_blocks)) * groups,
                group_sizes=[warm_blocks] * groups,
                block_indices=[0] * groups,
            ),
        )
        worker.wait({job_id})
        results = worker.get_finished()
        if len(results) != 1 or not results[0].success:
            raise RuntimeError("failed to warm vLLM KV promotion")
        return worker
