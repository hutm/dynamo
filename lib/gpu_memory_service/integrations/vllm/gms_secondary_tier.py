# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM native secondary-tier adapter for GMS persistent KV storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpu_memory_service.integrations.common.persistent_kv import (
    KVKey,
    KVLookup,
    KVTransfer,
    OperationId,
    PersistentKVTier,
)
from typing_extensions import override
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class GMSSecondaryTierManager(SecondaryTierManager):
    """Translate vLLM's native tier jobs to the engine-neutral GMS contract."""

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "gms",
        *,
        backend: PersistentKVTier | None = None,
        **config: Any,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if backend is None:
            from gpu_memory_service.integrations.common.gms_persistent_kv import (
                GMSPersistentKVTier,
            )

            if str(config.get("transport", "json")).lower() == "shm":
                config.setdefault(
                    "pool_path",
                    "/dev/shm/vllm_offload_"
                    f"{offloading_spec.vllm_config.instance_id}.mmap",
                )
            backend = GMSPersistentKVTier(
                primary_kv_view=primary_kv_view,
                **config,
            )
        self._backend = backend

    @staticmethod
    def _key(key: OffloadKey) -> KVKey:
        # OffloadKey already includes vLLM's native block hash and cache-group
        # identity. GMS stores it unchanged and does not rebuild a prefix tree.
        return KVKey(bytes(key))

    @classmethod
    def _transfer(cls, job: JobMetadata) -> KVTransfer:
        return KVTransfer(
            operation_id=OperationId(int(job.job_id)),
            keys=tuple(cls._key(key) for key in job.keys),
            slots=tuple(int(slot) for slot in job.block_ids),
        )

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        del req_context
        result = self._backend.lookup(self._key(key))
        if result is KVLookup.READY:
            return LookupResult.HIT
        if result is KVLookup.PENDING:
            return LookupResult.RETRY
        return LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        self._backend.submit_store(self._transfer(job_metadata))

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        self._backend.submit_load(self._transfer(job_metadata))

    @override
    def get_finished_jobs(self):
        return [
            JobResult(job_id=int(result.operation_id), success=result.success)
            for result in self._backend.poll_completed()
        ]

    @override
    def has_pending_work(self) -> bool:
        return self._backend.has_pending_work()

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        del req_context
        return RequestOffloadingContext()

    @override
    def drain_jobs(self) -> None:
        self._backend.drain()

    @override
    def shutdown(self) -> None:
        self._backend.close()


def register_gms_secondary_tier() -> None:
    """Register the adapter without changing vLLM source code."""
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

    try:
        SecondaryTierFactory.register_tier(
            "gms",
            __name__,
            GMSSecondaryTierManager.__name__,
        )
    except ValueError:
        registered = SecondaryTierFactory.get_tier_class({"type": "gms"})
        if registered is not GMSSecondaryTierManager:
            raise
