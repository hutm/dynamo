# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-neutral contract for a persistent KV secondary tier.

Inference engines retain ownership of their scheduling and GPU KV cache.  A
secondary tier owns durable copies addressed by opaque content keys and moves
bytes to or from slots in an attached engine-visible memory pool.

Implementations must keep ``lookup`` and ``submit_*`` non-blocking.  Transfer
completion is reported exactly once through ``poll_completed``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType

KVKey = NewType("KVKey", bytes)
OperationId = NewType("OperationId", int)


class KVLookup(Enum):
    """Availability of one content-addressed KV block."""

    MISS = auto()
    READY = auto()
    PENDING = auto()


@dataclass(frozen=True, slots=True)
class KVTransfer:
    """One batch transfer between an attached pool and the persistent tier.

    ``slots`` index rows in the memory pool attached by the implementation.
    A store reads those rows; a load writes them.  Keys and slots are paired by
    position and a batch has one exactly-once completion.
    """

    operation_id: OperationId
    keys: tuple[KVKey, ...]
    slots: tuple[int, ...]

    def __post_init__(self) -> None:
        if int(self.operation_id) < 0:
            raise ValueError("operation_id must be non-negative")
        if not self.keys:
            raise ValueError("a transfer must contain at least one key")
        if len(self.keys) != len(self.slots):
            raise ValueError("keys and slots must have equal lengths")
        if any(not bytes(key) for key in self.keys):
            raise ValueError("KV keys must not be empty")
        if any(slot < 0 for slot in self.slots):
            raise ValueError("slot indices must be non-negative")


@dataclass(frozen=True, slots=True)
class KVTransferResult:
    """Terminal result for one submitted transfer batch."""

    operation_id: OperationId
    success: bool


class PersistentKVTier(ABC):
    """Asynchronous content-addressed storage below an engine memory pool."""

    @abstractmethod
    def lookup(self, key: KVKey) -> KVLookup:
        """Return current local availability without blocking on I/O."""

    @abstractmethod
    def submit_store(self, transfer: KVTransfer) -> None:
        """Enqueue attached-pool to persistent-tier movement."""

    @abstractmethod
    def submit_load(self, transfer: KVTransfer) -> None:
        """Enqueue persistent-tier to attached-pool movement."""

    @abstractmethod
    def poll_completed(self) -> Iterable[KVTransferResult]:
        """Remove and return terminal results not reported previously."""

    def has_pending_work(self) -> bool:
        """Whether completion polling must continue with no scheduled work."""
        return False

    @abstractmethod
    def drain(self) -> None:
        """Wait until all submitted operations have reached a terminal state."""

    def close(self) -> None:
        """Release resources after ``drain``; idempotence is recommended."""
