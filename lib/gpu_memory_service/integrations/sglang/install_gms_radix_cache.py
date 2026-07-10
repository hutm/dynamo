# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch installer for `GMSRadixCache`.

Production deployments enable the GMS daemon-storage tier on
SGLang by setting:

    GMS_SGLANG_ENABLE_KV_RING=1
    GMS_SGLANG_DAEMON_SOCKET=/path/to/daemon.sock  (optional)
    GMS_SGLANG_ENGINE_ID=0                          (optional)

then importing this module BEFORE SGLang's `Scheduler` is
constructed (typically from the entry-point shim). On import,
`install()` wraps `Scheduler.__init__` to replace
`self.tree_cache` with `GMSRadixCache` after vanilla
construction.

Safety: if any step of the swap fails, the wrapper logs the
error and leaves `self.tree_cache` untouched — SGLang continues
with its vanilla RadixCache. Disabling the env var is also
sufficient to bypass the patch entirely.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False


def install() -> bool:
    """Apply the monkey-patch. Returns True iff a patch was
    actually installed (False if the env var is unset, the
    patch is already installed, or SGLang isn't importable)."""
    global _INSTALLED
    if _INSTALLED:
        return False
    directory_mode = os.environ.get("GMS_KV_DIRECTORY_MODE", "off").lower()
    if (
        os.environ.get("GMS_SGLANG_ENABLE_KV_RING") != "1"
        and directory_mode not in ("shadow", "authoritative")
    ):
        logger.debug(
            "SGLang GMS radix cache disabled; skipping install",
        )
        return False
    try:
        from sglang.srt.managers.scheduler import Scheduler
    except ImportError:
        logger.warning(
            "SGLang not importable; GMSRadixCache install skipped",
        )
        return False
    if getattr(Scheduler.__init__, "_gms_patched", False):
        _INSTALLED = True
        return False

    original_init = Scheduler.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        try:
            _swap_tree_cache(self)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[GMSRadixCache install] failed to swap "
                "tree_cache; SGLang scheduler continues with "
                "vanilla RadixCache",
            )

    _patched_init._gms_patched = True  # type: ignore[attr-defined]
    Scheduler.__init__ = _patched_init  # type: ignore[method-assign]
    _INSTALLED = True
    logger.info(
        "[GMSRadixCache install] patched Scheduler.__init__",
    )
    return True


def _swap_tree_cache(scheduler) -> None:
    """Replace `scheduler.tree_cache` with a `GMSRadixCache`
    instance built from the same `CacheInitParams`. Carries
    over no state — assumes this fires at scheduler init
    (tree is empty / freshly-built)."""
    from gpu_memory_service.integrations.sglang.gms_radix_cache import (
        make_gms_radix_cache_class,
    )

    old = getattr(scheduler, "tree_cache", None)
    if old is None:
        logger.warning(
            "[GMSRadixCache install] scheduler has no tree_cache " "yet; skipping swap",
        )
        return

    # Best-effort GDS wiring. Production deployments set the
    # daemon socket via env. If we can't reach the daemon, the
    # GMSRadixCache constructed below uses `gds=None` and
    # behaves identically to vanilla — safe degradation.
    gds = None
    gds_socket = os.environ.get("GMS_SGLANG_DAEMON_SOCKET")
    directory_socket = gds_socket or os.environ.get("GMS_KV_DIRECTORY_SOCKET")
    engine_id = os.environ.get("GMS_SGLANG_ENGINE_ID", "0")
    block_layout_fn = None
    if gds_socket and os.path.exists(gds_socket):
        try:
            gds, block_layout_fn = _build_gds_connector(
                allocator=old.token_to_kv_pool_allocator,
                daemon_socket=gds_socket,
                engine_id=engine_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[GMSRadixCache install] daemon wiring failed; "
                "falling back to vanilla-equivalent (no spill)",
            )
            gds = None
            block_layout_fn = None

    Cls = make_gms_radix_cache_class()
    # Re-create with the SAME params the vanilla cache was
    # built from. SGLang's RadixCache stashes these on `self`
    # (params attribute on latest main); fall back to manual
    # reconstruction for older versions.
    params = _params_from_existing(old)
    new = Cls(
        params,
        gds=gds,
        block_layout_fn=block_layout_fn,
        engine_id=engine_id,
        daemon_socket=directory_socket,
    )
    scheduler.tree_cache = new
    from dataclasses import fields, is_dataclass

    def cache_attrs(owner):
        names = set(vars(owner)) if hasattr(owner, "__dict__") else set()
        if is_dataclass(owner):
            names.update(field.name for field in fields(owner))
        return [
            (name, getattr(owner, name))
            for name in names
            if hasattr(owner, name)
        ]

    rebound = []
    owners = [
        (owner_name, owner)
        for owner_name, owner in vars(scheduler).items()
        if owner is not old
    ]
    for owner_name, owner in owners:
        for attr_name, value in cache_attrs(owner):
            if value is old:
                object.__setattr__(owner, attr_name, new)
                rebound.append(f"{owner_name}.{attr_name}")
    canary = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    canary = getattr(canary, "canary_manager", None)
    if canary is not None:
        canary.attach_radix_cache(new)
    stale = [
        f"{owner_name}.{attr_name}"
        for owner_name, owner in owners
        for attr_name, value in cache_attrs(owner)
        if value is old
    ]
    if stale:
        raise RuntimeError(f"stale SGLang tree-cache references: {stale}")
    logger.info(
        "[GMSRadixCache install] rebound cache references: %s",
        ", ".join(rebound) or "none",
    )
    try:
        from gpu_memory_service.integrations.common.transition_reclaim import (
            start_kv_transition_reclaim_watcher,
        )

        namespace_suffix = _lease_namespace_suffix(old.token_to_kv_pool_allocator)
        watcher = start_kv_transition_reclaim_watcher(
            "sglang",
            new.reclaim_cached_kv,
            namespace_suffix=namespace_suffix,
        )
        if watcher is not None:
            new._gms_transition_reclaim_watcher = watcher
    except Exception:  # noqa: BLE001
        logger.debug(
            "[GMSRadixCache install] transition reclaim watcher not started",
            exc_info=True,
        )
    logger.info(
        "[GMSRadixCache install] swapped tree_cache → " "GMSRadixCache (gds_active=%s)",
        bool(gds is not None and gds.is_available()),
    )


def _lease_namespace_suffix(allocator) -> str:
    try:
        return (
            f"{allocator.__class__.__name__}:"
            f"size{int(allocator.size)}:page{int(allocator.page_size)}"
        )
    except Exception:  # noqa: BLE001
        return "kv"


def _params_from_existing(old_cache):
    """Reconstruct CacheInitParams from an existing RadixCache
    instance. The latest-main API exposes them directly; older
    versions need attribute pickup."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    return CacheInitParams(
        disable=getattr(old_cache, "disable", False),
        req_to_token_pool=getattr(old_cache, "req_to_token_pool", None),
        token_to_kv_pool_allocator=getattr(
            old_cache,
            "token_to_kv_pool_allocator",
            None,
        ),
        page_size=getattr(old_cache, "page_size", 1),
        enable_kv_cache_events=getattr(
            old_cache,
            "enable_kv_cache_events",
            False,
        ),
    )


def _build_gds_connector(allocator, daemon_socket: str, engine_id: str):
    """Construct a `BlockGdsConnector` + layout function from
    SGLang's pool allocator. The layout function derivation
    is intentionally minimal here: we probe the allocator's
    `pool` attribute (if present) to get a per-layer base
    pointer and stride. Caller is responsible for handling
    model architectures where this assumption doesn't hold."""
    from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
    from gms_kv_ring.engines.handle import GMSKvRing

    layers_desc, block_layout_fn = _probe_layout(allocator)
    handle = GMSKvRing(
        engine_id=engine_id,
        daemon_socket=daemon_socket,
        layers=layers_desc,
    )
    gds = BlockGdsConnector(handle, block_layout_fn)
    return gds, block_layout_fn


def _probe_layout(allocator):
    """Best-effort introspection of SGLang's KV pool to build
    layer descriptors and a slot→bytes layout function.

    The shape varies by model architecture. This implementation
    handles the common MHA case where `allocator.pool.kv_buffer`
    is a list of per-layer tensors of shape
    `[num_total_slots, num_kv_heads * head_dim * 2]` (k+v
    interleaved).

    Returns (layers_desc, layout_fn) where:
      - layers_desc is the `layers` arg passed to GMSKvRing.attach
      - layout_fn(slot_idx) → [(layer_idx, byte_offset, byte_size), ...]

    Raises RuntimeError if the pool shape isn't recognized."""
    import torch

    pool = getattr(allocator, "pool", None)
    if pool is None:
        raise RuntimeError("allocator has no `pool` attribute — cannot probe")
    # SGLang stores per-layer tensors in `pool.k_buffer` /
    # `pool.v_buffer` (MHA) or `pool.kv_buffer` (some MLA
    # variants). Try the common cases.
    k_buf = getattr(pool, "k_buffer", None)
    v_buf = getattr(pool, "v_buffer", None)
    if k_buf is None or v_buf is None:
        # Fallback: try `kv_buffer` (a single list of tensors
        # with KV concatenated along the last dim).
        kv = getattr(pool, "kv_buffer", None)
        if kv is None:
            raise RuntimeError(
                "pool has neither k_buffer/v_buffer nor "
                "kv_buffer; unrecognized layout"
            )
        k_buf = kv
        v_buf = None

    layers_desc = []
    per_layer_meta = []  # (base_ptr, slot_stride_bytes)
    for layer_idx, kt in enumerate(k_buf):
        if not isinstance(kt, torch.Tensor):
            continue
        base = int(kt.data_ptr())
        # Bytes per slot for K. v_buf may double this if
        # present.
        slot_stride = int(kt.element_size() * kt[0].numel())
        size = int(slot_stride)
        if v_buf is not None:
            # Pack K + V as adjacent ranges with the same layer
            # index — keeps the layout fn returning a single
            # tuple per layer.
            vt = v_buf[layer_idx]
            size += int(vt.element_size() * vt[0].numel())
        layers_desc.append(
            {
                "layer_idx": layer_idx,
                "va": base,
                "size": size * int(kt.shape[0]),
                "stride": size,
            }
        )
        per_layer_meta.append((base, slot_stride))

    def layout_fn(slot_idx):
        # Per-layer byte range for a single slot, K + V
        # adjacent.
        out = []
        for layer_idx, (base, k_stride) in enumerate(per_layer_meta):
            offset = int(slot_idx) * k_stride * 2  # K + V
            out.append((layer_idx, offset, k_stride * 2))
        return out

    return layers_desc, layout_fn


# Auto-install on import if env is set. Operators can wire
# this via `import gpu_memory_service.integrations.sglang.install_gms_radix_cache`
# at SGLang entrypoint startup.
if os.environ.get("GMS_SGLANG_ENABLE_KV_RING") == "1":
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception(
            "[GMSRadixCache install] auto-install failed",
        )
