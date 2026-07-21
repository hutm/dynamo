# GMS v8 rebase — validation status

Base: upstream ai-dynamo/dynamo main `0237635a42` (ancestor of HEAD).
Engines bumped to latest main (vLLM/SGLang) and delta-rebuilt on 2×B200.

## Unit / integration (local, worker-env python)

| Suite | Result |
|---|---|
| `lib/gpu_memory_service/tests` + `lib/gms_kv_ring/tests` | **585 passed, 16 skipped** (CUDA-only skips) |
| `components/src/dynamo/common` GMS tests | **86 passed** |
| `components/src/dynamo/{vllm,sglang,trtllm}/tests` | **782 passed, 7 skipped, 7 failed** |

### The 7 failures are pre-existing engine-bump drift, NOT GMS regressions

All 7 live in files **byte-identical to pristine upstream main** (`git diff
0237635a42 HEAD` empty) with zero GMS content. They fail because the freshly
bumped vLLM/SGLang changed contracts that upstream dynamo has not yet caught
up to — they fail identically on pristine dynamo main built against these
engines:

- `test_vllm_kv_events_api.py::test_block_{stored,removed}_fields` — vLLM
  `BlockStored`/`BlockRemoved` field order changed (test docstring: "If vLLM
  adds/removes/reorders fields, this test will fail").
- `test_vllm_renderer_api.py::test_engine_core_struct_contract` — vLLM
  `EngineCoreRequest` fields changed; test says update
  `components/src/dynamo/frontend/vllm_processor.py`.
- `test_vllm_worker_handler.py::TestDecodeWorkerMultimodalBranching` (2) —
  vLLM multimodal handler API drift.
- `test_fpm_contract.py` (2) — environmental: `sglang.srt` is not importable
  in the vLLM worker-env used by the shared runner (cross-engine test).

Harness note: upstream async tests need `-o asyncio_mode=auto` (the repro
runner uses `-c /dev/null`, stripping the pyproject setting).

## Local shadow-failover e2e (2×B200, repro-bulwark-failover.sh)

`test_gms_shadow_engine_failover_<engine>`: start primary + shadow GMS engines,
SIGKILL the primary, assert a shadow adopts the GMS-owned KV across the crash and
serves real post-failover tokens.

| Engine | Result |
|---|---|
| sglang | **1 passed** (118s) — shadow takeover, KV stays RW (alloc=57) across crash |
| vllm   | **1 passed** (135s) — geometry pin `num_gpu_blocks_override=19967`, shadow serves after failover |

Two engine-bump items surfaced and were resolved:
- **sglang (code fix):** the bump renamed `ModelRunner.init_memory_pool` ->
  `alloc_memory_pool` and moved the pre-model-load memory baseline to the instance
  attribute `self.pre_model_load_memory`. The GMS memory-saver patch died at import
  with AttributeError, killing every worker. Fixed in
  `integrations/sglang/patches.py` (patch `alloc_memory_pool`, inflate
  `self.pre_model_load_memory`; keep the old path as fallback).
- **vllm (test-harness config, NOT a product regression):** the bump changed vLLM's
  memory profiling so cohort peers compute different KV block counts (20656 vs
  19967). The geometry pin (`num_gpu_blocks_override`) resolves this, but only when
  `GMS_VLLM_SHARED_KV=1`/`GMS_VLLM_KV_LEASES=1` are set. Production k8s manifests
  (`bulwark-failover-pod.yaml.tmpl`, `pytest-failover-pod.yaml.tmpl`) already set
  these; only the local repro script omitted the vLLM-specific envs (it set the
  sglang ones). Passing them makes the local vllm e2e green.

## K8s single-pod failover e2e (Layer 2, in-cluster, DRA 1-GPU)

`test_gms_authoritative_hbm_failover_<engine>` in one store-native pod (mounts the
Nix store + tests from the rootfs PVC): primary + shadow GMS engines, SIGKILL the
primary, assert the shadow adopts the authoritative persistent HBM KV and serves.

| Engine | Result |
|---|---|
| sglang | **1 passed** — geometry pin `Adjusted 4057 -> 5528 pages`, shadow serves |
| vllm   | **1 passed** — geometry pin `num_gpu_blocks_override=22054`, shadow serves |

The in-cluster authoritative test exercises prefix caching + strict persistent-HBM
adoption, so it surfaced engine-bump regressions the lighter local test did not.
Four real GMS/SGLang fixes were required (all committed):

1. `alloc_memory_pool` — SGLang renamed `ModelRunner.init_memory_pool` and moved the
   pre-model-load baseline to `self.pre_model_load_memory`.
2. `cache_finished_req` — SGLang made it take required kw-only `kv_len_to_handle` and
   dropped `req._cache_commit_len()`.
3. Geometry-pin re-hook — SGLang flattened `ModelRunnerKVCacheMixin` into
   `KVCacheConfigurator`; the pin's import failed (caught -> disabled), so shadows
   sized KV from their own post-attach profile and diverged from the primary.
4. Deterministic geometry re-derivation — after clamping `max_total_num_tokens` to the
   shared page count, re-derive `max_running_requests` + finalize pools from the
   pinned token count (mirroring SGLang's own resolve sequence) so the primary and
   shadow build byte-identical persistent allocations (the persistent alloc covers
   both token_to_kv_pool AND req_to_token_pool).

vLLM needed no code change for k8s (the pod template already sets
`GMS_VLLM_SHARED_KV=1` / `GMS_VLLM_KV_LEASES=1`).

TRT-LLM: deferred (engine not bumped — see TRTLLM-BUMP-DEFERRED.md); its k8s failover
is out of scope for this rebase.

## Reconciliations applied during rebase
- Daemon: all `cuMem*/cuda_*` ops routed through `self._vmm = get_vmm()`
  (upstream VMMDevice abstraction). Tests install a shared `FakeVMM`.
- `cli/server.py`: reverted half-merged upstream single-server `main()` back
  to our tag-based supervisor; `list_devices` now from `common.vmm.cuda_utils`.
- `test_vllm_unit.py`: 3-way re-merge — upstream's current non-GMS tests + our
  9 GMS tests.
