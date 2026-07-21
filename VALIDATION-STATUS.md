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

## Reconciliations applied during rebase
- Daemon: all `cuMem*/cuda_*` ops routed through `self._vmm = get_vmm()`
  (upstream VMMDevice abstraction). Tests install a shared `FakeVMM`.
- `cli/server.py`: reverted half-merged upstream single-server `main()` back
  to our tag-based supervisor; `list_devices` now from `common.vmm.cuda_utils`.
- `test_vllm_unit.py`: 3-way re-merge — upstream's current non-GMS tests + our
  9 GMS tests.
