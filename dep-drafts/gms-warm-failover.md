## Summary

When an inference engine crashes today, everything in its GPU memory is lost. Weights must be reloaded, which takes minutes for large models. The KV cache, every prompt prefix the engine has computed, disappears, so in-flight requests either fail or start over.

This DEP proposes keeping GPU memory alive outside the engine process, in the GPU Memory Service (GMS), and pairing each engine with a **warm standby** on the same GPUs. When the primary dies, the standby takes over in a few seconds. It reuses the weights and the cached KV that are already resident, and the frontend replays interrupted streams onto it.

The work is implemented in three PR trains:
- vLLM: #12053
- SGLang: #14704
- Production hardening: #15035

This issue is the umbrella design for all three.

## Motivation

**Problem.** Engine failures are common at scale: crashes, CUDA errors, OOMs, bad requests. Recovery today is a cold restart:

| Step | Cost for a 235B model at TP8/TP16 |
|---|---|
| Restart process, reload weights, capture graphs | minutes |
| Rebuild the KV cache | every cached prefix is recomputed |
| In-flight requests | migrated, but the full prompt is prefilled again |

Request migration hides the error, but the user still waits minutes, and capacity drops while the replica is down.

**Goal.** Recover from an engine-process failure in seconds, with:
- no client-visible errors;
- reuse of already-computed KV;
- no change to how vLLM or SGLang schedule, hash, or evict.

**Non-goals:**
- surviving GPU reset or node loss;
- durable or offloaded KV tiers (see the KVCR section);
- multi-tenant isolation.

## Proposal

### Idea

Separate the lifetime of GPU memory from the lifetime of the engine process.

```text
              frontend (migrates in-flight streams)
                 │                      │
          primary engine          standby engine (initialized, asleep)
                 │                      │
                 └──── map via CUDA VMM ─┘
                           │
          GMS daemon: owns weights + KV pool on each GPU
          + KV leases and generations + content directory
```

1. **GMS owns the memory.** A per-GPU daemon allocates the weights and the KV pool. Engines map them and never own them, so a crash releases nothing.
2. **Warm standby.** A second engine process on the same GPUs is fully initialized (weights mapped, CUDA graphs captured), then sleeps. The primary only starts serving once the standby is armed.
3. **KV survives in the engine's own format.** Each KV block is guarded by a lease with a generation number. Completed blocks are published to a small content directory keyed by prefix hash. On takeover, the standby adopts those blocks into its native prefix cache. The engine stays the source of truth; GMS does not mirror its index.
4. **Safety first: a dead or slow primary must never write into memory the successor uses.**
   - Takeover fences the old writer: it bumps the generations and retires the writer cohort, which is all ranks.
   - Pages the old primary may still touch are quarantined.
   - They are reclaimed only after the GPU proves the old process's work has stopped, using MPS client termination.
   - Anything ambiguous fails closed: a cache miss, or a refusal to start. It is never silent reuse.
5. **Request continuity.** The frontend migrates interrupted streams to the standby. The replay hits the reused KV instead of recomputing the prompt.

### The three PR trains

| Train | Tracker | What it delivers |
|---|---|---|
| vLLM | #12053 | Persistent allocation protocol and daemon (#12056, #14576), client and PyTorch attachment (#14577), KV lease ring, content directory, pool identity, BlockPool adoption (#13398), worker activation and failover primitives, five-process acceptance test (#12032) |
| SGLang | #14704 | The same model for SGLang: persistent KV identity, page leases, unified cache persistence, TP1 then TP2 failover, verified fencing, rank liveness, serving timeouts (#14706–#14719) |
| Hardening | #15035 | Common to both engines, in 5 PRs: startup verification and service fencing (#15036); ownership through takeover and MPS-proven quiescence (#15040); TP rank fail-stop and crash headroom (#15044); shadow prewarm, journaling, and migration continuity (#15048); frozen-predecessor two-phase takeover, GPU fault watchdog, and the standby-before-serving gate (#15050). Backend abstraction for v0 and v1 pools: #14814, #14818 |

**DEPs from this initiative.** This issue ties them together:

| DEP | Scope | Train |
|---|---|---|
| #14832 | Persistent GPU allocation lifetime across engine sessions | vLLM base (#12056, #14576) |
| #14833 | Failure-safe client and PyTorch attachment to persistent allocations | vLLM base (#14577) |
| #14828 | Pluggable KV recovery strategies: lease-based versus whole-pool | vLLM and SGLang, pool backends (#14814, #14818) |
| #15035 | Production hardening for persistent GMS KV failover | Hardening (#15036–#15050) |

## Relation to KVCR

KVCR, the KV Cache Runner behind DEP #11673 (KV Cache Controller), and this work both keep KV useful across failures. They cover different memory and different failures, and they are designed to compose.

| | GMS warm failover (this DEP) | KVCR |
|---|---|---|
| Memory | Engine-owned HBM (G1) KV pages and weights, kept on the same GPUs | Controller-owned tiers: host DRAM, SSD, object store (G2–G4), plus remote peers |
| Failure covered | Engine process dies; GPU and node healthy | Engine or GPU loss, cache capacity, cross-node reuse |
| Recovery path | Standby on the same GPUs maps the surviving pages (zero copy) | Restarted or other engine fetches blocks from lower tiers or peers |
| Recovery time | Seconds; no data movement | Bounded by transfer bandwidth |
| Resilience process | Per-GPU GMS daemon | Optional memory-service side process (the KVCR Guard) |

**How they fit together:**
1. **Ownership is disjoint.** GMS owns G1 pages that the engine uses directly. KVCR owns G2 and below. Neither manages the other's memory, which avoids two owners of the same HBM.
2. **Layered fallback.** After a takeover, the standby first adopts surviving G1 blocks from GMS. Misses can then be served from KVCR tiers or peers instead of being recomputed. If the GPU or node itself is lost, GMS has nothing to offer and KVCR, or a cold start, is the path.
3. **Shared indexing, open question.** Both publish what is cached, keyed by prefix hash: GMS through its content directory, KVCR through router inventory and KV events (see also #13044). We propose converging on one block-identity and event format, so the router sees G1 survivors and KVCR tiers in one view.
4. **Policy.** Choosing between standby takeover, a KVCR-backed restart, and a cold start belongs in the common recovery contract proposed in #15379. In that contract, this DEP is one strategy.

## Alternate Solutions

Several existing mechanisms are complements to standby failover, not replacements:

- **Snapshot (#12521, #13220): used together with failover.** The standby is restored from a snapshot of an initialized engine instead of booting from scratch, then attaches to the weights and KV that GMS holds. After a takeover, a fresh standby is re-armed the same way. Snapshot shortens the time to have a standby; GMS makes the takeover itself fast and keeps the KV.
- **KVCR (#11673): covers what GMS cannot.** It handles GPU or node loss and cache capacity through lower tiers and peers (see the KVCR section above).
- **Request migration: required building block.** The frontend's migration replays interrupted streams. GMS makes that replay hit cached KV instead of recomputing the prompt.
- **Cold restart: fallback.** Used when no standby or usable GMS state exists.

These are real alternatives:

- **Weight-only resident daemons**, such as vLLM's `vllm preload`. They make restarts fast for weights only, with no KV, no standby, and no fencing. GMS could act as the backend for such loaders.
- **Whole-pool recovery without per-block leases** (#14828). The same framework with a simpler strategy, but it needs stronger proof that the predecessor has stopped before any reuse.
- **Copying KV off the GPU when a failure happens.** Not viable after a crash, because the process is already gone.

## Requirements

- A replacement never serves a block whose generation is stale, and never writes before predecessor writers are retired and their pages classified.
- All TP ranks agree before allocation, adoption, reclamation, or takeover.
- Ambiguous or incompatible recovery state fails closed: a cache miss or a startup refusal.
- Coordination stays off the per-token hot path.
- Engine-native scheduling, hashing, eviction, and capacity semantics are preserved.

**Known limitations:**
- After a hard crash, the default `gpu-proof` policy keeps the predecessor's pages quarantined until a restart, because MPS cannot certify the dead client. An opt-in best-effort policy reclaims them after the process dies.
- SGLang cannot reproduce output byte for byte across a replay boundary, even without a fault.
- GPU reset and node loss are not covered.

## References

- Train trackers: #12053 (vLLM), #14704 (SGLang), #15035 (hardening; also a DEP)
- DEPs from this initiative: #14832 (allocation lifetime), #14833 (attachment), #14828 (recovery strategies), #15035 (hardening)
- Related: #11673 (KV Cache Controller / KVCR), #14888 (vLLM KV recovery after failover), #15379 (common recovery contract), #13044 (persistent KV events), #12521 (Snapshot-coupled GMS)
