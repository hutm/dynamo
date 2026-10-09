# DEP: Engine failover without losing the KV cache

## Summary

When an inference engine crashes today, its GPU memory is lost. Reloading the weights of a large model takes minutes, and the KV cache (every prompt prefix already computed) is gone, so in-flight requests fail or start over.

This DEP keeps GPU memory alive outside the engine and adds a **warm standby** engine on the same GPUs. When the main engine (the primary) dies, the standby takes over in about a second or two. It reuses the weights and KV cache that are still in GPU memory, and Dynamo resumes the interrupted requests on it.

Each failover deployment is self-contained: a local frontend, the primary, the standby, and the memory service, with no etcd or NATS. Clients see one stable endpoint. A takeover only changes which process answers behind it.

The work is split across three PR trains: vLLM (#12053), SGLang (#14704), and production hardening (#15035). This issue is the umbrella design for all three.

## Motivation

Engines fail often at scale: crashes, CUDA errors, out-of-memory, bad requests. Today recovery is a cold restart:

| Step | Cost for a 235B model at TP8/TP16 |
|---|---|
| Restart, reload weights, capture CUDA graphs | minutes |
| KV cache | lost; every prompt prefix is recomputed |
| In-flight requests | moved to another replica, which prefills the whole prompt again |

Dynamo already moves interrupted requests, so clients see no error, but they wait minutes, and the replica's capacity is gone meanwhile.

**Goal:** recover from an engine crash in seconds, with no client-visible errors, reusing the KV already computed, and without changing how vLLM or SGLang schedule, hash, or evict.

**Not in scope:** GPU reset or node loss, KV in CPU memory or storage (see "Relation to KVCR"), and multi-tenant isolation.

## Proposal

The core idea: GPU memory should outlive the engine process.

```text
 clients ──► local frontend (one stable endpoint; holds and resumes requests)
                  │   one logical worker, discovered through local files
                  ▼
        primary engine        standby engine (ready, asleep)
                  │                  │
                  └── map the same GPU memory ──┘
                              │
        GPU Memory Service (GMS): owns weights and KV cache on each GPU
```

**1. GMS owns the GPU memory.** A small daemon on each GPU allocates the weights and the KV cache. Engines only map that memory, so an engine crash frees nothing.

**2. A warm standby waits on the same GPUs.** It is fully started (weights mapped, CUDA graphs captured) and then sleeps. The primary starts serving only once its standby is ready.

**3. The KV cache survives in the engine's own format.** Every KV block carries a lease with a version number, and finished blocks are listed in a small directory by prefix hash. On takeover, the standby adds those blocks to its own prefix cache. The engine stays in charge of its cache; GMS does not keep a copy of its index.

**4. The old engine can't corrupt the standby.** When a crash is detected, every process of the old engine is stopped at once, on all GPU ranks. Memory it might still have been writing is set aside and reused only once that is known to be safe. If anything is unclear, the standby treats the data as a cache miss rather than reuse it.

**5. Requests keep going.** Interrupted requests continue on the standby using their cached KV, and new requests just wait a moment instead of failing.

**6. One worker, no extra infrastructure.** The primary and standby share one worker ID, so the frontend sees a single worker whose address changes on takeover. Discovery uses local files instead of etcd or NATS, and a deployment is just a few plain pods, with no operator required.

**7. Normal serving stays fast.** The bookkeeping that makes the KV cache recoverable is batched and kept off the critical path of each step, especially the step that releases the first token. Taking a lease never waits on a lock.

### The three PR trains

| Train | Tracker | What it delivers |
|---|---|---|
| vLLM | #12053 | The memory daemon and how engines attach to its memory (#12056, #14576, #14577); KV leases, the block directory, and adoption into vLLM's prefix cache (#13398); standby start-up and failover; an end-to-end test (#12032) |
| SGLang | #14704 | The same for SGLang: KV identity, page leases, cache persistence, failover from TP1 to TP2, fencing, rank health checks, serving timeouts (#14706–#14719) |
| Hardening | #15035 | Six PRs shared by both engines: safe start-up and low steady-state cost (#15036); ownership through takeover (#15040); stopping all ranks on a fault (#15044); warm-up, journaling, and keeping requests alive across the switch (#15048); a takeover that sets aside the old engine's memory, the isolation setting, a fast handoff, and waiting for new requests (#15050); one worker identity with local discovery (#15913). Memory-pool back ends: #14814, #14818 |

**DEPs from this work**, tied together by this issue:

| DEP | Topic | Train |
|---|---|---|
| #14832 | GPU memory that outlives engine sessions | vLLM (#12056, #14576) |
| #14833 | Safe engine attachment to that memory | vLLM (#14577) |
| #14828 | Ways to recover the KV cache: per block or whole pool | vLLM and SGLang (#14814, #14818) |
| #15035 | Production hardening | Hardening (#15036–#15050, #15913) |

## Relation to KVCR

KVCR (the KV Cache Runner behind DEP #11673) also keeps KV useful across failures. The two cover different memory and different failures, and they are designed to work together.

| | This DEP | KVCR |
|---|---|---|
| Memory | KV and weights in GPU memory, on the same GPUs | KV in CPU memory, SSD, object storage, or on other nodes |
| Failure covered | the engine process dies; GPU and node are fine | GPU or node loss, cache capacity, sharing across nodes |
| How it recovers | the standby maps the surviving memory; nothing is copied | a restarted or different engine fetches blocks from those tiers |
| Recovery time | seconds | limited by transfer speed |

How they fit together:
1. **No shared ownership.** GMS owns GPU memory; KVCR owns everything below it. Neither manages the other's memory.
2. **Layered fallback.** After a takeover, the standby first reuses the blocks GMS kept. Anything missing can come from KVCR instead of being recomputed. If the GPU or node is lost, KVCR or a cold start is the only path.
3. **One view of the cache (open question).** Both publish what they hold by prefix hash. We propose one shared block identity and event format, so the router sees both in one place (see also #13044).
4. **Choosing a strategy.** Picking between a standby takeover, a KVCR-backed restart, and a cold start belongs in the common recovery contract proposed in #15379.

## References

- Train trackers: #12053 (vLLM), #14704 (SGLang), #15035 (hardening; also a DEP)
- DEPs from this work: #14832, #14833, #14828, #15035
- Prior art: #11049 (Bulwark gateway and shared worker identity)
- Related: #11673 (KV Cache Controller / KVCR), #14888 (vLLM KV recovery after failover), #15379 (common recovery contract), #13044 (persistent KV events), #12521 (Snapshot with GMS)
