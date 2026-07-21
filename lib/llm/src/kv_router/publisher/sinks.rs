// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier};
use dynamo_runtime::transports::event_plane::EventPublisher;

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

/// Bound normal event-plane batches while always allowing one complete event.
///
/// The default NATS Core `max_payload` is 1 MiB. Production-shaped batches at
/// these caps, including sparse one-block events and one multimodal object with
/// one offset per stored block, remain below that limit in the wire-size
/// regression tests. Multimodal metadata is not intrinsically bounded, so an
/// exceptional batch can still exceed a deployment's configured transport
/// limit and fail under the existing best-effort publication semantics.
pub(super) const MAX_EVENT_PLANE_KV_EVENTS_PER_BATCH: usize = 128;
pub(super) const MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS: usize = 8_192;

pub(super) trait RouterEventBatchSink: Send + Sync {
    fn publish_events(&self, events: &[RouterEvent]) -> impl Future<Output = Result<()>> + Send;
}

#[derive(Default)]
struct PublishFailures {
    publishes: usize,
    events: usize,
    first_error: Option<anyhow::Error>,
}

impl PublishFailures {
    fn record(&mut self, event_count: usize, error: anyhow::Error) {
        self.publishes += 1;
        self.events += event_count;
        if self.first_error.is_none() {
            self.first_error = Some(error);
        }
    }

    fn into_result(self) -> Result<()> {
        let Some(first_error) = self.first_error else {
            return Ok(());
        };
        let summary = format!(
            "{} publish attempt(s) failed; {} event(s) dropped; first error: {first_error}",
            self.publishes, self.events
        );
        Err(first_error.context(summary))
    }
}

impl<P: RouterEventSink + Send + Sync> RouterEventBatchSink for P {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        let mut failures = PublishFailures::default();
        for event in events {
            if let Err(error) = self.publish_event(event).await {
                tracing::error!(
                    worker_id = event.worker_id,
                    event_id = event.event.event_id,
                    error = %error,
                    "Failed to publish KV event"
                );
                failures.record(1, error);
            }
        }
        failures.into_result()
    }
}

pub(super) async fn emit_router_event<P: RouterEventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    router_event: RouterEvent,
) {
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(
            worker_id = router_event.worker_id,
            error = %e,
            "Failed to apply event to local indexer"
        );
    }
    if let Err(e) = publisher.publish_event(&router_event).await {
        tracing::error!(
            worker_id = router_event.worker_id,
            error = %e,
            "Failed to publish event"
        );
    }
}

pub(super) async fn emit<P: RouterEventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
    output: &mut Vec<RouterEvent>,
) {
    emit_router_event(
        publisher,
        local_indexer,
        RouterEvent::with_storage_tier(worker_id, event, storage_tier),
    )
    .await;
}
