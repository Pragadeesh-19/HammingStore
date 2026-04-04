package io.hammingstore.spring.watch;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;
import java.util.logging.Logger;

public final class WatchRegistry implements AutoCloseable {

    private static final Logger log = Logger.getLogger(WatchRegistry.class.getName());
    private static final int DEFAULT_QUEUE_CAPACITY = 10_000;
    private static final int DEFAULT_DELIVERY_THREADS = 4;

    private record WatchEntry(Watchpoint watchpoint, float[] centroid) {}

    private final CopyOnWriteArrayList<WatchEntry> entries =
            new CopyOnWriteArrayList<>();

    private final Map<String, CopyOnWriteArrayList<HammingWatchListener>> specificListeners =
            new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<HammingWatchListener> wildcardListeners =
            new CopyOnWriteArrayList<>();

    private final BlockingQueue<WatchMatch> matchQueue;
    private final ExecutorService deliveryPool;

    private final LongAdder totalDetected = new LongAdder();
    private final LongAdder totalDelivered = new LongAdder();
    private final LongAdder totalDropped = new LongAdder();

    public WatchRegistry() {
        this(DEFAULT_QUEUE_CAPACITY, DEFAULT_DELIVERY_THREADS);
    }

    public WatchRegistry(final int queueCapacity, final int deliveryThreads) {
        this.matchQueue = new ArrayBlockingQueue<>(queueCapacity);
        this.deliveryPool = Executors.newVirtualThreadPerTaskExecutor();
        for (int i = 0; i < deliveryThreads; i++) {
            deliveryPool.submit(this::drainLoop);
        }
    }

    public void register(final Watchpoint watchpoint) {
        Objects.requireNonNull(watchpoint, "watchpoint");
        entries.removeIf(e -> e.watchpoint().id().equals(watchpoint.id()));
        entries.add(new WatchEntry(watchpoint, watchpoint.centroid()));
        log.info(String.format(
                "WatchRegistry: registered '%s' (id=%s  dims=%d  info>=%.2f  critical>=%.2f)",
                watchpoint.label(), watchpoint.id(), watchpoint.dims(),
                watchpoint.infoThreshold(), watchpoint.criticalThreshold()));
    }

    public boolean unregister(final String watchpointId) {
        Objects.requireNonNull(watchpointId, "watchpointId");
        return entries.removeIf(e -> e.watchpoint().id().equals(watchpointId));
    }

    public List<Watchpoint> watchpoints() {
        return entries.stream().map(WatchEntry::watchpoint).toList();
    }

    public int watchpointCount() { return entries.size(); }

    public void subscribe(final String watchpointId,
                          final HammingWatchListener listener) {
        Objects.requireNonNull(watchpointId, "watchpointId");
        Objects.requireNonNull(listener, "listener");
        if ("*".equals(watchpointId)) {
            wildcardListeners.add(listener);
        } else {
            specificListeners
                    .computeIfAbsent(watchpointId, k -> new CopyOnWriteArrayList<>())
                    .add(listener);
        }
    }

    public void unsubscribe(final String watchpointId,
                            final HammingWatchListener listener) {
        if ("*".equals(watchpointId)) {
            wildcardListeners.remove(listener);
        } else {
            final var list = specificListeners.get(watchpointId);
            if (list != null) list.remove(listener);
        }
    }

    public void check(final long entityId, final float[] embedding) {
        if (embedding == null || embedding.length == 0) return;
        final Object[] snapshot = entries.toArray();

        for (final Object obj : snapshot) {
            final WatchEntry entry    = (WatchEntry) obj;
            final Watchpoint wp       = entry.watchpoint();
            final float[]    centroid = entry.centroid();

            final int dims = Math.min(embedding.length, centroid.length);

            float dot = 0f;
            for (int d = 0; d < dims; d++)
                dot += embedding[d] * centroid[d];

            final double similarity = Math.max(0.0, Math.min(1.0, dot));
            final WatchSeverity severity = wp.severityFor(similarity);

            if (severity == null) continue;

            final int hammingDist = (int) ((1.0 - similarity) * 5024.0);

            totalDetected.increment();

            final WatchMatch match = new WatchMatch(
                    entityId,
                    wp.id(),
                    wp.label(),
                    similarity,
                    hammingDist,
                    severity,
                    Instant.now());

            if (!matchQueue.offer(match)) {
                totalDropped.increment();
                log.warning(String.format(
                        "WatchRegistry: queue full — dropped match entity=%d watchpoint=%s",
                        entityId, wp.id()));
            }
        }
    }

    private void drainLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                final WatchMatch match = matchQueue.take();
                deliver(match);
            } catch (final InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (final Exception e) {
                log.warning("WatchRegistry: delivery error: " + e.getMessage());
            }
        }
    }

    private void deliver(final WatchMatch match) {
        for (final HammingWatchListener l : wildcardListeners)
            safeCall(l, match);

        final var specific = specificListeners.get(match.watchpointId());
        if (specific != null)
            for (final HammingWatchListener l : specific)
                safeCall(l, match);

        totalDelivered.increment();
    }

    private void safeCall(final HammingWatchListener listener,
                          final WatchMatch match) {
        try {
            listener.onMatch(match);
        } catch (final Exception e) {
            log.warning(String.format(
                    "WatchRegistry: %s threw on match entity=%d watchpoint=%s: %s",
                    listener.getClass().getSimpleName(),
                    match.entityId(), match.watchpointId(),
                    e.getMessage()));
        }
    }

    public long totalDetected()  { return totalDetected.sum(); }
    public long totalDelivered() { return totalDelivered.sum(); }
    public long totalDropped()   { return totalDropped.sum(); }
    public int  pendingMatches() { return matchQueue.size(); }

    @Override
    public void close() {
        deliveryPool.shutdownNow();
        try {
            if (!deliveryPool.awaitTermination(2, TimeUnit.SECONDS)) {
                log.warning("WatchRegistry: delivery pool did not terminate within 2s");
            }
        } catch (final InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
