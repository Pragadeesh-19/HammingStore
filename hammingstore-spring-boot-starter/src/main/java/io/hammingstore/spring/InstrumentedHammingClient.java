package io.hammingstore.spring;

import io.hammingstore.client.*;
import io.hammingstore.client.query.*;
import io.hammingstore.spring.watch.WatchRegistry;
import io.micrometer.core.instrument.Timer;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

public final class InstrumentedHammingClient implements AutoCloseable{

    private final HammingClient delegate;
    private final HammingMetricsBinder binder;
    private volatile WatchRegistry watchRegistry;

    public InstrumentedHammingClient(final HammingClient delegate,
                                     final HammingMetricsBinder binder) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must be not null");
        this.binder = Objects.requireNonNull(binder,"binder must be not null");
    }

    public HammingClient unwrap() { return delegate; }

    public void setWatchRegistry(final WatchRegistry watchRegistry) {
        this.watchRegistry = watchRegistry;
    }

    public void storeFloat(final long entityId, final float[] embedding) {
        binder.storeFloatTimer().record(() -> delegate.storeFloat(entityId, embedding));

        final WatchRegistry wr = this.watchRegistry;
        if (wr != null) wr.check(entityId, embedding);
    }

    public void storeFloat(final String entityName, final float[] embedding) {
        binder.storeFloatTimer().record(() -> delegate.storeFloat(entityName, embedding));

        final WatchRegistry wr = this.watchRegistry;
        if (wr != null) wr.check(Entity.defaultId(entityName), embedding);
    }

    public void storeEdge(final Edge edge) {
        binder.storeEdgeTimer().record(() -> delegate.storeEdge(edge));
    }

    public BatchResult storeBatch(final List<StoreRequest> requests) {
        binder.batchSizeSummary().record(requests.size());
        final BatchResult result = delegate.storeBatch(requests);
        binder.batchFailureCounter().increment(result.failureCount());

        final WatchRegistry wr = this.watchRegistry;
        if (wr != null) {
            for (final StoreRequest req : requests) {
                wr.check(req.entityId(), req.embedding());
            }
        }
        return result;
    }

    public BatchResult storeEdges(final List<Edge> edges) {
        binder.batchSizeSummary().record(edges.size());
        return delegate.storeEdges(edges);
    }

    public void retract(final long entityId) {
        binder.retractTimer().record(() -> delegate.retract(entityId));
    }

    public void retract(final String entityName) {
        binder.retractTimer().record(() -> delegate.retract(entityName));
    }

    public void checkpoint() {
        binder.checkpointTimer().record(delegate::checkpoint);
    }

    public TimedChainQueryBuilder from(final long entityId) {
        return new TimedChainQueryBuilder(delegate.from(entityId), binder.chainTimer());
    }

    public TimedChainQueryBuilder from(final String entityName) {
        return new TimedChainQueryBuilder(delegate.from(entityName), binder.chainTimer());
    }

    public HopBuilder hop(final long entityId, final long relationId) {
        return delegate.hop(entityId, relationId);
    }

    public HopBuilder hop(final String entityName, final String relation) {
        return delegate.hop(entityName, relation);
    }

    public TimedAnalogyBuilder analogy(final String subjectA, final String objectA) {
        return new TimedAnalogyBuilder(
                delegate.analogy(subjectA, objectA), binder.analogyTimer());

    }

    public TimedRelationBuilder relation(final long relationId, final long objectId) {
        return new TimedRelationBuilder(
                delegate.relation(relationId, objectId), binder.relationTimer());
    }

    public TimedRelationBuilder relation(final String relation, final String objectName) {
        return new TimedRelationBuilder(
                delegate.relation(relation, objectName), binder.relationTimer());
    }

    public TimedBundleBuilder bundle(final long... entityIds) {
        return new TimedBundleBuilder(delegate.bundle(entityIds), binder.bundleTimer());
    }

    public TimedBundleBuilder bundle(final String... names) {
        return new TimedBundleBuilder(delegate.bundle(names), binder.bundleTimer());
    }

    public TimedSearchBuilder search(final float[] embedding) {
        return new TimedSearchBuilder(delegate.search(embedding), binder.searchTimer());
    }

    public boolean ping() {
        return Boolean.TRUE.equals(binder.pingTimer().record(delegate::ping));
    }

    public ServerStatus status() {
        return binder.statusTimer().record(delegate::status);
    }

    @Override
    public void close() {
        delegate.close();
    }

    @Override
    public String toString() {
        return "InstrumentedHammingClient{delegate=" + delegate + "}";
    }

    public static final class TimedChainQueryBuilder {

        private final ChainQueryBuilder delegate;
        private final Timer timer;

        TimedChainQueryBuilder(final ChainQueryBuilder delegate, final Timer timer) {
            this.delegate = delegate;
            this.timer    = timer;
        }

        public TimedChainQueryBuilder via(final long relationId) {
            return new TimedChainQueryBuilder(delegate.via(relationId), timer);
        }

        public TimedChainQueryBuilder via(final String relation) {
            return new TimedChainQueryBuilder(delegate.via(relation), timer);
        }

        public TimedChainQueryBuilder topK(final int k) {
            return new TimedChainQueryBuilder(delegate.topK(k), timer);
        }

        /** Terminal call — this is where the gRPC call happens and time is recorded. */
        public List<Entity> execute() {
            return timer.record(delegate::execute);
        }
    }

    public static final class TimedAnalogyBuilder {
        private final AnalogyBuilder delegate;
        private final Timer timer;

        TimedAnalogyBuilder(final AnalogyBuilder delegate, final Timer timer) {
            this.delegate = delegate;
            this.timer    = timer;
        }

        /** Sets the C argument in A:B :: C:? */
        public TimedAnalogyBuilder isTo(final String name) {
            return new TimedAnalogyBuilder(delegate.isTo(name), timer);
        }

        public TimedAnalogyBuilder isTo(final long entityId) {
            return new TimedAnalogyBuilder(delegate.isTo(entityId), timer);
        }

        public TimedAnalogyBuilder topK(final int k) {
            return new TimedAnalogyBuilder(delegate.topK(k), timer);
        }

        /** Terminal call — gRPC call happens here and is timed. */
        public List<Entity> execute() {
            return timer.record(delegate::execute);
        }

        public CompletableFuture<List<Entity>> executeAsync() {
            return CompletableFuture.supplyAsync(this::execute);
        }
    }

    public static final class TimedRelationBuilder {
        private final RelationBuilder delegate;
        private final Timer timer;

        TimedRelationBuilder(final RelationBuilder delegate, final Timer timer) {
            this.delegate = delegate;
            this.timer    = timer;
        }

        public TimedRelationBuilder topK(final int k) {
            return new TimedRelationBuilder(delegate.topK(k), timer);
        }

        public List<Entity> execute() {
            return timer.record(delegate::execute);
        }

        public CompletableFuture<List<Entity>> executeAsync() {
            return CompletableFuture.supplyAsync(this::execute);
        }
    }

    public static final class TimedBundleBuilder {
        private final BundleBuilder delegate;
        private final Timer timer;

        TimedBundleBuilder(final BundleBuilder delegate, final Timer timer) {
            this.delegate = delegate;
            this.timer = timer;
        }

        public TimedBundleBuilder topK(final int k) {
            return new TimedBundleBuilder(delegate.topK(k), timer);
        }

        public List<Entity> execute() {
            return timer.record(delegate::execute);
        }

        public CompletableFuture<List<Entity>> executeAsync() {
            return CompletableFuture.supplyAsync(this::execute);
        }
    }

    public static final class TimedSearchBuilder {
        private final SearchBuilder delegate;
        private final Timer timer;

        TimedSearchBuilder(final SearchBuilder delegate, final Timer timer) {
            this.delegate = delegate;
            this.timer    = timer;
        }

        public TimedSearchBuilder topK(final int k) {
            return new TimedSearchBuilder(delegate.topK(k), timer);
        }

        public TimedSearchBuilder minSimilarity(final double minSimilarity) {
            return new TimedSearchBuilder(delegate.minSimilarity(minSimilarity), timer);
        }

        public List<SearchResult> execute() {
            return timer.record(delegate::execute);
        }

        public CompletableFuture<List<SearchResult>> executeAsync() {
            return CompletableFuture.supplyAsync(this::execute);
        }
    }
}
