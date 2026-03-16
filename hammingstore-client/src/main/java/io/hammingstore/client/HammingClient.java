package io.hammingstore.client;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.hammingstore.client.config.EntityIdResolver;
import io.hammingstore.client.config.HammingClientConfig;
import io.hammingstore.client.exception.HammingException;
import io.hammingstore.client.internal.ErrorTranslator;
import io.hammingstore.client.internal.GrpcChannelFactory;
import io.hammingstore.client.internal.ProtoMapper;
import io.hammingstore.client.query.*;
import io.hammingstore.grpc.proto.*;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.function.Function;

public final class HammingClient implements AutoCloseable {

    private final HammingClientConfig config;
    private final ManagedChannel channel;
    private final HammingStoreGrpc.HammingStoreBlockingStub stub;
    private final ExecutorService batchExecutor;

    private HammingClient(final HammingClientConfig config) {
        this.config  = config;
        this.channel = GrpcChannelFactory.create(config);
        this.stub    = HammingStoreGrpc.newBlockingStub(channel)
                .withDeadlineAfter(
                        config.requestTimeout().toMillis(),
                        TimeUnit.MILLISECONDS);
        this.batchExecutor = Executors.newFixedThreadPool(config.poolSize(), r -> {
            final Thread t = new Thread(r, "hammingstore-batch");
            t.setDaemon(true);
            return t;
        });
    }

    public static Builder builder() {
        return new Builder();
    }

    public ChainQueryBuilder from(final String entityName) {
        return from(config.entityIdResolver().resolve(entityName));
    }

    public ChainQueryBuilder from(final long entityId) {
        return new ChainQueryBuilder(entityId, this::executeChain);
    }

    public HopBuilder hop(final String entityName, final String relation) {
        return new HopBuilder(
                config.entityIdResolver().resolve(entityName),
                config.relationIdResolver().resolve(relation),
                this::executeHop);
    }

    public HopBuilder hop(final long entityId, final long relationId) {
        return new HopBuilder(entityId, relationId, this::executeHop);
    }

    public RelationBuilder relation(final String relation, final String objectName) {
        return new RelationBuilder(
                config.relationIdResolver().resolve(relation),
                config.entityIdResolver().resolve(objectName),
                this::executeRelation);
    }

    public RelationBuilder relation(final long relationId, final long objectId) {
        return new RelationBuilder(relationId, objectId, this::executeRelation);
    }

    public BundleBuilder bundle(final String... names) {
        return new BundleBuilder(this::executeBundle, names);
    }

    public BundleBuilder bundle(final long... entityIds) {
        return new BundleBuilder(this::executeBundle, entityIds);
    }

    public AnalogyBuilder analogy(final String subjectA, final String objectA) {
        return new AnalogyBuilder(
                config.entityIdResolver().resolve(subjectA),
                config.entityIdResolver().resolve(objectA),
                this::executeAnalogy);
    }

    public SearchBuilder search(final float[] embedding) {
        return new SearchBuilder(embedding, this::executeSearch);
    }

    public void storeFloat(final String entityName, final float[] embedding) {
        storeFloat(config.entityIdResolver().resolve(entityName), embedding);
    }

    public void storeFloat(final long entityId, final float[] embedding) {
        Objects.requireNonNull(embedding, "embedding must not be null");
        try {
            stub.storeFloat(ProtoMapper.toStoreFloatRequest(entityId, embedding));
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    public BatchResult storeBatch(final List<StoreRequest> requests) {
        Objects.requireNonNull(requests, "requests must not be null");
        return executeBatch(
                requests,
                req -> { storeFloat(req.entityId(), req.embedding()); },
                StoreRequest::entityId);
    }

    public void storeEdge(final Edge edge) {
        Objects.requireNonNull(edge, "edge must not be null");
        try {
            stub.storeTypedEdge(ProtoMapper.toStoreTypedEdgeRequest(
                    edge.subject().id(),
                    edge.relationId(),
                    edge.object().id()));
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    public BatchResult storeEdges(final List<Edge> edges) {
        Objects.requireNonNull(edges, "edges must not be null");
        return executeBatch(
                edges,
                edge -> { storeEdge(edge); },
                edge -> edge.subject().id());
    }

    public void retract(final String entityName) {
        retract(config.entityIdResolver().resolve(entityName));
    }

    public void retract(final long entityId) {
        try {
            stub.retract(ProtoMapper.toRetractRequest(entityId));
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    public void checkpoint() {
        try {
            stub.checkpoint(ProtoMapper.toCheckpointRequest(true));
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    public boolean ping() {
        try {
            stub.getProjectionConfig(GetProjectionConfigRequest.newBuilder().build());
            return true;
        } catch (final StatusRuntimeException e) {
            return false;
        }
    }

    public ServerStatus status() {
        try {
            final ProjectionConfig cfg = stub
                    .getProjectionConfig(GetProjectionConfigRequest.newBuilder().build());
            return ServerStatus.of(
                    true,
                    cfg.getInputDimensions(),
                    cfg.getSeed(),
                    "UNKNOWN"
            );
        } catch (final StatusRuntimeException e) {
            return ServerStatus.dead();
        }
    }

    @Override
    public void close() {
        batchExecutor.shutdown();
        try {
            if (!batchExecutor.awaitTermination(5, TimeUnit.SECONDS))
                batchExecutor.shutdownNow();
        } catch (final InterruptedException e) {
            batchExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        GrpcChannelFactory.shutdown(channel);
    }

    @Override
    public String toString() {
        return "HammingClient{endpoint=" + config.endpoint()
                + ", tls=" + config.tls()
                + ", poolSize=" + config.poolSize()
                + ", timeout=" + config.requestTimeout() + "}";
    }

    private List<Entity> executeChain(final long startEntityId,
                                      final long[] relationIds,
                                      final int k) {
        try {
            final QueryChainResponse response = stub.queryChain(
                    ProtoMapper.toQueryChainRequest(startEntityId, relationIds, k));
            return ProtoMapper.toEntities(response.getResultsList());
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    private List<Entity> executeHop(final long entityId, final long relationId, final int k) {
        try {
            final QueryHopResponse res = stub.queryHop(ProtoMapper.toQueryHopRequest(entityId, relationId, k));
            return ProtoMapper.toEntities(res.getResultsList());
        } catch (final StatusRuntimeException e) { throw ErrorTranslator.translate(e, config.endpoint()); }
    }

    private List<Entity> executeRelation(final long relationId, final long objectId, final int k) {
        try {
            final QueryRelationResponse res = stub.queryRelation(ProtoMapper.toQueryRelationRequest(relationId, objectId, k));
            return ProtoMapper.toEntities(res.getResultsList());
        } catch (final StatusRuntimeException e) { throw ErrorTranslator.translate(e, config.endpoint()); }
    }

    private List<Entity> executeBundle(final long[] ids, final int k) {
        try {
            final QuerySetResponse res = stub.querySet(ProtoMapper.toQuerySetRequest(ids, k));
            return ProtoMapper.toEntities(res.getResultsList());
        } catch (final StatusRuntimeException e) { throw ErrorTranslator.translate(e, config.endpoint()); }
    }

    private List<Entity> executeAnalogy(final long subjectAId,
                                        final long objectAId,
                                        final long subjectBId,
                                        final int  k) {
        try {
            final QueryAnalogyResponse response = stub.queryAnalogy(
                    ProtoMapper.toQueryAnalogyRequest(subjectAId, objectAId, subjectBId, k));
            return ProtoMapper.toEntities(response.getResultsList());
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    private List<SearchResult> executeSearch(final float[] embedding, final int k) {
        try {
            final SearchFloatResponse response = stub.searchFloat(
                    ProtoMapper.toSearchFloatRequest(embedding, k));
            return ProtoMapper.toSearchResults(response.getResultsList());
        } catch (final StatusRuntimeException e) {
            throw ErrorTranslator.translate(e, config.endpoint());
        }
    }

    private <T> BatchResult executeBatch(final List<T> items,
                                         final Consumer<T> action,
                                         final Function<T, Long> idExtractor) {
        if (items.isEmpty()) return BatchResult.of(0, List.of());

        final List<Future<Void>> futures  = new ArrayList<>(items.size());
        final List<BatchFailure> failures = new ArrayList<>();

        for (final T item : items) {
            futures.add(batchExecutor.submit(() -> {
                action.accept(item);
                return null;
            }));
        }

        int successCount = 0;
        for (int i = 0; i < futures.size(); i++) {
            try {
                futures.get(i).get();
                successCount++;
            } catch (final Exception e) {
                final HammingException cause = ErrorTranslator.wrap(e);
                failures.add(new BatchFailure(idExtractor.apply(items.get(i)), cause));
            }
        }
        return BatchResult.of(successCount, failures);
    }

    public static final class Builder {

        private String host = "localhost";
        private int port = 50051;
        private boolean tls = false;
        private int poolSize = HammingClientConfig.DEFAULT_POOL_SIZE;
        private Duration timeout = HammingClientConfig.DEFAULT_TIMEOUT;
        private EntityIdResolver entityIdResolver  = EntityIdResolver.DEFAULT;
        private EntityIdResolver relationIdResolver = EntityIdResolver.forRelations(EntityIdResolver.DEFAULT);

        private Builder() {}

        public Builder endpoint(final String host, final int port) {
            this.host = Objects.requireNonNull(host, "host must not be null");
            this.port = port;
            return this;
        }

        public Builder tls() {
            this.tls = true;
            return this;
        }

        public Builder plaintext() {
            this.tls = false;
            return this;
        }

        public Builder poolSize(final int poolSize) {
            if (poolSize < 1) throw new IllegalArgumentException("PoolSize must be >= 1, got: " + poolSize);
            this.poolSize = poolSize;
            return this;
        }

        public Builder timeout(final Duration timeout) {
            this.timeout = Objects.requireNonNull(timeout, "timeout must not be null");
            return this;
        }

        public Builder entityIdResolver(final EntityIdResolver resolver) {
            this.entityIdResolver = Objects.requireNonNull(resolver, "resolver must not be null");
            return this;
        }

        public HammingClient build() {
            final HammingClientConfig config = new HammingClientConfig(
                    host, port, tls, poolSize, timeout,
                    entityIdResolver, relationIdResolver);
            return new HammingClient(config);
        }
    }
}
