package io.hammingstore.client.query;

import io.hammingstore.client.Entity;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

public final class HopBuilder {

    private final long entityId;
    private final long relationId;
    private final int k;
    private final Executor executor;

    public interface Executor {
        List<Entity> executeHop(long entityId, long relationId, int k);
    }

    public HopBuilder(final long entityId, final long relationId, final Executor executor) {
        this(entityId, relationId, 10, executor);
    }

    private HopBuilder(final long entityId,
                       final long relationId,
                       final int k,
                       final Executor executor) {
        this.entityId = entityId;
        this.relationId = relationId;
        this.k = k;
        this.executor = Objects.requireNonNull(executor, "executor must not be null");
    }

    public HopBuilder topK(final int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0, got: " + k);
        return new HopBuilder(entityId, relationId, k, executor);
    }

    public List<Entity> execute() {
        return executor.executeHop(entityId, relationId, k);
    }

    public CompletableFuture<List<Entity>> executeAsync() {
        return CompletableFuture.supplyAsync(this::execute);
    }
}
