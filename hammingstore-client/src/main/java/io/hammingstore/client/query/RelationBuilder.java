package io.hammingstore.client.query;

import io.hammingstore.client.Entity;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

public final class RelationBuilder {

    private final long relationId;
    private final long objectId;
    private final int k;
    private final Executor executor;

    public interface Executor {
        List<Entity> executeRelation(long relationId, long objectId, int k);
    }

    public RelationBuilder(final long relationId,
                           final long objectId,
                           final Executor executor) {
        this(relationId, objectId, 10, executor);
    }

    private RelationBuilder(final long relationId,
                            final long objectId,
                            final int k,
                            final Executor executor) {
        this.relationId = relationId;
        this.objectId = objectId;
        this.k = k;
        this.executor = Objects.requireNonNull(executor,"executor must not be null");
    }

    public RelationBuilder topK(final int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0, got: " + k);
        return new RelationBuilder(relationId, objectId, k, executor);
    }

    public List<Entity> execute() {
        return executor.executeRelation(relationId, objectId, k);
    }

    public CompletableFuture<List<Entity>> executeAsync() {
        return CompletableFuture.supplyAsync(this::execute);
    }
}
