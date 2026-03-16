package io.hammingstore.client.query;

import io.hammingstore.client.Entity;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Immutable fluent builder for chain traversal queries.
 *
 * <p>Each call to {@link #via} returns a NEW builder instance - this builder
 * is immutable and safe to share across threads.
 *
 * <pre>
 *   List&lt;Entity&gt; result = client.from("Paris")
 *       .via("capitalOf")
 *       .via("locatedIn")
 *       .topK(5)
 *       .execute();
 * </pre>
 */
public final class ChainQueryBuilder {

    private final long startEntityId;
    private final long[] relationIds;
    private final int k;
    private final Executor executor;

    public interface Executor {
        List<Entity> executeChain(long startEntityId, long[] relationIds, int k);
    }

    public ChainQueryBuilder(final long startEntityId, final Executor executor) {
        this(startEntityId, new long[0], 10, executor);
    }

    private ChainQueryBuilder(final long startEntityId,
                              final long[] relationIds,
                              final int k,
                              final Executor executor) {
        this.startEntityId = startEntityId;
        this.relationIds   = relationIds;
        this.k = k;
        this.executor = Objects.requireNonNull(executor, "executor must not be null");;
    }

    public ChainQueryBuilder via(final String relation) {
        Objects.requireNonNull(relation, "relation must not be null");
        return via(Entity.defaultId("REL:" + relation));
    }

    public ChainQueryBuilder via(final long relationId) {
        final long[] newRelIds = new long[relationIds.length + 1];
        System.arraycopy(relationIds, 0, newRelIds, 0, relationIds.length);
        newRelIds[relationIds.length] = relationId;
        return new ChainQueryBuilder(startEntityId, newRelIds, k, executor);
    }

    public ChainQueryBuilder topK(final int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0, got: " + k);
        return new ChainQueryBuilder(startEntityId, relationIds, k, executor);
    }

    public List<Entity> execute() {
        validateHops();
        return executor.executeChain(startEntityId, relationIds, k);
    }

    public CompletableFuture<List<Entity>> executeAsync() {
        validateHops();
        return CompletableFuture.supplyAsync(this::execute);
    }

    private void validateHops() {
        if (relationIds.length == 0) {
            throw new IllegalStateException(
                    "No hops added - call .via(relation) at least once before executing.");
        }
    }

}
