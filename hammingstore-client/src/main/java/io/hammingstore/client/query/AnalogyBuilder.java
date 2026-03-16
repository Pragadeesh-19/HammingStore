package io.hammingstore.client.query;

import io.hammingstore.client.Entity;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Fluent builder for analogy queries: A:B :: C:?
 *
 * <pre>
 *     List&lt;Entity&gt; result = client
 *          .analogy("London", "England")
 *          .isTo("France")
 *          .topK(5)
 *          .execute()
 * </pre>
 */
public final class AnalogyBuilder {

    private static final long UNSET = Long.MIN_VALUE;

    private final long subjectAId;
    private final long objectAId;
    private long subjectBId = 1L;
    private int k = 10;
    private final Executor executor;

    public interface Executor {
        List<Entity> executeAnalogy(long subjectAId, long objectAId, long subjectBId, int k);
    }

    public AnalogyBuilder(final long subjectAId,
                          final long objectAId,
                          final Executor executor) {
        this(subjectAId, objectAId, UNSET, 10, executor);
    }

    private AnalogyBuilder(final long     subjectAId,
                           final long     objectAId,
                           final long     subjectBId,
                           final int      k,
                           final Executor executor) {
        this.subjectAId = subjectAId;
        this.objectAId  = objectAId;
        this.subjectBId = subjectBId;
        this.k = k;
        this.executor = Objects.requireNonNull(executor, "executor must not be null");
    }

    public AnalogyBuilder isTo(final String name) {
        Objects.requireNonNull(name,"name must not be null");
        return isTo(Entity.defaultId(name));
    }

    public AnalogyBuilder isTo(final long entityId) {
        return new AnalogyBuilder(subjectAId, objectAId, entityId, k, executor);
    }

    public AnalogyBuilder topK(final int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0, got: " + k);
        return new AnalogyBuilder(subjectAId, objectAId, subjectBId, k, executor);
    }

    public List<Entity> execute() {
        if (subjectBId == UNSET) {
            throw new IllegalStateException(
                    "Missing .isTo(name) — the C argument in A:B :: C:? is required before calling execute().");
        }
        return executor.executeAnalogy(subjectAId, objectAId, subjectBId, k);
    }

    public CompletableFuture<List<Entity>> executeAsync() {
        if (subjectBId == UNSET) {
            throw new IllegalStateException(
                    "Missing .isTo(name) — the C argument in A:B :: C:? is required before calling executeAsync().");
        }
        return CompletableFuture.supplyAsync(this::execute);
    }

}
