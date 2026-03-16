package io.hammingstore.client.query;

import io.hammingstore.client.SearchResult;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Fluent builder for nearest-neighbour search queries.
 *
 * <pre>
 *     List&lt;SearchResult&gt; results = client
 *          .search(embedding)
 *          .topK(10)
 *          .minSimilarity(0.7)
 *          .execute()
 * </pre>
 */
public final class SearchBuilder {

    private final float[] embedding;
    private int k = 10;
    private double minSimilarity = 0.0;
    private final Executor executor;

    public interface Executor {
        List<SearchResult> executeSearch(float[] embedding, int k);
    }

    public SearchBuilder(final float[] embedding, final Executor executor) {
        this(
                Objects.requireNonNull(embedding, "embedding must not be null").clone(),
                10,
                0.0,
                Objects.requireNonNull(executor, "executor must not be null"));
    }

    private SearchBuilder(final float[] embedding,
                          final int k,
                          final double minSimilarity,
                          final Executor executor) {
        this.embedding = embedding;
        this.k = k;
        this.minSimilarity = minSimilarity;
        this.executor = executor;
    }

    public SearchBuilder topK(final int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0, got: " + k);
        return new SearchBuilder(embedding, k, minSimilarity, executor);
    }

    public SearchBuilder minSimilarity(final double minSimilarity) {
        if (minSimilarity < 0.0 || minSimilarity > 1.0) {
            throw new IllegalArgumentException(
                    "minSimilarity must be in [0.0, 1.0], got: " + minSimilarity);
        }
        return new SearchBuilder(embedding, k, minSimilarity, executor);
    }

    public List<SearchResult> execute() {
        final List<SearchResult> raw = executor.executeSearch(embedding, k);
        if (minSimilarity <= 0.0) return raw;
        final double threshold = minSimilarity;
        return raw.stream()
                .filter(r -> r.similarity() >= threshold)
                .toList();
    }

    public CompletableFuture<List<SearchResult>> executeAsync() {
        return CompletableFuture.supplyAsync(this::execute);
    }
}
