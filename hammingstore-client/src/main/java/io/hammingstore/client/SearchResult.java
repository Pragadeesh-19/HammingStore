package io.hammingstore.client;

import java.util.Objects;

public final class SearchResult {

    public static final long TOTAL_BITS = 10_048L;

    private final Entity entity;
    private final double similarity;
    private final long hammingDistance;

    private SearchResult(final Entity entity,
                         final double similarity,
                         final long hammingDistance) {
        this.entity = Objects.requireNonNull(entity, "entity must not be null");
        this.similarity = similarity;
        this.hammingDistance = hammingDistance;
    }

    public static SearchResult of(final Entity entity,
                                  final double similarity,
                                  final long   hammingDistance) {
        return new SearchResult(entity, similarity, hammingDistance);
    }

    public Entity entity() { return entity; }

    public double similarity() { return similarity; }

    public long hammingDistance() { return hammingDistance; }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (!(o instanceof SearchResult other)) return false;
        return entity.equals(other.entity)
                && Double.compare(similarity, other.similarity) == 0
                && hammingDistance == other.hammingDistance;
    }

    @Override
    public int hashCode() {
        return Objects.hash(entity, similarity, hammingDistance);
    }

    @Override
    public String toString() {
        return "SearchResult{entity=" + entity
                + ", similarity=" + String.format("%.4f", similarity)
                + ", hammingDistance=" + hammingDistance + "}";
    }
}
