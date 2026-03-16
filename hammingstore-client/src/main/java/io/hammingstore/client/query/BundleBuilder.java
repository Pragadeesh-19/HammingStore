package io.hammingstore.client.query;

import io.hammingstore.client.Entity;

import java.util.*;
import java.util.concurrent.CompletableFuture;

public final class BundleBuilder {

    private final List<Long>  entityIds;
    private final int k;
    private final Executor executor;

    public interface Executor {
        List<Entity> executeBundle(long[] entityIds, int k);
    }

    public BundleBuilder(final Executor executor, final String... names) {
        this(
                Objects.requireNonNull(executor, "executor must not be null"),
                Arrays.stream(names).mapToLong(Entity::defaultId).boxed().toList(),
                10);
    }

    public BundleBuilder(final Executor executor, final long... entityIds) {
        this (
                Objects.requireNonNull(executor, "executor must not be null"),
                Arrays.stream(entityIds).boxed().toList(),
                10);
    }

    private BundleBuilder(final Executor executor, final List<Long> entityIds, final int k) {
        this.executor = Objects.requireNonNull(executor, "executor must not be null");
        this.entityIds = Collections.unmodifiableList(entityIds);
        this.k = k;
    }

    public BundleBuilder add(final String name) {
        return add(Entity.defaultId(Objects.requireNonNull(name, "name must not be null")));
    }

    public BundleBuilder add(final long entityId) {
        final List<Long> newIds = new ArrayList<>(entityIds);
        newIds.add(entityId);
        return new BundleBuilder(executor, newIds, k);
    }

    public BundleBuilder topK(final int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0, got: " + k);
        return new BundleBuilder(executor, entityIds, k);
    }

    public List<Entity> execute() {
        if (entityIds.size() < 2) {
            throw new IllegalStateException(
                    "Bundle requires at least 2 entities, got: " + entityIds.size()
                            + ". Use .add(name) to add more, or use client.search() for single-entity lookup.");
        }
        return executor.executeBundle(
                entityIds.stream().mapToLong(Long::longValue).toArray(), k);
    }

    public CompletableFuture<List<Entity>> executeAsync() {
        return CompletableFuture.supplyAsync(this::execute);
    }
}
