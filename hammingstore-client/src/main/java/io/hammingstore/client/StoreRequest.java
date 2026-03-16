package io.hammingstore.client;

import java.util.Objects;

public final class StoreRequest {

    private final long entityId;
    private final String name;
    private final float[] embedding;

    private StoreRequest(final long entityId, final String name, final float[] embedding) {
        this.entityId  = entityId;
        this.name = name;
        this.embedding = Objects.requireNonNull(embedding,"embedding must not be null").clone();
    }

    public static StoreRequest of(final String name, final float[] embedding) {
        Objects.requireNonNull(name, "name must not be null");
        return new StoreRequest(Entity.defaultId(name), name, embedding);
    }

    public static StoreRequest of(final long entityId, final float[] embedding) {
        return new StoreRequest(entityId, null, embedding);
    }

    public long entityId() { return entityId; }
    public String name() { return name; }
    public float[] embedding() { return embedding.clone(); }

    @Override
    public String toString() {
        return name != null
                ? "StoreRequest{id=" + entityId + ", name='" + name + "'}"
                : "StoreRequest{id=" + entityId + "}";
    }
}
