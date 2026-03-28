package io.hammingstore.client;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public final class Entity {

    private final long id;
    private final String name;
    private final float[] embedding;
    private final Map<String,Object> metadata;

    private Entity(final Builder builder) {
        this.id = builder.id;
        this.name = builder.name;
        this.embedding = builder.embedding == null
                ? null
                : builder.embedding.clone();
        this.metadata  = builder.metadata == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new HashMap<>(builder.metadata));
    }

    public static Entity of(final long id) {
        return new Builder(id).build();
    }

    public static Entity named(final String name) {
        Objects.requireNonNull(name, "name must not be null");
        return new Builder(defaultId(name)).name(name).build();
    }

    public static Builder builder(final long id) {
        return new Builder(id);
    }

    public long id() { return id; }

    public String name() { return name; }

    public float[] embedding() {
        return embedding == null ? null : embedding.clone();
    }

    public Map<String, Object> metadata() { return metadata; }

    public boolean hasName() { return name != null; }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (!(o instanceof Entity other)) return false;
        return id == other.id;
    }

    @Override
    public int hashCode() { return Long.hashCode(id); }

    @Override
    public String toString() {
        return hasName()
                ? "Entity{id=" + id + ", name='" + name + "'}"
                : "Entity{id=" + id + "}";
    }

    public static long defaultId(final String name) {
        try {
            final java.security.MessageDigest sha =
                    java.security.MessageDigest.getInstance("SHA-256");
            final byte[] hash = sha.digest(name.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            long value = 0L;
            for (int i = 0; i < 8; i++) {
                value = (value << 8) | (hash[i] & 0xFFL);
            }
            return value & 0x7FFF_FFFF_FFFF_FFFFL;
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    public static final class Builder {
        private final long id;
        private String name;
        private float[] embedding;
        private Map<String,Object> metadata;

        private Builder(final long id) { this.id = id; }

        public Builder name(final String name) {
            this.name = name;
            return this;
        }

        public Builder embedding(final float[] embedding) {
            this.embedding = embedding;
            return this;
        }

        public Builder metadata(final Map<String,Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder meta(final String key, final Object value) {
            if (this.metadata == null) this.metadata = new HashMap<>();
            this.metadata.put(key, value);
            return this;
        }

        public Entity build() { return new Entity(this); }
    }
}
