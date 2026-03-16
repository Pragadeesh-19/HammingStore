package io.hammingstore.client.config;

import java.time.Duration;
import java.util.Objects;

public final class HammingClientConfig {

    public static final int DEFAULT_POOL_SIZE = Math.min(Runtime.getRuntime().availableProcessors(), 8);

    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private final String host;
    private final int port;
    private final boolean tls;
    private final int poolSize;
    private final Duration requestTimeout;
    private final EntityIdResolver entityIdResolver;
    private final EntityIdResolver  relationIdResolver;

    public HammingClientConfig(
            final String host,
            final int port,
            final boolean tls,
            final int poolSize,
            final Duration requestTimeout,
            final EntityIdResolver entityIdResolver,
            final EntityIdResolver relationIdResolver) {

        this.host = Objects.requireNonNull(host, "host must not be null");
        this.port = port;
        this.tls = tls;
        this.poolSize = poolSize;
        this.requestTimeout = Objects.requireNonNull(requestTimeout, "requestTimeout must not be null");
        this.entityIdResolver = Objects.requireNonNull(entityIdResolver, "entityIdResolver must not be null");
        this.relationIdResolver = Objects.requireNonNull(relationIdResolver, "relationIdResolver must not be null");

        if (port < 1 || port > 65535) throw new IllegalArgumentException("port must be 1-65535, got: " + port);
        if (poolSize < 1) throw new IllegalArgumentException("poolSize must be >= 1, got: " + poolSize);
    }

    public String host() {
        return host;
    }

    public int port() { return port; }
    public boolean tls() { return tls; }
    public int poolSize() { return poolSize; }
    public Duration requestTimeout() { return requestTimeout; }
    public EntityIdResolver entityIdResolver() { return entityIdResolver; }
    public EntityIdResolver relationIdResolver() { return relationIdResolver; }

    public String endpoint() { return host + ":" + port; }

    @Override
    public String toString() {
        return "HammingClientConfig{endpoint=" + endpoint()
                + ", tls=" + tls
                + ", poolSize=" + poolSize
                + ", timeout=" + requestTimeout + "}";
    }
}
