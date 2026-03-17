package io.hammingstore.spring;

import io.hammingstore.client.HammingClient;
import io.hammingstore.client.ServerStatus;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicReference;

public final class HammingHealthIndicator implements HealthIndicator {

    private final HammingClient client;
    private final String endpoint;
    private final Duration cacheTtl;

    private final AtomicReference<CachedHealth> cache = new AtomicReference<>(null);

    public HammingHealthIndicator(final HammingClient client, final HammingProperties props) {
        this.client = client;
        this.endpoint = props.getEndpoint();
        this.cacheTtl = props.getHealthCacheTtl();
    }

    @Override
    public Health health() {
        final CachedHealth cached = cache.get();
        if (cached != null && !cached.isExpired(cacheTtl)) {
            return cached.health;
        }

        final Health fresh = queryHealth();
        cache.set(new CachedHealth(fresh, Instant.now()));
        return fresh;
    }

    private Health queryHealth() {
        final ServerStatus status = client.status();
        if (!status.alive()) {
            return Health.down()
                    .withDetail("endpoint", endpoint)
                    .withDetail("reason", "Server unreachable")
                    .build();
        }
        return Health.up()
                .withDetail("endpoint", endpoint)
                .withDetail("dims", status.inputDimensions())
                .withDetail("seed", status.projectionSeed())
                .withDetail("mode", status.mode())
                .build();
    }

    private static final class CachedHealth {

        private final Health  health;
        private final Instant cachedAt;

        CachedHealth(final Health health, final Instant cachedAt) {
            this.health   = health;
            this.cachedAt = cachedAt;
        }

        boolean isExpired(final Duration ttl) {
            return Instant.now().isAfter(cachedAt.plus(ttl));
        }
    }
}
