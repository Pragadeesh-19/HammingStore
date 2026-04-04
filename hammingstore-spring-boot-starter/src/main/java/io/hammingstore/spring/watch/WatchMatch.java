package io.hammingstore.spring.watch;

import java.time.Instant;
import java.util.Objects;

public record WatchMatch(
        long entityId,
        String watchpointId,
        String watchpointLabel,
        double similarity,
        int approximateHammingDistance,
        WatchSeverity severity,
        Instant detectedAt
) {
    public WatchMatch {
        Objects.requireNonNull(watchpointId, "watchpointId");
        Objects.requireNonNull(watchpointLabel, "watchpointLabel");
        Objects.requireNonNull(severity, "severity");
        Objects.requireNonNull(detectedAt, "detectedAt");
    }

    public String summary() {
        return String.format("[%s] Entity %d matched '%s' (similarity=%.4f, hammingDist=%d, at=%s)",
                severity, entityId, watchpointLabel,
                similarity, approximateHammingDistance, detectedAt);
    }
}
