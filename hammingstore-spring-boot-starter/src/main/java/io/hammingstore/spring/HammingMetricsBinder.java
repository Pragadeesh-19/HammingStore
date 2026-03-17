package io.hammingstore.spring;

import io.hammingstore.client.HammingClient;
import io.micrometer.core.instrument.*;
import org.springframework.beans.factory.SmartInitializingSingleton;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public final class HammingMetricsBinder implements SmartInitializingSingleton {

    public static final String TAG_OPERATION = "operation";
    public static final String TAG_ENDPOINT = "endpoint";

    public static final List<String> OPERATIONS = List.of(
            "store_float", "store_edge", "retract", "checkpoint",
            "chain", "hop", "relation", "analogy", "bundle", "search",
            "ping", "status"
    );

    private final MeterRegistry registry;
    private final HammingClient client;
    private final String prefix;
    private final Tags commonTags;

    private Timer storeFloatTimer;
    private Timer storeEdgeTimer;
    private Timer retractTimer;
    private Timer checkpointTimer;
    private Timer chainTimer;
    private Timer hopTimer;
    private Timer relationTimer;
    private Timer analogyTimer;
    private Timer bundleTimer;
    private Timer searchTimer;
    private Timer pingTimer;
    private Timer statusTimer;
    private DistributionSummary batchSizeSummary;
    private Counter batchFailureCounter;

    public HammingMetricsBinder(final MeterRegistry registry,
                                final HammingClient client,
                                final HammingProperties props) {
        this.registry = registry;
        this.client = client;
        this.prefix = props.getMetricPrefix();
        this.commonTags = Tags.of(Tag.of(TAG_ENDPOINT, props.getEndpoint()));
    }

    @Override
    public void afterSingletonsInstantiated() {
        storeFloatTimer = operationTimer("store_float", "Time to binarise and store a float embedding");
        storeEdgeTimer = operationTimer("store_edge", "Time to store a typed edge");
        retractTimer = operationTimer("retract", "Time to tombstone an entity");
        checkpointTimer = operationTimer("checkpoint", "Time to flush and fdatasync to disk");
        chainTimer = operationTimer("chain", "Multi-hop chain traversal latency");
        hopTimer = operationTimer("hop", "Single typed-edge traversal latency");
        relationTimer = operationTimer("relation", "Reverse relation lookup latency");
        analogyTimer = operationTimer("analogy", "Analogy query latency");
        bundleTimer = operationTimer("bundle", "Majority-vote bundle query latency");
        searchTimer = operationTimer("search", "Nearest-neighbour search latency");
        pingTimer = operationTimer("ping", "Server liveness check latency");
        statusTimer = operationTimer("status", "Server status + projection config latency");

        batchSizeSummary = DistributionSummary.builder(prefix + ".batch.size")
                .description("Number of items submitted per batch write operation")
                .tags(commonTags)
                .register(registry);

        batchFailureCounter = Counter.builder(prefix + ".batch.failures")
                .description("Total number of entities that failed in batch write operations")
                .tags(commonTags)
                .register(registry);

        final AtomicBoolean alive = new AtomicBoolean(true);
        Gauge.builder(prefix + ".connection.alive", client, c -> c.ping() ? 1.0 : 0.0)
                .description("1.0 if the HammingStore server is reachable, 0.0 if not")
                .tags(commonTags)
                .register(registry);
    }

    public Timer storeFloatTimer() { return storeFloatTimer; }
    public Timer storeEdgeTimer() { return storeEdgeTimer; }
    public Timer retractTimer() { return retractTimer; }
    public Timer checkpointTimer() { return checkpointTimer; }
    public Timer chainTimer() { return chainTimer; }
    public Timer hopTimer() { return hopTimer; }
    public Timer relationTimer() { return relationTimer; }
    public Timer analogyTimer() { return analogyTimer; }
    public Timer bundleTimer() { return bundleTimer; }
    public Timer searchTimer() { return searchTimer; }
    public Timer pingTimer() { return pingTimer; }
    public Timer statusTimer() { return statusTimer; }
    public DistributionSummary batchSizeSummary() { return batchSizeSummary; }
    public Counter batchFailureCounter(){ return batchFailureCounter; }

    private Timer operationTimer(final String operation, final String description) {
        return Timer.builder(prefix + ".operation.duration")
                .description(description)
                .tags(commonTags.and(Tag.of(TAG_OPERATION, operation)))
                .publishPercentileHistogram()
                .register(registry);
    }
}
