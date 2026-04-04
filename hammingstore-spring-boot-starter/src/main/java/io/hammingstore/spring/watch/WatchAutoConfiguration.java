package io.hammingstore.spring.watch;

import io.hammingstore.client.HammingClient;
import io.hammingstore.spring.InstrumentedHammingClient;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;

import java.util.Map;
import java.util.logging.Logger;

@AutoConfiguration
@ConditionalOnClass(HammingClient.class)
@ConditionalOnProperty(
        prefix = "hammingstore.watch",
        name = "enabled",
        havingValue = "true",
        matchIfMissing = true)
public class WatchAutoConfiguration implements SmartInitializingSingleton {

    private static final Logger log = Logger.getLogger(WatchAutoConfiguration.class.getName());

    private final ApplicationContext context;
    private WatchRegistry registry;

    public WatchAutoConfiguration(final ApplicationContext context) {
        this.context = context;
    }

    @Bean
    @ConditionalOnMissingBean(WatchRegistry.class)
    public WatchRegistry watchRegistry() {
        registry = new WatchRegistry();
        return registry;
    }

    @Override
    public void afterSingletonsInstantiated() {
        if (registry == null) registry = context.getBean(WatchRegistry.class);

        int subscribed = subscribeListeners();
        wireIntoClient();
        wireMicrometer();

        log.info(String.format(
                "WatchAutoConfiguration: ready. watchpoints=%d  subscribers=%d  queueCapacity=10000",
                registry.watchpointCount(), subscribed));
    }

    private int subscribeListeners() {
        int count = 0;
        final Map<String, Object> beans =
                context.getBeansWithAnnotation(WatchFor.class);
        for (final Map.Entry<String, Object> entry : beans.entrySet()) {
            final Object bean = entry.getValue();
            if (!(bean instanceof HammingWatchListener listener)) {
                log.warning("@WatchFor on '" + entry.getKey() +
                        "' but bean does not implement HammingWatchListener — skipping");
                continue;
            }
            final WatchFor annotation =
                    bean.getClass().getAnnotation(WatchFor.class);
            for (final String id : annotation.value()) {
                registry.subscribe(id, listener);
                log.info("  subscribed '" + entry.getKey() +
                        "' to watchpoint '" + id + "'");
                count++;
            }
        }
        return count;
    }

    private void wireIntoClient() {
        try {
            final InstrumentedHammingClient client =
                    context.getBean(InstrumentedHammingClient.class);
            client.setWatchRegistry(registry);
            log.info("WatchAutoConfiguration: wired into InstrumentedHammingClient");
            return;
        } catch (final Exception ignored) {}

        log.warning("WatchAutoConfiguration: InstrumentedHammingClient not found. " +
                "Watch API requires InstrumentedHammingClient to intercept storeFloat calls. " +
                "Add io.micrometer:micrometer-core to your dependencies to enable it, " +
                "or call watchRegistry.check(entityId, embedding) manually after each store.");
    }

    private void wireMicrometer() {
        try {
            final io.micrometer.core.instrument.MeterRegistry meterRegistry =
                    context.getBean(io.micrometer.core.instrument.MeterRegistry.class);

            io.micrometer.core.instrument.Gauge
                    .builder("hammingstore.watch.queue.size",
                            registry, WatchRegistry::pendingMatches)
                    .description("Pending WatchMatch events in delivery queue")
                    .register(meterRegistry);

            io.micrometer.core.instrument.Gauge
                    .builder("hammingstore.watch.watchpoints",
                            registry, WatchRegistry::watchpointCount)
                    .description("Number of registered watchpoints")
                    .register(meterRegistry);

            io.micrometer.core.instrument.FunctionCounter
                    .builder("hammingstore.watch.detected.total",
                            registry, WatchRegistry::totalDetected)
                    .description("Total watchpoint matches detected")
                    .register(meterRegistry);

            io.micrometer.core.instrument.FunctionCounter
                    .builder("hammingstore.watch.delivered.total",
                            registry, WatchRegistry::totalDelivered)
                    .description("Total watchpoint matches delivered to listeners")
                    .register(meterRegistry);

            io.micrometer.core.instrument.FunctionCounter
                    .builder("hammingstore.watch.dropped.total",
                            registry, WatchRegistry::totalDropped)
                    .description("Total watchpoint matches dropped (queue full)")
                    .register(meterRegistry);

            log.info("WatchAutoConfiguration: Micrometer metrics registered");
        } catch (final Exception ignored) {
        }
    }
}
