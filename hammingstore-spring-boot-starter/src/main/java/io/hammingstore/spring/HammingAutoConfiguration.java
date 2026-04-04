package io.hammingstore.spring;

import io.hammingstore.client.HammingClient;
import io.hammingstore.client.ServerStatus;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;

import java.util.logging.Logger;

@AutoConfiguration
@ConditionalOnClass(HammingClient.class)
@ConditionalOnProperty(prefix = "hammingstore", name = "endpoint")
@EnableConfigurationProperties(HammingProperties.class)
public class HammingAutoConfiguration {

    private static final Logger log = Logger.getLogger(HammingAutoConfiguration.class.getName());

    @Bean
    @ConditionalOnMissingBean(HammingClient.class)
    public HammingClient hammingClient(final HammingProperties props) {
        log.info("HammingStore: creating client → endpoint=" + props.getEndpoint()
                + "  tls=" + props.isTls()
                + "  poolSize=" + props.getPoolSize()
                + "  timeout=" + props.getTimeout());

        final HammingClient.Builder builder = HammingClient.builder()
                .endpoint(props.host(), props.port())
                .poolSize(props.getPoolSize())
                .timeout(props.getTimeout());

        if (props.isTls()) builder.tls();
        else               builder.plaintext();

        return builder.build();
    }

    @Bean
    @Primary
    @ConditionalOnMissingBean(InstrumentedHammingClient.class)
    @ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
    public InstrumentedHammingClient instrumentedHammingClient(
            final HammingClient client,
            final ObjectProvider<MeterRegistry> registryProvider,
            final HammingProperties props) {
        final io.micrometer.core.instrument.MeterRegistry registry =
                registryProvider.getIfAvailable(SimpleMeterRegistry::new);
        log.info("HammingStore: wrapping client with InstrumentedHammingClient");
        final HammingMetricsBinder binder = new HammingMetricsBinder(registry, client, props);
        binder.afterSingletonsInstantiated();
        return new InstrumentedHammingClient(client, binder);
    }

    @Bean
    @ConditionalOnMissingBean(HammingShutdownHook.class)
    public HammingShutdownHook hammingShutdownHook(final HammingClient client) {
        return new HammingShutdownHook(client);
    }

    @Bean
    @ConditionalOnMissingBean(HammingHealthIndicator.class)
    @ConditionalOnClass(name = "org.springframework.boot.actuate.health.HealthIndicator")
    public HammingHealthIndicator hammingHealthIndicator(final HammingClient client,
                                                         final HammingProperties props) {
        return new HammingHealthIndicator(client, props);
    }

    @Bean
    @ConditionalOnMissingBean(HammingMetricsBinder.class)
    @ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
    @ConditionalOnBean(type = "io.micrometer.core.instrument.MeterRegistry")
    public HammingMetricsBinder hammingMetricsBinder(
            final io.micrometer.core.instrument.MeterRegistry registry,
            final HammingClient client,
            final HammingProperties props) {
        return new HammingMetricsBinder(registry, client, props);
    }

    @Bean
    @ConditionalOnMissingBean(StartupVerifier.class)
    @ConditionalOnProperty(prefix = "hammingstore", name = "verify-on-startup", havingValue = "true")
    public StartupVerifier hammingStartupVerifier(final HammingClient client,
                                                  final HammingProperties props) {
        return new StartupVerifier(client, props);
    }

    public static final class StartupVerifier implements SmartInitializingSingleton {

        private static final Logger log = Logger.getLogger(StartupVerifier.class.getName());

        private final HammingClient client;
        private final HammingProperties props;

        public StartupVerifier(final HammingClient client, final HammingProperties props) {
            this.client = client;
            this.props  = props;
        }

        @Override
        public void afterSingletonsInstantiated() {
            log.info("HammingStore: verifying server projection config...");
            final ServerStatus status = client.status();
            status.assertCompatible(props.getExpectedDims(), props.getExpectedSeed());
            log.info("HammingStore: server verified — dims=" + status.inputDimensions()
                    + "  seed=" + status.projectionSeed()
                    + "  mode=" + status.mode());
        }
    }
}
