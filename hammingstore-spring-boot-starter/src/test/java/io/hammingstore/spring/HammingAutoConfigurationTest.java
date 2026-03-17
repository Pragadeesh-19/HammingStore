package io.hammingstore.spring;

import io.hammingstore.client.HammingClient;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

class HammingAutoConfigurationTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(HammingAutoConfiguration.class));


    @Test
    void autoConfigurationIsInactive_whenEndpointNotConfigured() {
        runner.run(ctx ->
                assertThat(ctx).doesNotHaveBean(HammingClient.class));
    }

    @Test
    void autoConfigurationIsActive_whenEndpointIsConfigured() {
        runner
                .withPropertyValues("hammingstore.endpoint=localhost:50051")
                .run(ctx ->
                        assertThat(ctx).hasSingleBean(HammingClient.class));
    }

    @Test
    void userDefinedClientBean_takesPreference_overAutoConfigured() {
        runner
                .withPropertyValues("hammingstore.endpoint=localhost:50051")
                .withBean(HammingClient.class, () ->
                        HammingClient.builder().endpoint("custom-host", 9999).plaintext().build())
                .run(ctx -> {
                    assertThat(ctx).hasSingleBean(HammingClient.class);
                    // The auto-configured bean is suppressed — no duplicate
                });
    }

    @Test
    void properties_bindCorrectly_fromApplicationProperties() {
        runner
                .withPropertyValues(
                        "hammingstore.endpoint=prod-host:50051",
                        "hammingstore.tls=true",
                        "hammingstore.pool-size=16",
                        "hammingstore.timeout=PT10S",
                        "hammingstore.expected-dims=384",
                        "hammingstore.expected-seed=42",
                        "hammingstore.metric-prefix=myapp.hamming"
                )
                .run(ctx -> {
                    final HammingProperties props = ctx.getBean(HammingProperties.class);
                    assertThat(props.getEndpoint()).isEqualTo("prod-host:50051");
                    assertThat(props.isTls()).isTrue();
                    assertThat(props.getPoolSize()).isEqualTo(16);
                    assertThat(props.getTimeout().getSeconds()).isEqualTo(10);
                    assertThat(props.getExpectedDims()).isEqualTo(384);
                    assertThat(props.getExpectedSeed()).isEqualTo(42L);
                    assertThat(props.getMetricPrefix()).isEqualTo("myapp.hamming");
                });
    }

    @Test
    void properties_host_parsedCorrectly() {
        runner
                .withPropertyValues("hammingstore.endpoint=my-server:50051")
                .run(ctx -> {
                    final HammingProperties props = ctx.getBean(HammingProperties.class);
                    assertThat(props.host()).isEqualTo("my-server");
                    assertThat(props.port()).isEqualTo(50051);
                });
    }

    @Test
    void properties_tls_defaultsToTrue() {
        runner
                .withPropertyValues("hammingstore.endpoint=localhost:50051")
                .run(ctx -> {
                    final HammingProperties props = ctx.getBean(HammingProperties.class);
                    assertThat(props.isTls()).isTrue();
                });
    }

    @Test
    void shutdownHook_isRegistered_whenClientIsActive() {
        runner
                .withPropertyValues("hammingstore.endpoint=localhost:50051")
                .run(ctx ->
                        assertThat(ctx).hasSingleBean(HammingShutdownHook.class));
    }

    @Test
    void shutdownHook_isNotRegistered_whenNoEndpoint() {
        runner.run(ctx ->
                assertThat(ctx).doesNotHaveBean(HammingShutdownHook.class));
    }

    @Test
    void startupVerifier_notRegistered_byDefault() {
        runner
                .withPropertyValues("hammingstore.endpoint=localhost:50051")
                .run(ctx ->
                        assertThat(ctx).doesNotHaveBean(
                                HammingAutoConfiguration.StartupVerifier.class));
    }

    @Test
    void startupVerifier_notRegistered_whenVerifyOnStartupFalse() {
        runner
                .withPropertyValues(
                        "hammingstore.endpoint=localhost:50051",
                        "hammingstore.verify-on-startup=false"
                )
                .run(ctx ->
                        assertThat(ctx).doesNotHaveBean(
                                HammingAutoConfiguration.StartupVerifier.class));
    }
}
