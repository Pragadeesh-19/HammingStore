package io.hammingstore.spring;

import org.springframework.boot.context.properties.ConfigurationProperties;
import java.time.Duration;

@ConfigurationProperties(prefix = "hammingstore")
public class HammingProperties {

    private String endpoint;

    private boolean tls = true;

    private int poolSize = Math.min(Runtime.getRuntime().availableProcessors(), 8);

    private Duration timeout = Duration.ofSeconds(30);

    private int expectedDims = 0;

    private long expectedSeed = 0L;

    private boolean verifyOnStartup = false;

    private String metricPrefix = "hammingstore";

    private Duration healthCacheTtl = Duration.ofSeconds(10);

    public String host() {
        return parsedEndpoint()[0];
    }

    public int port() {
        return Integer.parseInt(parsedEndpoint()[1]);
    }

    private String[] parsedEndpoint() {
        if (endpoint == null || endpoint.isBlank()) {
            throw new IllegalStateException(
                    "hammingstore.endpoint is required but was not configured.");
        }
        final String[] parts = endpoint.split(":", 2);
        if (parts.length != 2) {
            throw new IllegalStateException(
                    "hammingstore.endpoint must be in host:port format, got: " + endpoint);
        }
        return parts;
    }

    public String getEndpoint() {
        return endpoint;
    }

    public void setEndpoint(String endpoint) {
        this.endpoint = endpoint;
    }

    public boolean isTls() {
        return tls;
    }

    public void setTls(boolean tls) {
        this.tls = tls;
    }

    public int getPoolSize() {
        return poolSize;
    }

    public void setPoolSize(int poolSize) {
        this.poolSize = poolSize;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public void setTimeout(Duration timeout) {
        this.timeout = timeout;
    }

    public int getExpectedDims() {
        return expectedDims;
    }

    public void setExpectedDims(int expectedDims) {
        this.expectedDims = expectedDims;
    }

    public long getExpectedSeed() {
        return expectedSeed;
    }

    public void setExpectedSeed(long expectedSeed) {
        this.expectedSeed = expectedSeed;
    }

    public boolean isVerifyOnStartup() {
        return verifyOnStartup;
    }

    public void setVerifyOnStartup(boolean verifyOnStartup) {
        this.verifyOnStartup = verifyOnStartup;
    }

    public String getMetricPrefix() {
        return metricPrefix;
    }

    public void setMetricPrefix(String metricPrefix) {
        this.metricPrefix = metricPrefix;
    }

    public Duration getHealthCacheTtl() {
        return healthCacheTtl;
    }

    public void setHealthCacheTtl(Duration healthCacheTtl) {
        this.healthCacheTtl = healthCacheTtl;
    }

    @Override
    public String toString() {
        return "HammingProperties{endpoint='" + endpoint + "'"
                + ", tls=" + tls
                + ", poolSize=" + poolSize
                + ", timeout=" + timeout
                + ", verifyOnStartup=" + verifyOnStartup + "}";
    }
}
