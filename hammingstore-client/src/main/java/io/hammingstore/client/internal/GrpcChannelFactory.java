package io.hammingstore.client.internal;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.hammingstore.client.config.HammingClientConfig;

import java.util.concurrent.TimeUnit;

public final class GrpcChannelFactory {

    private static final int SHUTDOWN_TIMEOUT_SECONDS = 5;

    private GrpcChannelFactory() {}

    public static ManagedChannel create(final HammingClientConfig config) {
        final ManagedChannelBuilder<?> builder = ManagedChannelBuilder
                .forAddress(config.host(), config.port());

        if (config.tls()) {
            builder.useTransportSecurity();
        } else {
            builder.usePlaintext();
        }

        return builder.build();
    }

    public static void shutdown(final ManagedChannel channel) {
        channel.shutdown();
        try {
            if (!channel.awaitTermination(SHUTDOWN_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                channel.shutdownNow();
            }
        } catch (InterruptedException ignored) {
            channel.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
