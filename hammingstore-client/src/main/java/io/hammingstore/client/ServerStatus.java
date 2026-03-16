package io.hammingstore.client;

import java.util.Objects;

public final class ServerStatus {

    private final boolean alive;
    private final int inputDimensions;
    private final long projectionSeed;
    private final String mode;

    private ServerStatus(final boolean alive,
                         final int inputDimensions,
                         final long projectionSeed,
                         final String mode) {
        this.alive = alive;
        this.inputDimensions = inputDimensions;
        this.projectionSeed = projectionSeed;
        this.mode = Objects.requireNonNull(mode,"mode must not be null");
    }

    public static ServerStatus of(final boolean alive,
                                  final int inputDimensions,
                                  final long projectionSeed,
                                  final String mode) {
        return new ServerStatus(alive, inputDimensions, projectionSeed, mode);
    }

    public static ServerStatus dead() {
        return new ServerStatus(false, 0, 0L, "UNREACHABLE");
    }

    public boolean alive() { return alive; }

    public int inputDimensions() { return inputDimensions; }

    public long projectionSeed() { return projectionSeed; }

    public String mode() { return mode; }

    public void assertCompatible(final int expectedDims, final long expectedSeed) {
        if (!alive) {
            throw new IllegalStateException("Server is not reachable — cannot verify compatibility.");
        }
        if (inputDimensions != expectedDims) {
            throw new IllegalStateException(
                    "Projection dimension mismatch: server=" + inputDimensions
                            + ", expected=" + expectedDims
                            + ". Stored vectors will not be comparable across different dimensions.");
        }
        if (projectionSeed != expectedSeed) {
            throw new IllegalStateException(
                    "Projection seed mismatch: server=" + projectionSeed
                            + ", expected=" + expectedSeed
                            + ". Vectors encoded with different seeds are not comparable.");
        }
    }

    @Override
    public String toString() {
        return "ServerStatus{alive=" + alive
                + ", dims=" + inputDimensions
                + ", seed=" + projectionSeed
                + ", mode=" + mode + "}";
    }
}
