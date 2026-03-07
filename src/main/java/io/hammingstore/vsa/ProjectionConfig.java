package io.hammingstore.vsa;

public record ProjectionConfig(long seed, int inputDimensions, int outputBits) {

    public static final long DEFAULT_SEED = 0xC0FFEE_FACADE_42L;

    public static final int DIMS_MINILM   = 384;

    public static final int DIMS_OPENAI_SMALL = 1536;

    public static final int DIMS_OPENAI_LARGE = 3072;

    public static final int OUTPUT_BITS = 10_048;

    public static ProjectionConfig of(final int inputDimensions) {
        return new ProjectionConfig(DEFAULT_SEED, inputDimensions, OUTPUT_BITS);
    }

    public static ProjectionConfig of(final long seed, final int inputDimensions) {
        return new ProjectionConfig(seed, inputDimensions, OUTPUT_BITS);
    }

    public ProjectionConfig {
        if (inputDimensions <= 0)
            throw new IllegalArgumentException("inputDimensions must be > 0");
        if (outputBits != OUTPUT_BITS)
            throw new IllegalArgumentException("outputBits must be " + OUTPUT_BITS);
    }
}
