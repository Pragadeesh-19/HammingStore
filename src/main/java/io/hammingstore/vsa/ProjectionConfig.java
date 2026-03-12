package io.hammingstore.vsa;

/**
 * Immutable configuration for the random Gaussian projection matrix used to
 * binarize float embeddings into hypervectors.
 *
 * <p>The projection matrix is fully determined by {@link #seed} and
 * {@link #inputDimensions}. The {@code ProjectionConfig} instances with the same
 * seed and input dimensions will produce identical projection matrices, and therefore
 * identical binary vectors from the float embeddings. Changing either value
 * after vectors have been stored invalidates every stored vector.
 *
 * <p>{@link #outputBits} is fixed at {@link #OUTPUT_BITS} for all configurations.
 * It exists as a record component only so that the full configuration can be
 * transmitted over the wire (e.g. via {@code GetProjectionConfig} RPC) and
 * reconstructed at the client. The compact constructor rejects any other value.
 *
 * <p>Use the factory methods {@link #of(int)} or {@link #of(long, int)} rather than
 * the canonical constructor directly.
 */
public record ProjectionConfig(long seed, int inputDimensions, int outputBits) {

    /**
     * Default PRNG seed used when no seed is specified.
     * the value is arbitrary - any fixed constant would work equally well.
     */
    public static final long DEFAULT_SEED = 0xC0FFEE_FACADE_42L;

    /** Embedding dimension for {@code sentence-transformers/all-MiniLM-L6-v2} */
    public static final int DIMS_MINILM = 384;

    /**
     * The only supported output dimensionality: {@value} bits.
     *
     * <p>This value is fixed by the node memory layout in {@code HNSWConfig}:
     * each graph node stores exactly 157 × 8 = 1,256 bytes of vector data,
     * which holds exactly 10,048 bits. Changing this constant would require
     * a breaking change to the on-disk format.
     */
    public static final int OUTPUT_BITS = 10_048;

    /**
     * Compact constructor - validates all fields.
     *
     * @throws IllegalArgumentException if {@code inputDimensions} is not positive,
     *                                  or if {@code outputBits} is not {@link #OUTPUT_BITS}
     */
    public ProjectionConfig {
        if (inputDimensions <= 0)
            throw new IllegalArgumentException("inputDimensions must be > 0");
        if (outputBits != OUTPUT_BITS)
            throw new IllegalArgumentException("outputBits must be " + OUTPUT_BITS);
    }

    /**
     * Creates a configuration using the {@link #DEFAULT_SEED} and the given
     * input dimension.
     *
     * @param inputDimensions the float embedding dimension (e.g. {@link #DIMS_MINILM})
     * @return a new {@code ProjectionConfig}
     */
    public static ProjectionConfig of(final int inputDimensions) {
        return new ProjectionConfig(DEFAULT_SEED, inputDimensions, OUTPUT_BITS);
    }

    /**
     * Creates a configuration using an explicit seed and input dimension.
     *
     * @param seed the PRNG seed for Gaussian matrix generation
     * @param inputDimensions the float embedding dimension.
     * @return a new {@code ProjectionConfig}
     */
    public static ProjectionConfig of(final long seed, final int inputDimensions) {
        return new ProjectionConfig(seed, inputDimensions, OUTPUT_BITS);
    }
}
