package io.hammingstore.vsa;

import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Binarizes float embeddings into fixed-size binary hypervectors using a random
 * Gaussian projection matrix (locality-sensitive hashing).
 *
 * <p>The projection matrix {@code M} has shape
 * {@code OUTPUT_BITS X inputDimensions}, where each entry is drawn independently
 * from a standard normal distribution N(0, 1). To encode an embedding {@code v}:
 * <pre>
 *     bit[i] = 1 if dot(M[i], v) &gt; 0
 *     bit[i] = 0 otherwise
 * </pre>
 *
 * <p>This prevents the angular similarity between vectors: two embeddings with
 * high cosine similarity will produce binary vectors with low hamming distance.
 * The probability of a bit agreeing between 2 vectors is
 * {@code 1 - arccos(cosine_similarity / pi)}
 *
 * <p>The matrix is fully determined by the seed and input dimension in
 * {@link ProjectionConfig}. Two encoders constructed with identical configs will
 * produce identical binary vectors for the same float input. Changing the seed
 * or dimension after vectors have been stored invalidates all stored vectors.
 *
 * <p>Instances are thead-safe: {@link #encode} is stateless beyond reading the
 * immutable projection matrix.
 */
public final class RandomProjectionEncoder {

    private static final String PREFERRED_RNG = "Xoshiro256StarStar";

    private final MemorySegment   matrix;
    private final ProjectionConfig config;

    /**
     * The byte stride between rows of the projection matrix.
     * One row = one hyperplane = {@code inputDimensions} floats.
     */
    private final long rowStride;

    /**
     * Creates an encoder whose projection matrix is generated from {@code config}.
     *
     * @param allocator the allocator from which to request matrix storage
     * @param config    the projection configuration (seed, input dimension, output bits)
     * @throws IllegalStateException if the required PRNG is unavailable on this JVM
     */
    public RandomProjectionEncoder(
            final OffHeapAllocator allocator,
            final ProjectionConfig config) {
        this.config    = config;
        this.rowStride = (long) config.inputDimensions() * Float.BYTES;

        final long matrixBytes = (long) ProjectionConfig.OUTPUT_BITS * rowStride;
        this.matrix = allocator.allocateRawSegment(matrixBytes, Float.BYTES);

        populateMatrix(config.seed());
    }

    /**
     * Creates an encoder using the default seed and the given input dimension.
     *
     * @param allocator the allocator from which to request matrix storage
     * @param inputDimensions the float embedding dimension
     */
    public RandomProjectionEncoder(
            final OffHeapAllocator allocator,
            final int inputDimensions) {
        this(allocator, ProjectionConfig.of(inputDimensions));
    }

    /**
     * Projects {@code embeddings} through the Gaussian matrix and writes the resulting
     * binary hypervector into {@code target}.
     *
     * <p>Each output bit is the sign of the dot product between the embedding and
     * the corresponding row of the projection matrix. This is a single encode call
     * and allocates nothing on the heap.
     *
     * @param embedding the float embedding to binarize; length must be equal
     *                  {@link ProjectionConfig#inputDimensions()}
     * @param targetBinaryVector destination segment; must be exactly
     *                           {@link BinaryVector#VECTOR_BYTES} bytes
     * @throws IllegalArgumentException if {@code embedding} has the wrong length
     */
    public void encode(final float[] embedding, final MemorySegment targetBinaryVector) {
        if (embedding.length != config.inputDimensions()) {
            throw new IllegalArgumentException(
                    "Embedding length " + embedding.length +
                            " does not match encoder inputDimensions " + config.inputDimensions());
        }

        for (int w = 0; w < BinaryVector.VECTOR_LONGS; w++) {
            long word = 0L;
            for (int b = 0; b < Long.SIZE; b++) {
                final int hyperplaneIdx = w * Long.SIZE + b;
                final long rowBase = (long) hyperplaneIdx * rowStride;

                float dot = 0.0f;
                for (int d= 0; d<config.inputDimensions(); d++) {
                    dot += embedding[d] *
                            matrix.get(ValueLayout.JAVA_FLOAT_UNALIGNED,
                                    rowBase + (long) d * Float.BYTES);
                }

                if (dot > 0.0f) {
                    word |= (1L << b);
                }
            }

            targetBinaryVector.set(ValueLayout.JAVA_LONG_UNALIGNED,
                    (long) w * Long.BYTES, word);
        }
    }

    /**
     * Returns the projection configuration this encoder was built from.
     * Useful for transmitting config to clients via {@code GetProjectionConfig}.
     *
     */
    public ProjectionConfig config() {
        return config;
    }

    /**
     * Fills the projection matrix with samples from N(0, 1) using the
     * Box-Muller transform.
     *
     * <p>Box-muller converts two independent uniform samples (u1, u2) into two
     * independent standard normal samples (z0, z1):
     * <pre>
     *     z0 = sqrt(-2 * ln(u1) * cos(2pi * u2))
     *     z1 = sqrt(-2 * ln(u1) * sin(2pi * u2))
     * </pre>
     * Samples are generated pairs for efficiency. If the total float count is odd,
     * the final sample is generated using only z0.
     *
     * <p>u1 is clamped away from zero before taking the logarithm to avoid
     * producing {@code -Infinity}.
     *
     * @throws IllegalStateException if the required PRNG is unavailable on this JVM
     */
    private void populateMatrix(final long seed) {
        final RandomGenerator rng = createRng(seed);

        final long totalFloats = (long) ProjectionConfig.OUTPUT_BITS * config.inputDimensions();
        long offset = 0L;
        long remaining = totalFloats;

        while (remaining >= 2) {
            double u1 = rng.nextDouble();
            double u2 = rng.nextDouble();

            if (u1 < Double.MIN_VALUE) u1 = Double.MIN_VALUE;

            final double magnitude = Math.sqrt(-2.0 * Math.log(u1));
            final double angle = 2.0 * Math.PI * u2;
            final float z0 = (float) (magnitude * Math.cos(angle));
            final float z1 = (float) (magnitude * Math.sin(angle));

            matrix.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, z0);
            matrix.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset + Float.BYTES, z1);
            offset += 2L * Float.BYTES;
            remaining -= 2;
        }

        if (remaining == 1) {
            double u1 = rng.nextDouble();
            if (u1 < Double.MIN_VALUE) u1 = Double.MIN_VALUE;
            final float z0 = (float) (Math.sqrt(-2.0 * Math.log(u1))
                    * Math.cos(2.0 * Math.PI * rng.nextDouble()));
            matrix.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, z0);
        }
    }

    /**
     * Returns a seeded {@link RandomGenerator} for projection matrix population.
     *
     * <p>Tries {@value #PREFERRED_RNG} first. That algorithm lives in the
     * {@code jdk.random} module which may be absent in stripped JREs (e.g. custom
     * {@code jlink} images or some Windows JDK distributions). If unavailable, falls
     * back to {@link SplittableRandom}, which is always present in {@code java.base}
     * and is itself a high-quality PRNG suitable for this use case.
     *
     * <p>Both paths are fully deterministic for a given seed, so projection matrices
     * produced by either path are reproducible — but they are <em>not identical</em>
     * across paths. A stored index must always be searched with the same JVM
     * configuration it was built on. This is enforced by {@link ProjectionConfig}'s
     * seed check in {@link io.hammingstore.persist.EngineSnapshot#validateProjectionConfig}.
     */
    private static RandomGenerator createRng(final long seed) {
        try {
            return RandomGeneratorFactory.of(PREFERRED_RNG).create(seed);
        } catch (IllegalArgumentException ignored) {
            return new SplittableRandom(seed);
        }
    }
}
