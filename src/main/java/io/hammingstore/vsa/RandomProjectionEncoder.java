package io.hammingstore.vsa;

import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class RandomProjectionEncoder {

    private static final int OUTPUT_BITS = ProjectionConfig.OUTPUT_BITS;
    private static final int OUTPUT_LONGS = BinaryVector.VECTOR_LONGS;
    private static final int BITS_PER_LONG = Long.SIZE;

    private final MemorySegment matrix;
    private final int inputDimensions;
    private final ProjectionConfig config;
    private final long rowStride;

    public RandomProjectionEncoder(
            final OffHeapAllocator allocator,
            final ProjectionConfig config) {

        this.config          = config;
        this.inputDimensions = config.inputDimensions();
        this.rowStride       = (long) inputDimensions * Float.BYTES;

        final long matrixBytes = (long) OUTPUT_BITS * rowStride;
        this.matrix = allocator.allocateRawSegment(matrixBytes, Float.BYTES);

        populateMatrix(config.seed());
    }

    public RandomProjectionEncoder(
            final OffHeapAllocator allocator,
            final int inputDimensions) {
        this(allocator, ProjectionConfig.of(inputDimensions));
    }

    public void encode(final float[] embedding, final MemorySegment targetBinaryVector) {
        if (embedding.length != inputDimensions) {
            throw new IllegalArgumentException(
                    "Embedding length " + embedding.length +
                            " does not match encoder inputDimensions " + inputDimensions);
        }

        for (int w = 0; w < OUTPUT_LONGS; w++) {
            long word = 0L;
            for (int b = 0; b < BITS_PER_LONG; b++) {
                final int hyperplaneIdx = w * BITS_PER_LONG + b;
                final long rowBase = (long) hyperplaneIdx * rowStride;

                float dot = 0.0f;
                for (int d= 0; d<inputDimensions; d++) {
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

    public ProjectionConfig config() {
        return config;
    }

    public int inputDimensions() {
        return inputDimensions;
    }

    private void populateMatrix(final long seed) {
        RandomGenerator rng;
        try {
            rng = RandomGeneratorFactory.of("Xoshiro256StarStar").create(seed);
        } catch (IllegalArgumentException e) {
            rng = RandomGeneratorFactory.of("Random").create(seed);
        }

        final long totalFloats = (long) OUTPUT_BITS * inputDimensions;
        long offset = 0L;
        long remaining = totalFloats;

        while (remaining >= 2) {
            double u1 = rng.nextDouble();
            double u2 = rng.nextDouble();

            if (u1 < Double.MIN_VALUE) u1 = Double.MIN_VALUE;
            final double mag = Math.sqrt(-2.0 * Math.log(u1));
            final double angle = 2.0 * Math.PI * u2;
            final float z0 = (float) (mag * Math.cos(angle));
            final float z1 = (float) (mag * Math.sin(angle));

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
}
