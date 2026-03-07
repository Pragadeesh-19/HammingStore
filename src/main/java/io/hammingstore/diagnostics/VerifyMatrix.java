package io.hammingstore.diagnostics;

import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.vsa.ProjectionConfig;
import io.hammingstore.vsa.RandomProjectionEncoder;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class VerifyMatrix {

    public static void main(String[] args) {
        final ProjectionConfig config = ProjectionConfig.of(
                ProjectionConfig.DEFAULT_SEED,
                ProjectionConfig.DIMS_MINILM   // 384
        );

        System.out.println("seed=" + config.seed());
        System.out.println("inputDimensions=" + config.inputDimensions());

        final OffHeapAllocator alloc = new OffHeapAllocator(500L);
        final RandomProjectionEncoder encoder = new RandomProjectionEncoder(alloc, config);

        // Access the matrix via reflection to print first 10 floats
        try {
            final var field = RandomProjectionEncoder.class.getDeclaredField("matrix");
            field.setAccessible(true);
            final MemorySegment matrix = (MemorySegment) field.get(encoder);

            System.out.println("First 10 matrix floats:");
            for (int i = 0; i < 10; i++) {
                float v = matrix.get(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) i * Float.BYTES);
                System.out.printf("  matrix[%d] = %.10f%n", i, v);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
