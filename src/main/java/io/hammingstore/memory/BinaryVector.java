package io.hammingstore.memory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A fixed size 10,048 bit binary hypervector backed by off-heap memory.
 *
 * <p>Hypervectors are the fundamental unit of storage and computation in this engine.
 * Representing them as packed-64 bit longs rather than byte arrays allows the JVM to
 * exploit hardware popcount instructions for Hamming distance, and keeps the memory
 * footprint ~32x smaller than the equivalent float embedding.
 *
 * <p>The underlying {@link MemorySegment} is owned by the allocator that created it.
 * This record does not manage lifecycle - callers must ensure the segment outlives
 * any {@code BinaryVector} wrapping it.
 *
 */
public record BinaryVector(MemorySegment segment) {

    /**
     * Number of 64-bit words in one hypervector
     * 10,048 bits / 64 bits per long = 157 longs
     */
    public static final int VECTOR_LONGS = 157;

    /**
     * Byte size of one hypervector: {@value} bytes.
     * Derived from {@link #VECTOR_LONGS} x 8 bytes per long.
     */
    public static final long VECTOR_BYTES = (long) VECTOR_LONGS * Long.BYTES;

    /**
     * Compact constructor - validates that the segment is exactly the right size.
     *
     * @throws IllegalArgumentException if {@code segment} is null of not exactly
     *                                  {@link #VECTOR_BYTES} bytes
     */
    public BinaryVector {
        if (segment == null) {
            throw new IllegalArgumentException("segment must not be null");
        }
        if (segment.byteSize() != VECTOR_BYTES) {
            throw new IllegalArgumentException(
                    "Segment must be exactly " + VECTOR_BYTES + " bytes, got " + segment.byteSize()
            );
        }
    }

    /**
     * Reads one 64-bit word from this hypervector
     *
     * @param index word index in {@code 0}, {@link #VECTOR_LONGS}
     * @return the long value at that word position.
     * @throws IllegalArgumentException if {@code index} is out of range
     */
    public long getLong(final int index) {
        if (index < 0 || index >= VECTOR_LONGS) {
            throw new IllegalArgumentException("index " + index + " out of [0, " + VECTOR_LONGS + ")");
        }
        return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, (long) index * Long.BYTES);
    }

    /**
     * Writes one 64-bit word into this hypervector
     *
     * @param index word index in {@code 0}, {@link #VECTOR_LONGS}
     * @param value the long value to write
     * @throws IllegalArgumentException if {@code index} is out of range
     */
    public void setLong(final int index, final long value) {
        if (index < 0 || index >= VECTOR_LONGS) {
            throw new IllegalArgumentException("index " + index + " out of [0, " + VECTOR_LONGS + ")");
        }
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED, (long) index * Long.BYTES, value);
    }
}
