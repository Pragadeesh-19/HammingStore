package io.hammingstore.math;

import io.hammingstore.memory.BinaryVector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class VectorMath {

    static final int LONGS = BinaryVector.VECTOR_LONGS;
    public static final long TOTAL_BITS = (long) LONGS * Long.SIZE;

    private VectorMath() {
        throw new AssertionError("VectorMath is a static utility class");
    }

    public static void bind(
            final MemorySegment src1,
            final MemorySegment src2,
            final MemorySegment result) {

        for (int i = 0; i < LONGS; i++) {
            final long off  = (long) i * Long.BYTES;
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, off,
                    src1.get(ValueLayout.JAVA_LONG_UNALIGNED, off)
                            ^ src2.get(ValueLayout.JAVA_LONG_UNALIGNED, off));
        }
    }

    public static void bundle3(
            final MemorySegment v1,
            final MemorySegment v2,
            final MemorySegment v3,
            final MemorySegment result) {
        for (int i = 0; i < LONGS; i++) {
            final long off = (long) i * Long.BYTES;
            final long a   = v1.get(ValueLayout.JAVA_LONG_UNALIGNED, off);
            final long b   = v2.get(ValueLayout.JAVA_LONG_UNALIGNED, off);
            final long c   = v3.get(ValueLayout.JAVA_LONG_UNALIGNED, off);
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, off,
                    (a & b) | (a & c) | (b & c));
        }
    }

    public static void bundle(final MemorySegment[] vectors, final MemorySegment result) {
        final int n = vectors.length;
        if (n == 0) throw new IllegalArgumentException("Cannot bundle empty vector array");
        if (n == 1) {
            MemorySegment.copy(vectors[0], 0L, result, 0L, BinaryVector.VECTOR_BYTES);
            return;
        }
        final int threshold = n/2;

        final int[] bitCounts = new int[64];

        for (int w = 0; w < LONGS; w++) {
            final long byteOff = (long) w * Long.BYTES;
            for (int b = 0; b < 64; b ++)  bitCounts[b] = 0;

            for (int v = 0; v < n; v++) {
                final long word = vectors[v].get(ValueLayout.JAVA_LONG_UNALIGNED, byteOff);
                for (int b = 0; b < 64; b++) {
                    bitCounts[b] += (int) ((word >>> b) & 1L);
                }
            }

            long resultWord = 0L;
            for (int b = 0; b < 64; b++) {
                if (bitCounts[b] > threshold) {
                    resultWord |= (1L << b);
                }
            }
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, byteOff, resultWord);
        }
    }

    public static void accumulateTally(final MemorySegment src, final int[] tally) {
        for (int w = 0; w < LONGS; w++) {
            final long byteOff = (long) w * Long.BYTES;
            final long word = src.get(ValueLayout.JAVA_LONG_UNALIGNED, byteOff);
            final int  base = w * 64;
            for (int b = 0; b < 64; b++) {
                tally[base + b] += (int) ((word >>> b) & 1L);
            }
        }
    }

    public static void thresholdTally(
            final int[] tally,
            final int nVectors,
            final MemorySegment result) {
        final int threshold = nVectors / 2;
        for (int w = 0; w < LONGS; w++) {
            final int  base       = w * 64;
            final long byteOff    = (long) w * Long.BYTES;
            long resultWord = 0L;
            for (int b = 0; b < 64; b++) {
                if (tally[base + b] > threshold) {
                    resultWord |= (1L << b);
                }
            }
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, byteOff, resultWord);
        }
    }

    public static void permute1(final MemorySegment src, final MemorySegment result) {

        final long word0 = src.get(ValueLayout.JAVA_LONG_UNALIGNED, 0L);
        final long wrapBit = (word0 >>> 63) & 1L;

        for (int i = 0; i< LONGS -1; i++) {
            final long curr = src.get(ValueLayout.JAVA_LONG_UNALIGNED, (long) i * Long.BYTES);
            final long next = src.get(ValueLayout.JAVA_LONG_UNALIGNED, (long) (i + 1) * Long.BYTES);
            final long carryIn  = (next >>> 63) & 1L;
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, (long) i * Long.BYTES,
                    (curr << 1) | carryIn);
        }

        final long last = src.get(ValueLayout.JAVA_LONG_UNALIGNED, (long)(LONGS - 1) * Long.BYTES);
        result.set(ValueLayout.JAVA_LONG_UNALIGNED, (long)(LONGS - 1) * Long.BYTES,
                (last << 1) | wrapBit);
    }

    public static void permuteN(
            final MemorySegment src,
            final int n,
            final MemorySegment result,
            final MemorySegment scratch) {
        if (n == 0) {
            MemorySegment.copy(src, 0L, result, 0L, BinaryVector.VECTOR_BYTES);
            return;
        }
        if (n==1) {
            permute1(src, result);
            return;
        }

        permute1(src, result);
        for (int i = 1; i < n; i++) {
            if ((i & 1) == 1) {
                permute1(result, scratch);
            } else {
                permute1(scratch, result);
            }
        }

        if ((n & 1) == 0) {
            MemorySegment.copy(scratch, 0L, result, 0L, BinaryVector.VECTOR_BYTES);
        }
    }

    public static long hammingDistance(final MemorySegment v1, final MemorySegment v2) {
        long dist = 0L;
        for (int i=0; i<LONGS; i++) {
            final long off = (long) i * Long.BYTES;
            dist += Long.bitCount(
                    v1.get(ValueLayout.JAVA_LONG_UNALIGNED, off)
                            ^ v2.get(ValueLayout.JAVA_LONG_UNALIGNED, off));
        }
        return dist;
    }

    public static double similarity(final long hammingDist) {
        return 1.0 - ((double) hammingDist / (double) TOTAL_BITS);
    }

    public static long similarityToMaxDistance(final double minSimilarity) {
        return (long) ((1.0 - minSimilarity) * TOTAL_BITS);
    }
}
