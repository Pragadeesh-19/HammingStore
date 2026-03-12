package io.hammingstore.math;

import io.hammingstore.memory.BinaryVector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Static utility methods for binary hypervector arithmetic.
 *
 * <p>All operations work directly on {@link MemorySegment} instances representing
 * packed 10,048-bit hypervector's ({@value BinaryVector#VECTOR_LONGS} x 64-bit longs),
 * avoiding heap allocations on hot search and encode paths.
 *
 * <p><b>VSA operation summary:</b>
 * <ul>
 *   <li>{@link #bind} - XOR binding. Encodes associations; result is dissimilar to inputs.</li>
 *   <li>{@link #accumulateTally} / {@link #thresholdTally} - Streaming majority vote bundling
 *        Encodes sets; result is similar to all inputs.</li>
 *   <li>{@link #permute1} / {@link #permuteN} - Cyclic left-shift. Encodes sequence
 *       position; permuted vectors are dissimilar to the original.</li>
 *   <li>{@link #hammingDistance} - Measures dissimilarity between two hypervector's.</li>
 *   <li>{@link #similarity} - Converts Hamming distance to a [0, 1] similarity score.</li>
 * </ul>
 */
public final class VectorMath {

    /** Number of 64-bit longs per hypervector. Used only within this class. */
    static final int LONGS = BinaryVector.VECTOR_LONGS;

    /**
     * Total number of bits per hypervector: {@value}.
     * Used externally for similarity normalisation and binarisation bounds.
     */
    public static final long TOTAL_BITS = (long) LONGS * Long.SIZE;

    private VectorMath() {
        throw new AssertionError("VectorMath is a static utility class");
    }

    /**
     * Binds two hypervectors via bitwise XOR, storing the result in {@code result}.
     *
     * <p>Binding is the VSA operation for encoding associations (e.g., key -> value).
     * The result is approximately orthogonal to both inputs and can be unbound by
     * XOR-ing with either operand: {@code bind(bind(a, b), b) ~ a}.
     *
     * <p>All three segments must be exactly {@link BinaryVector#VECTOR_BYTES} bytes.
     * {@code result} may alias {@code src1} or {@code src2}.
     *
     * @param src1   first operand
     * @param src2   second operand
     * @param result destination segment; receives {@code src1 XOR src2}
     */
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

    /**
     * Accumulates per-bit vote counts from {@code src} into {@code tally}.
     *
     * <p>This is the incremental form of bundling: call this once per input vector,
     * then call {@link #thresholdTally} to finalise. Useful when vectors are streamed
     * and cannot all be held in memory simultaneously.
     *
     * <p>{@code tally} must have length ≥ {@link #TOTAL_BITS}.
     *
     * <p><em>Note: verify this method is still in use before the next refactor pass.
     * It was not called in any reviewed file at the time of this clean-up.</em>
     *
     * @param src   the hypervector to tally
     * @param tally per-bit vote accumulator; updated in place
     */
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

    /**
     * Converts a completed per-bit tally into a bundled hypervector via majority vote.
     *
     * <p>A result bit is {@code 1} if strictly more than {@code nVectors / 2} votes
     * were accumulated for that bit. Pair with {@link #accumulateTally} for streaming
     * bundle over large collections.
     *
     * <p><em>Note: verify this method is still in use before the next refactor pass.
     * It was not called in any reviewed file at the time of this clean-up.</em>
     *
     * @param tally per-bit vote counts, as filled by {@link #accumulateTally}
     * @param nVectors total number of vectors that were tallied
     * @param result destination segment; receives the majority-vote hypervector
     */
    public static void thresholdTally(
            final int[] tally,
            final int nVectors,
            final MemorySegment result) {
        final int threshold = nVectors / 2;
        for (int w = 0; w < LONGS; w++) {
            final int  base = w * 64;
            final long byteOff = (long) w * Long.BYTES;
            long resultWord = 0L;
            for (int b = 0; b < 64; b++) {
                if (tally[base + b] > threshold) {
                    resultWord |= (1L << b);
                }
            }
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, byteOff, resultWord);
        }
    }

    /**
     * Applies a single cyclic left-shift permutation to {@code src}, writing the
     * result to {@code result}.
     *
     * <p>Permutation is the VSA operation for encoding sequence position. A vector
     * permuted {@code n} times encodes position {@code n} and is approximately
     * orthogonal to the same vector permuted any other number of times.
     *
     * <p>The shift wraps the most-significant bit of the first word around to the
     * least-significant bit of the last word, preserving the total bit count.
     *
     * <p>{@code result} must not alias {@code src}.
     *
     * <p><em>Note: verify this method is still in use before the next refactor pass.
     * It was not called in any reviewed file at the time of this clean-up.</em>
     *
     * @param src the hypervector to permute
     * @param result destination segment; receives the permuted hypervector
     */
    public static void permute1(final MemorySegment src, final MemorySegment result) {

        final long word0 = src.get(ValueLayout.JAVA_LONG_UNALIGNED, 0L);
        final long wrapBit = (word0 >>> 63) & 1L;

        for (int i = 0; i< LONGS -1; i++) {
            final long curr = src.get(ValueLayout.JAVA_LONG_UNALIGNED, (long) i * Long.BYTES);
            final long next = src.get(ValueLayout.JAVA_LONG_UNALIGNED, (long) (i + 1) * Long.BYTES);
            final long carryIn = (next >>> 63) & 1L;
            result.set(ValueLayout.JAVA_LONG_UNALIGNED, (long) i * Long.BYTES,
                    (curr << 1) | carryIn);
        }

        final long last = src.get(ValueLayout.JAVA_LONG_UNALIGNED, (long)(LONGS - 1) * Long.BYTES);
        result.set(ValueLayout.JAVA_LONG_UNALIGNED, (long)(LONGS - 1) * Long.BYTES,
                (last << 1) | wrapBit);
    }

    /**
     * Applies {@code n} cyclic left-shift permutations to {@code src}, writing the
     * final result to {@code result}.
     *
     * <p>For {@code n == 0}, {@code result} receives an exact copy of {@code src}.
     * For {@code n == 1}, delegates directly to {@link #permute1}.
     * For {@code n > 1}, alternates between {@code result} and {@code scratch} to
     * avoid allocating intermediate buffers.
     *
     * <p>{@code result} and {@code scratch} must not alias each other or {@code src}.
     *
     * <p><em>Note: verify this method is still in use before the next refactor pass.
     * It was not called in any reviewed file at the time of this clean-up.</em>
     *
     * @param src the hypervector to permute
     * @param n number of cyclic left-shift positions; must be ≥ 0
     * @param result destination segment; receives the permuted hypervector
     * @param scratch temporary buffer; must be {@link BinaryVector#VECTOR_BYTES} bytes
     */
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

    /**
     * Computes the Hamming distance between two hypervectors.
     *
     * <p>Hamming distance is the number of bit positions at which the two vectors
     * differ. The maximum possible distance for these vectors is {@link #TOTAL_BITS}.
     *
     * @param v1 first hypervector; must be exactly {@link BinaryVector#VECTOR_BYTES} bytes
     * @param v2 second hypervector; must be exactly {@link BinaryVector#VECTOR_BYTES} bytes
     * @return the number of differing bits in [{@code 0}, {@link #TOTAL_BITS}]
     */
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

    /**
     * Converts a Hamming distance to a normalised similarity score in [{@code 0.0}, {@code 1.0}].
     *
     * <p>A distance of {@code 0} returns {@code 1.0} (identical vectors).
     * A distance of {@link #TOTAL_BITS} returns {@code 0.0} (maximally dissimilar).
     * Random hypervectors are expected to have distance ≈ {@code TOTAL_BITS / 2},
     * yielding similarity ≈ {@code 0.5}.
     *
     * @param hammingDist the Hamming distance, as returned by {@link #hammingDistance}
     * @return the similarity score
     */
    public static double similarity(final long hammingDist) {
        return 1.0 - ((double) hammingDist / (double) TOTAL_BITS);
    }

    /**
     * Converts a minimum similarity threshold to the maximum Hamming distance that
     * satisfies it.
     *
     * <p>Use this to convert a user-facing similarity cutoff into a distance bound
     * for range queries.
     *
     * @param minSimilarity the minimum acceptable similarity in [{@code 0.0}, {@code 1.0}]
     * @return the maximum Hamming distance corresponding to {@code minSimilarity}
     */
    public static long similarityToMaxDistance(final double minSimilarity) {
        return (long) ((1.0 - minSimilarity) * TOTAL_BITS);
    }
}
