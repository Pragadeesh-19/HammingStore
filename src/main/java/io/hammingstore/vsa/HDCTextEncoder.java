package io.hammingstore.vsa;

import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class HDCTextEncoder {

    public static final int ALPHABET_SIZE = 256;

    public static final int DEFAULT_NGRAM = 3;

    private static final long ITEM_MEMORY_BYTES =
            (long) ALPHABET_SIZE * BinaryVector.VECTOR_BYTES;

    private final MemorySegment itemMemory;

    public HDCTextEncoder(final OffHeapAllocator allocator, final long seed) {
        this.itemMemory = allocator.allocateRawSegment(ITEM_MEMORY_BYTES, Long.BYTES);
        populateItemMemory(seed);
    }

    public HDCTextEncoder(final OffHeapAllocator allocator) {
        this(allocator, 0xDEADBEEFCAFEL);
    }

    public static EncoderScratch allocateScratch(final OffHeapAllocator allocator) {
        return new EncoderScratch(allocator);
    }

    public static final class EncoderScratch {

        public final MemorySegment permA;
        public final MemorySegment permB;
        public final MemorySegment ngramVec;
        public final int[] tally;

        private EncoderScratch(final OffHeapAllocator allocator) {
            this.permA    = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.permB    = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.ngramVec = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.tally    = new int[BinaryVector.VECTOR_LONGS * 64]; // 10,048 ints = 40 KB
        }
    }

    public void encode(
            final String text,
            final int nGramSize,
            final MemorySegment target,
            final EncoderScratch scratch) {
        encode(text.getBytes(StandardCharsets.UTF_8), nGramSize, target, scratch);
    }

    private void encode(
            final byte[] bytes,
            final int nGramSize,
            final MemorySegment target,
            final EncoderScratch scratch) {

        if (bytes == null || bytes.length == 0) {
            target.fill((byte) 0);
            return;
        }

        if (nGramSize < 1) throw new IllegalArgumentException("nGramSize must be >= 1");

        final int tallyLen = scratch.tally.length;
        for (int t = 0; t < tallyLen; t++) scratch.tally[t] = 0;

        final int nGrams   = bytes.length - nGramSize + 1;
        int nGramCount = 0;

        if (nGrams <= 0) {
            final long charVecOffset = charOffset(bytes[0] & 0xFF);
            final MemorySegment charVec = itemMemory.asSlice(charVecOffset, BinaryVector.VECTOR_BYTES);
            VectorMath.accumulateTally(charVec, scratch.tally);
            nGramCount = 1;
        } else {
            for (int i = 0; i < nGrams; i++) {
                buildNGramVector(bytes, i, nGramSize, scratch);
                VectorMath.accumulateTally(scratch.ngramVec, scratch.tally);
                nGramCount++;
            }
        }

        final int effectiveCount = (nGramCount % 2 == 0) ? nGramCount + 1 : nGramCount;
        VectorMath.thresholdTally(scratch.tally, effectiveCount, target);
    }

    public MemorySegment itemVector(final int byteValue) {
        return itemMemory.asSlice(charOffset(byteValue & 0xFF), BinaryVector.VECTOR_BYTES);
    }

    private void buildNGramVector(
            final byte[] bytes,
            final int offset,
            final int nGramSize,
            final EncoderScratch scratch) {

        final int lastCharIdx = offset + nGramSize - 1;
        MemorySegment.copy(
                itemMemory, charOffset(bytes[lastCharIdx] & 0xFF),
                scratch.ngramVec, 0L,
                BinaryVector.VECTOR_BYTES);

        for (int j = nGramSize - 2; j >= 0; j--) {
            final int charVal   = bytes[offset + j] & 0xFF;
            final int permCount = nGramSize - 1 - j;

            final MemorySegment charVec =
                    itemMemory.asSlice(charOffset(charVal), BinaryVector.VECTOR_BYTES);

            VectorMath.permuteN(charVec, permCount, scratch.permA, scratch.permB);
            VectorMath.bind(scratch.ngramVec, scratch.permA, scratch.ngramVec);
        }
    }

    private static long charOffset(final int c) {
        return (long) c * BinaryVector.VECTOR_BYTES;
    }

    private void populateItemMemory(final long seed) {
        RandomGenerator rng;
        try {
            rng = RandomGeneratorFactory.of("Xoshiro256StarStar").create(seed);
        } catch (IllegalArgumentException e) {
            rng = RandomGeneratorFactory.of("Random").create(seed);
        }

        for (int c = 0; c < ALPHABET_SIZE; c++) {
            final long base = charOffset(c);
            for (int w = 0; w < BinaryVector.VECTOR_LONGS; w++) {
                itemMemory.set(ValueLayout.JAVA_LONG_UNALIGNED,
                        base + (long) w * Long.BYTES,
                        rng.nextLong());
            }
        }
    }
}
