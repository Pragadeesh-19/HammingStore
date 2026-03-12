package io.hammingstore.vsa;

import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Zero-ML text encoder: converts raw text into binary hypervectors using
 * Hyperdimensional Computing (HDC) character n-gram encoding.
 *
 * <h2>Why this class exists</h2>
 * <p>The primary encoding path in this engine uses float embeddings from a neural
 * model (e.g. MiniLM) compressed via {@link RandomProjectionEncoder}. That path
 * requires an external model and cannot handle domains where no pretrained model
 * exists — DNA sequences, machine log files, proprietary identifier schemes, etc.
 *
 * <p>This encoder provides a fully self-contained alternative: no model download,
 * no Python dependency, no GPU. It operates purely on byte n-grams and is
 * therefore suitable for any domain where character-level similarity is meaningful.
 *
 * <h2>How it works</h2>
 * <ol>
 *   <li><b>Item memory:</b> each byte value (0–255) is assigned a unique random
 *       hypervector at construction time. These vectors form the alphabet.</li>
 *   <li><b>N-gram binding:</b> for each n-gram of bytes, the character vectors are
 *       XOR-bound with positional permutations so that "abc" and "bca" produce
 *       different vectors.</li>
 *   <li><b>Bundling:</b> all n-gram vectors are combined via majority-vote tally
 *       into a single hypervector representing the full text.</li>
 * </ol>
 *
 * <p>Texts with similar byte n-gram content produce similar (low Hamming distance)
 * hypervectors, enabling fuzzy string matching and language-agnostic semantic hashing.
 *
 * <h2>Current status - not yet wired into the gRPC server</h2>
 * <p>This encoder is intentionally kept but not yet exposed as a server-side
 * encoding path. To make it accessible to clients, the following work is needed:
 * <ul>
 *   <li>Add a {@code --encoder=hdc} flag to {@code HammingServer} startup args.</li>
 *   <li>Add {@code StoreText} / {@code SearchText} RPCs to {@code hammingstore.proto}.</li>
 *   <li>Wire those RPCs through {@code HammingGrpcService} to this encoder.</li>
 * </ul>
 *
 * <p>Instances are thread-safe for concurrent {@link #encode} calls provided
 * each calling thread supplies its own {@link EncoderScratch} buffer.
 */
public final class HDCTextEncoder {

    /**
     * Number of distinct byte values in the item memory alphabet: {@value}.
     * Covers the full unsigned byte range (0–255).
     */
    public static final int ALPHABET_SIZE = 256;

    /**
     * Recommended n-gram size for general-purpose text encoding: {@value}
     *
     * <p>Trigrams balance vocabulary coverage against sensitivity.
     * Pass this value to {@link #encode} unless you have a specific reason
     * to use a different n-gram size.
     */
    public static final int DEFAULT_NGRAM = 3;

    /**
     * Default seed used when no seed is provided.
     * Any fixed value works; this one is arbitrary.
     */
    private static final long DEFAULT_SEED = 0xDEADBEEFCAFEL;

    /**
     * Total byte size of the item memory: one hypervector per alphabet character.
     * {@value ALPHABET_SIZE} x {@link BinaryVector#VECTOR_BYTES} bytes
     */
    private static final long ITEM_MEMORY_BYTES =
            (long) ALPHABET_SIZE * BinaryVector.VECTOR_BYTES;

    /**
     * Name of the preferred PRNG algorithm for item memory generation.
     * Xoshiro256StarStar is fast and has excellent statistical properties.
     */
    private static final String PREFERRED_RNG = "Xoshiro256StarStar";

    private final MemorySegment itemMemory;

    /**
     * Creates an encoder whose item memory is seeded with the given value.
     *
     * <p>Two encoder created with the same {@code seed} will produce identical
     * item memories and therefore output hypervectors for the same input.
     *
     * @param allocator the allocator from which request item memory storage
     * @param seed the PRNG seed for item memory generation
     * @throws IllegalStateException if no suitable PRNG is available on this JVM.
     */
    public HDCTextEncoder(final OffHeapAllocator allocator, final long seed) {
        this.itemMemory = allocator.allocateRawSegment(ITEM_MEMORY_BYTES, Long.BYTES);
        populateItemMemory(seed);
    }

    public HDCTextEncoder(final OffHeapAllocator allocator) {
        this(allocator, DEFAULT_SEED);
    }

    public static EncoderScratch allocateScratch(final OffHeapAllocator allocator) {
        return new EncoderScratch(allocator);
    }

    /**
     * Pre-threaded scratch buffers for intermediate encoding steps.
     *
     * <p>Keeping these off-heap avoids GC pressure on the encode hot path.
     * Fields are package-private: they are an implementation detail of
     * {@link HDCTextEncoder} and should not be accessed directly by callers.
     */
    public static final class EncoderScratch {

        public final MemorySegment permA;
        public final MemorySegment permB;
        public final MemorySegment ngramVec;

        /**
         * Per-bit vote tally across all n-grams
         * Length = {@link BinaryVector#VECTOR_LONGS} x 64 = 10,048 ints ~ 40,192 byte
         */
        public final int[] tally;

        private EncoderScratch(final OffHeapAllocator allocator) {
            this.permA    = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.permB    = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.ngramVec = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.tally    = new int[BinaryVector.VECTOR_LONGS * 64]; // 10,048 ints = 40 KB
        }
    }

    /**
     * Encodes {@code text} into binary hypervector using n-gram binding.
     *
     * <p>The result is written into {@code target}. The {@code scratch} buffer is
     * used for intermediate computation and its contents after this call are
     * undefined.
     *
     * @param text the text to encode; must not be null
     * @param nGramSize the n-gram size; must be >= 1. use {@link #DEFAULT_NGRAM} is unsure.
     * @param target destination segment; must be exactly {@link BinaryVector#VECTOR_BYTES} bytes
     * @param scratch if {@code nGramSize} is less than 1
     */
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

    /**
     * Returns the item memory vector for the given byte value.
     *
     * <p>The returned segment is a view into the item memory - not a copy
     * callers must write to it.
     *
     * @param byteValue the byte value in [0, 255]
     * @return a read only view of the item vector for that byte.
     */
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
        final RandomGenerator rng;
        try {
            rng = RandomGeneratorFactory.of(PREFERRED_RNG).create(seed);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(
                    "Required PRNG '" + PREFERRED_RNG + "' is not available on this JVM. "
                            + "Ensure you are running Java 17 or later.", e);
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
