import io.hammingstore.graph.VectorGraphRepository;
import io.hammingstore.hnsw.HNSWIndex;
import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.vsa.ProjectionConfig;
import io.hammingstore.vsa.RandomProjectionEncoder;
import io.hammingstore.vsa.SymbolicReasoner;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end smoke test for the full engine pipeline.
 *
 * Tests are split by vector type:
 *
 *   Float path  — uses the RandomProjectionEncoder (LSH). Tests HNSW search
 *                 and float ingestion. Geometric similarity is preserved by
 *                 the projection; Euclidean vector arithmetic is valid here.
 *
 *   Binary path — uses storeBinary with explicitly constructed hypervectors.
 *                 VSA operations (bind, analogy, queryChain) require pure binary
 *                 input. Applying XOR arithmetic to LSH-projected float vectors
 *                 is mathematically unsound because the projection is not
 *                 homomorphic over XOR.
 */
class SmokeTest {

    private static final int  DIMS = 128;
    private static final long SEED = ProjectionConfig.DEFAULT_SEED;

    // Float-path entity IDs
    private static final long F_PARIS   = 10L;
    private static final long F_FRANCE  = 11L;
    private static final long F_LONDON  = 12L;
    private static final long F_ENGLAND = 13L;

    // Binary-path entity IDs (separate namespace)
    private static final long B_PARIS    = 20L;
    private static final long B_FRANCE   = 21L;
    private static final long B_LONDON   = 22L;
    private static final long B_ENGLAND  = 23L;
    private static final long B_CAPITAL  = 24L;
    private static final long B_EUROPE   = 25L;
    private static final long B_LOCATED  = 26L;

    @Test
    void floatStoreAndHnswSearch() {
        final ProjectionConfig cfg = ProjectionConfig.of(SEED, DIMS);
        try (final VectorGraphRepository repo = new VectorGraphRepository(500L, cfg)) {

            final Random rng = new Random(42L);
            final float[] paris  = gaussianUnit(rng, DIMS);
            final float[] france = gaussianUnit(rng, DIMS);
            final float[] london = gaussianUnit(rng, DIMS);
            repo.store(F_PARIS,   paris);
            repo.store(F_FRANCE,  france);
            repo.store(F_LONDON,  london);

            assertEquals(3L, repo.nodeCount());

            final HNSWIndex.SearchResults r = repo.searchHNSW(paris, 3);
            assertTrue(r.count() > 0);
            assertEquals(F_PARIS, r.entityId(0), "Paris must be nearest to itself");
            assertTrue(r.similarity(0) > 0.9);

            for (int i = 1; i < r.count(); i++) {
                assertTrue(r.distance(i - 1) <= r.distance(i));
            }

            try (Arena arena = Arena.ofConfined()) {
                final RandomProjectionEncoder enc =
                        new RandomProjectionEncoder(new OffHeapAllocator(32L), cfg);
                final MemorySegment bin =
                        arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
                enc.encode(paris, bin);

                final HNSWIndex.SearchResults rb = repo.searchHNSWBinary(bin, 3);
                assertTrue(rb.count() > 0);
                assertEquals(F_PARIS, rb.entityId(0));
            }
        }
    }

    @Test
    void binaryVsaPipeline() {
        try (
                Arena arena = Arena.ofConfined();
                VectorGraphRepository repo  = new VectorGraphRepository(2_000L)
        ) {
            final SymbolicReasoner reasoner = new SymbolicReasoner(repo);

            final MemorySegment france   = randomBinary(arena, 1L);
            final MemorySegment england  = randomBinary(arena, 2L);
            final MemorySegment europe   = randomBinary(arena, 3L);
            final MemorySegment capitalOf = randomBinary(arena, 4L);
            final MemorySegment locatedIn = randomBinary(arena, 5L);
            final MemorySegment paris    = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
            final MemorySegment london   = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);

            VectorMath.bind(france,  capitalOf, paris);
            VectorMath.bind(england, capitalOf, london);

            repo.storeBinary(B_FRANCE,  france);
            repo.storeBinary(B_ENGLAND, england);
            repo.storeBinary(B_EUROPE,  europe);
            repo.storeBinary(B_CAPITAL, capitalOf);
            repo.storeBinary(B_LOCATED, locatedIn);
            repo.storeBinary(B_PARIS,   paris);
            repo.storeBinary(B_LONDON,  london);

            assertEquals(7L, repo.nodeCount());

            repo.bindRelationalEdge(B_PARIS,  B_FRANCE,  B_CAPITAL);
            repo.bindRelationalEdge(B_LONDON, B_ENGLAND, B_CAPITAL);
            assertEquals(9L, repo.nodeCount());

            final HNSWIndex.SearchResults analogy =
                    reasoner.queryAnalogy(B_LONDON, B_ENGLAND, B_FRANCE, 5);
            assertTrue(analogy.count() > 0);

            boolean parisInTop5 = false;
            for (int i = 0; i < analogy.count(); i++) {
                if (analogy.entityId(i) == B_PARIS) { parisInTop5 = true; break; }
            }
            assertTrue(parisInTop5,
                    "Paris must appear in analogy top-5. Top result: " + analogy.entityId(0));

            final HNSWIndex.SearchResults set =
                    reasoner.querySet(new long[]{B_PARIS, B_LONDON}, 3);
            assertTrue(set.count() > 0);
            for (int i = 0; i < set.count(); i++) {
                final double sim = set.similarity(i);
                assertTrue(sim >= 0.0 && sim <= 1.0);
            }

            repo.storeTypedEdge(B_PARIS,  B_CAPITAL, B_FRANCE);
            repo.storeTypedEdge(B_LONDON, B_CAPITAL, B_ENGLAND);
            repo.storeTypedEdge(B_FRANCE, B_LOCATED, B_EUROPE);

            final HNSWIndex.SearchResults hop =
                    reasoner.queryHop(B_PARIS, B_CAPITAL, 5);
            assertTrue(hop.count() > 0);
            for (int i = 0; i < hop.count(); i++) {
                assertTrue(hop.similarity(i) >= 0.0 && hop.similarity(i) <= 1.0);
            }

            final HNSWIndex.SearchResults chain =
                    reasoner.queryChain(B_PARIS, new long[]{B_CAPITAL, B_LOCATED}, 5);
            assertTrue(chain.count() > 0);
            for (int i = 0; i < chain.count(); i++) {
                assertTrue(chain.similarity(i) >= 0.0 && chain.similarity(i) <= 1.0);
            }

            assertThrows(IllegalArgumentException.class,
                    () -> reasoner.queryChain(B_PARIS, new long[]{}, 3));

            repo.retract(B_PARIS);
            assertThrows(IllegalArgumentException.class,
                    () -> repo.bindRelationalEdge(B_PARIS, B_FRANCE, B_CAPITAL));
            assertThrows(IllegalArgumentException.class,
                    () -> repo.storeTypedEdge(B_PARIS, B_CAPITAL, B_FRANCE));
        }
    }

    @Test
    void vectorMathPrimitives() {
        try (Arena arena = Arena.ofConfined()) {
            final MemorySegment allOnes = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
            final MemorySegment allZeros = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
            final MemorySegment out = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);

            allOnes.fill((byte) 0xFF);
            allZeros.fill((byte) 0x00);

            assertEquals(VectorMath.TOTAL_BITS, VectorMath.hammingDistance(allOnes, allZeros));
            assertEquals(0L, VectorMath.hammingDistance(allOnes, allOnes));

            VectorMath.bind(allOnes, allZeros, out);
            VectorMath.bind(out, allZeros, out);
            assertEquals(0L, VectorMath.hammingDistance(allOnes, out));

            assertEquals(1.0, VectorMath.similarity(0L), 1e-9);
            assertEquals(0.0, VectorMath.similarity(VectorMath.TOTAL_BITS), 1e-9);
            assertTrue(VectorMath.similarity(VectorMath.TOTAL_BITS / 4)
                    > VectorMath.similarity(VectorMath.TOTAL_BITS / 2));

            final MemorySegment asymmetric = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
            asymmetric.fill((byte) 0x00);
            asymmetric.set(ValueLayout.JAVA_LONG_UNALIGNED, 0L, -1L); // first 8 bytes = 0xFF
            VectorMath.permute1(asymmetric, out);
            assertNotEquals(0L, VectorMath.hammingDistance(asymmetric, out));
        }
    }


    /**
     * Allocates a random binary hypervector from a seeded RNG.
     * The seed ensures reproducibility across runs.
     */
    private static MemorySegment randomBinary(final Arena arena, final long seed) {
        final MemorySegment seg = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
        final Random rng = new Random(seed);
        for (int i = 0; i < BinaryVector.VECTOR_LONGS; i++) {
            seg.set(ValueLayout.JAVA_LONG_UNALIGNED, (long) i * Long.BYTES, rng.nextLong());
        }
        return seg;
    }

    private static float[] gaussianUnit(final Random rng, final int dims) {
        final float[] v = new float[dims];
        float sumSq = 0f;
        for (int i = 0; i < dims; i++) {
            v[i] = (float) rng.nextGaussian();
            sumSq += v[i] * v[i];
        }
        final float norm = (float) Math.sqrt(sumSq);
        for (int i = 0; i < dims; i++) v[i] /= norm;
        return v;
    }
}
