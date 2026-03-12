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
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class SmokeTest {

    private static final int  DIMS = 128;
    private static final long SEED = ProjectionConfig.DEFAULT_SEED;

    private static final long ID_PARIS   = 1L;
    private static final long ID_FRANCE  = 2L;
    private static final long ID_LONDON  = 3L;
    private static final long ID_ENGLAND = 4L;
    private static final long ID_CAPITAL = 5L;

    @Test
    void fullPipeline() {
        final ProjectionConfig cfg = ProjectionConfig.of(SEED, DIMS);

        try (final VectorGraphRepository repo = new VectorGraphRepository(1_000L, cfg)) {

            final SymbolicReasoner reasoner = new SymbolicReasoner(repo);
            final Random rng = new Random(42L);

            final float[] capitalOf = randomGaussianUnit(rng, DIMS);
            final float[] france    = randomGaussianUnit(rng, DIMS);
            final float[] england   = randomGaussianUnit(rng, DIMS);
            final float[] paris     = l2Normalize(addScaled(france,  capitalOf, 0.8f));
            final float[] london    = l2Normalize(addScaled(england, capitalOf, 0.8f));

            repo.store(ID_PARIS,   paris);
            repo.store(ID_FRANCE,  france);
            repo.store(ID_LONDON,  london);
            repo.store(ID_ENGLAND, england);
            repo.store(ID_CAPITAL, capitalOf);

            assertEquals(5L, repo.nodeCount(),
                    "nodeCount() must equal number of stored entities");

            repo.bindRelationalEdge(ID_PARIS,  ID_FRANCE,  ID_CAPITAL);
            repo.bindRelationalEdge(ID_LONDON, ID_ENGLAND, ID_CAPITAL);

            assertEquals(7L, repo.nodeCount(),
                    "Two composite edge nodes must be added to the graph");

            final HNSWIndex.SearchResults hnswResults = repo.searchHNSW(paris, 5);

            assertTrue(hnswResults.count() > 0, "HNSW search must return results");
            assertEquals(ID_PARIS, hnswResults.entityId(0),
                    "Paris must be nearest to its own embedding");
            assertTrue(hnswResults.similarity(0) > 0.9,
                    "Self-match similarity must exceed 0.9; got " + hnswResults.similarity(0));
            for (int i = 1; i < hnswResults.count(); i++) {
                assertTrue(hnswResults.distance(i - 1) <= hnswResults.distance(i),
                        "Results must be sorted ascending by Hamming distance");
            }

            try (Arena arena = Arena.ofConfined()) {
                final RandomProjectionEncoder encoder =
                        new RandomProjectionEncoder(new OffHeapAllocator(32L), cfg);
                final MemorySegment binaryParis =
                        arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
                encoder.encode(paris, binaryParis);

                final HNSWIndex.SearchResults binaryResults =
                        repo.searchHNSWBinary(binaryParis, 3);

                assertTrue(binaryResults.count() > 0,
                        "Binary search must return results");
                assertEquals(ID_PARIS, binaryResults.entityId(0),
                        "Binary search must find Paris as nearest to its own binary vector");
            }

            final HNSWIndex.SearchResults analogyResults =
                    reasoner.queryAnalogy(ID_LONDON, ID_ENGLAND, ID_FRANCE, 5);

            assertTrue(analogyResults.count() > 0, "Analogy query must return results");

            boolean parisInTopK = false;
            for (int i = 0; i < analogyResults.count(); i++) {
                if (analogyResults.entityId(i) == ID_PARIS) {
                    parisInTopK = true;
                    break;
                }
            }
            assertTrue(parisInTopK,
                    "Paris must appear in top-5 analogy results for London:England::?:France. "
                            + "Top result was entityId=" + analogyResults.entityId(0));

            final HNSWIndex.SearchResults setResults =
                    reasoner.querySet(new long[]{ID_PARIS, ID_LONDON}, 3);

            assertTrue(setResults.count() > 0, "Set query must return results");
            for (int i = 0; i < setResults.count(); i++) {
                final double sim = setResults.similarity(i);
                assertTrue(sim >= 0.0 && sim <= 1.0,
                        "Similarity must be in [0.0, 1.0]; got " + sim);
            }

            try (Arena arena = Arena.ofConfined()) {
                final MemorySegment allOnes  = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
                final MemorySegment allZeros = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
                final MemorySegment out      = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);

                allOnes.fill((byte) 0xFF);
                allZeros.fill((byte) 0x00);

                assertEquals(VectorMath.TOTAL_BITS,
                        VectorMath.hammingDistance(allOnes, allZeros),
                        "Hamming(all-1, all-0) must equal TOTAL_BITS");

                assertEquals(0L,
                        VectorMath.hammingDistance(allOnes, allOnes),
                        "Hamming(v, v) must be 0");

                VectorMath.bind(allOnes, allZeros, out);
                assertEquals(VectorMath.TOTAL_BITS,
                        VectorMath.hammingDistance(out, allZeros),
                        "bind(all-1, all-0) must produce all-1");

                // bind is its own inverse: bind(bind(v,k),k) == v
                VectorMath.bind(allOnes, allZeros, out);
                VectorMath.bind(out, allZeros, out);
                assertEquals(0L,
                        VectorMath.hammingDistance(allOnes, out),
                        "bind must be its own inverse");

                final double simMax = VectorMath.similarity(0L);
                final double simMid = VectorMath.similarity(VectorMath.TOTAL_BITS / 2);
                final double simMin = VectorMath.similarity(VectorMath.TOTAL_BITS);
                assertEquals(1.0, simMax, 1e-9, "similarity(0) must be 1.0");
                assertEquals(0.0, simMin, 1e-9, "similarity(TOTAL_BITS) must be 0.0");
                assertTrue(simMax > simMid && simMid > simMin,
                        "similarity must be monotonically decreasing with distance");

                VectorMath.permute1(allOnes, out);
                assertEquals(0L,
                        VectorMath.hammingDistance(allOnes, out),
                        "permute1(all-1) must equal all-1");

                final MemorySegment asymmetric = arena.allocate(BinaryVector.VECTOR_BYTES, Long.BYTES);
                asymmetric.fill((byte) 0x00);
                MemorySegment.copy(allOnes, 0L, asymmetric, 0L, Long.BYTES);
                VectorMath.permute1(asymmetric, out);
                assertNotEquals(0L,
                        VectorMath.hammingDistance(asymmetric, out),
                        "permute1 must change a non-uniform vector");
            }

            repo.retract(ID_PARIS);

            assertThrows(IllegalArgumentException.class,
                    () -> repo.bindRelationalEdge(ID_PARIS, ID_FRANCE, ID_CAPITAL),
                    "Retracted entity as subject must throw");

            assertThrows(IllegalArgumentException.class,
                    () -> repo.bindRelationalEdge(ID_LONDON, ID_PARIS, ID_CAPITAL),
                    "Retracted entity as object must throw");
        }
    }

    /** Uniform random unit vector via Gaussian sampling then L2 normalisation. */
    private static float[] randomGaussianUnit(final Random rng, final int dims) {
        final float[] v = new float[dims];
        for (int i = 0; i < dims; i++) v[i] = (float) rng.nextGaussian();
        return l2Normalize(v);
    }

    /** Returns a new array: {@code a + scale * b}. Does not mutate inputs. */
    private static float[] addScaled(final float[] a, final float[] b, final float scale) {
        final float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] + scale * b[i];
        return result;
    }

    /** Mutates {@code v} in-place to unit length and returns it. */
    private static float[] l2Normalize(final float[] v) {
        float sumSq = 0f;
        for (final float x : v) sumSq += x * x;
        final float norm = (float) Math.sqrt(sumSq);
        if (norm > 0f) for (int i = 0; i < v.length; i++) v[i] /= norm;
        return v;
    }
}
