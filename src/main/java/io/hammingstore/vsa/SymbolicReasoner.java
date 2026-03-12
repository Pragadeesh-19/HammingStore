package io.hammingstore.vsa;

import io.hammingstore.graph.VectorGraphRepository;
import io.hammingstore.hnsw.HNSWIndex;
import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.memory.SparseEntityIndex;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Executes VSA (Vector Symbolic Architecture) reasoning queries over stored
 * binary hypervector's.
 *
 * <p>VSA reasoning works by composing entity vectors algebraically and searching
 * for the nearest stored vector to the result. Three query types are supported:
 * <ul>
 *     <li>{@link #queryAnalogy} - analogical reasoning: A:B :: C:? via XOR arithmetic</li>
 *     <li>{@link #queryRelation} - relation decoding: given a relation and object, find the subject</li>
 *     <li>{@link #querySet} - set membership: bundle N entity vectors, find the nearest concept</li>
 * </ul>
 *
 * <p>All query methods are thread-safe. Each thread maintains its own
 * {@link ReasonerScratch} via a {@link ThreadLocal}, avoiding allocation and
 * synchronisation on the hot query path.
 *
 * <p>Note: these queries operate on VSA hypervector's stored via
 * {@code StoreBinary} or {@code BindEdge}. Entities stored via
 * {@code StoreFloat} use neural embeddings and should be queried via
 * {@code SearchFloat} instead.
 *
 */
public final class SymbolicReasoner {

    /**
     * Per-thread scratch buffers for intermediate VSA computation.
     * Fields are package-private: they are an implementation detail of
     * {@link SymbolicReasoner} and must not be accessed directly by callers.
     */
    public static final class ReasonerScratch {
        public final MemorySegment scratchA;
        public final MemorySegment scratchB;

        /**
         * Per-bit vote tally for majority-vote bundling.
         * Length = {@link BinaryVector#VECTOR_LONGS} × 64 = 10,048 ints ≈ 40,192 bytes.
         */
        public final int[] tally;

        public ReasonerScratch(final OffHeapAllocator allocator) {
            this.scratchA = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.scratchB = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.tally    = new int[BinaryVector.VECTOR_LONGS * 64];
        }
    }

    private final VectorGraphRepository repository;
    private final ThreadLocal<ReasonerScratch> scratchPool;

    /**
     * Creates a reasoner backed by {@code repository}.
     *
     * @param repository the repository to resolve entity vectors from and search against
     * @throws NullPointerException if {@code repository} is null
     */
    public SymbolicReasoner(final VectorGraphRepository repository) {
        Objects.requireNonNull(repository, "repository must not be null");
        this.repository  = repository;
        this.scratchPool = ThreadLocal.withInitial(() ->
                new ReasonerScratch(new OffHeapAllocator(16L))
        );
    }

    /**
     * Analogical reasoning via XOR: A:B :: C:?
     *
     * <p>Computes {@code bind(bind(subjectA, objectA), subjectB)} and searches
     * for the nearest stored vector. In a well-formed VSA knowledge graph, the
     * result should be the entity that stands in the same relation to
     * {@code subjectB} as {@code objectA} stands to {@code subjectA}.
     *
     * <p>Example: London:England :: ?:France → Paris
     *
     * @param subjectAId entity ID of A (e.g. London)
     * @param objectAId  entity ID of B (e.g. England)
     * @param subjectBId entity ID of C (e.g. France)
     * @param k number of results to return
     * @return the k nearest entities to the analogy vector
     * @throws IllegalArgumentException if any entity ID is not found in the index
     */
    public HNSWIndex.SearchResults queryAnalogy(
            final long subjectAId,
            final long objectAId,
            final long subjectBId,
            final int k) {

        final ReasonerScratch ctx = scratchPool.get();
        bindAnalogy(subjectAId, objectAId, subjectBId, ctx.scratchA, ctx.scratchB);
        return repository.searchHNSWBinary(ctx.scratchA, k);
    }

    /**
     * Relation decoding: given a relation and an object, find the subject.
     *
     * <p>Computes {@code bind(relation, object)} and searches for the nearest
     * stored vector. Used to answer queries of the form "what X has relation R to Y?"
     *
     * @param relationId entity ID of the relation hypervector
     * @param objectId   entity ID of the object
     * @param k number of results to return
     * @return the k nearest entities to the decoded subject vector
     * @throws IllegalArgumentException if any entity ID is not found in the index
     */
    public HNSWIndex.SearchResults queryRelation(
            final long relationId,
            final long objectId,
            final int k) {

        final ReasonerScratch ctx = scratchPool.get();
        final MemorySegment relation = resolveVector(relationId, "relation");
        final MemorySegment object   = resolveVector(objectId,   "object");
        VectorMath.bind(relation, object, ctx.scratchA);
        return repository.searchHNSWBinary(ctx.scratchA, k);
    }

    /**
     * Set reasoning: bundle N entity vectors and find the nearest concept.
     *
     * <p>Accumulates a majority-vote tally across all entity vectors and searches
     * for the nearest stored vector to the result. The bundled vector is similar
     * to all its constituents, so the result represents the "prototype" of the set.
     *
     * @param entityIds non-empty array of entity IDs to bundle
     * @param k number of results to return
     * @return the k nearest entities to the bundled set vector
     * @throws IllegalArgumentException if {@code entityIds} is null or empty,
     *                                  or if any entity ID is not found in the index
     */
    public HNSWIndex.SearchResults querySet(final long[] entityIds, final int k) {
        if (entityIds == null || entityIds.length == 0) {
            throw new IllegalArgumentException("entityIds must be non-empty");
        }

        final ReasonerScratch ctx = scratchPool.get();

        final int tallyLen = ctx.tally.length;
        for (int t = 0; t < tallyLen; t++) ctx.tally[t] = 0;

        for (final long entityId : entityIds) {
            final MemorySegment vec = resolveVector(entityId, "entity[" + entityId + "]");
            VectorMath.accumulateTally(vec, ctx.tally);
        }

        final int effectiveCount = entityIds.length % 2 == 0
                ? entityIds.length + 1
                : entityIds.length;
        VectorMath.thresholdTally(ctx.tally, effectiveCount, ctx.scratchA);

        return repository.searchHNSWBinary(ctx.scratchA, k);
    }

    /**
     * Computes the analogy vector {@code bind(bind(subjectA, objectA), subjectB)}
     * and writes the result into {@code result}, using {@code scratch} as a
     * temporary buffer.
     */
    private void bindAnalogy(
            final long subjectAId,
            final long objectAId,
            final long subjectBId,
            final MemorySegment result,
            final MemorySegment scratch) {

        final MemorySegment subjectA = resolveVector(subjectAId, "subjectA");
        final MemorySegment objectA = resolveVector(objectAId, "objectA");
        final MemorySegment subjectB = resolveVector(subjectBId, "subjectB");

        VectorMath.bind(subjectA, objectA, scratch);
        VectorMath.bind(scratch, subjectB, result);
    }

    /**
     * Looks up the binary hypervector for {@code entityId} in the vector store.
     *
     * @param entityId the entity whose vector to retrieve
     * @param role human-readable name for this entity's role in the query,
     *             used in error messages
     * @return the hypervector segment for the entity
     * @throws IllegalArgumentException if the entity is not found in the index
     */
    private MemorySegment resolveVector(final long entityId, final String role) {
        final long hash   = SparseEntityIndex.mixHash64(entityId);
        final long offset = repository.entityIndex().getOffset(hash);
        if (offset < 0L) {
            throw new IllegalArgumentException(
                    "Entity not found in index for role=" + role + " id=" + entityId);
        }
        return repository.vectorStore().sliceAt(offset);
    }

}
