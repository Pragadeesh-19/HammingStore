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
 * <h2>Query Types</h2>
 * <ul>
 *     <li>{@link #queryAnalogy} - analogical reasoning: A:B :: C:? via XOR</li>
 *     <li>{@link #queryRelation} - relation decoding: given relation+object, find subject</li>
 *     <li>{@link #querySet} - set prototype: bundle N entities, find nearest concept</li>
 *     <li>{@link #queryTypedRelation} - typed edge query using role-encoded binding</li>
 *     <li>{@link #queryHop} - single-hop traversal along a typed edge</li>
 *     <li>{@link #queryChain} - multi-hop chain: follow a sequence of typed edges</li>
 * </ul>
 *
 * <h2>Typed relations - role-encoded binding</h2>
 * <p>Untyped XOR binding ({@code subject XOR relation XOR object}) places all
 * triples in the smae vector space. Two different relation types interfere during
 * decoding whenever their relation vectors are not perfectly orthogonal - and
 * random hypervectors are only approximately orthogonal.
 *
 * <p>Role-encoded binding applies a deterministic cyclic bit permutation to
 * each participant before combining.
 *
 * <pre>
 *     edge = bind(permute(subject,  SUBJECT_SHIFT),
 *             bind(permute(relation, RELATION_SHIFT),
 *                  permute(object,   OBJECT_SHIFT)))
 * </pre>
 * Subject, relation, and object occupy disjoint bit regions. To decode the
 * object given a known subject and relation:
 * <pre>
 *     query  = bind(permute(subject, SUBJECT_SHIFT), permute(relation, RELATION_SHIFT))
 *     result ≈ nearest(query)   (object is the stored vector closest to query)
 * </pre>
 *
 * <h2>Multi-hop chain traversal</h2>
 * <pre>
 *     Paris --[locatedIn]--> France --[locatedIn]--> Europe
 *     queryChain(Paris, [locatedIn, locatedIn], k) → Europe
 * </pre>
 * Intermediate hops are resolved greedily (rank-1). the final hop returns top-k.
 *
 * <h2>Thread safety</h2>
 * <p>All public methods are thread-safe via per-thread {@link ReasonerScratch}
 * buffers allocated by a {@link ThreadLocal}. No locks are held; the repository's
 * own lock governs concurrent inserts.
 */
public final class SymbolicReasoner {

    /**
     * Cyclic shift applied to the subject role before binding.
     * All three shifts must be distinct.
     */
    private static final int SUBJECT_SHIFT = 1;

    /** Cyclic shift applied to the relation role before binding. */
    private static final int RELATION_SHIFT = 2;

    /** Cyclic shift applied to the object role before binding. */
    private static final int OBJECT_SHIFT   = 3;


    /**
     * Per-thread scratch buffers for intermediate VSA computation.
     * Fields are package-private: they are an implementation detail of
     * {@link SymbolicReasoner} and must not be accessed directly by callers.
     */
    public static final class ReasonerScratch {
        final MemorySegment scratchA;
        final MemorySegment scratchB;
        final MemorySegment scratchC;

        /**
         * Per-bit vote tally for majority-vote bundling.
         * Length = {@link BinaryVector#VECTOR_LONGS} × 64 = 10,048 ints ≈ 40,192 bytes.
         */
        public final int[] tally;

        public ReasonerScratch(final OffHeapAllocator allocator) {
            this.scratchA = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.scratchB = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.scratchC = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
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
     * Typed relation query: find entities linked to {@code subjectId} via
     * {@code relationId} using role-encoded binding.
     *
     * <p>Role encoding eliminates cross-relation interference by assigning
     * each role a distinct cyclic bit shift before XOR binding. The query:
     * <pre>
     *     query = bind(permute(subject, SUBJECT_SHIFT), permute(relation, RELATION_SHIFT))
     * </pre>
     * The nearest stored entity to {@code query} is the object of the
     * (subject, relation) edge.
     *
     * <p><b>Pre-requisites:</b> edges must have been stored via
     * {@link VectorGraphRepository#storeTypedEdge}, not {@code bindRelationalEdge}.
     *
     * @param subjectId entityid of the subject
     * @param relationId entity ID of the relation type.
     * @param k number of results.
     * @throws IllegalArgumentException if any entity ID is not found.
     */
    public HNSWIndex.SearchResults queryTypedRelation(
            final long subjectId,
            final long relationId,
            final int k) {
        final ReasonerScratch ctx = scratchPool.get();

        VectorMath.permuteN(
                resolveVector(subjectId, "subject"),
                SUBJECT_SHIFT, ctx.scratchB, ctx.scratchA);
        VectorMath.permuteN(
                resolveVector(relationId, "relation"),
                RELATION_SHIFT, ctx.scratchC, ctx.scratchA);

        VectorMath.bind(ctx.scratchB, ctx.scratchC, ctx.scratchA);
        return repository.searchHNSWBinary(ctx.scratchA, k);
    }

    /**
     * Single-hop typed traversal: follow the edge labelled {@code relationId}
     * from {@code entityId} and return the top-{@code k} entities.
     *
     * <p>Identical to {@link #queryTypedRelation} - named for clarity when composing
     * multy-hop chains.
     *
     * @param entityId the starting entity
     * @param relationId the relation type to traverse
     * @param k number of results.
     * @throws IllegalArgumentException if any entity ID is not found.
     */
    public HNSWIndex.SearchResults queryHop(
            final long entityId,
            final long relationId,
            final int k) {
        return queryTypedRelation(entityId, relationId, k);
    }

    /**
     * Multi-hop chain traversal: follow a sequence of typed edges from
     * {@code startEntityId}.
     *
     * <p>Each intermediate hop resolves greedily to the rank-1 result.
     * The final hop returns the full top-{@code k} result set.
     *
     * <p>Example:
     * <pre>
     *   Paris --[locatedIn]--> France --[locatedIn]--> Europe
     *   queryChain(Paris, [locatedIn, locatedIn], 3) -> [Europe, ...]
     * </pre>
     *
     * <p><b>Greedy caveat:</b> a wrong intermediate entity causes all subsequent
     * hops to diverge. For critical traversals, run multiple chains and
     * intersect results.
     *
     * @param startEntityId starting entity ID
     * @param relationIds ordered relation type IDs; non-null, non-empty
     * @param k number of results from the final hop
     * @return k nearest entities at the end of the chain
     * @throws IllegalArgumentException if {@code relationIds} is null or empty,
     *                                  any entity ID is not found, or an
     *                                  intermediate hop returns no results
     */
    public HNSWIndex.SearchResults queryChain(
            final long startEntityId,
            final long[] relationIds,
            final int k) {
        if (relationIds == null || relationIds.length == 0) {
            throw new IllegalArgumentException("relationIds must be non-empty");
        }

        long currentEntityId = startEntityId;

        for (int hop = 0; hop < relationIds.length - 1; hop++) {
            final HNSWIndex.SearchResults step =
                    queryTypedRelation(currentEntityId, relationIds[hop], 1);

            if (step.count() == 0) {
                throw new IllegalArgumentException(
                        "Chain broke at hop" + hop
                        + ": no result for entityID=" + currentEntityId
                        + " via relationID=" + relationIds[hop]);
            }
            currentEntityId = step.entityId(0);
        }

        return queryTypedRelation(
                currentEntityId,
                relationIds[relationIds.length - 1],
                k);
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
        final long hash = SparseEntityIndex.mixHash64(entityId);
        final long offset = repository.entityIndex().getOffset(hash);
        if (offset < 0L || offset == Long.MIN_VALUE) {
            throw new IllegalArgumentException(
                    "Entity not found in index for role=" + role + " id=" + entityId);
        }
        return repository.vectorStore().sliceAt(offset);
    }

}
