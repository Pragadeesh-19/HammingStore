package io.hammingstore.vsa;

import io.hammingstore.graph.VectorGraphRepository;
import io.hammingstore.hnsw.HNSWIndex;
import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.memory.SparseEntityIndex;

import java.lang.foreign.MemorySegment;

public final class SymbolicReasoner {

    public static final class ReasonerScratch {
        public final MemorySegment scratchA;
        public final MemorySegment scratchB;
        public final int[] tally;

        public ReasonerScratch(final OffHeapAllocator allocator) {
            this.scratchA = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.scratchB = allocator.allocateRawSegment(BinaryVector.VECTOR_BYTES, Long.BYTES);
            this.tally    = new int[BinaryVector.VECTOR_LONGS * 64]; // 10,048 ints
        }
    }

    private final VectorGraphRepository repository;
    private final ThreadLocal<ReasonerScratch> scratchPool;

    public SymbolicReasoner(final VectorGraphRepository repository) {
        if (repository == null) throw new NullPointerException("repository must not be null");
        this.repository  = repository;
        this.scratchPool = ThreadLocal.withInitial(() ->
                new ReasonerScratch(new OffHeapAllocator(16L))
        );
    }

    public void bindAnalogy(
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

    public HNSWIndex.SearchResults queryAnalogy(
            final long subjectAId,
            final long objectAId,
            final long subjectBId,
            final int k) {

        final ReasonerScratch ctx = scratchPool.get();
        bindAnalogy(subjectAId, objectAId, subjectBId, ctx.scratchA, ctx.scratchB);
        return repository.searchHNSWBinary(ctx.scratchA, k);
    }

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

    public HNSWIndex.SearchResults queryRelationType(
            final long subjectId,
            final long objectId,
            final int k) {

        final ReasonerScratch ctx = scratchPool.get();
        final MemorySegment subject = resolveVector(subjectId, "subject");
        final MemorySegment object  = resolveVector(objectId,  "object");
        VectorMath.bind(subject, object, ctx.scratchA);
        return repository.searchHNSWBinary(ctx.scratchA, k);
    }

    public double similarity(final long entityAId, final long entityBId) {
        final MemorySegment a = resolveVector(entityAId, "entityA");
        final MemorySegment b = resolveVector(entityBId, "entityB");
        return VectorMath.similarity(VectorMath.hammingDistance(a, b));
    }

    public VectorGraphRepository repository() {
        return repository;
    }

    private MemorySegment resolveVector(final long entityId, final String role) {
        final long hash   = SparseEntityIndex.xxHash3Stub(entityId);
        final long offset = repository.entityIndex().getOffset(hash);
        if (offset < 0L) {
            throw new IllegalArgumentException(
                    "Entity not found in index for role=" + role + " id=" + entityId);
        }
        return repository.vectorStore().sliceAt(offset);
    }

}
