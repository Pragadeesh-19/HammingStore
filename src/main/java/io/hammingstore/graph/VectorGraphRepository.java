package io.hammingstore.graph;

import io.hammingstore.hnsw.HNSWConfig;
import io.hammingstore.hnsw.HNSWIndex;
import io.hammingstore.hnsw.HNSWLayer;
import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.memory.OffHeapVectorStore;
import io.hammingstore.memory.SparseEntityIndex;
import io.hammingstore.persist.EngineSnapshot;
import io.hammingstore.persist.MappedFileAllocator;
import io.hammingstore.vsa.ProjectionConfig;
import io.hammingstore.vsa.RandomProjectionEncoder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.locks.StampedLock;

public final class VectorGraphRepository implements AutoCloseable{

    public record SearchResult(long entityOffset, long hammingDistance, double similarity)
            implements Comparable<SearchResult> {
        @Override
        public int compareTo(final SearchResult o) {
            return Long.compare(this.hammingDistance, o.hammingDistance);
        }
    }

    private static final long SLOT_BYTES = BinaryVector.VECTOR_BYTES;

    private final OffHeapVectorStore vectorStore;
    private final SparseEntityIndex entityIndex;
    private final MemorySegment bindScratch;
    private final HNSWIndex hnswIndex;
    private final RandomProjectionEncoder encoder;
    private final StampedLock lock = new StampedLock();

    private final MappedFileAllocator mappedAllocator;
    private final Path dataDir;

    private final ThreadLocal<MemorySegment> encodeScratch;

    public VectorGraphRepository(final long maxVectors, final ProjectionConfig projectionConfig) {
        if (maxVectors <= 1) throw new IllegalArgumentException("maxVectors must be > 1");

        final OffHeapAllocator flatAllocator = new OffHeapAllocator(maxVectors);
        this.vectorStore = new OffHeapVectorStore(flatAllocator, maxVectors);
        this.entityIndex = new SparseEntityIndex(flatAllocator, maxVectors);
        this.bindScratch = flatAllocator.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        this.encoder = new RandomProjectionEncoder(flatAllocator, projectionConfig);
        this.hnswIndex = new HNSWIndex(maxVectors);
        this.mappedAllocator = null;
        this.dataDir = null;
        this.encodeScratch = ThreadLocal.withInitial(() ->
                flatAllocator.allocateRawSegment(SLOT_BYTES, Long.BYTES));
    }

    public VectorGraphRepository(final long maxVectors) {
        this(maxVectors, ProjectionConfig.of(ProjectionConfig.DIMS_MINILM));
    }

    public static VectorGraphRepository openFromDisk(
            final Path dataDir,
            final ProjectionConfig projectionConfig,
            final long maxVectors) {
        try {
            return openFromDiskInternal(dataDir, projectionConfig, maxVectors);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to open disk-backed repository", e);
        }
    }

    private static VectorGraphRepository openFromDiskInternal(
            final Path dataDir,
            final ProjectionConfig projectionConfig,
            final long maxVectors) throws IOException {

        // Read existing snapshot or create fresh state
        EngineSnapshot snap = EngineSnapshot.readFrom(dataDir);
        final boolean isNew = (snap == null);

        if (isNew) {
            final int layerCount = HNSWIndex.computeMaxLayers(maxVectors);
            snap = EngineSnapshot.fresh(
                    projectionConfig.inputDimensions(),
                    projectionConfig.seed(),
                    layerCount);
        } else {
            snap.validateProjectionConfig(
                    projectionConfig.inputDimensions(), projectionConfig.seed());
        }

        final MappedFileAllocator mfa = new MappedFileAllocator(dataDir);

        // ---- Flat vector store ----
        final long vectorStoreBytes = maxVectors * SLOT_BYTES;
        final MemorySegment vectorStoreSeg = mfa.map("vector_store.dat", vectorStoreBytes);
        final OffHeapVectorStore vectorStore = OffHeapVectorStore.fromMapped(
                vectorStoreSeg, maxVectors, snap.vectorCursorSlots);

        // ---- Entity index ----
        final long entityIndexBytes = SparseEntityIndex.tableByteSize(maxVectors);
        final long entityIndexCap   = SparseEntityIndex.tableCapacity(maxVectors);
        final MemorySegment entityIndexSeg = mfa.map("entity_index.dat", entityIndexBytes);
        final SparseEntityIndex entityIndex = SparseEntityIndex.fromMapped(
                entityIndexSeg, entityIndexCap, maxVectors, snap.indexEntryCount);

        // ---- HNSW layers ----
        final int hnswLayerCount = HNSWIndex.computeMaxLayers(maxVectors);
        final HNSWLayer[] layers = new HNSWLayer[hnswLayerCount];

        for (int l = 0; l < hnswLayerCount; l++) {
            final long layerCapacity = HNSWIndex.computeLayerCapacity(maxVectors, l);

            // Node storage file for this layer
            final MemorySegment nodesSeg = mfa.map(
                    "hnsw_nodes_L" + l + ".dat",
                    layerCapacity * HNSWConfig.NODE_BYTES);

            // Node index file for this layer
            final long nodeIndexBytes = SparseEntityIndex.tableByteSize(layerCapacity);
            final long nodeIndexCap   = SparseEntityIndex.tableCapacity(layerCapacity);
            final MemorySegment nodeIndexSeg = mfa.map("hnsw_index_L" + l + ".dat", nodeIndexBytes);

            final long committedNodes = (l < EngineSnapshot.MAX_LAYERS)
                    ? snap.layerNodeCounts[l] : 0L;
            final long committedIdxSz = (l < EngineSnapshot.MAX_LAYERS)
                    ? snap.nodeIndexSizes[l]  : 0L;

            final SparseEntityIndex nodeIdx = SparseEntityIndex.fromMapped(
                    nodeIndexSeg, nodeIndexCap, layerCapacity, committedIdxSz);

            layers[l] = HNSWLayer.fromMapped(nodesSeg, nodeIdx, layerCapacity, l, committedNodes);
        }

        final HNSWIndex hnswIndex = new HNSWIndex(
                layers, maxVectors,
                snap.hnswEntryPointOffset,
                snap.hnswTopLayer);

        // ---- Encoder (always RAM — regenerated from seed, fast) ----
        final OffHeapAllocator encoderAllocator = new OffHeapAllocator(
                (long) Math.ceil((double) projectionConfig.outputBits()
                        * projectionConfig.inputDimensions() * Float.BYTES / SLOT_BYTES) + 2L);
        final RandomProjectionEncoder encoder =
                new RandomProjectionEncoder(encoderAllocator, projectionConfig);

        // ---- Bind scratch (tiny, RAM) ----
        final OffHeapAllocator scratchAllocator = new OffHeapAllocator(4L);
        final MemorySegment bindScratch = scratchAllocator.allocateRawSegment(SLOT_BYTES, Long.BYTES);

        return new VectorGraphRepository(
                vectorStore, entityIndex, hnswIndex, encoder, bindScratch, mfa, dataDir);
    }

    private VectorGraphRepository(
            final OffHeapVectorStore vectorStore,
            final SparseEntityIndex  entityIndex,
            final HNSWIndex          hnswIndex,
            final RandomProjectionEncoder encoder,
            final MemorySegment      bindScratch,
            final MappedFileAllocator mappedAllocator,
            final Path               dataDir) {
        this.vectorStore = vectorStore;
        this.entityIndex = entityIndex;
        this.hnswIndex = hnswIndex;
        this.encoder = encoder;
        this.bindScratch = bindScratch;
        this.mappedAllocator  = mappedAllocator;
        this.dataDir = dataDir;
        this.encodeScratch = ThreadLocal.withInitial(() -> {
            final OffHeapAllocator a = new OffHeapAllocator(2L);
            return a.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        });
    }

    public void store(final long entityId, final float[] denseEmbedding) {
        final MemorySegment scratch = encodeScratch.get();
        encoder.encode(denseEmbedding, scratch);
        storeBinary(entityId, scratch);
    }

    public void storeBinary(final long entityId, final MemorySegment binaryVec) {
        final long stamp = lock.writeLock();
        try {
            final long slot = vectorStore.allocateSlot();
            vectorStore.copyInto(slot, binaryVec);
            entityIndex.put(SparseEntityIndex.xxHash3Stub(entityId), slot);
            hnswIndex.insertBinary(entityId, binaryVec);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    public void bindRelationalEdge(
            final long subjectId,
            final long objectId,
            final long relationshipId) {

        final long stamp = lock.writeLock();
        try {
            final MemorySegment sv = resolveVector(subjectId,      "subject");
            final MemorySegment ov = resolveVector(objectId,       "object");
            final MemorySegment rv = resolveVector(relationshipId, "relationship");

            VectorMath.bind(sv, rv, bindScratch);
            VectorMath.bind(bindScratch, ov, bindScratch);

            final long edgeSlot = vectorStore.allocateSlot();
            vectorStore.copyInto(edgeSlot, bindScratch);

            final long compositeId = subjectId ^ relationshipId ^ objectId;
            entityIndex.put(SparseEntityIndex.xxHash3Stub(compositeId), edgeSlot);
            hnswIndex.insertBinary(compositeId, bindScratch);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    public void retract(final long entityId) {
        final long stamp = lock.writeLock();
        try {
            final long hash   = SparseEntityIndex.xxHash3Stub(entityId);
            final long offset = entityIndex.getOffset(hash);
            if (offset >= 0) {
                entityIndex.put(hash, Long.MIN_VALUE); // tombstone
            }
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    public void checkpoint() {
        checkpoint(false);
    }

    public void checkpoint(final boolean durable) {
        if (mappedAllocator == null) return; // RAM mode — no-op

        // Snapshot cursor values under read lock so we get a consistent view
        final long vecCursor;
        final long idxCount;
        final int  hnswTopLayer;
        final long hnswEntryPt;
        final long[] layerNodeCounts = new long[EngineSnapshot.MAX_LAYERS];
        final long[] nodeIdxSizes    = new long[EngineSnapshot.MAX_LAYERS];

        final long stamp = lock.readLock();
        try {
            vecCursor    = vectorStore.allocatedSlots();
            idxCount     = entityIndex.size();
            hnswTopLayer = hnswIndex.activeLayerCount() - 1;
            hnswEntryPt  = hnswIndex.entryPointOffset();
            final int layerCount = hnswIndex.activeLayerCount();
            for (int l = 0; l < Math.min(layerCount, EngineSnapshot.MAX_LAYERS); l++) {
                layerNodeCounts[l] = hnswIndex.layer(l).nodeCount();
                nodeIdxSizes[l]    = hnswIndex.layer(l).nodeIndex().size();
            }
        } finally {
            lock.unlockRead(stamp);
        }

        // Phase 1: flush dirty mmap pages BEFORE writing the snapshot
        mappedAllocator.force(durable);

        // Phase 2: write snapshot atomically (temp file + rename)
        final EngineSnapshot snap = new EngineSnapshot(
                1,
                encoder.config().inputDimensions(),
                encoder.config().seed(),
                vecCursor,
                idxCount,
                hnswTopLayer,
                hnswIndex.activeLayerCount(),
                hnswEntryPt,
                layerNodeCounts,
                nodeIdxSizes);
        try {
            snap.writeTo(dataDir);
        } catch (IOException e) {
            throw new UncheckedIOException("checkpoint() failed to write snapshot", e);
        }
    }

    public HNSWIndex.SearchResults searchHNSW(final float[] queryEmbedding, final int k) {
        final MemorySegment scratch = encodeScratch.get();
        encoder.encode(queryEmbedding, scratch);
        return hnswIndex.searchBinary(scratch, k);
    }

    public HNSWIndex.SearchResults searchHNSWBinary(final MemorySegment binaryQuery, final int k) {
        return hnswIndex.searchBinary(binaryQuery, k);
    }

    public HNSWIndex.SearchResults searchHNSWBruteForce(final float[] queryEmbeddings, final int k) {
        final MemorySegment scratch = encodeScratch.get();
        encoder.encode(queryEmbeddings, scratch);
        return hnswIndex.bruteForceSearch(scratch, k);
    }

    public SearchResult[] searchNearest(
            final MemorySegment queryVector,
            final double similarityThreshold,
            final int maxResults) {

        if (queryVector.byteSize() != SLOT_BYTES)
            throw new IllegalArgumentException("queryVector must be " + SLOT_BYTES + " bytes");
        if (maxResults <= 0)
            throw new IllegalArgumentException("maxResults must be > 0");

        long stamp = lock.tryOptimisticRead();
        SearchResult[] result = bruteForce(queryVector, similarityThreshold, maxResults);
        if (!lock.validate(stamp)) {
            stamp = lock.readLock();
            try { result = bruteForce(queryVector, similarityThreshold, maxResults); }
            finally { lock.unlockRead(stamp); }
        }
        return result;
    }

    public ProjectionConfig projectionConfig() {
        return encoder.config();
    }

    public RandomProjectionEncoder encoder() { return encoder; }

    public HNSWIndex hnswIndex() { return hnswIndex; }

    public OffHeapVectorStore vectorStore() { return vectorStore; }

    public SparseEntityIndex entityIndex() { return entityIndex; }

    public boolean isDiskBacked() { return mappedAllocator != null; }

    @Override
    public void close() {
        hnswIndex.close();
        if (mappedAllocator != null) mappedAllocator.close();
    }

    private SearchResult[] bruteForce(
            final MemorySegment query,
            final double similarityThreshold,
            final int maxResults) {

        final long maxDist   = VectorMath.similarityToMaxDistance(similarityThreshold);
        final long populated = vectorStore.allocatedSlots();
        final MemorySegment store = vectorStore.rawStorage();

        final long[] offsets   = new long[maxResults];
        final long[] distances = new long[maxResults];
        int count = 0; long worstDist = maxDist;

        for (long slot = 0; slot < populated; slot++) {
            final long byteOff = slot * SLOT_BYTES;
            final long dist    = VectorMath.hammingDistance(
                    query, store.asSlice(byteOff, SLOT_BYTES));
            if (dist > worstDist) continue;
            if (count < maxResults) {
                offsets[count] = byteOff; distances[count] = dist; count++;
                if (count == maxResults) worstDist = maxInArray(distances, count);
            } else {
                final int wi = indexOfMax(distances, count);
                if (dist < distances[wi]) {
                    offsets[wi] = byteOff; distances[wi] = dist;
                    worstDist = maxInArray(distances, count);
                }
            }
        }

        final SearchResult[] r = new SearchResult[count];
        for (int i = 0; i < count; i++) {
            r[i] = new SearchResult(
                    offsets[i], distances[i], VectorMath.similarity(distances[i]));
        }
        Arrays.sort(r);
        return r;
    }

    private MemorySegment resolveVector(final long entityId, final String role) {
        final long offset = entityIndex.getOffset(SparseEntityIndex.xxHash3Stub(entityId));
        if (offset < 0L || offset == Long.MIN_VALUE) {
            throw new IllegalArgumentException(
                    "Entity not found (or retracted) for " + role + " id=" + entityId);
        }
        return vectorStore.sliceAt(offset);
    }

    private static long maxInArray(final long[] a, final int len) {
        long m = Long.MIN_VALUE;
        for (int i = 0; i < len; i++) if (a[i] > m) m = a[i];
        return m;
    }

    private static int indexOfMax(final long[] a, final int len) {
        int idx = 0;
        for (int i = 1; i < len; i++) if (a[i] > a[idx]) idx = i;
        return idx;
    }

}
