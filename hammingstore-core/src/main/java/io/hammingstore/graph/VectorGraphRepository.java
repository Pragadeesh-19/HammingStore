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
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.StampedLock;
import java.util.logging.Logger;

/**
 * Central access point for storing and retrieving binary hypervectors.
 *
 * <p>Combines three subsystems into a single thread-safe unit:
 * <ul>
 *   <li><b>Flat vector store</b> ({@link OffHeapVectorStore}) - sequential
 *       off-heap slab; one slot per stored entity.</li>
 *   <li><b>Entity index</b> ({@link SparseEntityIndex}) - hash map from
 *       entity ID to slot byte offset in the vector store.</li>
 *   <li><b>HNSW graph</b> ({@link HNSWIndex}) - proximity graph for ANN search.
 *       Each HNSW node stores a direct {@code vectorStoreOffset}, so distance
 *       computation is a single arithmetic lookup — no hash on the hot path.</li>
 * </ul>
 *
 * <h2>Write ordering for SCHEMA_V2 correctness</h2>
 * <p>{@link #storeBinary} follows a strict order:
 * <ol>
 *   <li>{@code slot = vectorStore.allocateSlot()} — reserve the slot</li>
 *   <li>{@code vectorStore.copyInto(slot, binaryVec)} — write the vector</li>
 *   <li>{@code entityIndex.put(hash, slot)} — register entity → slot mapping</li>
 *   <li>{@code hnswIndex.insertBinary(entityId, binaryVec, slot)} — insert into
 *       graph; {@code slot} is written directly into every new node</li>
 * </ol>
 * Step 4 passes {@code slot} explicitly so that allocateNode() in HNSWLayer.java
 * can embed it in the node with no further lookups.
 * 
 * <h2>Thread safety</h2>
 * <p>All public methods are thread-safe via {@link StampedLock}. Writes acquire
 * an exclusive write lock; reads use an optimistic reads with a fallback to a shared
 * read lock. Note that {@code insertBinary} inside {@link HNSWIndex} also holds its
 * own internal lock - the two locks are different instances so no deadlock is possible,
 * but the index lock is redundant whenever the repository write lock is
 * already held. this is a known inefficiency, not a correctness issue.
 *
 * <h2>Persistence</h2>
 * <p>Use {@link #openFromDisk} for disk backed repository whose data survives
 * process restarts. Use the public constructors for an ephemeral in-RAM repository.
 * In RAM mode, {@link #checkpoint} is no-op.
 *
 */
public final class VectorGraphRepository implements AutoCloseable{

    private static final Logger log = Logger.getLogger(VectorGraphRepository.class.getName());

    /**
     * Byte size of one vector slot in the flat store.
     * Alias for {@link BinaryVector#VECTOR_BYTES} kept private to reduce noise
     * inside this class.
     */
    private static final long SLOT_BYTES = BinaryVector.VECTOR_BYTES;
    private static final long EDGE_LOG_BYTES = 16L;

    private static final int SUBJECT_SHIFT = 1;
    private static final int RELATION_SHIFT = 2;
    private static final int OBJECT_SHIFT   = 3;

    private final ConcurrentHashMap<Long, Long> edgeLookup = new ConcurrentHashMap<>();
    private final MemorySegment edgeLog;
    private long edgeCursor;

    private final OffHeapVectorStore vectorStore;
    private final SparseEntityIndex entityIndex;
    private final MemorySegment bindScratch;
    private final MemorySegment bindScratch2;
    private final MemorySegment bindScratch3;
    private final MemorySegment bindScratch4;
    private final HNSWIndex hnswIndex;
    private final RandomProjectionEncoder encoder;
    private final StampedLock lock = new StampedLock();
    private final MappedFileAllocator mappedAllocator;
    private final Path dataDir;
    private final ThreadLocal<MemorySegment> encodeScratch;

    /**
     * Creates an in-RAM repository with the given capacity and projection config.
     *
     * @param maxVectors maximum number of vectors the repository can hold
     * @param projectionConfig the encoder configuration (input dimensions, seed)
     * @throws IllegalArgumentException if {@code maxVectors} ≤ 1
     */
    public VectorGraphRepository(final long maxVectors, final ProjectionConfig projectionConfig) {
        if (maxVectors <= 1) throw new IllegalArgumentException("maxVectors must be > 1");

        final OffHeapAllocator allocator = new OffHeapAllocator(maxVectors);
        this.vectorStore = new OffHeapVectorStore(allocator, maxVectors);
        this.entityIndex = new SparseEntityIndex(allocator, maxVectors);
        this.bindScratch = allocator.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        this.bindScratch2 = allocator.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        this.bindScratch3 = allocator.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        this.bindScratch4 = allocator.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        this.encoder = new RandomProjectionEncoder(allocator, projectionConfig);
        this.hnswIndex = new HNSWIndex(maxVectors, vectorStore);
        this.mappedAllocator = null;
        this.dataDir = null;
        this.edgeLog = null;
        this.edgeCursor = 0L;
        this.encodeScratch = ThreadLocal.withInitial(() ->
                allocator.allocateRawSegment(SLOT_BYTES, Long.BYTES));
    }

    public VectorGraphRepository(final long maxVectors) {
        this(maxVectors, ProjectionConfig.of(ProjectionConfig.DIMS_MINILM));
    }

    /**
     * Opens a disk-backed repository, or creates a new one
     * if no snapshot exists there yet.
     *
     * <p>Uses memory-mapped files for zero-copy persistence. Data survives
     * process restarts. Call {@link #checkpoint} periodically and on shutdown
     * to commit in-flight state to the snapshot file.
     *
     * @param dataDir directory where snapshot and data files live.
     * @param projectionConfig the encoder configuration; must match the snapshot
     *                         if one exists, otherwise it is used to create a fresh one
     * @param maxVectors maximum number of vectors the repository can hold.
     * @throws UncheckedIOException if the snapshot or data files cannot be read or created.
     */
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

    /**
     * Encodes {@code denseEmbedding} using the configured {@link RandomProjectionEncoder}
     * and stores the resulting binary vector under {@code entityId}.
     *
     * @param entityId the entity identifier
     * @param denseEmbedding the float embedding to binarise and store;
     *                       length must match {@link ProjectionConfig#inputDimensions()}
     */
    public void store(final long entityId, final float[] denseEmbedding) {
        final MemorySegment scratch = encodeScratch.get();
        encoder.encode(denseEmbedding, scratch);
        storeBinary(entityId, scratch);
    }

    /**
     * Stores a pre-binarised hypervector under {@code entityId}.
     *
     * @param entityId the entity identifier
     * @param binaryVec the binary hypervector; must be exactly {@link BinaryVector#VECTOR_BYTES} bytes
     */
    public void storeBinary(final long entityId, final MemorySegment binaryVec) {
        final long stamp = lock.writeLock();
        try {
            final long slot = vectorStore.allocateSlot();
            vectorStore.copyInto(slot, binaryVec);
            entityIndex.put(SparseEntityIndex.mixHash64(entityId), slot);
            hnswIndex.insertBinary(entityId, binaryVec, slot);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /**
     * Binds a subject-relation-object triple into a composite edge vector and
     * inserts it into the graph.
     *
     * <p>The edge vector is computed as {@code bind(bind(subject, relation), object)}
     * using XOR binding. The composite entity ID is {@code subjectId ^ relationshipId ^ objectId}.
     *
     * @param subjectId entity ID of the subject
     * @param objectId entity ID of the object
     * @param relationshipId entity ID of the relation
     * @throws IllegalArgumentException if any of the three entity IDs is not found in the index
     */
    public void bindRelationalEdge(
            final long subjectId,
            final long objectId,
            final long relationshipId) {

        final long stamp = lock.writeLock();
        try {
            final MemorySegment sv = resolveVector(subjectId,"subject");
            final MemorySegment ov = resolveVector(objectId,"object");
            final MemorySegment rv = resolveVector(relationshipId,"relationship");

            VectorMath.bind(sv, rv, bindScratch);
            VectorMath.bind(bindScratch, ov, bindScratch);

            final long edgeSlot = vectorStore.allocateSlot();
            vectorStore.copyInto(edgeSlot, bindScratch);

            final long compositeId = subjectId ^ relationshipId ^ objectId;
            entityIndex.put(SparseEntityIndex.mixHash64(compositeId), edgeSlot);
            hnswIndex.insertBinary(compositeId, bindScratch, edgeSlot);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /**
     * Stores a role-encoded typed edge: {@code subject -[relation]-> object}.
     *
     * <p>Unlike {@link #bindRelationalEdge}, which uses flat XOR binding,
     * this method applies a distinct cyclic bit permutation to each role:
     * <pre>
     *     edge = bind(permute(subject,  SUBJECT_SHIFT),
     *             bind(permute(relation, RELATION_SHIFT),
     *                  permute(object,   OBJECT_SHIFT)))
     * </pre>
     * This eliminates cross relation interferences when multiple relation types
     * share the same vector space.
     *
     * <p>Edges stored here must be queried via
     * {@link io.hammingstore.vsa.SymbolicReasoner#queryTypedRelation} or
     * {@link io.hammingstore.vsa.SymbolicReasoner#queryChain}, not via the
     * untyped {@code queryRelation} or {@code queryAnalogy}.
     *
     * @param subjectId entity ID of the subject; must be already be stored.
     * @param relationId entityId of the relation type; must be already stored.
     * @param objectId entityID of the object. must be already stored.
     * @throws IllegalArgumentException if any of the three entity IDs is not found.
     */
    public void storeTypedEdge(
            final long subjectId,
            final long relationId,
            final long objectId) {
        final long stamp = lock.writeLock();
        try {
            final MemorySegment sv = resolveVector(subjectId,"subject");
            final MemorySegment rv = resolveVector(relationId,"relation");
            final MemorySegment ov = resolveVector(objectId,"object");

            VectorMath.permuteN(sv, SUBJECT_SHIFT,  bindScratch,  bindScratch4);
            VectorMath.permuteN(rv, RELATION_SHIFT, bindScratch2, bindScratch4);
            VectorMath.bind(bindScratch, bindScratch2, bindScratch);

            final long edgeSlot = vectorStore.allocateSlot();
            vectorStore.copyInto(edgeSlot, bindScratch);

            final long compositeId = subjectId ^ relationId ^ objectId;
            entityIndex.put(SparseEntityIndex.mixHash64(compositeId), edgeSlot);
            hnswIndex.insertBinary(compositeId, bindScratch, edgeSlot);

            final long edgeKey = subjectId ^ relationId;
            appendEdgeToLog(edgeKey, objectId);
            edgeLookup.put(edgeKey, objectId);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /**
     * Tombstones {@code entityId} in the entity index so it no longer resolves
     * to a vector. Does not remove it from the HNSW graph (graph nodes are
     * immutable after insertion).
     *
     * @param entityId the entity to retract
     */
    public void retract(final long entityId) {
        final long stamp = lock.writeLock();
        try {
            final long hash   = SparseEntityIndex.mixHash64(entityId);
            final long offset = entityIndex.getOffset(hash);
            if (offset >= 0) {
                entityIndex.put(hash, Long.MIN_VALUE);
            }
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /**
     * Commits the current state to disk (no-op in RAM mode).
     *
     * <p>Writes an atomic SCHEMA_V2 snapshot. The HNSW entry point is stored
     * as an entity ID so the snapshot is node-layout-independent.
     *
     * @param durable if {@code true}, performs a fsync to guarantee data is
     *                on durable storage before returning
     * @throws UncheckedIOException if the snapshot file cannot be written
     */
    public void checkpoint(final boolean durable) {
        if (mappedAllocator == null) return;

        final long vecCursor;
        final long idxCount;
        final int  hnswTopLayer;
        final long hnswEntryPtEntityId;
        final long[] layerNodeCounts = new long[EngineSnapshot.MAX_LAYERS];
        final long[] nodeIdxSizes    = new long[EngineSnapshot.MAX_LAYERS];
        final long committedEdgeCursor;

        final long stamp = lock.readLock();
        try {
            vecCursor = vectorStore.allocatedSlots();
            idxCount = entityIndex.size();
            hnswTopLayer = hnswIndex.activeLayerCount() - 1;
            hnswEntryPtEntityId = hnswIndex.entryPointEntityId();
            committedEdgeCursor = edgeCursor;
            final int layerCount = hnswIndex.activeLayerCount();
            for (int l = 0; l < Math.min(layerCount, EngineSnapshot.MAX_LAYERS); l++) {
                layerNodeCounts[l] = hnswIndex.layer(l).nodeCount();
                nodeIdxSizes[l] = hnswIndex.layer(l).nodeIndex().size();
            }
        } finally {
            lock.unlockRead(stamp);
        }

        mappedAllocator.force(durable);

        final EngineSnapshot snap = new EngineSnapshot(
                2,
                encoder.config().inputDimensions(),
                encoder.config().seed(),
                vecCursor,
                idxCount,
                hnswTopLayer,
                hnswIndex.activeLayerCount(),
                hnswEntryPtEntityId,
                layerNodeCounts,
                nodeIdxSizes,
                committedEdgeCursor);
        try {
            snap.writeTo(dataDir);
        } catch (IOException e) {
            throw new UncheckedIOException("checkpoint() failed to write snapshot", e);
        }
    }

    /**
     * Encodes {@code queryEmbedding} and searches the HNSW graph for the {@code k}
     * nearest stored vectors.
     *
     * @param queryEmbedding the float query embedding
     * @param k number of nearest neighbours to return
     * @return search results sorted by ascending Hamming distance
     */
    public HNSWIndex.SearchResults searchHNSW(final float[] queryEmbedding, final int k) {
        final MemorySegment scratch = encodeScratch.get();
        encoder.encode(queryEmbedding, scratch);
        return hnswIndex.searchBinary(scratch, k);
    }

    /**
     * Searches the HNSW graph for the {@code k} nearest stored vectors to a
     * pre-binarised query.
     *
     * @param binaryQuery the binary query hypervector;
     *                    must be exactly {@link BinaryVector#VECTOR_BYTES} bytes
     * @param k number of nearest neighbours to return
     * @return search results sorted by ascending Hamming distance
     */
    public HNSWIndex.SearchResults searchHNSWBinary(final MemorySegment binaryQuery, final int k) {
        return hnswIndex.searchBinary(binaryQuery, k);
    }

    /**
     * Returns the projection configuration this repository was built with.
     * Used by the gRPC service to return encoder metadata to clients.
     */
    public ProjectionConfig projectionConfig() {
        return encoder.config();
    }

    public long nodeCount() {
        return hnswIndex.layer(0).nodeCount();
    }

    public OffHeapVectorStore vectorStore() { return vectorStore; }

    public SparseEntityIndex entityIndex() { return entityIndex; }

    public boolean isDiskBacked() { return mappedAllocator != null; }

    public Long lookupEdge(final long subjectId, final long relationId) {
        return edgeLookup.get(subjectId ^ relationId);
    }

    @Override
    public void close() {
        hnswIndex.close();
        if (mappedAllocator != null) mappedAllocator.close();
    }

    private void appendEdgeToLog(final long edgeKey, final long objectId) {
        if (edgeLog == null) return;

        final long byteOffset = edgeCursor * EDGE_LOG_BYTES;
        if (byteOffset + EDGE_LOG_BYTES > edgeLog.byteSize()) {
            if (edgeCursor % 10_000 == 0) {
                log.warning(String.format(
                        "edge_log.dat is full (%,d entries). "
                                + "Edge will not survive restart. "
                                + "Increase --max-vectors to persist more edges.",
                        edgeCursor));
            }
            return;
        }

        edgeLog.set(ValueLayout.JAVA_LONG_UNALIGNED, byteOffset, edgeKey);
        edgeLog.set(ValueLayout.JAVA_LONG_UNALIGNED, byteOffset + 8, objectId);
        edgeCursor++;
    }

    private MemorySegment resolveVector(final long entityId, final String role) {
        final long offset = entityIndex.getOffset(SparseEntityIndex.mixHash64(entityId));
        if (offset < 0L || offset == Long.MIN_VALUE) {
            throw new IllegalArgumentException(
                    "Entity not found (or retracted) for " + role + " id=" + entityId);
        }
        return vectorStore.sliceAt(offset);
    }


    private static VectorGraphRepository openFromDiskInternal(
            final Path dataDir,
            final ProjectionConfig projectionConfig,
            final long maxVectors) throws IOException {

        final Optional<EngineSnapshot> maybeSnap = EngineSnapshot.readFrom(dataDir);
        final EngineSnapshot snap;

        if (maybeSnap == null || maybeSnap.isEmpty()) {
            snap = EngineSnapshot.fresh(
                    projectionConfig.inputDimensions(),
                    projectionConfig.seed(),
                    HNSWIndex.computeMaxLayers(maxVectors));
        } else {
            snap = maybeSnap.get();
            snap.validateProjectionConfig(
                    projectionConfig.inputDimensions(), projectionConfig.seed());
        }

        final MappedFileAllocator mfa = new MappedFileAllocator(dataDir);

        final MemorySegment vectorStoreSeg = mfa.map(
                "vector_store.dat",maxVectors * SLOT_BYTES);
        final OffHeapVectorStore vectorStore = OffHeapVectorStore.fromMapped(
                vectorStoreSeg, maxVectors, snap.vectorCursorSlots());

        final MemorySegment entityIndexSeg = mfa.map(
                "entity_index.dat", SparseEntityIndex.tableByteSize(maxVectors));
        final SparseEntityIndex entityIndex = SparseEntityIndex.fromMapped(
                entityIndexSeg,
                SparseEntityIndex.tableCapacity(maxVectors),
                maxVectors,
                snap.indexEntryCount());

        final int hnswLayerCount = HNSWIndex.computeMaxLayers(maxVectors);
        final HNSWLayer[] layers = new HNSWLayer[hnswLayerCount];
        final long[] layerNodeCounts = snap.layerNodesCounts();
        final long[] nodeIndexSizes  = snap.nodeIndexSizes();

        for (int l = 0; l < hnswLayerCount; l++) {
            final long layerCapacity  = HNSWIndex.computeLayerCapacity(maxVectors, l);
            final long committedNodes = (l < EngineSnapshot.MAX_LAYERS)
                    ? layerNodeCounts[l] : 0L;
            final long committedIdxSz = (l < EngineSnapshot.MAX_LAYERS)
                    ? nodeIndexSizes[l]  : 0L;

            final MemorySegment nodesSeg = mfa.map(
                    "hnsw_nodes_L" + l + ".dat",
                    layerCapacity * HNSWConfig.NODE_BYTES);

            final MemorySegment nodeIndexSeg = mfa.map(
                    "hnsw_index_L" + l + ".dat",
                    SparseEntityIndex.tableByteSize(layerCapacity));

            final SparseEntityIndex nodeIdx = SparseEntityIndex.fromMapped(
                    nodeIndexSeg,
                    SparseEntityIndex.tableCapacity(layerCapacity),
                    layerCapacity,
                    committedIdxSz);

            layers[l] = HNSWLayer.fromMapped(
                    nodesSeg, nodeIdx, layerCapacity, l, committedNodes, vectorStore);
        }

        final long entryPointEntityId = snap.hnswEntryPointEntityId();
        final long entryPointOffset;
        if (entryPointEntityId == -1L) {
            entryPointOffset = -1L;
        } else {
            final int topLayer = snap.hnswTopLayer();
            final long resolved = layers[topLayer].findOffset(entryPointEntityId);
            entryPointOffset = (resolved >= 0) ? resolved : -1L;
        }

        final HNSWIndex hnswIndex = new HNSWIndex(
                layers, maxVectors,
                entryPointOffset, entryPointEntityId,
                snap.hnswTopLayer());

        final long encoderSlots = (long) Math.ceil(
                (double) projectionConfig.outputBits()
                        * projectionConfig.inputDimensions()
                        * Float.BYTES / SLOT_BYTES) + 2L;
        final RandomProjectionEncoder encoder = new RandomProjectionEncoder(
                new OffHeapAllocator(encoderSlots), projectionConfig);

        final OffHeapAllocator scratchAlloc = new OffHeapAllocator(5L);
        final MemorySegment bindScratch  = scratchAlloc.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        final MemorySegment bindScratch2 = scratchAlloc.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        final MemorySegment bindScratch3 = scratchAlloc.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        final MemorySegment bindScratch4 = scratchAlloc.allocateRawSegment(SLOT_BYTES, Long.BYTES);

        final long edgeLogBytes = maxVectors * EDGE_LOG_BYTES;
        final MemorySegment edgeLog = mfa.map("edge_log.dat", edgeLogBytes);

        final long committedEdges = snap.edgeLookupCursor();
        final ConcurrentHashMap<Long, Long> edgeLookup = new ConcurrentHashMap<>(
                (int) Math.min(committedEdges * 2, Integer.MAX_VALUE));

        if (committedEdges > 0) {
            log.info(String.format(
                    "Replaying %,d edges from edge_log.dat...", committedEdges));
            for (long i = 0; i < committedEdges; i++) {
                final long offset = i * EDGE_LOG_BYTES;
                final long edgeKey = edgeLog.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
                final long objectId = edgeLog.get(ValueLayout.JAVA_LONG_UNALIGNED, offset + 8);
                edgeLookup.put(edgeKey, objectId);
            }
            log.info(String.format(
                    "Edge replay complete: %,d edges restored in O(N). "
                            + "Chain traversal is fully operational.",
                    edgeLookup.size()));
        } else {
            log.info("edge_log.dat: no committed edges (fresh database or V2 migration). "
                    + "Re-ingest edges to populate the persistent edge log.");
        }

        return new VectorGraphRepository(
                vectorStore, entityIndex, hnswIndex, encoder,
                bindScratch, bindScratch2, bindScratch3, bindScratch4,
                mfa, dataDir,
                edgeLog, committedEdges, edgeLookup);
    }

    private VectorGraphRepository(
            final OffHeapVectorStore vectorStore,
            final SparseEntityIndex  entityIndex,
            final HNSWIndex hnswIndex,
            final RandomProjectionEncoder encoder,
            final MemorySegment bindScratch,
            final MemorySegment bindScratch2,
            final MemorySegment bindScratch3,
            final MemorySegment bindScratch4,
            final MappedFileAllocator mappedAllocator,
            final Path dataDir,
            final MemorySegment edgeLog,
            final long edgeCursor,
            final ConcurrentHashMap<Long, Long> edgeLookupSeed) {
        this.vectorStore = vectorStore;
        this.entityIndex = entityIndex;
        this.hnswIndex = hnswIndex;
        this.encoder = encoder;
        this.bindScratch = bindScratch;
        this.bindScratch2 = bindScratch2;
        this.bindScratch3 = bindScratch3;
        this.bindScratch4 = bindScratch4;
        this.mappedAllocator  = mappedAllocator;
        this.dataDir = dataDir;
        this.edgeLog = edgeLog;
        this.edgeCursor = edgeCursor;
        this.edgeLookup.putAll(edgeLookupSeed);
        this.encodeScratch = ThreadLocal.withInitial(() -> {
            final OffHeapAllocator a = new OffHeapAllocator(2L);
            return a.allocateRawSegment(SLOT_BYTES, Long.BYTES);
        });
    }
}
