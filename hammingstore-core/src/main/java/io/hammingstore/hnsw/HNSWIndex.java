package io.hammingstore.hnsw;

import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.memory.OffHeapVectorStore;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.StampedLock;

/**
 * Hierarchical Navigable Small World (HNSW) graph index for binary hypervectors.
 *
 * <p>HNSW is a proximity graph algorithm that supports approximate nearest-neighbour
 * search in sub-linear time. The graph is organized into layers: layer 0 contains
 * all nodes; each higher layer is an exponentially sparser subset that acts as an
 * express lane to skip larger distances quickly. Inserts and searches both start at
 * the top layer and work downward, refining the candidate set at each level.
 *
 * <p>Distance is measured as hamming distance between 10,048-bit binary vectors.
 * As of SCHEMA_V2, each node stores a direct {@code vectorStoreOffset} pointing
 * into {@link io.hammingstore.memory.OffHeapVectorStore}. Distance computation is:
 * <pre>
 *   offset = node.getVectorStoreOffset()
 *   vec = vectorStore.sliceAt(offset)
 *   dist = VectorMath.hammingDistance(vec, query)
 * </pre>
 * No hash, no hash-map lookup, no lambda.
 *
 * <h2>Thread safety</h2>
 * <p>Concurrent reads are supported via optimistically read with {@link StampedLock}.
 * Writes acquire an exclusive write lock. Each thread maintains its own
 * {@link SearchContext} via {@link ThreadLocal}, so no per-query allocation occurs on the
 * hot path after warmup.
 *
 * <h2>Persistence</h2>
 * <p>The two argument restore constructor accepts pre-built {@link HNSWLayer} arrays
 * loaded from memoryMapped files by {@code VectorGraphRepository}. The {@code allocator}
 * is set to {@code null} in that path; {@link #close()} handles this safely.
 */
public final class HNSWIndex implements AutoCloseable {

    /**
     * The result of a nearest-neighbour search: parallel arrays of entity IDs,
     * Hamming distances, and cosine-equivalent similarities, with a count of
     * how many results were actually populated.
     */
    public record SearchResults(
            long[] entityIds,
            long[] distance,
            double[] similarities,
            int count
    ) {
        public long entityId(final int i) { return entityIds[i]; }
        public long distance(final int i) { return distance[i]; }
        public double similarity(final int i) { return similarities[i]; }
    }

    private final OffHeapAllocator allocator;
    private final HNSWLayer[] layers;
    private final int maxLayerCount;
    private final long maxNodes;
    private final StampedLock lock = new StampedLock();

    private volatile long entryPointOffset = -1L;
    private volatile long entryPointEntityId = -1L;
    private final AtomicInteger topLayer = new AtomicInteger(0);

    private final ThreadLocal<SearchContext> searchContextPool;

    /**
     * Creates a new empty index with fresh off-heap storage.
     *
     * @param maxNodes maximum number of vectors the index can hold
     * @param vectorStore the canonical vector slab shared with VectorGraphRepository
     * @throws IllegalArgumentException if {@code maxNodes} is not positive
     */
    public HNSWIndex(final long maxNodes, final OffHeapVectorStore vectorStore) {
        if (maxNodes <= 0) throw new IllegalArgumentException("maxNodes must be > 0");
        this.maxNodes = maxNodes;
        this.allocator = new OffHeapAllocator(maxNodes);
        this.maxLayerCount = computeMaxLayers(maxNodes);
        this.layers = new HNSWLayer[maxLayerCount];

        for (int l = 0; l < maxLayerCount; l++) {
            layers[l] = new HNSWLayer(allocator, computeLayerCapacity(maxNodes, l), l, vectorStore);
        }

        this.searchContextPool = buildContextPool(maxNodes);
    }

    /**
     * Restores an index from pre-built layers loaded from memory-mapped files.
     * Used exclusively by {@link io.hammingstore.graph.VectorGraphRepository}
     * during snapshot restore. Layers already carry their own {@link OffHeapVectorStore}
     * reference (injected at {@link HNSWLayer#fromMapped} time).
     *
     * @param prebuildLayers layers loaded from mapped files.
     * @param maxNodes maximum capacity the index was originally built for
     * @param initialEntryPointOffset byte offset of the entry point node in layer 0
     * @param initialEntryPointEntityId entityId of the entry point (-1 if empty)
     * @param initialTopLayer the highest populated layer at snapshot time.
     */
    public HNSWIndex(
            final HNSWLayer[] prebuildLayers,
            final long maxNodes,
            final long initialEntryPointOffset,
            final long initialEntryPointEntityId,
            final int initialTopLayer) {
        this.layers = prebuildLayers;
        this.maxLayerCount = prebuildLayers.length;
        this.maxNodes = maxNodes;
        this.allocator = null;
        this.entryPointOffset = initialEntryPointOffset;
        this.entryPointEntityId = initialEntryPointEntityId;
        this.topLayer.set(initialTopLayer);

        this.searchContextPool = buildContextPool(maxNodes);
    }

    /**
     * Inserts an entity into the HNSW graph.
     *
     * <p>The entity's vector must already be stored in {@link OffHeapVectorStore}
     * and the {@code vectorStoreOffset} must be the byte offset returned by
     * {@link OffHeapVectorStore#allocateSlot()}. This offset is written directly
     * into each node so that distance computation requires no hash lookup.
     *
     * <p>The {@code binaryVec} parameter is used as the query vector during graph
     * traversal (greedy descent and beam search), avoiding a vectorStore lookup
     * on the insert path itself.
     *
     * @param entityId the entity identifier to associate with this vector
     * @param binaryVec the binary hypervector; must be exactly
     *                  {@link BinaryVector#VECTOR_BYTES} bytes
     * @param vectorStoreOffset byte offset of {@code binaryVec} in OffHeapVectorStore
     * @throws IllegalArgumentException if the vector has the wrong size
     */
    public void insertBinary(final long entityId, final MemorySegment binaryVec, final long vectorStoreOffset) {
        if (binaryVec.byteSize() != BinaryVector.VECTOR_BYTES) {
            throw new IllegalArgumentException(
                    "binaryVec must be " + BinaryVector.VECTOR_BYTES + " bytes");
        }
        final long stamp = lock.writeLock();
        try {
            insertInternal(entityId, binaryVec, vectorStoreOffset);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /**
     * Searches for the {@code k} nearest stored vectors to {@code binaryQuery}
     * using HNSW graph traversal.
     *
     * <p>Uses an optimistic read lock: if no write occurred during the search
     * the result is returned immediately. If a concurrent write was detected,
     * the search is retried under a full read lock.
     *
     * @param binaryQuery the query hypervector; must be exactly
     *                    {@link BinaryVector#VECTOR_BYTES} bytes
     * @param k number of nearest neighbours to return
     * @return the k nearest results sorted by ascending Hamming distance
     * @throws IllegalArgumentException if the query has the wrong size or k ≤ 0
     */
    public SearchResults searchBinary(final MemorySegment binaryQuery, final int k) {
        if (binaryQuery.byteSize() != BinaryVector.VECTOR_BYTES) {
            throw new IllegalArgumentException(
                    "binaryQuery must be " + BinaryVector.VECTOR_BYTES + " bytes");
        }
        if (k <= 0) throw new IllegalArgumentException("k must be > 0");

        long stamp = lock.tryOptimisticRead();
        SearchResults result = searchInternal(binaryQuery, k);
        if (!lock.validate(stamp)) {
            stamp = lock.readLock();
            try {
                result = searchInternal(binaryQuery, k);
            } finally {
                lock.unlockRead(stamp);
            }
        }
        return result;
    }

    public int activeLayerCount() { return topLayer.get() + 1; }

    public long entryPointOffset() { return entryPointOffset; }

    public long entryPointEntityId() { return entryPointEntityId; }

    public long size() { return layers[0].nodeCount(); }

    public HNSWLayer layer(final int l) {
        if (l < 0 || l >= maxLayerCount) throw new IndexOutOfBoundsException("layer " + l);
        return layers[l];
    }

    @Override
    public void close() {
        if (allocator != null) {
            allocator.close();
        }
    }

    /**
     * Computes the number of HNSW layers needed for a dataset of {@code n} vectors.
     * Formula: {@code min(ceil(log(n) / log(M)) + 2, MAX_LAYERS)}.
     */
    public static int computeMaxLayers(final long n) {
        if (n <= 1) return 1;
        return Math.min((int) Math.ceil(Math.log(n) / Math.log(HNSWConfig.M)) + 2,
                HNSWConfig.MAX_LAYERS);
    }

    /**
     * Computes the node capacity for layer {@code l} of an index with {@code n} total nodes.
     * Layer 0 holds all nodes. Upper layers hold exponentially fewer nodes.
     * A 20% margin is added to absorb random-level variance.
     */
    public static long computeLayerCapacity(final long n, final int l) {
        if (l == 0) return n;
        return Math.max(16L, (long) (n * Math.exp(-l) * 1.2));
    }

    private void insertInternal(final long entityId, final MemorySegment binaryVec, final long vectorStoreOffset) {
        final int nodeMaxLayer = randomLevel();
        final long currentEntryOffset = entryPointOffset;
        final int currentTopLayer = topLayer.get();
        long entryOffset = currentEntryOffset;

        if (entryOffset == -1L) {
            for (int l = nodeMaxLayer; l >= 0; l--) {
                layers[l].allocateNode(entityId, nodeMaxLayer, vectorStoreOffset);
            }
            topLayer.set(nodeMaxLayer);
            entryPointOffset = layers[nodeMaxLayer].findOffset(entityId);
            entryPointEntityId = entityId;
            return;
        }

        for (int l = currentTopLayer; l > nodeMaxLayer; l--) {
            entryOffset = layers[l].greedySearch(entryOffset, binaryVec);
            final long lower = projectOffsetToLayer(layers[l], entryOffset, l - 1);
            if (lower >= 0) {
                entryOffset = lower;
            } else {
                final long fallback = layers[l - 1].findOffset(
                        layers[l].viewAt(entryPointOffset).getEntityId());
                if (fallback >= 0) entryOffset = fallback;
                break;
            }
        }

        final int insertTopLayer = Math.min(nodeMaxLayer, currentTopLayer);
        for (int l = nodeMaxLayer; l > insertTopLayer; l--) {
            layers[l].allocateNode(entityId, nodeMaxLayer, vectorStoreOffset);
        }

        final SearchContext ctx = searchContextPool.get();

        for (int l = insertTopLayer; l >= 0; l--) {
            final long newNodeOffset = layers[l].allocateNode(entityId, nodeMaxLayer, vectorStoreOffset);

            ctx.candidates.reset();
            ctx.results.reset();
            ctx.visited.reset();

            final int ef = (l == 0) ? HNSWConfig.EF_CONSTRUCTION : HNSWConfig.M;
            final int mMax = (l == 0) ? HNSWConfig.M_LAYER_ZERO : HNSWConfig.M;
            layers[l].efSearch(entryOffset, binaryVec, ef, ctx.results, ctx.candidates, ctx.visited);
            layers[l].connectNeighbors(newNodeOffset, ctx.results, mMax);

            if (l > 0 && ctx.results.size() > 0) {
                ctx.results.sortAscending();
                final long bestEntityId = layers[l].viewAt(ctx.results.offsetAt(0)).getEntityId();
                final long lowerOffset = layers[l - 1].findOffset(bestEntityId);
                entryOffset = (lowerOffset >= 0) ? lowerOffset : newNodeOffset;
            }
        }

        if (nodeMaxLayer > currentTopLayer) {
            topLayer.set(nodeMaxLayer);
            final long ep = layers[nodeMaxLayer].findOffset(entityId);
            if (ep >= 0) {
                entryPointOffset = ep;
                entryPointEntityId = entityId;
            }
        }
    }

    private SearchResults searchInternal(final MemorySegment query, final int k) {
        final long currentEntryOffset = entryPointOffset;
        if (currentEntryOffset == -1L) {
            return new SearchResults(new long[0], new long[0], new double[0], 0);
        }

        final int currentTopLayer = topLayer.get();
        long entryOffset = currentEntryOffset;

        for (int l = currentTopLayer; l > 0; l--) {
            entryOffset = layers[l].greedySearch(entryOffset, query);
            final long entityIdAtBest = layers[l].viewAt(entryOffset).getEntityId();
            final long lower = layers[l-1].findOffset(entityIdAtBest);
            if (lower >= 0) {
                entryOffset = lower;
            } else {
                final long entryEntityId = layers[l].viewAt(currentEntryOffset).getEntityId();
                final long layer0Offset = layers[0].findOffset(entryEntityId);
                entryOffset = (layer0Offset >= 0) ? layer0Offset : 0L;
                break;
            }
        }

        final int ef = Math.max(k, HNSWConfig.EF_SEARCH);
        final SearchContext ctx = searchContextPool.get();
        ctx.candidates.reset();
        ctx.results.reset();
        ctx.visited.reset();

        final TopKBuffer resultsBuffer = (ctx.results.capacity() >= ef)
                ? ctx.results : new TopKBuffer(ef);
        layers[0].efSearch(entryOffset, query, ef, resultsBuffer, ctx.candidates, ctx.visited);
        resultsBuffer.sortAscending();

        final int resultCount = Math.min(resultsBuffer.size(), k);
        final long[] entityIds = new long[resultCount];
        final long[] distances = new long[resultCount];
        final double[] similarities = new double[resultCount];

        for (int i = 0; i < resultCount; i++) {
            final long nodeOffset = resultsBuffer.offsetAt(i);
            entityIds[i] = layers[0].viewAt(nodeOffset).getEntityId();
            distances[i] = resultsBuffer.distanceAt(i);
            similarities[i] = VectorMath.similarity(distances[i]);
        }
        return new SearchResults(entityIds, distances, similarities, resultCount);
    }

    private static long maxOf(final long[] a, final int len) {
        long m = Long.MIN_VALUE;
        for (int i = 0; i < len; i++) if (a[i] > m) m = a[i];
        return m;
    }

    private static int indexOfMax(final long[] a, final int len) {
        int idx = 0;
        for (int i = 1; i < len; i++) if (a[i] > a[idx]) idx = i;
        return idx;
    }

    private long projectOffsetToLayer(
            final HNSWLayer source, final long offset, final int targetIdx) {
        if (targetIdx < 0 || targetIdx >= maxLayerCount) return -1L;
        return layers[targetIdx].findOffset(source.viewAt(offset).getEntityId());
    }

    private int randomLevel() {
        final double r = ThreadLocalRandom.current().nextDouble();
        final double safe = r == 0.0 ? Double.MIN_VALUE : r;
        return Math.min((int) (-Math.log(safe) * HNSWConfig.ML), maxLayerCount - 1);
    }

    private ThreadLocal<SearchContext> buildContextPool(final long maxNodes) {
        return ThreadLocal.withInitial(() ->
                new SearchContext(
                        new CandidateMinHeap(HNSWConfig.EF_CONSTRUCTION),
                        new TopKBuffer(HNSWConfig.EF_CONSTRUCTION),
                        new VisitedTracker(new OffHeapAllocator(maxNodes), maxNodes)
                )
        );
    }

    /**
     * Per-thread scratch buffers reused across search and insert operations.
     *
     * <p>All three components are allocated once per thread and reset between
     * operations via {@link CandidateMinHeap#reset()}, {@link TopKBuffer#reset()},
     * and {@link VisitedTracker#reset()}.
     *
     * <p>This is a record so all fields are {@code final} and the compiler
     * enforces that no component is accidentally shared across threads.
     */
    private record SearchContext(
            CandidateMinHeap candidates,
            TopKBuffer results,
            VisitedTracker visited
    ) {}
}
