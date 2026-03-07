package io.hammingstore.hnsw;

import io.hammingstore.math.VectorMath;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.StampedLock;

public final class HNSWIndex implements AutoCloseable {

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
    private AtomicInteger topLayer = new AtomicInteger(0);
    private final ThreadLocal<SearchContext> searchContextPool;

    public HNSWIndex(final long maxNodes) {
        if (maxNodes <= 0) throw new IllegalArgumentException("maxNodes must be > 0");
        this.maxNodes      = maxNodes;
        this.allocator     = new OffHeapAllocator(maxNodes);
        this.maxLayerCount = computeMaxLayers(maxNodes);
        this.layers        = new HNSWLayer[maxLayerCount];

        for (int l = 0; l < maxLayerCount; l++) {
            layers[l] = new HNSWLayer(allocator, computeLayerCapacity(maxNodes, l), l);
        }

        final long totalNodes = maxNodes;
        this.searchContextPool = ThreadLocal.withInitial(() ->
                new SearchContext(
                        new CandidateMinHeap(HNSWConfig.EF_CONSTRUCTION),
                        new TopKBuffer(HNSWConfig.EF_CONSTRUCTION),
                        new VisitedTracker(new OffHeapAllocator(totalNodes), totalNodes)
                )
        );
    }

    public HNSWIndex(
            final HNSWLayer[] prebuildLayers,
            final long maxNodes,
            final long initialEntryPointOffset,
            final int initialTopLayer) {
        this.layers = prebuildLayers;
        this.maxLayerCount = prebuildLayers.length;
        this.maxNodes = maxNodes;
        this.allocator = null;
        this.entryPointOffset = initialEntryPointOffset;
        this.topLayer = new AtomicInteger(initialTopLayer);

        final long totalNodes = maxNodes;
        this.searchContextPool = ThreadLocal.withInitial(() ->
                new SearchContext(
                        new CandidateMinHeap(HNSWConfig.EF_CONSTRUCTION),
                        new TopKBuffer(HNSWConfig.EF_CONSTRUCTION),
                        new VisitedTracker(new OffHeapAllocator(totalNodes), totalNodes)
                )
        );
    }

    public void insertBinary(final long entityId, final MemorySegment binaryVec) {
        if (binaryVec.byteSize() != BinaryVector.VECTOR_BYTES) {
            throw new IllegalArgumentException(
                    "binaryVec must be " + BinaryVector.VECTOR_BYTES + " bytes");
        }
        final long stamp = lock.writeLock();
        try {
            insertInternal(entityId, binaryVec);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    public void insert(final long entityId, final float[] denseEmbedding) {
        final SearchContext ctx = searchContextPool.get();
        binarizeInto(denseEmbedding, ctx.binaryScratch);
        insertBinary(entityId, ctx.binaryScratch);
    }

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

    public SearchResults search(final float[] queryEmbedding, final int k) {
        final SearchContext ctx = searchContextPool.get();
        binarizeInto(queryEmbedding, ctx.binaryScratch);
        return searchBinary(ctx.binaryScratch, k);
    }

    public SearchResults bruteForceSearch(final MemorySegment query, final int k) {
        if (k <= 0) throw new IllegalArgumentException("K must be > 0");

        long stamp = lock.tryOptimisticRead();
        SearchResults result = bruteForceInternal(query, k);
        if (!lock.validate(stamp)) {
            stamp = lock.readLock();
            try {
                result = bruteForceInternal(query, k);
            } finally {
                lock.unlockRead(stamp);
            }
        }
        return result;
    }

    public int activeLayerCount() { return topLayer.get() + 1; }

    public long entryPointOffset() { return entryPointOffset; }

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

    private void insertInternal(final long entityId, final MemorySegment binaryVec) {
        final int  nodeMaxLayer = randomLevel();
        final long currentEntryOffset = entryPointOffset;
        final int  currentTopLayer = topLayer.get();
        long       entryOffset = currentEntryOffset;

        if (entryOffset != -1L) {
            for (int l = currentTopLayer; l > nodeMaxLayer; l--) {
                entryOffset = layers[l].greedySearch(entryOffset, binaryVec);
                final long lower = projectOffsetToLayer(layers[l], entryOffset, l - 1);
                if (lower >= 0) {
                    entryOffset = lower;
                } else {
                    final long fallback = layers[l-1].findOffset(
                            layers[l].viewAt(entryPointOffset).getEntityId());
                    if (fallback >= 0) entryOffset = fallback;
                    break;
                }
            }
        }

        final SearchContext ctx = searchContextPool.get();
        final int insertTopLayer = (entryOffset == -1L) ? 0 : Math.min(nodeMaxLayer, currentTopLayer);

        for (int l = insertTopLayer; l >= 0; l--) {
            final long newNodeOffset = layers[l].allocateNode(entityId, binaryVec, nodeMaxLayer);

            if (entryOffset == -1L) {
                entryPointOffset = newNodeOffset;
                topLayer.set(nodeMaxLayer);
                return;
            }

            ctx.candidates.reset();
            ctx.results.reset();
            ctx.visited.reset();

            final int ef = (l == 0) ? HNSWConfig.EF_CONSTRUCTION : HNSWConfig.M;
            layers[l].efSearch(entryOffset, binaryVec, ef, ctx.results, ctx.candidates, ctx.visited);

            final int mMax = (l == 0) ? HNSWConfig.M_LAYER_ZERO : HNSWConfig.M;
            layers[l].connectNeighbors(newNodeOffset, ctx.results, mMax);

            if (l > 0 && ctx.results.size() > 0) {
                ctx.results.sortAscending();
                final long bestEntityId = layers[l].viewAt(ctx.results.offsetAt(0)).getEntityId();
                final long lowerOffset  = layers[l - 1].findOffset(bestEntityId);
                entryOffset = (lowerOffset >= 0) ? lowerOffset : newNodeOffset;
            }
        }

        if (nodeMaxLayer > currentTopLayer) {
            topLayer.set(nodeMaxLayer);
            entryPointOffset = layers[nodeMaxLayer].findOffset(entityId);
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
        final long[]   entityIds    = new long[resultCount];
        final long[]   distances    = new long[resultCount];
        final double[] similarities = new double[resultCount];

        for (int i = 0; i < resultCount; i++) {
            final long nodeOffset = resultsBuffer.offsetAt(i);
            entityIds[i]    = layers[0].viewAt(nodeOffset).getEntityId();
            distances[i]    = resultsBuffer.distanceAt(i);
            similarities[i] = VectorMath.similarity(distances[i]);
        }
        return new SearchResults(entityIds, distances, similarities, resultCount);
    }

    private SearchResults bruteForceInternal(final MemorySegment query, final int k) {
        final long nodeCount = layers[0].nodeCount();
        if (nodeCount == 0) {
            return new SearchResults(new long[0], new long[0], new double[0], 0);
        }

        final int cap = (int) Math.min(nodeCount, k);
        final long[] topOffsets   = new long[cap];
        final long[] topDistances = new long[cap];
        int found = 0;
        long worstDist = Long.MAX_VALUE;

        for (long slot = 0; slot < nodeCount; slot++) {
            final long offset = slot * HNSWConfig.NODE_BYTES;
            final long dist   = layers[0].viewAt(offset).getVectorDistance(query);

            if (found < cap) {
                topOffsets[found]   = offset;
                topDistances[found] = dist;
                found++;
                if (found == cap) {
                    worstDist = maxOf(topDistances, cap);
                }
            } else if (dist < worstDist) {
                final int wi = indexOfMax(topDistances, cap);
                topOffsets[wi]   = offset;
                topDistances[wi] = dist;
                worstDist = maxOf(topDistances, cap);
            }
        }

        for (int i = 1; i < found; i++) {
            final long keyOff  = topOffsets[i];
            final long keyDist = topDistances[i];
            int j = i - 1;
            while (j >= 0 && topDistances[j] > keyDist) {
                topOffsets[j + 1]   = topOffsets[j];
                topDistances[j + 1] = topDistances[j];
                j--;
            }
            topOffsets[j + 1]   = keyOff;
            topDistances[j + 1] = keyDist;
        }

        final long[]   entityIds    = new long[found];
        final double[] similarities = new double[found];
        for (int i = 0; i < found; i++) {
            entityIds[i]    = layers[0].viewAt(topOffsets[i]).getEntityId();
            similarities[i] = VectorMath.similarity(topDistances[i]);
        }
        return new SearchResults(entityIds, topDistances, similarities, found);
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

    private static int randomLevel() {
        final double r = ThreadLocalRandom.current().nextDouble();
        final double safe = r == 0.0 ? Double.MIN_VALUE : r;
        return Math.min((int) (-Math.log(safe) * HNSWConfig.ML), HNSWConfig.MAX_LAYERS - 1);
    }

    private static void binarizeInto(final float[] embedding, final MemorySegment dest) {
        dest.fill((byte) 0);
        final int floatCount = Math.min(embedding.length, (int) VectorMath.TOTAL_BITS);
        for (int i = 0; i < floatCount; i++) {
            if (embedding[i] > 0.0f) {
                final int  wordIndex = i >>> 6;
                final int  bitIndex  = i & 63;
                final long byteOff   = (long) wordIndex * Long.BYTES;
                dest.set(ValueLayout.JAVA_LONG_UNALIGNED, byteOff,
                        dest.get(ValueLayout.JAVA_LONG_UNALIGNED, byteOff) | (1L << bitIndex));
            }
        }
    }

    public static int computeMaxLayers(final long n) {
        if (n <= 1) return 1;
        return Math.min((int) Math.ceil(Math.log(n) / Math.log(HNSWConfig.M)) + 2,
                HNSWConfig.MAX_LAYERS);
    }

    public static long computeLayerCapacity(final long n, final int l) {
        if (l == 0) return n;
        return Math.max(16L, (long) (n * Math.exp(-l) * 1.2));
    }

    private record SearchContext(
            CandidateMinHeap candidates,
            TopKBuffer       results,
            VisitedTracker   visited
    ) {
        static MemorySegment binaryScratch;
        SearchContext {
            binaryScratch = new OffHeapAllocator(1L).allocateRawSegment(
                    BinaryVector.VECTOR_BYTES, Long.BYTES);
        }
    }
}
