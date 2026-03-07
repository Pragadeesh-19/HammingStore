package io.hammingstore.hnsw;

import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.memory.SparseEntityIndex;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicLong;

public final class HNSWLayer {

    private final MemorySegment storage;
    private final SparseEntityIndex nodeIndex;
    private final AtomicLong cursor;
    private final long capacity;
    private final int layerIndex;

    public HNSWLayer(
            final OffHeapAllocator allocator,
            final long capacity,
            final int layerIndex) {

        if (capacity <= 0) throw new IllegalArgumentException("capacity must be > 0");
        this.capacity   = capacity;
        this.layerIndex = layerIndex;
        this.cursor     = new AtomicLong(0L);
        this.storage    = allocator.allocateRawSegment(
                capacity * HNSWConfig.NODE_BYTES, Long.BYTES);
        this.nodeIndex  = new SparseEntityIndex(allocator, capacity);
    }

    public static HNSWLayer fromMapped(
            final MemorySegment nodeStorage,
            final SparseEntityIndex nodeIndex,
            final long capacity,
            final int layerIndex,
            final long initialNodeCount) {
        return new HNSWLayer(nodeStorage, nodeIndex, capacity, layerIndex, initialNodeCount);
    }

    private HNSWLayer(
            final MemorySegment storage,
            final SparseEntityIndex nodeIndex,
            final long capacity,
            final int layerIndex,
            final long initialNodeCount) {
        this.storage = storage;
        this.nodeIndex = nodeIndex;
        this.capacity = capacity;
        this.layerIndex = layerIndex;
        this.cursor = new AtomicLong(initialNodeCount);
    }

    public long allocateNode(
            final long entityId,
            final MemorySegment vectorSrc,
            final int nodeMaxLayer) {

        final long slotIndex = cursor.getAndIncrement();
        if (slotIndex >= capacity) {
            cursor.decrementAndGet();
            throw new IllegalStateException(
                    "HNSWLayer[" + layerIndex + "] full: capacity=" + capacity);
        }
        final long offset = slotIndex * HNSWConfig.NODE_BYTES;
        final HNSWNodeView view = viewAt(offset);

        view.setEntityId(entityId);
        view.setVector(vectorSrc);
        view.setNodeLayer(nodeMaxLayer);
        view.initNeighbors();

        nodeIndex.put(SparseEntityIndex.xxHash3Stub(entityId), offset);
        return offset;
    }

    public long greedySearch(long entryOffset, final MemorySegment query) {
        long currentOffset = entryOffset;

        while (true) {
            final HNSWNodeView current = viewAt(currentOffset);
            final long currentDist    = current.getVectorDistance(query);
            long bestOffset = currentOffset;
            long bestDist   = currentDist;

            final int neighborCount = current.getNeighborCount();
            for (int i = 0; i < neighborCount; i++) {
                final long neighborOffset = current.getNeighbor(i);
                if (neighborOffset == HNSWConfig.EMPTY_NEIGHBOR) continue;

                final long d = viewAt(neighborOffset).getVectorDistance(query);
                if (d < bestDist) {
                    bestDist   = d;
                    bestOffset = neighborOffset;
                }
            }

            if (bestOffset == currentOffset) {
                return currentOffset; // local minimum
            }
            currentOffset = bestOffset;
        }
    }

    public void efSearch(
            final long entryOffset,
            final MemorySegment query,
            final int ef,
            final TopKBuffer results,
            final CandidateMinHeap candidates,
            final VisitedTracker visited) {

        final long entryDist = viewAt(entryOffset).getVectorDistance(query);
        candidates.push(entryOffset, entryDist);
        results.offer(entryOffset, entryDist);
        visited.visit(entryOffset);

        while (!candidates.isEmpty()) {
            final long cDist   = candidates.peekMinDistance();
            final long worstResult = results.peekMaxDistance();

            if (cDist > worstResult) break;

            final long cOffset = candidates.popMinOffset();
            final HNSWNodeView cView = viewAt(cOffset);
            final int neighborCount  = cView.getNeighborCount();

            for (int i = 0; i < neighborCount; i++) {
                final long nOffset = cView.getNeighbor(i);
                if (nOffset == HNSWConfig.EMPTY_NEIGHBOR) continue;
                if (visited.isVisited(nOffset)) continue;

                visited.visit(nOffset);
                final long nDist = viewAt(nOffset).getVectorDistance(query);

                if (nDist < worstResult || !results.isFull()) {
                    if (!candidates.isEmpty() && candidates.size() < ef) {
                        candidates.push(nOffset, nDist);
                    } else if (nDist < candidates.peekMinDistance() || candidates.size() < ef) {
                        // Only push if we have room (bounded by ef)
                        try { candidates.push(nOffset, nDist); }
                        catch (IllegalStateException ignored) { /* candidates at capacity */ }
                    }
                    results.offer(nOffset, nDist);
                }
            }
        }
    }

    public void connectNeighbors(
            final long newNodeOffset,
            final TopKBuffer candidates,
            final int mMax) {

        candidates.sortAscending(); // best (lowest dist) first
        final HNSWNodeView newNode = viewAt(newNodeOffset);

        final int toConnect = Math.min(candidates.size(), mMax);
        for (int i = 0; i < toConnect; i++) {
            final long neighborOffset = candidates.offsetAt(i);
            newNode.addNeighbor(neighborOffset);

            final HNSWNodeView neighbor = viewAt(neighborOffset);
            if (!neighbor.addNeighbor(newNodeOffset)) {
                pruneNeighborList(neighbor, newNodeOffset, mMax);
            }
        }
    }

    public long findOffset(final long entityId) {
        return nodeIndex.getOffset(SparseEntityIndex.xxHash3Stub(entityId));
    }

    public long nodeCount() {
        return cursor.get();
    }

    public SparseEntityIndex nodeIndex() {
        return nodeIndex;
    }

    public MemorySegment rawStorage() {
        return storage;
    }

    public int layerIndex() {
        return layerIndex;
    }

    HNSWNodeView viewAt(final long offset) {
        return new HNSWNodeView(storage.asSlice(offset, HNSWConfig.NODE_BYTES));
    }

    private void pruneNeighborList(
            final HNSWNodeView node,
            final long candidateOffset,
            final int mMax) {

        final int existing = node.getNeighborCount();
        final long[] scratchOffsets   = new long[mMax + 1];
        final long[] scratchDistances = new long[mMax + 1];
        int scratchSize = 0;

        final MemorySegment nodeVec = node.getVectorSlice();
        for (int i = 0; i < existing; i++) {
            final long off = node.getNeighbor(i);
            if (off != HNSWConfig.EMPTY_NEIGHBOR) {
                scratchOffsets[scratchSize]   = off;
                scratchDistances[scratchSize] = viewAt(off).getVectorDistance(nodeVec);
                scratchSize++;
            }
        }
        scratchOffsets[scratchSize]   = candidateOffset;
        scratchDistances[scratchSize] = viewAt(candidateOffset).getVectorDistance(nodeVec);
        scratchSize++;

        for (int i = 1; i < scratchSize; i++) {
            final long keyOff  = scratchOffsets[i];
            final long keyDist = scratchDistances[i];
            int j = i - 1;
            while (j >= 0 && scratchDistances[j] > keyDist) {
                scratchOffsets[j + 1]   = scratchOffsets[j];
                scratchDistances[j + 1] = scratchDistances[j];
                j--;
            }
            scratchOffsets[j + 1]   = keyOff;
            scratchDistances[j + 1] = keyDist;
        }

        node.setNeighborCount(0);
        node.initNeighbors();
        final int newCount = Math.min(scratchSize, mMax);
        for (int i = 0; i < newCount; i++) {
            node.addNeighbor(scratchOffsets[i]);
        }
    }

}
