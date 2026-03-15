package io.hammingstore.hnsw;

import io.hammingstore.memory.OffHeapAllocator;
import io.hammingstore.memory.SparseEntityIndex;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One layer of the HNSW proximity graph.
 *
 * <p>Each layer holds a flat array of fixed-size nodes in off-heap memory.
 * Every node contains an entity ID, a binary hypervector, a neighbour count,
 * the node's max layer assignment, and up to {@link HNSWConfig#M} neighbour
 * offsets. See {@link HNSWConfig} for the exact byte layout.
 *
 * <p>An accompanying {@link SparseEntityIndex} maps entity IDs to their byte
 * offsets within this layer's storage segment, enabling O(1) entity lookup.
 *
 * <h2>Visibility contract</h2>
 * <p>Methods used only by {@link HNSWIndex} (same package) are package-private.
 * Methods used by {@code VectorGraphRepository} for persistence are public.
 */
public final class HNSWLayer {

    private final MemorySegment storage;
    private final SparseEntityIndex nodeIndex;
    private final AtomicLong cursor;
    private final long capacity;
    private final int layerIndex;

    /**
     * Creates a new empty layer with freshly allocated off-heap storage.
     *
     * @param allocator the allocator from which node storage is requested
     * @param capacity maximum number of nodes this layer can hold
     * @param layerIndex the layer number (0 = base layer, higher = sparser)
     * @throws IllegalArgumentException if {@code capacity} is not positive
     */
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

    /**
     * Restores a layer from memory-mapped storage produced by a previous run.
     * Called by {@code VectorGraphRepository} during snapshot restore.
     *
     * @param nodeStorage the mapped segment containing serialised nodes
     * @param nodeIndex the restored entity→offset index
     * @param capacity the maximum node count this layer was built for
     * @param layerIndex the layer number
     * @param initialNodeCount the number of nodes already written at restore time
     * @return the restored layer
     */
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

    /**
     * Allocates and initialises a new node slot for {@code entityId}.
     *
     * @param entityId the entity this node represents
     * @param vectorSrc the binary hypervector to store in the node
     * @param nodeMaxLayer the highest layer this entity was assigned to
     * @return the byte offset of the newly allocated node within {@code storage}
     * @throws IllegalStateException if the layer is already at capacity
     */
     long allocateNode(
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

        nodeIndex.put(SparseEntityIndex.mixHash64(entityId), offset);
        return offset;
    }

    /**
     * Greedily moves to the nearest neighbour of the current node until no
     * improvement is found (local minimum).
     *
     * <p>Used during insertion to descend through upper layers quickly before
     * the full beam search begins at the insertion layers.
     *
     * @param entryOffset byte offset of the starting node
     * @param query the query hypervector
     * @return the byte offset of the locally nearest node found
     */
     long greedySearch(long entryOffset, final MemorySegment query) {
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
                return currentOffset;
            }
            currentOffset = bestOffset;
        }
    }

    /**
     * Beam search (ef-Search) from {@code entryOffset}: explores the graph
     * using a frontier of up to {@code ef} candidates and accumulates the best
     * results in {@code results}.
     *
     * <p>Algorithm (from the original HNSW paper, Algorithm 2):
     * <pre>
     *   candidates <- {entry}        // min-heap ordered by distance to query
     *   results <- {entry}        // top-k buffer (max-heap, bounded by ef)
     *   visited <- {entry}
     *
     *   while candidates not empty:
     *     c ← candidates.popMin()
     *     if dist(c, query) &gt; results.worstDistance: break   // pruning condition
     *     for each neighbour n of c:
     *       if n not visited:
     *         visited.add(n)
     *         if dist(n, query) &lt; results.worstDistance or results not full:
     *           candidates.push(n)
     *           results.offer(n)
     * </pre>
     *
     * @param entryOffset byte offset of the entry-point node
     * @param query the query hypervector
     * @param ef maximum frontier size (controls recall vs. speed tradeoff)
     * @param results accumulator for the best candidates found; must be reset before call
     * @param candidates pre-allocated frontier heap; must be reset before call
     * @param visited pre-allocated visited tracker; must be reset before call
     */
     void efSearch(
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
                    results.offer(nOffset, nDist);
                    if (candidates.size() < ef) {
                        try {
                            candidates.push(nOffset, nDist);
                        } catch (IllegalStateException ignored) {

                        }
                    }
                }
            }
        }
    }

    /**
     * Wires {@code newNodeOffset} to its best neighbours from {@code candidates}
     * and performs reciprocal back-linking. If a neighbour's list is already full,
     * {@link #pruneNeighborList} selects the best {@code mMax} connections.
     *
     * @param newNodeOffset byte offset of the newly inserted node
     * @param candidates the candidate set from {@link #efSearch}; will be sorted
     * @param mMax maximum allowed neighbours per node at this layer
     */
     void connectNeighbors(
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
        return nodeIndex.getOffset(SparseEntityIndex.mixHash64(entityId));
    }

    public long nodeCount() {
        return cursor.get();
    }

    public SparseEntityIndex nodeIndex() {
        return nodeIndex;
    }

    HNSWNodeView viewAt(final long offset) {
        return new HNSWNodeView(storage.asSlice(offset, HNSWConfig.NODE_BYTES));
    }

    /**
     * Replaces {@code node}'s neighbour list with the best {@code mMax}
     * connections from its current neighbours plus {@code candidateOffset}.
     *
     * <p>Collects all existing neighbours and the new candidate into a scratch
     * array, sorts by distance to {@code node}'s vector, and writes the
     * closest {@code mMax} back to the node. This is a simplified heuristic;
     * the original paper's RNG pruning could be applied here for better
     * graph diversity at the cost of additional complexity.
     *
     * @param node the node whose neighbour list is being pruned
     * @param candidateOffset byte offset of the incoming candidate to consider
     * @param mMax maximum number of neighbours to keep
     */
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
            final long keyOff = scratchOffsets[i];
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
