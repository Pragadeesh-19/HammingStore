package io.hammingstore.hnsw;

import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Tracks which HNSW nodes have been visited during a single beam search.
 *
 * <h2>Generation-counter design</h2>
 * <p>A naive visited set would require clearing an array of {@code maxNodes} longs
 * before every search - O(N) per query. This implementation avoids that cost
 * entirely using a generation counter:
 * <ul>
 *   <li>Each slot in {@code stamps} holds the generation number at which that node
 *       was last visited.</li>
 *   <li>A node is considered visited if and only if {@code stamps[slot] == generation}.</li>
 *   <li>{@link #reset()} increments {@code generation} by one. All existing stamps
 *       immediately become stale - the equivalent of clearing the entire array -
 *       in O(1).</li>
 * </ul>
 *
 * <p>The only edge case is generation overflow: when {@code generation} reaches
 * {@link Long#MAX_VALUE}, the stamps array is explicitly zeroed and the counter
 * resets to 1. At one search per microsecond this occurs after roughly
 * 292,000 years, so it is purely a correctness safeguard.
 *
 * <p>This class is package-private: it is an implementation detail of
 * {@link HNSWIndex} and {@link HNSWLayer}.
 */
final class VisitedTracker {

    private final MemorySegment stamps;
    private long generation;

    /**
     * Creates a tracker that can cover up to {@code maxNodes} distinct nodes.
     *
     * @param allocator the allocator from which to request stamp storage
     * @param maxNodes the maximum number of nodes in the layer being searched;
     *                 must match the layer capacity so that every valid node
     *                 offset maps to a slot within {@code stamps}
     * @throws IllegalArgumentException if {@code maxNodes} is not positive
     */
     VisitedTracker(final OffHeapAllocator allocator, final long maxNodes) {
        if (maxNodes <= 0) throw new IllegalArgumentException("maxNodes must be > 0");
        this.generation = 1L;
        this.stamps = allocator.allocateRawSegment(maxNodes * Long.BYTES, Long.BYTES);
    }

    /**
     * Marks the node at {@code nodeOffset} as visited in the current search.
     *
     * @param nodeOffset byte offset of the node within its layer's storage segment;
     *                   must be a multiple of {@link HNSWConfig#NODE_BYTES}
     */
     void visit(final long nodeOffset) {
        final long slot = nodeOffset / HNSWConfig.NODE_BYTES;
        stamps.set(ValueLayout.JAVA_LONG_UNALIGNED, slot * Long.BYTES, generation);
    }

    /**
     * Returns {@code true} if the node at {@code nodeOffset} has been visited
     * in the current search generation.
     *
     * @param nodeOffset byte offset of the node within its layer's storage segment
     */
     boolean isVisited(final long nodeOffset) {
        final long slot = nodeOffset / HNSWConfig.NODE_BYTES;
        return stamps.get(ValueLayout.JAVA_LONG_UNALIGNED, slot * Long.BYTES) == generation;
    }

    /**
     * Resets the tracker for a new search in O(1).
     *
     * <p>Increments the generation counter, instantly invalidating all existing
     * stamps without touching the stamps array. If the counter would overflow
     * {@link Long#MAX_VALUE}, the stamps array is explicitly zeroed and the
     * counter resets to 1.
     */
     void reset() {
        if (generation == Long.MAX_VALUE) {
            stamps.fill((byte) 0);
            generation = 1L;
        } else {
            generation++;
        }
    }
}
