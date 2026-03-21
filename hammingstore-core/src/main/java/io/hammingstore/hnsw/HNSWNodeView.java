package io.hammingstore.hnsw;

import io.hammingstore.memory.OffHeapVectorStore;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A structured view over a single HNSW node stored in off-heap memory.
 *
 * <p>A node is a fixed-size record of exactly {@link HNSWConfig#NODE_BYTES} bytes,
 * laid out as described in {@link HNSWConfig}. This class provides typed accessors
 * that read and write directly into the underlying {@link MemorySegment} with no
 * heap allocation - it is a zero-copy lens over the raw storage.
 *
 * <h2>SCHEMA_V2 layout</h2>
 * <p>The binary vector has been removed from the node. In its place, the node
 * stores a {@link HNSWConfig#NODE_OFFSET_VECTOR_STORE_OFFSET vectorStoreOffset}
 * - a direct byte offset into {@link io.hammingstore.memory.OffHeapVectorStore}.
 * This allows {@link HNSWLayer#distanceTo} to fetch the vector in a single
 * arithmetic operation with no hash lookup.
 *
 * <p>Instances are lightweight and short-lived. They are created on demand by
 * {@link HNSWLayer#viewAt(long)} and discarded after each graph operation.
 *
 * <p>This class is package-private: it is an implementation detail of the
 * {@code io.hammingstore.hnsw} package and must not be exposed to callers.
 */
final class HNSWNodeView {

    private final MemorySegment segment;

    /**
     * Wraps {@code segment} as a node view.
     *
     * @param segment the backing segment; must be exactly {@link HNSWConfig#NODE_BYTES} bytes
     * @throws IllegalArgumentException if the segment has the wrong size
     */
     HNSWNodeView(final MemorySegment segment) {
        if (segment.byteSize() != HNSWConfig.NODE_BYTES) {
            throw new IllegalArgumentException(
                    "Segment must be exactly " + HNSWConfig.NODE_BYTES
                            + " bytes (SCHEMA_V2 node layout), got: "
                            + segment.byteSize());
        }
        this.segment = segment;
    }

    /**
     * Returns the direct byte offset of this node's binary vector within
     * {@link io.hammingstore.memory.OffHeapVectorStore}.
     *
     * <p>This is the hot-loop accessor. It is in the same 280-byte cache line
     * as the entity ID and neighbor count, so it is typically already in L1
     * when {@link HNSWLayer#distanceTo} is called.
     */
    long getVectorStoreOffset() {
         return segment.get(ValueLayout.JAVA_LONG_UNALIGNED,
                 HNSWConfig.NODE_OFFSET_VECTOR_STORE_OFFSET);
    }

    /**
     * Writes the direct byte offset of this node's binary vector.
     * Called exactly once, from {@link HNSWLayer#allocateNode}, at insert time.
     *
     * @param offset the byte offset returned by {@link OffHeapVectorStore#allocateSlot()}
     */
    void setVectorStoreOffset(final long offset) {
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED,
                HNSWConfig.NODE_OFFSET_VECTOR_STORE_OFFSET, offset);
    }

     long getEntityId() {
        return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, HNSWConfig.NODE_OFFSET_ENTITY_ID);
    }

    void setEntityId(final long entityId) {
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED, HNSWConfig.NODE_OFFSET_ENTITY_ID, entityId);
    }

     int getNeighborCount() {
        return segment.get(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT);
    }

    void setNeighborCount(final int count) {
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT, count);
    }

     void setNodeLayer(final int layer) {
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NODE_LAYER, layer);
    }

    /**
     * Returns the byte offset of the neighbour at {@code index}, or
     * {@link HNSWConfig#EMPTY_NEIGHBOR} if the slot is unused.
     *
     * @throws IndexOutOfBoundsException if {@code index} is out of {@code [0, M)}
     */
     long getNeighbor(final int index) {
        checkNeighborIndex(index);
        return segment.get(ValueLayout.JAVA_LONG_UNALIGNED,
                HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) index * Long.BYTES);
    }

    /**
     * Appends {@code offset} to the neighbour list if space remains.
     *
     * @return {@code true} if the neighbour was added; {@code false} if the list
     *         was already full (caller must then prune)
     */
     boolean addNeighbor(final long offset) {
        final int count = getNeighborCount();
        if (count >= HNSWConfig.M) {
            return false;
        }
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED,
                HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) count * Long.BYTES, offset);
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT, count + 1);
        return true;
    }

    /**
     * Resets the neighbour count to zero and fills all {@link HNSWConfig#M} slots
     * with {@link HNSWConfig#EMPTY_NEIGHBOR}. Called immediately after a new node
     * is allocated to ensure no stale data from a previous run is interpreted as
     * a live neighbour.
     */
     void initNeighbors() {
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT, 0);
        for (int i = 0; i < HNSWConfig.M; i++) {
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED,
                    HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) i * Long.BYTES,
                    HNSWConfig.EMPTY_NEIGHBOR);
        }
    }

    private static void checkNeighborIndex(final int index) {
        if (index < 0 || index >= HNSWConfig.M) {
            throw new IndexOutOfBoundsException(
                    "Neighbor index " + index + " out of [0, " + HNSWConfig.M + ")");
        }
    }

}
