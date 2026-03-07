package io.hammingstore.hnsw;

import io.hammingstore.math.VectorMath;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class HNSWNodeView {

    private final MemorySegment segment;

    public HNSWNodeView(final MemorySegment segment) {
        if (segment.byteSize() != HNSWConfig.NODE_BYTES) {
            throw new IllegalArgumentException(
                    "Segment must be exactly " + HNSWConfig.NODE_BYTES + " bytes, got: " + segment.byteSize());
        }
        this.segment = segment;
    }

    public long getEntityId() {
        return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, HNSWConfig.NODE_OFFSET_ENTITY_ID);
    }

    public void setEntityId(final long entityId) {
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED, HNSWConfig.NODE_OFFSET_ENTITY_ID, entityId);
    }

    public MemorySegment getVectorSlice() {
        return segment.asSlice(HNSWConfig.NODE_OFFSET_VECTOR, HNSWConfig.VECTOR_BYTES);
    }

    public void setVector(final MemorySegment source) {
        MemorySegment.copy(source, 0L, segment, HNSWConfig.NODE_OFFSET_VECTOR, HNSWConfig.VECTOR_BYTES);
    }

    public int getNeighborCount() {
        return segment.get(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT);
    }

    public void setNeighborCount(final int count) {
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT, count);
    }

    public int getNodeLayer() {
        return segment.get(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NODE_LAYER);
    }

    public void setNodeLayer(final int layer) {
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NODE_LAYER, layer);
    }

    public long getNeighbor(final int index) {
        checkNeighborIndex(index);
        return segment.get(ValueLayout.JAVA_LONG_UNALIGNED,
                HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) index * Long.BYTES);
    }

    public void setNeighbor(final int index, final long offset) {
        checkNeighborIndex(index);
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED,
                HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) index * Long.BYTES, offset);
    }

    public boolean addNeighbor(final long offset) {
        final int count = getNeighborCount();
        if (count >= HNSWConfig.M) {
            return false;
        }
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED,
                HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) count * Long.BYTES, offset);
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT, count + 1);
        return true;
    }

    public void initNeighbors() {
        segment.set(ValueLayout.JAVA_INT_UNALIGNED, HNSWConfig.NODE_OFFSET_NEIGHBOR_COUNT, 0);
        for (int i = 0; i < HNSWConfig.M; i++) {
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED,
                    HNSWConfig.NODE_OFFSET_NEIGHBORS_START + (long) i * Long.BYTES,
                    HNSWConfig.EMPTY_NEIGHBOR);
        }
    }

    public long getVectorDistance(final MemorySegment query) {
        return VectorMath.hammingDistance(getVectorSlice(), query);
    }

    public MemorySegment rawSegment() {
        return segment;
    }

    private static void checkNeighborIndex(final int index) {
        if (index < 0 || index >= HNSWConfig.M) {
            throw new IndexOutOfBoundsException(
                    "Neighbor index " + index + " out of [0, " + HNSWConfig.M + ")");
        }
    }

}
