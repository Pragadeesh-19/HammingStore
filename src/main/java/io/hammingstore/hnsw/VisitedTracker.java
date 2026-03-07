package io.hammingstore.hnsw;

import io.hammingstore.memory.OffHeapAllocator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class VisitedTracker {

    private final MemorySegment stamps;
    private final long maxNodes;
    private long generation;

    public VisitedTracker(final OffHeapAllocator allocator, final long maxNodes) {
        if (maxNodes <= 0) throw new IllegalArgumentException("maxNodes must be > 0");
        this.maxNodes = maxNodes;
        this.generation = 1L;
        this.stamps = allocator.allocateRawSegment(maxNodes * Long.BYTES, Long.BYTES);
    }

    public void visit(final long nodeOffset) {
        final long index = nodeOffset / HNSWConfig.NODE_BYTES;
        stamps.set(ValueLayout.JAVA_LONG_UNALIGNED, index * Long.BYTES, generation);
    }

    public boolean isVisited(final long nodeOffset) {
        final long index = nodeOffset / HNSWConfig.NODE_BYTES;
        return stamps.get(ValueLayout.JAVA_LONG_UNALIGNED, index * Long.BYTES) == generation;
    }

    public void reset() {
        if (generation == Long.MAX_VALUE) {
            stamps.fill((byte) 0);
            generation = 1L;
        } else {
            generation++;
        }
    }

    public long generation() { return generation; }

    public long maxNodes() { return maxNodes; }


}
