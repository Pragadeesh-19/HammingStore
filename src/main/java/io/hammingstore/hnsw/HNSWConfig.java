package io.hammingstore.hnsw;

import io.hammingstore.memory.BinaryVector;

public final class HNSWConfig {

    public static final int M = 32;

    public static final int M_LAYER_ZERO = M;

    public static final int EF_CONSTRUCTION = 200;

    public static final int EF_SEARCH = 64;

    public static final int MAX_LAYERS = 16;

    public static final double ML = 1.0/ Math.log(M);

    public static final long NODE_OFFSET_ENTITY_ID       = 0L;

    public static final long NODE_OFFSET_VECTOR          = 8L;

    public static final long NODE_OFFSET_NEIGHBOR_COUNT  = 1264L;

    public static final long NODE_OFFSET_NODE_LAYER      = 1268L;

    public static final long NODE_OFFSET_NEIGHBORS_START = 1272L;

    public static final long NODE_BYTES = 1536L;

    public static final long EMPTY_NEIGHBOR = -1L;

    public static final long VECTOR_BYTES = BinaryVector.VECTOR_BYTES;

    public static final long NEIGHBOUR_ARRAY_BYTES = (long) M * Long.BYTES;

    private HNSWConfig() {
        throw new AssertionError("HNSWConfig is a static constants class");
    }
}
