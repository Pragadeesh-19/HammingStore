package io.hammingstore.hnsw;

/**
 * Compile-time constants for the HNSW graph index.
 *
 * <p>All tuneable parameters ({@link #M}, {@link #EF_CONSTRUCTION}, etc.) and all
 * structural layout constants ({@link #NODE_BYTES}, {@link #NODE_OFFSET_VECTOR_STORE_OFFSET},
 * etc.) live here so they have a single authoritative source across
 * {@link HNSWLayer}, {@link HNSWNodeView}, {@link HNSWIndex} and
 * {@link io.hammingstore.hnsw.VisitedTracker}.
 *
 * <h2>On-disk layout contract - SCHEMA_V2</h2>
 * <p>The {@code NODE_OFFSET_*} and {@code NODE_BYTES} constants define the exact
 * binary layout of every node written to the memory-mapped layer files. Changing
 * any of these values invalidates all persisted indexes. Never change them without
 * a corresponding migration and a version bump in
 * {@link io.hammingstore.persist.EngineSnapshot}.
 *
 * <h2>Node memory layout - SCHEMA_V2 (280 bytes per node)</h2>
 * <pre>
 * Offset  Size   Field
 * ------  ----   -----
 *      0     8   entityId              (long)
 *      8     8   vectorStoreOffset     (long) - direct byte offset into OffHeapVectorStore
 *     16     4   neighborCount         (int)
 *     20     4   nodeLayer             (int)
 *     24   256   neighborOffsets       (32 longs × 8 bytes = M × Long.BYTES)
 *                                      ─────────────────────────────────────
 *                                      Total: 280 bytes (naturally 8-byte aligned)
 * </pre>
 *
 * <h2>Design rationale — why vectorStoreOffset lives in the node</h2>
 * <p>During {@link HNSWLayer#efSearch}, the engine examines hundreds of nodes and
 * up to 32 neighbours per node. Every examination requires a Hamming distance
 * computation, which requires the stored binary vector.
 *
 * <p>Storing {@code vectorStoreOffset} directly in the node means fetching the vector
 * is a single arithmetic operation: {@code vectorStore.sliceAt(vectorStoreOffset)}.
 * No hash computation, no hash-map lookup, no pointer chase through
 * {@link io.hammingstore.memory.SparseEntityIndex}. The offset is in the same
 * 280-byte cache line as the neighbour list, so it is frequently already in L1
 * when the distance call is made.
 *
 * <p>This is 8 bytes more per node than the pure-metadata layout (272 bytes), but
 * eliminates up to 1,472 hash-map lookups per search query that would otherwise
 * destroy L3 cache locality. The memory cost is negligible; the latency benefit
 * is decisive.
 *
 * <h2>Breaking change from SCHEMA_V1</h2>
 * <p>SCHEMA_V1 embedded the full 1,256-byte binary vector in the node
 * (NODE_BYTES = 1,536). SCHEMA_V2 replaces it with an 8-byte direct offset
 * pointer, reducing node size by 5.49x. Existing SCHEMA_V1 snapshot files are
 * rejected at startup with a migration message.
 */
public final class HNSWConfig {

    /**
     * Maximum number of bidirectional links per node in layers 1 and above.
     * Higher M improves recall at the cost of memory and insert time.
     * The value 32 is the recommended default for high-dimensional binary vectors.
     */
    public static final int M = 32;

    /**
     * Maximum number of bidirectional links per node in layer 0.
     *
     * <p><b>Note:</b> the original HNSW paper (Malkov &amp; Yashunin, 2018)
     * recommends {@code M_LAYER_ZERO = 2 × M} because layer 0 is searched
     * exhaustively during the final beam-search phase and benefits from higher
     * connectivity. This engine intentionally sets {@code M_LAYER_ZERO = M} to
     * keep the node layout uniform across all layers — every node is exactly
     * {@link #NODE_BYTES} bytes regardless of which layer it lives in. This
     * simplifies persistence and off-heap memory management at the cost of a
     * small recall reduction on very large datasets.
     *
     * <p>Do not change this value to {@code 2 * M} without also updating
     * {@link #NODE_BYTES}, {@link #NODE_OFFSET_NEIGHBORS_START}, and all
     * dependent offset constants, as well as bumping the snapshot version.
     */
    public static final int M_LAYER_ZERO = M;

    /**
     * Size of the candidate set during index construction.
     * Higher values improve graph quality at the cost of insert throughput.
     * 200 is the recommended default.
     */
    public static final int EF_CONSTRUCTION = 200;

    /**
     * Minimum candidate set size during search.
     * The actual ef used is {@code max(k, EF_SEARCH)} where k is the number
     * of results requested. Higher values improve recall at the cost of
     * query latency.
     */
    public static final int EF_SEARCH = 64;

    /**
     * Maximum number of graph layers. Layer assignment is sampled from an
     * exponential distribution; in practice fewer than 8 layers are used for
     * datasets up to ~10M vectors with M=32.
     */
    public static final int MAX_LAYERS = 16;

    /**
     * Level generation multiplier: {@code 1 / ln(M)}.
     * A random level {@code l} is drawn as {@code floor(-ln(uniform(0,1)) x ML)},
     * giving an exponential distribution with expected layer occupancy ratio of
     * {@code 1/M} between consecutive layers.
     */
    public static final double ML = 1.0/ Math.log(M);

    public static final long NODE_OFFSET_ENTITY_ID = 0L;

    /**
     * Byte offset of the vectorStoreOffset field (long, 8 bytes).
     * Holds the direct byte offset of this entity's binary vector inside
     * {@link io.hammingstore.memory.OffHeapVectorStore}. Written once at
     * {@link HNSWLayer#allocateNode} time and never updated.
     */
    public static final long NODE_OFFSET_VECTOR_STORE_OFFSET = 8L;

    /**
     * Byte offset of the neighborCount field (int, 4 bytes).
     * Follows vectorStoreOffset: 8 + 8 = 16.
     */
    public static final long NODE_OFFSET_NEIGHBOR_COUNT = 16L;

    /**
     * Byte offset of the nodeLayer field (int, 4 bytes).
     * Follows neighborCount: 16 + 4 = 20.
     */
    public static final long NODE_OFFSET_NODE_LAYER = 20L;

    /**
     * Byte offset of the neighbor-offset array within a node.
     * Immediately follows the node-layer field (4-byte int): 20 + 4 = 24.
     * The array contains {@link #M} longs, each holding the byte
     * offset of a neighbor node within the same layer's storage segment, or
     * {@link #EMPTY_NEIGHBOR} if the slot is unused.
     */
    public static final long NODE_OFFSET_NEIGHBORS_START = 24L;

    /**
     * Total size of one node in bytes — SCHEMA_V2.
     * Layout: 8 (entityId) + 8 (vectorStoreOffset) + 4 (count) + 4 (layer)
     *         + 256 (32 neighbours × 8) = 280.
     * Naturally 8-byte aligned. No padding required.
     */
    public static final long NODE_BYTES = 280L;

    public static final long EMPTY_NEIGHBOR = -1L;

    private HNSWConfig() {
        throw new AssertionError("HNSWConfig is a static constants class");
    }
}
