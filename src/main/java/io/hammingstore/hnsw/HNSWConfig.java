package io.hammingstore.hnsw;

import io.hammingstore.memory.BinaryVector;

/**
 * Compile-time constants for the HNSW graph index.
 *
 * <p>All tuneable parameters ({@link #M}, {@link #EF_CONSTRUCTION}, etc.) and all
 * structural layout constants ({@link #NODE_BYTES}, {@link #NODE_OFFSET_VECTOR},
 * etc.) live here so they have a single authoritative source across
 * {@link HNSWLayer}, {@link HNSWNodeView}, {@link HNSWIndex}, and
 * {@link io.hammingstore.hnsw.VisitedTracker}.
 *
 * <h2>On-disk layout contract</h2>
 * <p>The {@code NODE_OFFSET_*} and {@code NODE_BYTES} constants define the exact
 * binary layout of every node written to the memory-mapped layer files. Changing
 * any of these values invalidates all persisted indexes. Never change them without
 * a corresponding migration and a version bump in
 * {@link io.hammingstore.persist.EngineSnapshot}.
 *
 * <h2>Node memory layout (1,536 bytes per node)</h2>
 * <pre>
 * Offset  Size   Field
 * ------  ----   -----
 *      0     8   entity ID           (long)
 *      8  1256   binary vector       (10,048 bits = 157 longs × 8 bytes)
 *   1264     4   neighbor count      (int)
 *   1268     4   node layer          (int)
 *   1272   256   neighbor offsets    (32 longs × 8 bytes = M × Long.BYTES)
 *                                    ────────────────────────────────────
 *                                    Total: 1272 + 256 = 1,528 → padded to 1,536
 * </pre>
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

    public static final long NODE_OFFSET_ENTITY_ID       = 0L;

    public static final long NODE_OFFSET_VECTOR          = 8L;

    public static final long NODE_OFFSET_NEIGHBOR_COUNT  = 1264L;

    public static final long NODE_OFFSET_NODE_LAYER      = 1268L;

    /**
     * Byte offset of the neighbor-offset array within a node.
     * Immediately follows the node-layer field (4-byte int): 1,268 + 4 = 1,272.
     * The array contains {@link #M} longs (8 bytes each), each holding the byte
     * offset of a neighbor node within the same layer's storage segment, or
     * {@link #EMPTY_NEIGHBOR} if the slot is unused.
     */
    public static final long NODE_OFFSET_NEIGHBORS_START = 1272L;

    /**
     * Total size of one node in bytes.
     * Layout: 8 (entity ID) + 1,256 (vector) + 4 (count) + 4 (layer) + 256 (neighbors) = 1,528,
     * rounded up to 1,536 for alignment.
     * All layer storage segments are allocated as a multiple of this value.
     */
    public static final long NODE_BYTES = 1536L;

    public static final long EMPTY_NEIGHBOR = -1L;

    private HNSWConfig() {
        throw new AssertionError("HNSWConfig is a static constants class");
    }
}
