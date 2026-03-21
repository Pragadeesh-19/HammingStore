package io.hammingstore.persist;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Optional;

/**
 * An atomic, versioned snapshot of the engine's durable state.
 *
 * <p>A snapshot captures every cursor and counter needed to reconstruct in-memory
 * data structures after a restart: the vector store cursor, the entity index size,
 * and the per layer node count for HNSW graph. It also records the projection
 * configuration (embedding dimensions and seed) so that reopening the index with an
 * incompatible configuration is caught immediately rather than silently producing
 * wrong results.
 * <h2>SCHEMA_V2 change</h2>
 * <p>The field at byte offset 48 has changed semantics:
 * <ul>
 *   <li><b>SCHEMA_V1</b>: {@code hnswEntryPointOffset} - a byte offset into the
 *       top layer's node storage. Computed as {@code slotIndex × 1,536}.</li>
 *   <li><b>SCHEMA_V2</b>: {@code hnswEntryPointEntityId} - the entity ID of the
 *       entry-point node. On restore, the current byte offset is derived by
 *       calling {@code layers[topLayer].findOffset(entityId)}.</li>
 * </ul>
 * <p>Storing the entity ID makes the snapshot layout-independent: the byte
 * offset changes whenever {@link io.hammingstore.hnsw.HNSWConfig#NODE_BYTES}
 * changes, but the entity ID never does.
 *
 * <p><b>SCHEMA_V1 snapshots are not compatible</b> with SCHEMA_V2 node layout
 * (1,536 vs 272 bytes/node). {@link #readFrom} detects V1 and throws
 * {@link IllegalStateException} with a clear migration message.
 *
 * <h2>Binary layout — SCHEMA_V2 (256 bytes, little-endian)</h2>
 * <pre>
 * Offset  Size  Field
 * ------  ----  -----
 *      0     8  magic (0x4859504552494F4E = "HYPERION" in ASCII)
 *      8     4  schemaVersion (int) — must be 2
 *     12     4  inputDimensions (int)
 *     16     8  projectionSeed (long)
 *     24     8  vectorCursorSlots (long)
 *     32     8  indexEntryCount (long)
 *     40     4  hnswTopLayer (int)
 *     44     4  hnswLayerCount (int)
 *     48     8  hnswEntryPointEntityId (long, -1 = no entry point)
 *     56    64  layerNodeCounts[8] (8 × long)
 *    120    64  nodeIndexSizes[8] (8 × long)
 *    184    72  (reserved / padding to 256)
 * </pre>
 */
public final class EngineSnapshot {

    /**
     * magic number written as the first 8 bytes of every snapshot file.
     * Spells "HYPERION" in ASCII
     */
    private static final long MAGIC = 0x4859504552494F4EL;

    /** The only schema version this implementation can read or write */
    private static final int SCHEMA_V1 = 1;

    /**
     * SCHEMA_V2 — node layout without embedded vector (NODE_BYTES = 272).
     * Entry point stored as entity ID rather than byte offset.
     */
    private static final int SCHEMA_V2 = 2;

    /** Fixed byte size of the snapshot buffer */
    private static final int SNAPSHOT_BYTES = 256;

    /** Maximum number of HNSW layers recorded in a snapshot */
    public static final int MAX_LAYERS = 8;

    private static final String SNAPSHOT_FILENAME = "snapshot.dat";
    private static final String SNAPSHOT_TMP_FILENAME = "snapshot.dat.tmp";

    private final int schemaVersion;
    private final int inputDimensions;
    private final long projectionSeed;
    private final long vectorCursorSlots;
    private final long indexEntryCount;
    private final int hnswTopLayer;
    private final int hnswLayerCount;
    private final long hnswEntryPointEntityId;
    private final long[] layerNodeCounts;
    private final long[] nodeIndexSizes;

    /**
     * Constructs a snapshot with explicit values for all fields.
     *
     * <p>Prefer {@link #fresh} for new indexes. this constructor is used by
     * {@link #readFrom} when deserializing an existing snapshot from disk.
     *
     * @param layerNodeCounts per-layer HNSW node counts; cloned to prevent mutation.
     * @param nodeIndexSizes per-layer HNSW index size; cloned to prevent mutation.
     */
    public EngineSnapshot(
            final int schemaVersion,
            final int inputDimensions,
            final long projectionSeed,
            final long vectorCursorSlots,
            final long indexEntryCount,
            final int hnswTopLayer,
            final int hnswLayerCount,
            final long hnswEntryPointEntityId,
            final long[] layerNodeCounts,
            final long[] nodeIndexSizes) {
        this.schemaVersion = schemaVersion;
        this.inputDimensions = inputDimensions;
        this.projectionSeed = projectionSeed;
        this.vectorCursorSlots = vectorCursorSlots;
        this.indexEntryCount = indexEntryCount;
        this.hnswTopLayer = hnswTopLayer;
        this.hnswLayerCount = hnswLayerCount;
        this.hnswEntryPointEntityId = hnswEntryPointEntityId;
        this.layerNodeCounts = layerNodeCounts.clone();
        this.nodeIndexSizes = nodeIndexSizes.clone();
    }

    /**
     * Creates a zero-valued SCHEMA_V2 snapshot for a brand new index.
     *
     * <p>All cursors and counts are zero. The HNSW entry point is -1 to indicate
     * that no node has been inserted yet.
     *
     * @param inputDimensions the embedding dimension the server was started with
     * @param projectionSeed  the PRNG seed used to generate the projection matrix
     * @param hnswLayerCount  the number of HNSW layers in the index
     * @return a fresh SCHEMA_V2 snapshot ready to be written to disk
     */
    public static EngineSnapshot fresh(
            final int inputDimensions,
            final long projectionSeed,
            final int hnswLayerCount) {
        return new EngineSnapshot(
                SCHEMA_V2, inputDimensions, projectionSeed,
                0L, 0L, 0, hnswLayerCount,
                -1L,
                new long[MAX_LAYERS], new long[MAX_LAYERS]);
    }

    /**
     * Serialises this snapshot to {@code dir/snapshot.dat} via an atomic write.
     *
     * <p>The data is written to {@code snapshot.dat.tmp}, flushed to the OS buffer
     * cache, then atomically renamed over the live file. A crash at any point before
     * the rename leaves the previous snapshot intact and readable.
     *
     * @param dir the directory to write into; created if it does not exist
     * @throws IOException if the write or rename fails
     */
    public void writeTo(final Path dir) throws IOException {
        Files.createDirectories(dir);
        final Path tmpPath  = dir.resolve(SNAPSHOT_TMP_FILENAME);
        final Path snapPath = dir.resolve(SNAPSHOT_FILENAME);

        final ByteBuffer buf = ByteBuffer.allocate(SNAPSHOT_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);

        buf.putLong(MAGIC);
        buf.putInt(SCHEMA_V2);
        buf.putInt(inputDimensions);
        buf.putLong(projectionSeed);
        buf.putLong(vectorCursorSlots);
        buf.putLong(indexEntryCount);
        buf.putInt(hnswTopLayer);
        buf.putInt(hnswLayerCount);
        buf.putLong(hnswEntryPointEntityId);
        for (int i = 0; i < MAX_LAYERS; i++) buf.putLong(layerNodeCounts[i]);
        for (int i = 0; i < MAX_LAYERS; i++) buf.putLong(nodeIndexSizes[i]);
        buf.flip();

        try (FileChannel fc = FileChannel.open(tmpPath,
                StandardOpenOption.WRITE,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            while (buf.hasRemaining()) fc.write(buf);
            fc.force(false);
        }

        Files.move(tmpPath, snapPath,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING);
    }

    /**
     * Deserialises a snapshot from {@code dir/snapshot.dat}
     *
     * <p>Returns {@link Optional#empty()} if not snapshot exists, which indicates
     * a fresh index with no commited state.
     *
     * @param dir the directory to read from.
     * @return the deserialized snapshot, or empty if no snapshot file exists.
     * @throws IOException if the file exists but cannot be read.
     * @throws IllegalStateException if the magic number or schema version is invalid.
     */
    public static Optional<EngineSnapshot> readFrom(final Path dir) throws IOException {
        final Path snapshotPath = dir.resolve(SNAPSHOT_FILENAME);
        if (!Files.exists(snapshotPath)) return null;

        final ByteBuffer buf = ByteBuffer.allocate(SNAPSHOT_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);

        try (FileChannel fc = FileChannel.open(snapshotPath, StandardOpenOption.READ)) {
            while (buf.hasRemaining() && fc.read(buf) != -1) {}
        }
        buf.flip();

        final long magic = buf.getLong();
        if (magic != MAGIC) {
            throw new IllegalStateException(
                    "snapshot.dat has invalid magic: expected 0x"
                            + Long.toHexString(MAGIC) + ", got 0x" + Long.toHexString(magic));
        }

        final int schemaVersion = buf.getInt();

        if (schemaVersion == SCHEMA_V1) {
            throw new IllegalStateException(
                    "snapshot.dat is SCHEMA_V1 (node layout 1,536 bytes/node) and is not "
                            + "compatible with this build (SCHEMA_V2, 272 bytes/node). "
                            + "Migration required: stop the server, delete snapshot.dat and "
                            + "all hnsw_nodes_L*.dat files, then restart to re-index from the "
                            + "vector_store.dat which remains intact. "
                            + "See docs/MIGRATION_V1_V2.md for full instructions.");
        }

        if (schemaVersion != SCHEMA_V2) {
            throw new IllegalStateException(
                    "snapshot.dat schema version " + schemaVersion
                            + " is not supported (expected " + SCHEMA_V2 + ")");
        }

        final int inputDimensions = buf.getInt();
        final long projectionSeed = buf.getLong();
        final long vectorCursorSlots = buf.getLong();
        final long indexEntryCount = buf.getLong();
        final int hnswTopLayer = buf.getInt();
        final int hnswLayerCount = buf.getInt();
        final long hnswEntryPointOffset = buf.getLong();

        final long[] layerNodesCounts = new long[MAX_LAYERS];
        final long[] nodeIndexSizes = new long[MAX_LAYERS];
        for (int i = 0; i < MAX_LAYERS; i++) layerNodesCounts[i] = buf.getLong();
        for (int i = 0; i < MAX_LAYERS; i++) nodeIndexSizes[i]   = buf.getLong();

        return Optional.of(new EngineSnapshot(
                schemaVersion, inputDimensions, projectionSeed, vectorCursorSlots, indexEntryCount,
                hnswTopLayer, hnswLayerCount, hnswEntryPointOffset, layerNodesCounts, nodeIndexSizes));
    }

    /**
     * Verifies that this snapshot's projection configuration matches the provided values.
     *
     * <p>Called when reopening an existing index to ensure the server was not
     * restarted with a different embedding dimension or seed. Either mismatch would
     * silently corrupt all search results because stored binary vectors would have been
     * generated from a different projection matrix.
     *
     * @param inputDimensions the dimension to check against.
     * @param projectionSeed the seed to check against.
     * @throws IllegalStateException if either value does not match the snapshot.
     */
    public void validateProjectionConfig(final int inputDimensions, final long projectionSeed) {
        if (this.inputDimensions != inputDimensions) {
            throw new IllegalStateException(
                    "Snapshot inputDimensions=" + this.inputDimensions
                            + " does not match config inputDimensions=" + inputDimensions
                            + ". Cannot reopen index with a different embedding dimension.");
        }
        if (this.projectionSeed != projectionSeed) {
            throw new IllegalStateException(
                    "Snapshot projectionSeed does not match config seed. "
                            + "Stored vectors would be invalid with a different projection matrix.");
        }
    }

    public int schemaVersion() { return schemaVersion; }

    public int inputDimensions() { return inputDimensions; }

    public long projectionSeed() { return projectionSeed; }

    public long vectorCursorSlots() { return vectorCursorSlots; }

    public long indexEntryCount() { return indexEntryCount; }

    public  int hnswTopLayer() { return hnswTopLayer; }

    public int hnswLayerCount() { return hnswLayerCount; }

    public long   hnswEntryPointEntityId() { return hnswEntryPointEntityId; }

    public long[] layerNodesCounts() { return layerNodeCounts.clone(); }

    public long[] nodeIndexSizes() { return nodeIndexSizes.clone(); }

    @Override
    public String toString() {
        return "EngineSnapshot{vectors=" + vectorCursorSlots
                + ", indexEntries=" + indexEntryCount
                + ", hnswLayers=" + hnswLayerCount
                + ", hnswTopLayer=" + hnswTopLayer
                + ", entryPointEntityId=" + hnswEntryPointEntityId + "}";
    }
}
