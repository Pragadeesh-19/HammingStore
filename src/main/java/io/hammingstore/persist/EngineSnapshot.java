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
 *
 * <p>The snapshot is written via 2 phase commit: data is written to a temporary file
 * flushed to the OS buffer cache, then automatically renamed over the live file.
 * A crash between the flush and the rename leaves the previous snapshot intact.
 *
 * <p>Binary layout:  {@value #SNAPSHOT_BYTES} bytes, little-endian.
 * The first 8-bytes are a fixed magic number ({@value #MAGIC_HEX}) that guards
 * against reading unrelated files.
 */
public final class EngineSnapshot {

    /**
     * magic number written as the first 8 bytes of every snapshot file.
     * Spells "HYPERION" in ASCII
     */
    private static final long MAGIC = 0x4859504552494F4EL;
    private static final String MAGIC_HEX = "0x4859504552494F4EL";

    /** The only schema version this implementation can read or write */
    private static final int SCHEMA_V1 = 1;

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
    private final long hnswEntryPointOffset;
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
            final long hnswEntryPointOffset,
            final long[] layerNodeCounts,
            final long[] nodeIndexSizes) {
        this.schemaVersion = schemaVersion;
        this.inputDimensions = inputDimensions;
        this.projectionSeed = projectionSeed;
        this.vectorCursorSlots = vectorCursorSlots;
        this.indexEntryCount = indexEntryCount;
        this.hnswTopLayer = hnswTopLayer;
        this.hnswLayerCount = hnswLayerCount;
        this.hnswEntryPointOffset = hnswEntryPointOffset;
        this.layerNodeCounts = layerNodeCounts.clone();
        this.nodeIndexSizes = nodeIndexSizes.clone();
    }

    /**
     * Creates a zero-valued snapshot for a brand new index.
     *
     * <p>All cursors and counts are set to zero. The HNSW entry point is set to
     * {@code -1} to indicate that no node has been inserted yet.
     *
     * @param inputDimensions the embedding dimension the server was started with
     * @param projectionSeed the PRNG seed used to generate the projection matrix
     * @param hnswLayerCount the number of HNSW layers in the index.
     * @return a fresh snapshot ready to be written to disk.
     */
    public static EngineSnapshot fresh(
            final int inputDimensions,
            final long projectionSeed,
            final int hnswLayerCount) {
        return new EngineSnapshot(
                SCHEMA_V1, inputDimensions, projectionSeed,
                0L, 0L, 0, hnswLayerCount, -1L,
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
        buf.putInt(schemaVersion);
        buf.putInt(inputDimensions);
        buf.putLong(projectionSeed);
        buf.putLong(vectorCursorSlots);
        buf.putLong(indexEntryCount);
        buf.putInt(hnswTopLayer);
        buf.putInt(hnswLayerCount);
        buf.putLong(hnswEntryPointOffset);
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
        if (schemaVersion != SCHEMA_V1) {
            throw new IllegalStateException(
                    "snapshot.dat schema version " + schemaVersion + " is not supported (expected " + SCHEMA_V1 + ")");
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

    public long hnswEntryPointOffset() { return hnswEntryPointOffset; }

    public long[] layerNodesCounts() { return layerNodeCounts.clone(); }

    public long[] nodeIndexSizes() { return nodeIndexSizes.clone(); }

    @Override
    public String toString() {
        return "EngineSnapshot{vectors=" + vectorCursorSlots
                + ", indexEntries=" + indexEntryCount
                + ", hnswLayers=" + hnswLayerCount
                + ", hnswTopLayer=" + hnswTopLayer
                + ", entryPoint=" + hnswEntryPointOffset + "}";
    }
}
