package io.hammingstore.persist;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

public final class EngineSnapshot {

    private static final long MAGIC = 0x4859504552494F4EL;
    private static final int SCHEMA_V1 = 1;
    private static final int SNAPSHOT_BYTES = 256;
    public static final int MAX_LAYERS = 8;
    private static final String SNAP_FILE = "snapshot.dat";
    private static final String SNAP_TMP = "snapshot.dat.tmp";

    private final int schemaVersion;
    private final int inputDimensions;
    private final long projectionSeed;
    public final long vectorCursorSlots;
    public final long indexEntryCount;
    public final int hnswTopLayer;
    public final int hnswLayerCount;
    public final long hnswEntryPointOffset;
    public final long[] layerNodeCounts;
    public final long[] nodeIndexSizes;

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

    public static EngineSnapshot fresh(
            final int inputDimensions,
            final long projectionSeed,
            final int hnswLayerCount) {
        return new EngineSnapshot(
                SCHEMA_V1, inputDimensions, projectionSeed,
                0L, 0L, 0, hnswLayerCount, -1L,
                new long[MAX_LAYERS], new long[MAX_LAYERS]);
    }

    public void writeTo(final Path dir) throws IOException {
        Files.createDirectories(dir);
        final Path tmpPath  = dir.resolve(SNAP_TMP);
        final Path snapPath = dir.resolve(SNAP_FILE);

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
            fc.force(false); // flush data before rename
        }

        Files.move(tmpPath, snapPath,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING);
    }

    public static EngineSnapshot readFrom(final Path dir) throws IOException {
        final Path snapPath = dir.resolve(SNAP_FILE);
        if (!Files.exists(snapPath)) return null;

        final ByteBuffer buf = ByteBuffer.allocate(SNAPSHOT_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);

        try (FileChannel fc = FileChannel.open(snapPath, StandardOpenOption.READ)) {
            while (buf.hasRemaining() && fc.read(buf) != -1);
        }
        buf.flip();

        final long magic = buf.getLong();
        if (magic != MAGIC) {
            throw new IllegalStateException(
                    "snapshot.dat has invalid magic: expected 0x"
                            + Long.toHexString(MAGIC) + ", got 0x" + Long.toHexString(magic));
        }

        final int  sv = buf.getInt();
        if (sv != SCHEMA_V1) {
            throw new IllegalStateException(
                    "snapshot.dat schema version " + sv + " is not supported (expected " + SCHEMA_V1 + ")");
        }

        final int  inputDims = buf.getInt();
        final long seed = buf.getLong();
        final long vecCursor = buf.getLong();
        final long idxCount = buf.getLong();
        final int  topLayer = buf.getInt();
        final int  layerCount = buf.getInt();
        final long entryPt = buf.getLong();

        final long[] layerNodes = new long[MAX_LAYERS];
        final long[] idxSizes   = new long[MAX_LAYERS];
        for (int i = 0; i < MAX_LAYERS; i++) layerNodes[i] = buf.getLong();
        for (int i = 0; i < MAX_LAYERS; i++) idxSizes[i]   = buf.getLong();

        return new EngineSnapshot(
                sv, inputDims, seed, vecCursor, idxCount,
                topLayer, layerCount, entryPt, layerNodes, idxSizes);
    }

    public void validateProjectionConfig(final int inputDimensions, final long seed) {
        if (this.inputDimensions != inputDimensions) {
            throw new IllegalStateException(
                    "Snapshot inputDimensions=" + this.inputDimensions
                            + " does not match config inputDimensions=" + inputDimensions
                            + ". Cannot reopen index with a different embedding dimension.");
        }
        if (this.projectionSeed != seed) {
            throw new IllegalStateException(
                    "Snapshot projectionSeed does not match config seed. "
                            + "Stored vectors would be invalid with a different projection matrix.");
        }
    }

    @Override
    public String toString() {
        return "EngineSnapshot{vectors=" + vectorCursorSlots
                + ", indexEntries=" + indexEntryCount
                + ", hnswLayers=" + hnswLayerCount
                + ", hnswTopLayer=" + hnswTopLayer
                + ", entryPoint=" + hnswEntryPointOffset + "}";
    }
}
