package io.hammingstore.persist;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Opens and memory-maps data files in a single directory, managing all mapped
 * segments under one shared {@link Arena}
 *
 * <p>Each call to {@link #map} opens or creates a file, extends it to the requested
 * size if necessary, and returns a {@link MemorySegment} backed by the file's pages.
 * All segments remain valid until {@link #close()} is called.
 *
 * <p>On {@link #close()}, file channels are released first (freeing OS file handles),
 * then the arena is closed (unmapping all segments from the process address space).
 * This ordering ensures that a failure in any individual channel close does not
 * prevent the arena from being released.
 *
 * <p>This class is not thread-safe. Callers must ensure that {@link #map},
 * {@link #force}, and {@link #close} are not called concurrently.
 */
public final class MappedFileAllocator implements AutoCloseable {

    private final Path directory;
    private final Arena arena;
    private final List<Entry> entries = new ArrayList<>();

    private record Entry(String filename, FileChannel channel, MemorySegment segment) {}

    /**
     * Creates a new allocator rooted {@code directory}, creating the directory
     * if it does not already exist
     * @param directory the directory in which all files will be created.
     * @throws UncheckedIOException if the directory cannot be created.
     */
    public MappedFileAllocator(final Path directory) {
        try {
            Files.createDirectories(directory);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create data directory: " + directory, e);
        }
        this.directory = directory;
        this.arena = Arena.ofShared();
    }

    /**
     * Opens (or creates) {@code filename} in the root directory, extends it to
     * {@code byteSize} bytes if smaller, and returns a read-write memory mapped
     * segment backed by the file.
     *
     * <p>The returned segment is valid until this allocator is {@link #close()}d.
     * Multiple calls with the same filename returned independent segments over
     * the same file - callers are responsible for avoiding overlapping writes.
     *
     * @param filename the filename of the file to map, relative to the root directory.
     * @param byteSize the required byte size of the mapping; must be positive
     * @return a read-write {@link MemorySegment} backed by the file.
     * @throws IllegalArgumentException if {@code byteSize} is not positive.
     * @throws UncheckedIOException if the file cannot be opened or mapped.
     */
    public MemorySegment map(final String filename, final long byteSize) {
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be > 0");

        try {
            final Path filePath = directory.resolve(filename);
            final FileChannel fileChannel = FileChannel.open(filePath,
                    StandardOpenOption.READ,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.CREATE);

            if (fileChannel.size() < byteSize) {
                fileChannel.truncate(byteSize);
            }
            final MemorySegment segment = fileChannel.map(
                    FileChannel.MapMode.READ_WRITE, 0L, byteSize, arena);

            entries.add(new Entry(filename, fileChannel, segment));
            return segment;
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "Cannot map file " + filename + " in " + directory, e);
        }
    }

    /**
     * Flushes all mapped segments to the OS buffer cache, and optionally to disk.
     *
     * <p>Calling {@code force(false)} is sufficient before writting an
     * {@link EngineSnapshot} - the snapshots atomic rename provides the durability
     * guarantee. Call {@code force(true)} only when you need a hard fsync.
     *
     * @param durable if {@code true}, also issues a fsync on each backing file,
     *                flushing OS buffer cache pages to the physical storage device.
     * @throws UncheckedIOException if any segment or channel force fails.
     */
    public void force(final boolean durable) {
        for (final Entry entry : entries) {
            try {
                entry.segment().force();
                if (durable) {
                    entry.channel().force(false);
                }
            } catch (IOException ex) {
                throw new UncheckedIOException(
                        "Failed to force file: " + entry.filename(), ex);
            }
        }
    }

    /**
     * Closes all file channels, and releases all mapped segments by closing the arena.
     *
     * <p>Channels are closed before the arena so that OS file handles are freed
     * even if a subsequent arena closes fails. If any channel close throws, the
     * exception is recorded and the remaining channels are still closed before
     * re-throwing.
     *
     * @throws UncheckedIOException if any channel could not be closed.
     */
    @Override
    public void close() {
        UncheckedIOException deferred = null;
        for (final Entry entry: entries) {
            try {
                entry.channel.close();
            } catch (IOException e) {
                if (deferred == null) {
                    deferred = new UncheckedIOException(
                            "Failed to close channel: " + entry.filename(), e);
                }
            }
        }
        arena.close();
        if (deferred != null) throw deferred;
    }
}

