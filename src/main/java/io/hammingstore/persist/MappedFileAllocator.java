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

public final class MappedFileAllocator implements AutoCloseable {

    private final Path directory;
    private final Arena arena;
    private final List<Entry> entries = new ArrayList<>();

    private record Entry(String name, FileChannel channel, MemorySegment segment) {}

    public MappedFileAllocator(final Path directory) {
        try {
            Files.createDirectories(directory);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create data directory: " + directory, e);
        }
        this.directory = directory;
        this.arena = Arena.ofShared();
    }

    public MemorySegment map(final String filename, final long byteSize) {
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be > 0");

        try {
            final Path filePath = directory.resolve(filename);
            final FileChannel fc = FileChannel.open(filePath,
                    StandardOpenOption.READ,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.CREATE);

            if (fc.size() < byteSize) {
                fc.truncate(byteSize);
            }
            final MemorySegment seg = fc.map(
                    FileChannel.MapMode.READ_WRITE, 0L, byteSize, arena);

            entries.add(new Entry(filename, fc, seg));
            return seg;
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "Cannot map file " + filename + " in " + directory, e);
        }
    }

    public boolean isNewFile(final String fileName) {
        return !Files.exists(directory.resolve(fileName + ".pre_map_exists"));
    }

    public void force(final boolean durable) {
        for (final Entry e : entries) {
            try {
                e.segment().force(); // flush dirty pages to OS buffer cache
                if (durable) {
                    e.channel().force(false); // fsync — flush buffer cache to disk
                }
            } catch (IOException ex) {
                throw new UncheckedIOException(
                        "Failed to force file: " + e.name(), ex);
            }
        }
    }

    public Path directory() {
        return directory;
    }

    @Override
    public void close() {
        arena.close();
        UncheckedIOException last = null;
        for (final Entry e : entries) {
            try {
                e.channel().close();
            } catch (IOException ex) {
                last = new UncheckedIOException("Failed to close: " + e.name(), ex);
            }
        }
        if (last != null) throw last;
    }
}

