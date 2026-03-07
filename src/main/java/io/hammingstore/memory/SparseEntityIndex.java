package io.hammingstore.memory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public final class SparseEntityIndex {

    private static final long ENTRY_BYTES = 16L;
    private static final long OFFSET_KEY = 0L;
    private static final long OFFSET_VALUE = 8L;
    private static final long EMPTY_SENTINEL = 0L;

    private final MemorySegment table;
    private final long capacity;
    private final long mask;
    private long size;
    private final long maxEntities;

    public SparseEntityIndex(final OffHeapAllocator allocator, final long maxEntities) {
        if (maxEntities <= 0) {
            throw new IllegalArgumentException("maxEntities must be > 0");
        }

        this.maxEntities = maxEntities;
        this.capacity = nextPowerOfTwo(Math.max(4L, maxEntities * 2L));
        this.mask = capacity - 1L;
        this.table = allocator.allocateRawSegment(capacity * ENTRY_BYTES, Long.BYTES);
    }

    public static SparseEntityIndex fromMapped(
            final MemorySegment mappedTable,
            final long capacity,
            final long maxEntities,
            final long initialSize) {
        return new SparseEntityIndex(mappedTable, capacity, maxEntities, initialSize);
    }

    private SparseEntityIndex(
            final MemorySegment table,
            final long capacity,
            final long maxEntities,
            final long initialSize) {
        this.table = table;
        this.capacity = capacity;
        this.mask = capacity - 1L;
        this.maxEntities = maxEntities;
        this.size = initialSize;
    }

    public static long tableByteSize(final long maxEntities) {
        return nextPowerOfTwo(Math.max(4L, maxEntities * 2L)) * ENTRY_BYTES;
    }

    public static long tableCapacity(final long maxEntities) {
        return nextPowerOfTwo(Math.max(4L, maxEntities * 2L));
    }

    public void put(long hashedId, final long vectorOffset) {
        if (size >= maxEntities) {
            throw new IllegalStateException("SparseEntityIndex is full: size=" + size);
        }

        if (hashedId == EMPTY_SENTINEL) {
            hashedId = 1L;
        }
        long slot = hashedId & mask;
        while (true) {
            final long base = slot * ENTRY_BYTES;
            final long existingKey = table.get(ValueLayout.JAVA_LONG_UNALIGNED, base + OFFSET_KEY);

            if (existingKey == EMPTY_SENTINEL) {
                table.set(ValueLayout.JAVA_LONG_UNALIGNED, base + OFFSET_VALUE, vectorOffset);
                table.set(ValueLayout.JAVA_LONG_UNALIGNED, base + OFFSET_KEY, hashedId); // key last (visibility)
                size++;
                return;
            }
            if (existingKey == hashedId) {
                table.set(ValueLayout.JAVA_LONG_UNALIGNED, base + OFFSET_VALUE, vectorOffset);
                return;
            }
            slot = (slot + 1L) & mask;
        }
    }

    public long getOffset(long hashedId) {
        if (hashedId == EMPTY_SENTINEL) {
            hashedId = 1L;
        }

        long slot = hashedId & mask;
        while (true) {
            final long base = slot * ENTRY_BYTES;
            final long existingKey = table.get(ValueLayout.JAVA_LONG_UNALIGNED, base + OFFSET_KEY);
            if (existingKey == EMPTY_SENTINEL) {
                return -1L;
            }
            if (existingKey == hashedId) {
                return table.get(ValueLayout.JAVA_LONG_UNALIGNED, base + OFFSET_VALUE);
            }
            slot = (slot + 1L) & mask;
        }
    }

    public static long xxHash3Stub(long entityId) {
        entityId ^= entityId >>> 33;
        entityId *= 0xff51afd7ed558ccdL;
        entityId ^= entityId >>> 33;
        entityId *= 0xc4ceb9fe1a85ec53L;
        entityId ^= entityId >>> 33;
        return entityId;
    }

    public static long hashString(final String entityId) {
        final byte[] bytes = entityId.getBytes(StandardCharsets.UTF_8);
        long hash = 0xcbf29ce484222325L;
        for (final byte b : bytes) {
            hash ^= (b & 0xFFL);
            hash *= 0x100000001b3L;
        }
        return xxHash3Stub(hash);
    }

    public long size() {
        return size;
    }

    public long capacity() {
        return capacity;
    }

    private static long nextPowerOfTwo(final long v) {
        if (v <= 1L) return 1L;
        return Long.highestOneBit(v - 1L) << 1;
    }
}
