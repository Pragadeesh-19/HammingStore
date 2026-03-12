package io.hammingstore.memory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * An open-addressing, linear-probing hash map from a 64-bit hashed entity ID to a
 * 64 bit byte offset into a {@link OffHeapVectorStore}.
 *
 * <p>The table lives entirely in off-heap memory so that it can be memory-mapped
 * to disk without copying. each entry is 16 bytes: an 8 byte key (hashed ID)
 * followed by an 8 byte value (the vector store byte offset).
 *
 * <p><b>Key contract:</b> callers must hash entity IDs through {@link #mixHash64}
 * before passing them to {@link #put} or {@link #getOffset}. Raw sequential IDs
 * cluster badly under linear probing and degrade lookup to O(N).
 *
 * <p><b>Zero-key constrain:</b> the raw value {@code 0} is reserved as the empty
 * slot sentinel. Any hashed key that resolves to {@code 0} is remapped to {@code 1}.
 * this means entity IDs whose hash is exactly zero are stored under key {@code 1}.
 * Collisions from this remapping are vanishingly rare in practice.
 *
 * <p><b>Thread safety:</b> this class performs no internal synchronization.
 * concurrent writes must be serialized by the caller (typically via the write lock in
 * {@link io.hammingstore.graph.VectorGraphRepository}).
 * Concurrent reads after a write has completed are safe because key slots are written
 * last, acting as a visibility fence.
 */
public final class SparseEntityIndex {

    /** Byte size of one hash table entry: 8-byte key + 8-byte value*/
    private static final long ENTRY_BYTES = 16L;

    /** Byte offset of the key field within one entry */
    private static final long KEY_OFFSET = 0L;

    /** Byte offset of the value field within one entry */
    private static final long VALUE_OFFSET = 8L;

    /** Sentinel stored in the key field of an empty slot. */
    private static final long EMPTY_SLOT = 0L;

    private final MemorySegment table;
    private final long capacity;
    private final long mask;
    private long size;
    private final long maxEntities;

    /**
     * Creates a new RAM backed index sized to hold atleast {@code maxEntries} entries.
     *
     * <p>The underlying table is allocated at the next power of 2 above
     * {@code 2 x maxEntries} to keep the load factor below 0.5 and ensure O(1)
     * expected probe sequence.
     *
     * @param allocator the allocator from which to request table storage.
     * @param maxEntities maximum number of entries; must be positive.
     * @throws IllegalArgumentException if {@code maxEntries} is not positive.
     */
    public SparseEntityIndex(final OffHeapAllocator allocator, final long maxEntities) {
        if (maxEntities <= 0) {
            throw new IllegalArgumentException("maxEntities must be > 0");
        }

        this.maxEntities = maxEntities;
        this.capacity = nextPowerOfTwo(Math.max(4L, maxEntities * 2L));
        this.mask = capacity - 1L;
        this.table = allocator.allocateRawSegment(capacity * ENTRY_BYTES, Long.BYTES);
    }

    /**
     * creates an index over an existing memory mapped table segment, restoring a
     * previously commited entry count.
     *
     * <p>Intended for use by the persistence layer when reopening an index from disk.
     *
     * @param mappedTable the memory-mapped backing segment.
     * @param capacity number of slots in the table; must be a power of 2
     * @param maxEntities maximum number of entries before the index is considered full
     * @param initialSize number of entries already present before this session.
     * @return an index wrapping the mapped segment.
     */
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

    /**
     * @return the byte size required for a table that can hold {@code maxEntities} entries.
     * Use this to size a memory-mapped file before calling {@link #fromMapped}
     */
    public static long tableByteSize(final long maxEntities) {
        return nextPowerOfTwo(Math.max(4L, maxEntities * 2L)) * ENTRY_BYTES;
    }

    /**
     * @return the slot capacity of a table sized for {@code maxEntities} entries.
     * Use this when constructing a {@link SparseEntityIndex} over a mapped segment.
     */
    public static long tableCapacity(final long maxEntities) {
        return nextPowerOfTwo(Math.max(4L, maxEntities * 2L));
    }

    /**
     * Associates {@code vectorOffset} with {@code hashedId}, inserting a new entry
     * or overwriting an existing one.
     *
     * <p>The key must already be hashed through {@link #mixHash64}. passing a raw
     * sequential entity ID will produce severe probe-length degradation.
     *
     * @param hashedId the hashed entity ID to use as the key
     * @param vectorOffset the byte offset in the vector store to associate with this key
     * @throws IllegalStateException if the index is full
     */
    public void put(long hashedId, final long vectorOffset) {
        if (size >= maxEntities) {
            throw new IllegalStateException("SparseEntityIndex is full: size=" + size);
        }

        if (hashedId == EMPTY_SLOT) {
            hashedId = 1L;
        }
        long slot = hashedId & mask;
        while (true) {
            final long base = slot * ENTRY_BYTES;
            final long existingKey = table.get(ValueLayout.JAVA_LONG_UNALIGNED, base + KEY_OFFSET);

            if (existingKey == EMPTY_SLOT) {
                table.set(ValueLayout.JAVA_LONG_UNALIGNED, base + VALUE_OFFSET, vectorOffset);
                table.set(ValueLayout.JAVA_LONG_UNALIGNED, base + KEY_OFFSET, hashedId);
                size++;
                return;
            }
            if (existingKey == hashedId) {
                table.set(ValueLayout.JAVA_LONG_UNALIGNED, base + VALUE_OFFSET, vectorOffset);
                return;
            }
            slot = (slot + 1L) & mask;
        }
    }

    /**
     * Returns the vector store byte offset associated with {@code hashedId},
     * or {@code -1} if no entry exists for that key.
     *
     * <p>The key must already be hashed through {@link #mixHash64}.
     *
     * @param hashedId the hashed entity ID to look up
     * @return the associated byte offset, or {@code -1} if not found
     */
    public long getOffset(long hashedId) {
        if (hashedId == EMPTY_SLOT) {
            hashedId = 1L;
        }

        long slot = hashedId & mask;
        while (true) {
            final long base = slot * ENTRY_BYTES;
            final long existingKey = table.get(ValueLayout.JAVA_LONG_UNALIGNED, base + KEY_OFFSET);
            if (existingKey == EMPTY_SLOT) {
                return -1L;
            }
            if (existingKey == hashedId) {
                return table.get(ValueLayout.JAVA_LONG_UNALIGNED, base + VALUE_OFFSET);
            }
            slot = (slot + 1L) & mask;
        }
    }


    /**
     * Mixes a 64-bit integer key using the fmix64 finaliser from MurmurHash3.
     *
     * <p>This eliminates clustering from sequential or low-entropy entity IDs before
     * they enter the linear-probing table. The three xorshift-multiply rounds give
     * avalanche behaviour: every input bit affects every output bit.
     *
     * @param key the raw 64-bit entity ID
     * @return a well-distributed hash suitable for use as a table key
     */
    public static long mixHash64(long key) {
        key ^= key >>> 33;
        key *= 0xff51afd7ed558ccdL;
        key ^= key >>> 33;
        key *= 0xc4ceb9fe1a85ec53L;
        key ^= key >>> 33;
        return key;
    }

    /**
     * Returns the number of entries currently stored in this index
     */
    public long size() {
        return size;
    }

    /**
     * Returns the total slot capacity of the underlying hash table.
     */
    public long capacity() {
        return capacity;
    }

    private static long nextPowerOfTwo(final long v) {
        if (v <= 1L) return 1L;
        return Long.highestOneBit(v - 1L) << 1;
    }
}
