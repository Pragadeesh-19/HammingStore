package io.hammingstore.memory;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A flat, append only store for fixed size binary hypervectors in off-heap memory.
 *
 * <p>Each slot is exactly {@link BinaryVector#VECTOR_BYTES} bytes. Slots are
 * addressed by byte offset and allocated by incrementing an atomic cursor, making
 * allocation lock-free and O(1). There is no deletion - retraction is handled at
 * the index layer via tombstoning in {@link SparseEntityIndex}
 *
 * <p>This class is thread-safe for concurrent reads and concurrent {@link #allocatedSlots()}
 * calls. Callers are responsible for ensuring that a slot is fully written via
 * {@link #copyInto(long, MemorySegment)} before it is read by another thread.
 */
public final class OffHeapVectorStore {

    private static final long SLOT_BYTES = BinaryVector.VECTOR_BYTES;

    private final MemorySegment storage;
    private final long capacity;
    private final AtomicLong cursor;

    /**
     * Creates a new RAM-backed vector store.
     *
     * @param allocator the allocator from which to request storage.
     * @param capacity maximum number of vectors this store will hold; must be positive.
     * @throws IllegalArgumentException if {@code capacity} is not positive.
     */
    public OffHeapVectorStore(final OffHeapAllocator allocator, final long capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0, got: " + capacity);
        }
        this.capacity = capacity;
        this.storage = allocator.allocateRawSegment(capacity * SLOT_BYTES, Long.BYTES);
        this.cursor = new AtomicLong(0L);
    }

    /**
     * Creates a vector store over an existing memory-mapped segment, restoring
     * a previously commited cursor position
     *
     * <p>Intended for use by the persistence layer when reopening a store from
     * disk. The {@code mappedStorage} segment must be at least
     * {@code capacitySlots x SLOT_BYTES} bytes.
     *
     * @param mappedStorage the memory-mapped backing segment.
     * @param capacitySlots maximum number of vector slots in the segment.
     * @param initialCursorSlots number of slots already written before this session
     * @return a vector store wrapping the mapped segment.
     */
    public static OffHeapVectorStore fromMapped(
            final MemorySegment mappedStorage,
            final long capacitySlots,
            final long initialCursorSlots) {
        return new OffHeapVectorStore(mappedStorage, capacitySlots, initialCursorSlots);
    }

    private OffHeapVectorStore(
            final MemorySegment storage,
            final long capacitySlots,
            final long initialCursorSlots) {
        this.storage = storage;
        this.capacity = capacitySlots;
        this.cursor = new AtomicLong(initialCursorSlots);
    }

    /**
     * Reserves the next available slot and returns its byte offset
     *
     * <p>This method is lock-free and safe for concurrent callers. The returned
     * offset is valid for use with {@link #copyInto} and {@link #sliceAt}.
     *
     * @return the byte offset of the newly reserved slot.
     * @throws IllegalStateException if the store has reached its capacity.
     */
    public long allocateSlot() {
        final long slotIndex = cursor.getAndIncrement();
        if (slotIndex >= capacity) {
            cursor.decrementAndGet();
            throw new IllegalStateException(
                    "VectorStore capacity exhausted: max=" + capacity
            );
        }
        return slotIndex * SLOT_BYTES;
    }

    /**
     * Returns a slice of the backing storage for the slot at {@code byteOffset}.
     *
     * <p>The returned segment is a view into the backing storage - not a copy.
     * callers must not hold onto the slice beyond the lifetime of this store.
     *
     * @param byteOffset byte offset of the slot; must be slot-aligned.
     * @return a {@link BinaryVector#VECTOR_BYTES}- byte view of the slot
     * @throws IllegalArgumentException if the offset is out of range or misaligned.
     */
    public MemorySegment sliceAt(final long byteOffset) {
        validateOffset(byteOffset);
        return storage.asSlice(byteOffset, SLOT_BYTES);
    }

    /**
     * Copies {@link BinaryVector#VECTOR_BYTES} bytes from {@code source} into
     * the slot at {@code byteOffset}.
     *
     * @param byteOffset byte offset of the destination slot; must be slot-aligned
     * @param source segment to copy from; must be at least
     *               {@link BinaryVector#VECTOR_BYTES} bytes.
     * @throws IllegalArgumentException if the offset is out of range or misaligned.
     */
    public void copyInto(final long byteOffset, final MemorySegment source) {
        validateOffset(byteOffset);
        MemorySegment.copy(source, 0L, storage, byteOffset, SLOT_BYTES);
    }

    /**
     * @return the number of slots allocated so far (the current cursor position).
     * This is the number of slots that have been reserved, not necessarily written.
     */
    public long allocatedSlots() {
        return cursor.get();
    }

    /**
     * @return the raw backing segment for bulk operations such as hamming distance
     * scans. Callers must treat this as read-only outside of the store's own methods.
     */
    public MemorySegment rawStorage() {
        return storage;
    }

    private void validateOffset(final long byteOffset) {
        if (byteOffset < 0 || byteOffset * SLOT_BYTES > storage.byteSize()) {
            throw new IllegalArgumentException(
                    "byteOffset " + byteOffset + " out of range [0, " + storage.byteSize() + ")"
            );
        }
        if ((byteOffset % SLOT_BYTES) != 0) {
            throw new IllegalArgumentException(
                    "byteOffset " + byteOffset + " is not aligned to slot boundary (" + SLOT_BYTES + ")");
        }
    }
}
