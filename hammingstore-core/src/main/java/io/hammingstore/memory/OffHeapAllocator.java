package io.hammingstore.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Allocates off-heap native memory segments from a single GC-managed arena.
 *
 * <p>All segments allocated here share the same {@link Arena#ofAuto()} lifetime -
 * they are released when this allocator becomes unreachable, not when
 * {@link #close()} is called. This makes {@code OffHeapAllocator} suitable for
 * long-lived structures like vector stores and projection matrices that are owned
 * by a single top level component.
 *
 * <p>For structures that require explicit, deterministic release (such as
 * memory-mapped files), use {@link io.hammingstore.persist.MappedFileAllocator}
 * instead, which uses a shared arena with-defined close contract.
 */
public final class OffHeapAllocator implements AutoCloseable {

    private final Arena arena;

    /**
     * Creates a new allocator backed by GC-managed arena
     *
     * <p>The {@code capacityHint} parameter is informational. it is not enforced.
     * the underlying arena will serve any allocation regardless of this value.
     * Callers are responsible for not exceeding their intended capacity.
     *
     * @param capacityHint expected number of vector-sized allocations; must be positive.
     * @throws IllegalArgumentException if {@code capacityHint} is not positive
     */
    public OffHeapAllocator(final long capacityHint) {
        if (capacityHint <= 0) {
            throw new IllegalArgumentException("capacityHint must be > 0, got: " + capacityHint);
        }
        this.arena = Arena.ofAuto();
    }

    /**
     * Allocates a raw off-heap segment of the requested size and alignment.
     *
     * @param byteSize number of bytes to allocate; must be positive.
     * @param alignment byte alignment; must be a positive power of 2.
     * @return a zero-initialized off-heap {@link MemorySegment}
     * @throws IllegalArgumentException if {@code byteSize} is not positive, or
     *                                  {@code alignment} is not a positive power of 2
     */
    public MemorySegment allocateRawSegment(final long byteSize, final long alignment) {
        if (byteSize <= 0) {
            throw new IllegalArgumentException("byteSize must be > 0");
        }
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IllegalArgumentException("alignment must be a positive power of 2");
        }
        return arena.allocate(byteSize, alignment);
    }

    /**
     * No-op. memory is managed by GC-backed arena and released automatically
     * when this allocator becomes unreachable.
     */
    @Override
    public void close() {}
}
