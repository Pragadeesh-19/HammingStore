package io.hammingstore.hnsw;

/**
 * A fixed-capacity max-heap that retains the K nearest (lowest-distance) candidates
 * seen so far.
 *
 * <p>Used during HNSW beam search to track the best result set. The heap invariant
 * is: {@code distances[0]} is always the maximum (worst) distance in the buffer.
 * This lets the search prune unpromising candidates in O(1) via
 * {@link #peekMaxDistance()}, and replace the worst result in O(log K) via
 * {@link #offer}.
 *
 * <p>Three operations cover the entire search lifecycle:
 * <ul>
 *   <li>{@link #offer} - O(log K), replaces worst if better candidate found</li>
 *   <li>{@link #peekMaxDistance} - O(1), pruning threshold check</li>
 *   <li>{@link #sortAscending} - O(K log K), heapsort in-place, zero allocation</li>
 * </ul>
 *
 * <p>Instances are not thread-safe. Each search context owns a dedicated buffer
 * and resets it via {@link #reset()} between searches.
 */
final class TopKBuffer {

    private final long[] offsets;
    private final long[] distances;
    private final int capacity;
    private int size;

    /**
     * Creates a buffer retaining the best {@code k} candidates
     *
     * @param k the maximum number of candidates to retain
     * @throws IllegalArgumentException if {@code k} is not positive
     */
     TopKBuffer(final int k) {
        if (k<= 0) throw new IllegalArgumentException("k must be > 0");
        this.capacity = k;
        this.offsets = new long[k];
        this.distances = new long[k];
        this.size = 0;
    }

    /**
     * Offers a candidate to the buffer.
     *
     * <p>If the buffer is not yet full, the candidate is always accepted.
     * If it is full, the candidate is accepted only if its distance is less
     * than the current worst (max) distance in the buffer, which it then
     * replaces.
     *
     * @param offset   byte offset of the candidate node in layer storage
     * @param distance Hamming distance from the candidate to the query vector
     */
     void offer(final long offset, final long distance) {
        if (size < capacity) {
            offsets[size] = offset;
            distances[size] = distance;
            siftUp(size);
            size++;
        } else if (distance < distances[0]) {
            offsets[0] = offset;
            distances[0] = distance;
            siftDown(0, size);
        }
    }

    /**
     * Returns the worst (maximum) distance currently in the buffer without
     * modifying it. Returns {@link Long#MAX_VALUE} if the buffer is empty, so
     * callers can use this as a pruning threshold without a separate emptiness
     * check.
     */
     long peekMaxDistance() {
        return size == 0 ? Long.MAX_VALUE : distances[0];
    }

    public long offsetAt(final int i) { return offsets[i]; }

    public long distanceAt(final int i) { return distances[i]; }

    public int size() { return size; }

    public boolean isFull() { return size == capacity; }

    public int capacity() { return capacity; }

    public void reset() { size = 0; }

    /**
     * Sorts the buffer in-place by ascending distance (nearest first) using
     * heapsort.
     *
     * <p>The buffer is a valid max-heap on entry (maintained by {@link #offer}).
     * Standard heapsort extraction: repeatedly swap the root (maximum) to the
     * current end of the unsorted portion and sift the new root down. After
     * the final swap, {@code distances[0]} holds the minimum and
     * {@code distances[size-1]} holds the maximum.
     *
     * <p>Time: O(K log K). Space: O(1). No allocation.
     *
     * <p><b>Note:</b> this operation destroys the max-heap invariant. Do not
     * call {@link #offer} or {@link #peekMaxDistance} after sorting without
     * calling {@link #reset} first.
     */
    public void sortAscending() {
        for (int i=size - 1; i > 0; i--) {
            swap(0, i);
            siftDown(0, i);
        }
    }

    /** Sifts element at {@code i} up toward the root to restore the max-heap invariant. */
    private void siftUp(int i) {
        while (i > 0) {
            final int parent = (i - 1) >>> 1;
            if (distances[parent] >= distances[i]) break;
            swap(parent, i);
            i = parent;
        }
    }

    /** Sifts element at {@code i} down within {@code [0, heapSize)} to restore max-heap. */
    private void siftDown(int i, final int heapSize) {
        while (true) {
            int largest = i;
            final int left  = (i << 1) + 1;
            final int right = left + 1;
            if (left  < heapSize && distances[left]  > distances[largest]) largest = left;
            if (right < heapSize && distances[right] > distances[largest]) largest = right;
            if (largest == i) break;
            swap(i, largest);
            i = largest;
        }
    }

    private void swap(final int a, final int b) {
        final long tmpOff  = offsets[a];
        final long tmpDist = distances[a];
        offsets[a]   = offsets[b];
        distances[a] = distances[b];
        offsets[b]   = tmpOff;
        distances[b] = tmpDist;
    }

}
