package io.hammingstore.hnsw;

/**
 * A fixed-capacity binary min-heap ordered by Hamming distance.
 *
 * <p>Used as the exploration frontier during HNSW beam search. At each step,
 * the candidate with the smallest distance to the query is expanded first,
 * ensuring the search explores the most promising region of the graph before
 * less promising ones.
 *
 * <p>The heap stores two parallel arrays — {@code offsets} (byte offsets into
 * the layer's storage segment) and {@code distances} (Hamming distances to the
 * query vector) — kept in sync by every heap operation. The heap invariant is
 * maintained on {@code distances}: {@code distances[parent] ≤ distances[child]}
 * for all nodes.
 *
 * <p>Instances are not thread-safe. Each search context ({@code HNSWIndex.SearchContext})
 * owns a dedicated heap and resets it via {@link #reset()}
 */
public final class CandidateMinHeap {

    private final long[] offsets;
    private final long[] distances;
    private final int capacity;
    private int size;

    /**
     * Creates a heap with the given fixed capacity.
     *
     * @param capacity maximum number of candidates the heap can hold
     * @throws IllegalArgumentException if {@code capacity} is not positive
     */
    public CandidateMinHeap(final int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be > 0");
        this.capacity  = capacity;
        this.offsets   = new long[capacity];
        this.distances = new long[capacity];
        this.size      = 0;
    }

    /**
     * Inserts a candidate into the heap.
     *
     * @param offset byte offset of the candidate node in the layer's storage segment
     * @param distance Hamming distance from the candidate to the query vector
     * @throws IllegalStateException if the heap is already at capacity
     */
    public void push(final long offset, final long distance) {
        if (size >= capacity) {
            throw new IllegalStateException("CandidateMinHeap full at capacity=" + capacity);
        }
        offsets[size]   = offset;
        distances[size] = distance;
        siftUp(size);
        size++;
    }

    /**
     * Removes and returns the byte offset of the minimum-distance candidate.
     *
     * @return the byte offset of the nearest candidate
     * @throws IllegalStateException if the heap is empty
     */
    public long popMinOffset() {
        if (size == 0) throw new IllegalStateException("CandidateMinHeap is empty");
        final long result = offsets[0];
        size--;
        if (size > 0) {
            offsets[0]   = offsets[size];
            distances[0] = distances[size];
            siftDown(0);
        }
        return result;
    }

    /**
     * Returns the minimum distance without removing the candidate.
     * Returns {@link Long#MAX_VALUE} if the heap is empty, so callers can use
     * it as a pruning threshold without a separate emptiness check.
     */
    public long peekMinDistance() {
        return size == 0 ? Long.MAX_VALUE : distances[0];
    }

    public int size() { return size; }

    public boolean isEmpty() { return size == 0; }

    /**
     * Discards all candidates without releasing storage.
     * Called between searches to reuse the heap allocation across queries.
     */
    public void reset() { size = 0; }

    private void siftUp(int i) {
        while (i > 0) {
            final int parent = (i - 1) >>> 1;
            if (distances[parent] <= distances[i]) break;
            swap(parent, i);
            i = parent;
        }
    }

    private void siftDown(int i) {
        while (true) {
            int smallest = i;
            final int left  = (i << 1) + 1;
            final int right = left + 1;
            if (left  < size && distances[left]  < distances[smallest]) smallest = left;
            if (right < size && distances[right] < distances[smallest]) smallest = right;
            if (smallest == i) break;
            swap(i, smallest);
            i = smallest;
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
