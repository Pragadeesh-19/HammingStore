package io.hammingstore.hnsw;

public final class CandidateMinHeap {

    private final long[] offsets;
    private final long[] distances;
    private final int capacity;
    private int size;

    public CandidateMinHeap(final int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be > 0");
        this.capacity  = capacity;
        this.offsets   = new long[capacity];
        this.distances = new long[capacity];
        this.size      = 0;
    }

    public void push(final long offset, final long distance) {
        if (size >= capacity) {
            throw new IllegalStateException("CandidateMinHeap full at capacity=" + capacity);
        }
        offsets[size]   = offset;
        distances[size] = distance;
        siftUp(size);
        size++;
    }

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

    public long peekMinDistance() {
        return size == 0 ? Long.MAX_VALUE : distances[0];
    }

    public long popMinDistance() {
        if (size == 0) throw new IllegalStateException("CandidateMinHeap is empty");
        final long result = distances[0];
        size--;
        if (size > 0) {
            offsets[0]   = offsets[size];
            distances[0] = distances[size];
            siftDown(0);
        }
        return result;
    }

    public int size() { return size; }

    public boolean isEmpty() { return size == 0; }

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
