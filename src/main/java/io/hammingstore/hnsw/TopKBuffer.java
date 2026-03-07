package io.hammingstore.hnsw;

public final class TopKBuffer {

    private final long[] offsets;
    private final long[] distances;
    private final int capacity;
    private int size;

    public TopKBuffer(final int k) {
        if (k<= 0) throw new IllegalArgumentException("k must be > 0");
        this.capacity = k;
        this.offsets = new long[k];
        this.distances = new long[k];
        this.size = 0;
    }

    public void offer(final long offset, final long distance) {
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

    public long peekMaxDistance() {
        return size == 0 ? Long.MAX_VALUE : distances[0];
    }

    public long pollMaxOffset() {
        if (size == 0) throw new IllegalStateException("TopKBuffer is empty");
        final long result = offsets[0];
        size--;
        if (size > 0) {
            offsets[0]   = offsets[size];
            distances[0] = distances[size];
            siftDown(0, size);
        }
        return result;
    }

    public long offsetAt(final int i) { return offsets[i]; }

    public long distanceAt(final int i) { return distances[i]; }

    public int size() { return size; }

    public boolean isFull() { return size == capacity; }

    public int capacity() { return capacity; }

    public void reset() { size = 0; }

    public void sortAscending() {
        int n = size;
        for (int i=n-1; i > 0; i--) {
            swap(0, i);
            siftDown(0, i);
        }

        for (int i= (size/2) - 1; i >= 0; i--) {
            siftDown(i, size);
        }
    }

    private void siftUp(int i) {
        while (i > 0) {
            final int parent = (i - 1) >>> 1;
            if (distances[parent] >= distances[i]) break;
            swap(parent, i);
            i = parent;
        }
    }

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
