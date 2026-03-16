package io.hammingstore.client;

import io.hammingstore.client.exception.BatchException;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class BatchResult {

    private final int successCount;
    private final List<BatchFailure> failures;

    private BatchResult(final int successCount, final List<BatchFailure> failures) {
        this.successCount = successCount;
        this.failures = Collections.unmodifiableList(
                Objects.requireNonNull(failures,"failures must not be null"));
    }

    public static BatchResult of(final int successCount, final List<BatchFailure> failures) {
        return new BatchResult(successCount, failures);
    }

    public int successCount() { return successCount; }

    public int failureCount() { return failures.size(); }

    public int totalCount() { return successCount + failures.size(); }

    public boolean hasFailures() { return !failures.isEmpty(); }

    public boolean isFullSuccess() { return failures.isEmpty(); }

    public List<BatchFailure> failures() { return failures; }

    public void throwIfAnyFailure() {
        if (hasFailures()) {
            throw new BatchException(failures);
        }
    }

    @Override
    public String toString() {
        return "BatchResult{success=" + successCount
                + ", failures=" + failures.size()
                + ", total=" + totalCount() + "}";
    }


}
