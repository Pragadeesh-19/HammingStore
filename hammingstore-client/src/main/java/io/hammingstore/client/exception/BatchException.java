package io.hammingstore.client.exception;

import io.hammingstore.client.BatchFailure;

import java.util.Collections;
import java.util.List;

public final class BatchException extends HammingException {

    private final List<BatchFailure> failures;

    public BatchException(final List<BatchFailure> failures) {
        super(failures.size() + " entity/entities failed in batch operation. "
                + "First failure: " + failures.get(0));
        this.failures = Collections.unmodifiableList(failures);
    }

    public List<BatchFailure> failures() { return failures; }
}
