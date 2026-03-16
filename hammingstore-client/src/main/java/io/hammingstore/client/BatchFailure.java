package io.hammingstore.client;

import io.hammingstore.client.exception.HammingException;

import java.util.Objects;

public final class BatchFailure {

    private final long entityId;
    private final HammingException cause;

    public BatchFailure(final long entityId, final HammingException cause) {
        this.entityId = entityId;
        this.cause = Objects.requireNonNull(cause, "cause must not be null");
    }

    public long entityId() { return entityId; }

    public HammingException cause() { return cause; }

    @Override
    public String toString() {
        return "BatchFailure{entityId=" + entityId + ", reason=" + cause.getMessage() + "}";
    }
}
