package io.hammingstore.client.internal;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.hammingstore.client.exception.CapacityExceededException;
import io.hammingstore.client.exception.EntityNotFoundException;
import io.hammingstore.client.exception.HammingConnectException;
import io.hammingstore.client.exception.HammingException;

import java.util.concurrent.ExecutionException;

public final class ErrorTranslator {

    private ErrorTranslator() {}

    public static HammingException translate(final StatusRuntimeException e, final String endpoint) {
        final Status.Code code = e.getStatus().getCode();
        final String details = e.getStatus().getDescription() != null
                ? e.getStatus().getDescription()
                : e.getMessage();

        return switch (code) {
            case UNAVAILABLE, DEADLINE_EXCEEDED ->
                    new HammingConnectException(endpoint, e);

            case INVALID_ARGUMENT -> {
                final long entityId = parseEntityId(details);
                final String role = parseRole(details);
                if (entityId != -1L) {
                    yield new EntityNotFoundException(entityId, role, e);
                }
                yield new HammingException("Invalid argument: " + details, e);
            }

            case RESOURCE_EXHAUSTED ->
                    new CapacityExceededException(e);

            default ->
                    new HammingException("Server error [" + code + "]: " + details, e);
        };
    }

    public static HammingException wrap(final Exception e) {
        final Throwable cause = (e instanceof ExecutionException) ? e.getCause() : e;

        if (cause instanceof HammingException he) {
            return he;
        }
        if (cause instanceof StatusRuntimeException sre) {
            return translate(sre, "unknown");
        }
        return new HammingException(
                "Unexpected error during batch operation: "
                        + (cause != null ? cause.getMessage() : e.getMessage()),
                cause != null ? cause : e);
    }

    private static long parseEntityId(final String description) {
        if (description == null) return -1L;
        final int idxId = description.indexOf("id=");
        if (idxId == -1) return -1L;
        try {
            final String after = description.substring(idxId + 3).trim();
            final int    end   = after.indexOf(' ');
            final String raw   = end == -1 ? after : after.substring(0, end);
            return Long.parseLong(raw);
        } catch (NumberFormatException ignored) {
            return -1L;
        }
    }

    private static String parseRole(final String description) {
        if (description == null) return "unknown";
        final int idxRole = description.indexOf("role=");
        if (idxRole == -1) return "unknown";
        final String after = description.substring(idxRole + 5);
        final int end = after.indexOf(' ');
        return end == -1 ? after : after.substring(0, end);
    }
}
