package io.hammingstore.client.exception;

public class HammingException extends RuntimeException {

    public HammingException(final String message) {
        super(message);
    }

    public HammingException(final String message, final Throwable cause) {
        super(message, cause);
    }
}
