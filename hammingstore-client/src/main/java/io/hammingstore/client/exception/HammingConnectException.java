package io.hammingstore.client.exception;

public class HammingConnectException extends HammingException {
    public HammingConnectException(final String endpoint, final Throwable cause) {
        super("Cannot connect to HammingStore at " + endpoint, cause);
    }
}
