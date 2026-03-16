package io.hammingstore.client.exception;

public class CapacityExceededException extends HammingException {

    public CapacityExceededException(final Throwable cause) {
        super("HammingStore capacity exhausted — server cannot accept more vectors. "
                + "Restart with a higher --max-vectors value.", cause);
    }
}
