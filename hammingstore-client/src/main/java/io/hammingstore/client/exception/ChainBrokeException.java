package io.hammingstore.client.exception;

import io.hammingstore.client.Entity;

import java.util.Collections;
import java.util.List;

public class ChainBrokeException extends HammingException {

    private final int failedHop;
    private final List<Entity> partialPath;

    public ChainBrokeException(final int failedHop,
                               final List<Entity> partialPath,
                               final Throwable cause) {
        super("Chain traversal broke at hop " + failedHop
                + " - no results. Partial path length: " + partialPath.size(), cause);
        this.failedHop = failedHop;
        this.partialPath = Collections.unmodifiableList(partialPath);
    }

    public int failedHop() { return failedHop; }

    public List<Entity> partialPath() { return partialPath; }
}
