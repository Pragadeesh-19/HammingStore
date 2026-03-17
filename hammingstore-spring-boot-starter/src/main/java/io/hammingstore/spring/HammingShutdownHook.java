package io.hammingstore.spring;

import io.hammingstore.client.HammingClient;
import io.hammingstore.client.exception.HammingException;
import org.springframework.beans.factory.DisposableBean;

import java.util.logging.Level;
import java.util.logging.Logger;

public final class HammingShutdownHook implements DisposableBean {

    private static final Logger log = Logger.getLogger(HammingShutdownHook.class.getName());

    private final HammingClient client;

    public HammingShutdownHook(final HammingClient client) {
        this.client = client;
    }

    @Override
    public void destroy() {
        log.info("HammingStore shutdown: flushing checkpoint before closing channel...");
        try {
            client.checkpoint();
            log.info("HammingStore checkpoint complete.");
        } catch (final HammingException e) {
            log.log(Level.WARNING,
                    "HammingStore checkpoint failed during shutdown (server may be unreachable). "
                            + "Proceeding with channel close. Cause: " + e.getMessage(), e);
        } finally {
            client.close();
            log.info("HammingStore channel closed.");
        }
    }
}
