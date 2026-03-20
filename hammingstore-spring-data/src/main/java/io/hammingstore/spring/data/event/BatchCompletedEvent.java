package io.hammingstore.spring.data.event;

import io.hammingstore.client.BatchResult;
import org.springframework.context.ApplicationEvent;

public final class BatchCompletedEvent extends ApplicationEvent {

    private final BatchResult batchResult;
    private final String repositoryName;

    public BatchCompletedEvent(final Object source,
                               final BatchResult batchResult,
                               final String repositoryName) {
        super(source);
        this.batchResult = batchResult;
        this.repositoryName = repositoryName;
    }

    public BatchResult batchResult() { return batchResult; }
    public String repositoryName() { return repositoryName; }
}
