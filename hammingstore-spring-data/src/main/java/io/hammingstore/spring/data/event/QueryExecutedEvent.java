package io.hammingstore.spring.data.event;

import org.springframework.context.ApplicationEvent;

public final class QueryExecutedEvent extends ApplicationEvent {

    private final String operation;
    private final long latencyMs;
    private final int resultCount;
    private final String repositoryName;
    private final String methodName;

    public QueryExecutedEvent(final Object source,
                              final String operation,
                              final long latencyMs,
                              final int resultCount,
                              final String repositoryName,
                              final String methodName) {
        super(source);
        this.operation = operation;
        this.latencyMs = latencyMs;
        this.resultCount = resultCount;
        this.repositoryName = repositoryName;
        this.methodName = methodName;
    }

    public String operation() { return operation; }
    public long latencyMs() { return latencyMs; }
    public int resultCount() { return resultCount; }
    public String repositoryName() { return repositoryName; }
    public String methodName() { return methodName; }
}
