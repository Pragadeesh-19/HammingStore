package io.hammingstore.spring.data.event;

import org.springframework.context.ApplicationEvent;

public final class EntityDeletedEvent extends ApplicationEvent {

    private final long entity;
    private final String repositoryName;

    public EntityDeletedEvent(final Object source,
                              final long entityId,
                              final String repositoryName) {
        super(source);
        this.entity = entityId;
        this.repositoryName = repositoryName;
    }

    public long entityId() { return entity; }
    public String repositoryName() { return repositoryName; }
}
