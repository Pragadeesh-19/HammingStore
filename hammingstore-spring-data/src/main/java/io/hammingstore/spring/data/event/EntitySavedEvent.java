package io.hammingstore.spring.data.event;

import org.springframework.context.ApplicationEvent;

public final class EntitySavedEvent<T> extends ApplicationEvent {

    private final T entity;
    private final String repositoryName;

    public EntitySavedEvent(final Object source,
                            final T entity,
                            final String repositoryName) {
        super(source);
        this.entity = entity;
        this.repositoryName = repositoryName;
    }

    public T entity() { return entity;}

    public String repositoryName() { return repositoryName; }
}
