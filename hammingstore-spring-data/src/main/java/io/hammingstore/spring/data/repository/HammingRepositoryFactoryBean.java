package io.hammingstore.spring.data.repository;

import io.hammingstore.client.HammingClient;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;

public final class HammingRepositoryFactoryBean<R> implements FactoryBean {

    private final Class<R> repositoryInterface;
    private final Class<?> entityType;

    @Autowired
    private HammingClient client;
    @Autowired private ApplicationEventPublisher eventPublisher;

    public HammingRepositoryFactoryBean(final Class<R>  repositoryInterface,
                                        final Class<?>  entityType) {
        this.repositoryInterface = repositoryInterface;
        this.entityType = entityType;
    }

    @Override
    public R getObject() {
        return new HammingRepositoryFactory(client, eventPublisher)
                .createRepository(repositoryInterface, entityType);
    }

    @Override
    public Class<?> getObjectType() {
        return repositoryInterface;
    }

    @Override
    public boolean isSingleton() {
        return true;
    }
}
