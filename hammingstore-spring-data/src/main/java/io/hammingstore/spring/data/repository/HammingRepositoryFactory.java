package io.hammingstore.spring.data.repository;

import io.hammingstore.client.HammingClient;
import io.hammingstore.spring.data.annotation.HammingTransactional;
import io.hammingstore.spring.data.mapping.HammingEntityMapper;
import org.springframework.context.ApplicationEventPublisher;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class HammingRepositoryFactory {

    private static final Logger log = Logger.getLogger(HammingRepositoryFactory.class.getName());

    private final HammingClient client;
    private final ApplicationEventPublisher eventPublisher;

    public HammingRepositoryFactory(final HammingClient client,
                                    final ApplicationEventPublisher eventPublisher) {
        this.client = client;
        this.eventPublisher = eventPublisher;
    }

    @SuppressWarnings("unchecked")
    public <R> R createRepository(final Class<R> repositoryInterface,
                                  final Class<?> entityType) {
        final String repositoryName = repositoryInterface.getSimpleName();

        HammingEntityMapper.metadataFor(entityType);

        final HammingQueryExecutor executor = new HammingQueryExecutor(
                client, entityType, repositoryName, eventPublisher);

        final InvocationHandler handler = (proxy, method, args) -> {

            if (method.getDeclaringClass() == Object.class) {
                return switch (method.getName()) {
                    case "toString" -> repositoryInterface.getSimpleName()
                            + "Proxy@" + Integer.toHexString(System.identityHashCode(proxy));
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals"   -> proxy == args[0];
                    default         -> method.invoke(proxy, args);
                };
            }

            if (method.isDefault()) {
                return InvocationHandler.invokeDefault(proxy, method, args);
            }

            final boolean transactional =
                    method.isAnnotationPresent(HammingTransactional.class)
                            || repositoryInterface.isAnnotationPresent(HammingTransactional.class);

            try {
                final Object result = executor.execute(method, args);
                if (transactional) checkpoint(repositoryName, method.getName());
                return result;
            } catch (final RuntimeException e) {
                log.log(Level.WARNING, repositoryName + "." + method.getName()
                        + " threw: " + e.getMessage());
                throw e;
            }
        };

        log.info("HammingStore: created repository proxy — "
                + repositoryName + "<" + entityType.getSimpleName() + ">");

        return (R) Proxy.newProxyInstance(
                repositoryInterface.getClassLoader(),
                new Class<?>[]{ repositoryInterface },
                handler);
    }

    private void checkpoint(final String repositoryName, final String methodName) {
        try {
            client.checkpoint();
        } catch (final Exception e) {
            log.log(Level.WARNING,
                    "@HammingTransactional checkpoint failed after "
                            + repositoryName + "." + methodName + ": " + e.getMessage(), e);
        }
    }
}
