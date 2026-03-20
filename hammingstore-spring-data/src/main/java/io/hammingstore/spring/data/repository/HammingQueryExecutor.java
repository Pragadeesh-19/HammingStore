package io.hammingstore.spring.data.repository;

import io.hammingstore.client.*;
import io.hammingstore.client.config.EntityIdResolver;
import io.hammingstore.spring.data.annotation.*;
import io.hammingstore.spring.data.event.BatchCompletedEvent;
import io.hammingstore.spring.data.event.EntityDeletedEvent;
import io.hammingstore.spring.data.event.EntitySavedEvent;
import io.hammingstore.spring.data.event.QueryExecutedEvent;
import io.hammingstore.spring.data.mapping.HammingEntityMapper;
import org.springframework.context.ApplicationEventPublisher;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class HammingQueryExecutor {

    private static final Logger log = Logger.getLogger(HammingQueryExecutor.class.getName());

    private static final int DEFAULT_K = 10;

    private static final EntityIdResolver RELATION_RESOLVER =
            EntityIdResolver.forRelations(EntityIdResolver.DEFAULT);

    private final HammingClient client;
    private final Class<?> entityType;
    private final String repositoryName;
    private final ApplicationEventPublisher eventPublisher;

    public HammingQueryExecutor(final HammingClient client,
                                final Class<?> entityType,
                                final String repositoryName,
                                final ApplicationEventPublisher eventPublisher) {
        this.client = client;
        this.entityType = entityType;
        this.repositoryName = repositoryName;
        this.eventPublisher = eventPublisher;
    }

    public Object execute(final Method method, final Object[] args) {
        final long start = System.currentTimeMillis();
        final Object[] safeArgs = args != null ? args : new Object[0];

        try {
            final Object result = route(method, safeArgs);
            final long latency = System.currentTimeMillis() - start;
            publishQueryEvent(method, latency, resultCount(result));
            return adaptReturnType(method, result);
        } catch (final RuntimeException e) {
            final long latency = System.currentTimeMillis() - start;
            publishQueryEvent(method, latency, 0);
            return handleCircuitBreaker(method, e);
        }
    }

    private Object route(final Method method, final Object[] args) {
        if (method.isAnnotationPresent(Chain.class)) return executeChain(method, args);
        if (method.isAnnotationPresent(HopQuery.class)) return executeHop(method, args);
        if (method.isAnnotationPresent(Analogy.class)) return executeAnalogy(method, args);
        if (method.isAnnotationPresent(BundleQuery.class)) return executeBundle(method, args);
        if (method.isAnnotationPresent(RelationQuery.class)) return executeRelation(method, args);

        final DerivedQuery derived = DerivedQueryParser.parse(method.getName());
        if (derived != null) return executeDerived(derived, method, args);

        return executeCrud(method, args);
    }

    private List<Entity> executeChain(final Method method, final Object[] args) {
        final Chain ann = method.getAnnotation(Chain.class);
        final long id = extractLongArg(args, 0);
        final int k = extractTopK(method, args, DEFAULT_K);
        var builder = client.from(id);
        for (final String rel : ann.relations()) {
            builder = builder.via(rel);
        }
        return builder.topK(k).execute();
    }

    private List<Entity> executeHop(final Method method, final Object[] args) {
        final HopQuery ann = method.getAnnotation(HopQuery.class);
        final long entityId = extractLongArg(args, 0);
        final int k = extractTopK(method, args, DEFAULT_K);
        final long relId = RELATION_RESOLVER.resolve(ann.value());
        return client.hop(entityId, relId).topK(k).execute();
    }

    private List<Entity> executeAnalogy(final Method method, final Object[] args) {
        final Analogy ann = method.getAnnotation(Analogy.class);
        final int k = extractTopK(method, args, DEFAULT_K);
        final String targetEntity = requireStringArg(args, 0, "@Analogy", method.getName());
        return client.analogy(ann.subjectA(), ann.objectA()).isTo(targetEntity).topK(k).execute();
    }

    private List<Entity> executeBundle(final Method method, final Object[] args) {
        final BundleQuery ann = method.getAnnotation(BundleQuery.class);
        final int k = extractTopK(method, args, DEFAULT_K);
        return client.bundle(ann.entities()).topK(k).execute();
    }

    private List<Entity> executeRelation(final Method method, final Object[] args) {
        final RelationQuery ann = method.getAnnotation(RelationQuery.class);
        final String objectName = requireStringArg(args, 0, "@RelationQuery", method.getName());
        final int k = extractTopK(method, args, DEFAULT_K);
        final long relId = RELATION_RESOLVER.resolve(ann.value());
        final long objectId = EntityIdResolver.DEFAULT.resolve(objectName);
        return client.relation(relId, objectId).topK(k).execute();
    }

    private Object executeDerived(final DerivedQuery derived,
                                  final Method method,
                                  final Object[] args) {
        return switch (derived.type()) {
            case RELATION -> {
                final String objectName = requireStringArg(args, 0, "derived @RelationQuery", method.getName());
                final int k = args.length > 1 ? ((Number) args[1]).intValue() : DEFAULT_K;
                yield client.relation(derived.relation(), objectName).topK(k).execute();
            }
            case CHAIN -> {
                final long startId = extractLongArg(args, 0);
                final int  k = args.length > 1 ? ((Number) args[1]).intValue() : DEFAULT_K;
                var builder = client.from(startId);
                for (final String rel : derived.relations()) {
                    builder = builder.via(rel);
                }
                yield builder.topK(k).execute();
            }
        };
    }

    @SuppressWarnings("unchecked")
    private Object executeCrud(final Method method, final Object[] args) {
        return switch (method.getName()) {
            case "save" -> executeSave(args[0]);
            case "saveBatch" -> executeSaveBatch((List<Object>) args[0]);
            case "search" -> executeSearch(method, args);
            case "delete" -> { executeDelete(args[0]); yield null; }
            case "ping" -> client.ping();
            case "findById" -> throw new UnsupportedOperationException(
                    "findById is not supported by HammingStore. "
                            + "Use search() or a VSA query annotation to get entity IDs, "
                            + "then load full records from your primary database.");
            default -> throw new UnsupportedOperationException(
                    "Unknown repository method: '" + method.getName() + "' on "
                            + repositoryName + ". "
                            + "Annotate with @Chain, @HopQuery, @Analogy, @BundleQuery, "
                            + "or @RelationQuery; or use the derived query naming convention.");
        };
    }

    private Object executeSave(final Object entity) {
        HammingEntityMapper.populateAuditFields(entity, false);
        final long id = HammingEntityMapper.extractId(entity);
        final float[] embedding = HammingEntityMapper.extractEmbedding(entity);
        client.storeFloat(id, embedding);
        eventPublisher.publishEvent(new EntitySavedEvent<>(this, entity, repositoryName));
        return entity;
    }

    private BatchResult executeSaveBatch(final List<Object> entities) {
        final List<StoreRequest> requests = new ArrayList<>(entities.size());
        for (final Object e : entities) {
            HammingEntityMapper.populateAuditFields(e, false);
            requests.add(StoreRequest.of(
                    HammingEntityMapper.extractId(e),
                    HammingEntityMapper.extractEmbedding(e)));
        }
        final BatchResult result = client.storeBatch(requests);
        eventPublisher.publishEvent(new BatchCompletedEvent(this, result, repositoryName));
        return result;
    }

    private List<SearchResult> executeSearch(final Method method, final Object[] args) {
        final float[] embedding = (float[]) args[0];
        final int k = ((Number) args[1]).intValue();
        final double minSim = extractMinSimilarity(method, args);

        final var builder = client.search(embedding).topK(k);
        return minSim > 0.0 ? builder.minSimilarity(minSim).execute() : builder.execute();
    }

    private void executeDelete(final Object idArg) {
        final long id = ((Number) idArg).longValue();
        client.retract(id);
        eventPublisher.publishEvent(new EntityDeletedEvent(this, id, repositoryName));
    }

    private Object adaptReturnType(final Method method, final Object result) {
        if (method.getReturnType() == CompletableFuture.class) {
            return CompletableFuture.completedFuture(result);
        }

        // @Projection - map Entity objects to DTO interface proxies
        final Projection projection = method.getAnnotation(Projection.class);
        if (projection != null && result instanceof List<?> list) {
            final List<Object> projected = new ArrayList<>(list.size());
            for (final Object item : list) {
                if (item instanceof Entity entity) {
                    projected.add(HammingEntityMapper.project(entity, projection.value()));
                } else {
                    projected.add(item);
                }
            }
            return projected;
        }

        return result;
    }

    private Object handleCircuitBreaker(final Method method, final RuntimeException e) {
        final CircuitBreaker cb = method.getAnnotation(CircuitBreaker.class);
        if (cb == null || "throw".equals(cb.fallback())) throw e;
        if ("empty".equals(cb.fallback())) {
            log.log(Level.WARNING, "HammingStore circuit breaker 'empty' fallback for "
                    + repositoryName + "." + method.getName() + ": " + e.getMessage());
            return Collections.emptyList();
        }
        throw e;
    }

    private long extractLongArg(final Object[] args, final int index) {
        if (args[index] instanceof Long l) return l;
        if (args[index] instanceof Number n) return n.longValue();
        throw new IllegalArgumentException(
                "Expected long at parameter index " + index + ", got: "
                        + args[index].getClass().getSimpleName());
    }

    private String requireStringArg(final Object[] args, final int index,
                                    final String annotationName, final String methodName) {
        if (args.length <= index) {
            throw new IllegalArgumentException(
                    annotationName + " method '" + methodName
                            + "' requires a String entity name as parameter " + index + ".");
        }
        if (args[index] instanceof String s) {
            return s;
        }
        throw new IllegalArgumentException(
                annotationName + " method '" + methodName
                        + "' requires a String entity name at parameter " + index
                        + ", but received " + args[index].getClass().getSimpleName() + ". "
                        + "Pass the entity name as a String (e.g. \"France\"), not an ID.");
    }

    private int extractTopK(final Method method, final Object[] args, final int defaultK) {
        final Parameter[] params = method.getParameters();
        for (int i = 0; i < params.length; i++) {
            if (params[i].isAnnotationPresent(TopK.class) && i < args.length) {
                return ((Number) args[i]).intValue();
            }
        }
        return defaultK;
    }

    private double extractMinSimilarity(final Method method, final Object[] args) {
        final Parameter[] params = method.getParameters();
        for (int i = 0; i < params.length; i++) {
            if (params[i].isAnnotationPresent(MinSimilarity.class) && i < args.length) {
                return ((Number) args[i]).doubleValue();
            }
        }
        return -1.0;
    }

    private int resultCount(final Object result) {
        return result instanceof List<?> list ? list.size() : 0;
    }

    private void publishQueryEvent(final Method method, final long latencyMs, final int count) {
        eventPublisher.publishEvent(new QueryExecutedEvent(
                this, resolveOperation(method), latencyMs, count,
                repositoryName, method.getName()));
    }

    private String resolveOperation(final Method method) {
        if (method.isAnnotationPresent(Chain.class)) return "chain";
        if (method.isAnnotationPresent(HopQuery.class)) return "hop";
        if (method.isAnnotationPresent(Analogy.class)) return "analogy";
        if (method.isAnnotationPresent(BundleQuery.class)) return "bundle";
        if (method.isAnnotationPresent(RelationQuery.class)) return "relation";
        return method.getName();
    }
}
