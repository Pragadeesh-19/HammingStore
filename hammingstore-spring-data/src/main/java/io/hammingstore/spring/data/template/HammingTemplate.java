package io.hammingstore.spring.data.template;

import io.hammingstore.client.*;
import io.hammingstore.client.config.EntityIdResolver;
import io.hammingstore.spring.data.HammingOperations;
import io.hammingstore.spring.data.event.BatchCompletedEvent;
import io.hammingstore.spring.data.event.EntityDeletedEvent;
import io.hammingstore.spring.data.event.EntitySavedEvent;
import io.hammingstore.spring.data.event.QueryExecutedEvent;
import io.hammingstore.spring.data.mapping.HammingEntityMapper;
import org.springframework.context.ApplicationEventPublisher;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class HammingTemplate<T, ID> implements HammingOperations<T, ID> {

    private static final String TEMPLATE = "HammingTemplate";

    private static final EntityIdResolver RELATION_RESOLVER =
            EntityIdResolver.forRelations(EntityIdResolver.DEFAULT);

    protected final HammingClient client;
    protected final ApplicationEventPublisher eventPublisher;
    protected Class<T> entityType;

    public HammingTemplate(final HammingClient client,
                           final ApplicationEventPublisher eventPublisher) {
        this.client = Objects.requireNonNull(client,"client must not be null");
        this.eventPublisher = Objects.requireNonNull(eventPublisher, "eventPublisher must not be null");
    }

    public void setEntityType(final Class<T> entityType) {
        this.entityType = entityType;
    }

    @Override
    public T save(final T entity) {
        Objects.requireNonNull(entity, "entity must not be null");
        HammingEntityMapper.populateAuditFields(entity, false);
        final long id = HammingEntityMapper.extractId(entity);
        final float[] embedding = HammingEntityMapper.extractEmbedding(entity);
        client.storeFloat(id, embedding);
        eventPublisher.publishEvent(new EntitySavedEvent<>(this, entity, TEMPLATE));
        return entity;
    }

    @Override
    public BatchResult saveBatch(final List<T> entities) {
        Objects.requireNonNull(entities, "entities must not be null");
        final List<StoreRequest> requests = new ArrayList<>(entities.size());
        for (final T e : entities) {
            HammingEntityMapper.populateAuditFields(e, false);
            requests.add(StoreRequest.of(
                    HammingEntityMapper.extractId(e),
                    HammingEntityMapper.extractEmbedding(e)));
        }
        final BatchResult result = client.storeBatch(requests);
        eventPublisher.publishEvent(new BatchCompletedEvent(this, result, TEMPLATE));
        return result;
    }

    @Override
    public T findById(final ID id) {
        throw new UnsupportedOperationException(
                "findById is not supported by HammingStore. "
                        + "Use search() or a VSA query to get entity IDs, "
                        + "then load full records from your primary database.");
    }

    @Override
    public List<SearchResult> search(final float[] embedding, final int k) {
        Objects.requireNonNull(embedding, "embedding must not be null");
        return client.search(embedding).topK(k).execute();
    }

    @Override
    public void delete(final ID id) {
        final long entityId = ((Number) id).longValue();
        client.retract(entityId);
        eventPublisher.publishEvent(new EntityDeletedEvent(this, entityId, TEMPLATE));
    }

    @Override
    public boolean ping() {
        return client.ping();
    }

    public List<Entity> chain(final long startId, final int k, final String... relations) {
        final long start = System.currentTimeMillis();
        var builder = client.from(startId);
        for (final String rel : relations) {
            builder = builder.via(rel);
        }
        final List<Entity> results = builder.topK(k).execute();
        eventPublisher.publishEvent(new QueryExecutedEvent(
                this, "chain", System.currentTimeMillis() - start,
                results.size(), TEMPLATE, "chain"));
        return results;
    }

    public List<Entity> hop(final long entityId, final String relation, final int k) {
        final long start = System.currentTimeMillis();
        final long relId = RELATION_RESOLVER.resolve(relation);
        final List<Entity> results = client.hop(entityId, relId).topK(k).execute();
        eventPublisher.publishEvent(new QueryExecutedEvent(
                this, "hop", System.currentTimeMillis() - start,
                results.size(), TEMPLATE, "hop"));
        return results;
    }

    public List<Entity> analogy(final String subjectA, final String objectA,
                                final String subjectB, final int k) {
        return client.analogy(subjectA, objectA).isTo(subjectB).topK(k).execute();
    }
}
