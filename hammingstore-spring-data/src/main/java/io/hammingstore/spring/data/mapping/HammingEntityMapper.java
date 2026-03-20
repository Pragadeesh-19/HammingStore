package io.hammingstore.spring.data.mapping;

import io.hammingstore.client.Entity;
import io.hammingstore.spring.data.annotation.*;

import java.lang.reflect.Field;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

public final class HammingEntityMapper {

    private static final ConcurrentHashMap<Class<?>, EntityMetadata> CACHE =
            new ConcurrentHashMap<>();

    private HammingEntityMapper() {}

    public static long extractId(final Object entity) {
        Objects.requireNonNull(entity, "entity must not be null");
        try {
            return (long) metadataFor(entity.getClass()).idField().get(entity);
        } catch (final IllegalAccessException e) {
            throw new IllegalStateException(
                    "Cannot read @HammingId from " + entity.getClass().getSimpleName(), e);
        }
    }

    public static float[] extractEmbedding(final Object entity) {
        Objects.requireNonNull(entity, "entity must not be null");
        try {
            final float[] raw = (float[]) metadataFor(entity.getClass()).embeddingField().get(entity);
            if (raw == null) {
                throw new IllegalStateException(
                        "@HammingEmbedding field in " + entity.getClass().getSimpleName()
                                + " is null — embedding must be set before saving.");
            }
            return raw.clone();
        } catch (final IllegalAccessException e) {
            throw new IllegalStateException(
                    "Cannot read @HammingEmbedding from " + entity.getClass().getSimpleName(), e);
        }
    }

    public static <T> T fromClientEntity(final Entity clientEntity, final Class<T> type) {
        Objects.requireNonNull(clientEntity, "clientEntity must not be null");
        Objects.requireNonNull(type, "type must not be null");
        try {
            final T instance = type.getDeclaredConstructor().newInstance();
            final EntityMetadata meta = metadataFor(type);
            meta.idField().set(instance, clientEntity.id());
            if (meta.nameField() != null && clientEntity.name() != null) {
                meta.nameField().set(instance, clientEntity.name());
            }
            return instance;
        } catch (final ReflectiveOperationException e) {
            throw new IllegalStateException(
                    "Cannot instantiate " + type.getSimpleName()
                            + " — ensure it has a public no-arg constructor.", e);
        }
    }

    public static <T> List<T> fromClientEntities(final List<Entity> entities,
                                                 final Class<T> type) {
        final List<T> result = new ArrayList<>(entities.size());
        for (final Entity e : entities) {
            result.add(fromClientEntity(e, type));
        }
        return result;
    }

    public static void populateAuditFields(final Object entity, final boolean isNew) {
        Objects.requireNonNull(entity, "entity must not be null");
        final EntityMetadata meta = metadataFor(entity.getClass());
        final Instant now = Instant.now();
        try {
            if (isNew && meta.createdDateField() != null) {
                meta.createdDateField().set(entity, now);
            }
            if (meta.lastModifiedField() != null) {
                meta.lastModifiedField().set(entity, now);
            }
            if (meta.versionField() != null) {
                final long current = (long) meta.versionField().get(entity);
                meta.versionField().set(entity, current + 1L);
            }
        } catch (final IllegalAccessException e) {
            throw new IllegalStateException(
                    "Cannot set audit fields on " + entity.getClass().getSimpleName(), e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <P> P project(final Entity entity, final Class<P> projectionType) {
        Objects.requireNonNull(entity,"entity must not be null");
        Objects.requireNonNull(projectionType,"projectionType must not be null");
        if (!projectionType.isInterface()) {
            throw new IllegalArgumentException(
                    "@Projection target must be an interface, got: " + projectionType.getName());
        }
        return (P) java.lang.reflect.Proxy.newProxyInstance(
                projectionType.getClassLoader(),
                new Class<?>[]{ projectionType },
                (proxy, method, args) -> {
                    final String name = method.getName();
                    if (name.startsWith("get") && name.length() > 3) {
                        final String field = Character.toLowerCase(name.charAt(3)) + name.substring(4);
                        return switch (field) {
                            case "id"   -> entity.id();
                            case "name" -> entity.name();
                            default     -> null;
                        };
                    }
                    if ("toString".equals(name)) {
                        return projectionType.getSimpleName() + "@" + entity.id();
                    }
                    return null;
                });
    }

    public static EntityMetadata metadataFor(final Class<?> type) {
        return CACHE.computeIfAbsent(type, HammingEntityMapper::resolveMetadata);
    }

    private static EntityMetadata resolveMetadata(final Class<?> type) {
        Field idField = null;
        Field embeddingField = null;
        Field nameField = null;
        Field metadataField = null;
        Field createdDateField = null;
        Field lastModifiedField = null;
        Field versionField = null;

        Class<?> cursor = type;
        while (cursor != null && cursor != Object.class) {
            for (final Field field : cursor.getDeclaredFields()) {
                field.setAccessible(true);
                if (field.isAnnotationPresent(HammingId.class))           idField           = field;
                if (field.isAnnotationPresent(HammingEmbedding.class))    embeddingField    = field;
                if (field.isAnnotationPresent(HammingName.class))         nameField         = field;
                if (field.isAnnotationPresent(HammingMetadata.class))     metadataField     = field;
                if (field.isAnnotationPresent(HammingCreatedDate.class))  createdDateField  = field;
                if (field.isAnnotationPresent(HammingLastModified.class)) lastModifiedField = field;
                if (field.isAnnotationPresent(HammingVersion.class))      versionField      = field;
            }
            cursor = cursor.getSuperclass();
        }

        if (idField == null) {
            throw new IllegalStateException(
                    "No @HammingId field found in " + type.getName()
                            + ". Add @HammingId to the long field that holds the entity ID.");
        }
        if (embeddingField == null) {
            throw new IllegalStateException(
                    "No @HammingEmbedding field found in " + type.getName()
                            + ". Add @HammingEmbedding to the float[] field that holds the embedding.");
        }

        return new EntityMetadata(type, idField, embeddingField, nameField,
                metadataField, createdDateField, lastModifiedField, versionField);
    }
}
