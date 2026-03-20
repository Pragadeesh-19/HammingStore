package io.hammingstore.spring.data.mapping;

import java.lang.reflect.Field;
import java.util.Objects;

public final class EntityMetadata {

    private final Class<?> entityType;
    private final Field idField;
    private final Field embeddingField;
    private final Field nameField;
    private final Field metadataField;
    private final Field createdDateField;
    private final Field lastModifiedField;
    private final Field versionField;

    EntityMetadata(final Class<?> entityType,
                   final Field idField,
                   final Field embeddingField,
                   final Field nameField,
                   final Field metadataField,
                   final Field createdDateField,
                   final Field lastModifiedField,
                   final Field versionField) {
        this.entityType = Objects.requireNonNull(entityType);
        this.idField           = Objects.requireNonNull(idField);
        this.embeddingField    = Objects.requireNonNull(embeddingField);
        this.nameField         = nameField;
        this.metadataField     = metadataField;
        this.createdDateField  = createdDateField;
        this.lastModifiedField = lastModifiedField;
        this.versionField      = versionField;
    }

    public Class<?> entityType() { return entityType; }
    public Field idField() { return idField; }
    public Field embeddingField() { return embeddingField; }
    public Field nameField() { return nameField; }
    public Field metadataField() { return metadataField; }
    public Field createdDateField() { return createdDateField; }
    public Field lastModifiedField() { return lastModifiedField; }
    public Field versionField() { return versionField; }

    public boolean hasName() { return nameField != null; }
    public boolean hasMetadata() { return metadataField != null; }
    public boolean hasCreatedDate() { return createdDateField != null; }
    public boolean hasLastModified() { return lastModifiedField!= null; }
    public boolean hasVersion() { return versionField != null; }

    @Override
    public String toString() {
        return "EntityMetadata{type=" + entityType.getSimpleName() + "}";
    }
}
