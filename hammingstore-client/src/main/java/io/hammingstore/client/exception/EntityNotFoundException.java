package io.hammingstore.client.exception;

public class EntityNotFoundException extends HammingException {

    private final long entityId;
    private final String role;

    public EntityNotFoundException(final long entityId, final String role, final Throwable cause) {
        super("Entity not found in index: id=" + entityId + ", role=" + role, cause);
        this.entityId = entityId;
        this.role = role;
    }

    public EntityNotFoundException(final long entityId, final String role) {
        super("Entity not found in index: id=" + entityId + ", role=" + role);
        this.entityId = entityId;
        this.role = role;
    }

    public long entityId() { return entityId; }

    public String role() { return role; }
}
