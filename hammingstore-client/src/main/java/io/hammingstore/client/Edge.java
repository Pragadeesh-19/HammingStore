package io.hammingstore.client;

import java.util.Objects;

public final class Edge {

    private final Entity subject;
    private final String relation;
    private final Entity object;

    private Edge(final Entity subject, final String relation, final Entity object) {
        this.subject = Objects.requireNonNull(subject,"subject must not be null");
        this.relation = Objects.requireNonNull(relation,"relation must not be null");
        this.object = Objects.requireNonNull(object,"object must not be null");
        if (relation.isBlank()) throw new IllegalArgumentException("relation must not be blank");
    }

    public static Edge of(final Entity subject, final String relation, final Entity object) {
        return new Edge(subject, relation, object);
    }

    public static Edge named(final String subject, final String relation, final String object) {
        return new Edge(Entity.named(subject), relation, Entity.named(object));
    }

    public Entity subject() { return subject; }

    public String relation() { return relation; }

    public Entity object() { return object; }

    public long relationId() {
        return Entity.defaultId("REL:" + relation);
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (!(o instanceof Edge other)) return false;
        return subject.equals(other.subject)
                && relation.equals(other.relation)
                && object.equals(other.object);
    }

    @Override
    public int hashCode() {
        return Objects.hash(subject, relation, object);
    }

    @Override
    public String toString() {
        return subject + " -[" + relation + "]-> " + object;
    }
}
