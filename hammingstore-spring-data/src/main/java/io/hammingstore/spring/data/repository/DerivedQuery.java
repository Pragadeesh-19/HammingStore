package io.hammingstore.spring.data.repository;

import java.util.List;

public final class DerivedQuery {

    public enum Type { RELATION, CHAIN }

    private final Type type;
    private final String relation;
    private final List<String> relations;

    private DerivedQuery(final Type type, final String relation, final List<String> relations) {
        this.type = type;
        this.relation = relation;
        this.relations = relations;
    }

    public static DerivedQuery relation(final String relation) {
        return new DerivedQuery(Type.RELATION, relation, List.of());
    }

    public static DerivedQuery chain(final List<String> relations) {
        return new DerivedQuery(Type.CHAIN, null, List.copyOf(relations));
    }

    public Type type() { return type; }

    public String relation() { return relation; }

    public List<String> relations() { return relations; }

}
