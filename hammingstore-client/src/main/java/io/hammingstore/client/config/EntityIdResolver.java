package io.hammingstore.client.config;

import io.hammingstore.client.Entity;

@FunctionalInterface
public interface EntityIdResolver {

    long resolve(String name);

    EntityIdResolver DEFAULT = Entity::defaultId;

    static EntityIdResolver forRelations(final EntityIdResolver base) {
        return name -> base.resolve("REL:" + name);
    }
}
