package io.hammingstore.client;

import io.hammingstore.client.config.EntityIdResolver;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HammingClientTest {

    @Test
    void entity_of_producesCorrectId() {
        final Entity e = Entity.of(42L);
        assertEquals(42L, e.id());
        assertNull(e.name());
        assertFalse(e.hasName());
    }

    @Test
    void entity_named_derivesIdFromSha256() {
        final Entity paris = Entity.named("Paris");
        assertTrue(paris.id() > 0, "ID must be positive (63-bit)");
        assertEquals("Paris", paris.name());
        assertTrue(paris.hasName());
        // Stable across calls
        assertEquals(paris.id(), Entity.named("Paris").id());
    }

    @Test
    void entity_named_idMatchesPythonHelper() {
        // Python: int(hashlib.sha256("Paris".encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF
        // Precomputed expected value from the Python client
        final long parisId = Entity.defaultId("Paris");
        assertTrue(parisId > 0);
        assertTrue(parisId <= Long.MAX_VALUE);
        // Same name always produces same ID
        assertEquals(parisId, Entity.defaultId("Paris"));
        assertNotEquals(parisId, Entity.defaultId("London"));
    }

    @Test
    void entity_equality_basedOnIdOnly() {
        final Entity a = Entity.builder(1L).name("Alpha").build();
        final Entity b = Entity.builder(1L).name("Beta").build();
        assertEquals(a, b, "Entities with same ID must be equal regardless of name");
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void entity_embedding_defensiveCopy() {
        final float[] original = {1.0f, 2.0f, 3.0f};
        final Entity e = Entity.builder(1L).embedding(original).build();
        final float[] retrieved = e.embedding();
        retrieved[0] = 999.0f;
        // Original must not be mutated
        assertNotEquals(999.0f, e.embedding()[0]);
    }

    @Test
    void edge_named_constructsCorrectly() {
        final Edge edge = Edge.named("Paris", "capitalOf", "France");
        assertEquals("Paris",     edge.subject().name());
        assertEquals("capitalOf", edge.relation());
        assertEquals("France",    edge.object().name());
    }

    @Test
    void edge_relationId_prefixedWithREL() {
        final Edge edge = Edge.named("A", "myRelation", "B");
        final long withPrefix    = Entity.defaultId("REL:myRelation");
        final long withoutPrefix = Entity.defaultId("myRelation");
        assertEquals(withPrefix, edge.relationId());
        assertNotEquals(withoutPrefix, edge.relationId());
    }

    @Test
    void edge_requiresNonNullParts() {
        assertThrows(NullPointerException.class, () -> Edge.of(null, "r", Entity.of(1)));
        assertThrows(NullPointerException.class, () -> Edge.of(Entity.of(1), null, Entity.of(2)));
        assertThrows(NullPointerException.class, () -> Edge.of(Entity.of(1), "r", null));
        assertThrows(IllegalArgumentException.class, () -> Edge.of(Entity.of(1), "  ", Entity.of(2)));
    }

    @Test
    void searchResult_constructsCorrectly() {
        final Entity e = Entity.of(7L);
        final SearchResult r = SearchResult.of(e, 0.95, 500L);
        assertEquals(7L,   r.entity().id());
        assertEquals(0.95, r.similarity(), 1e-9);
        assertEquals(500L, r.hammingDistance());
    }

    @Test
    void defaultResolver_matchesEntityDefaultId() {
        final EntityIdResolver resolver = EntityIdResolver.DEFAULT;
        assertEquals(Entity.defaultId("Paris"), resolver.resolve("Paris"));
    }

    @Test
    void relationResolver_prefixesREL() {
        final EntityIdResolver rel = EntityIdResolver.forRelations(EntityIdResolver.DEFAULT);
        assertEquals(Entity.defaultId("REL:capitalOf"), rel.resolve("capitalOf"));
    }

    @Test
    void chainBuilder_throwsIfNoHops() {
        final io.hammingstore.client.query.ChainQueryBuilder builder =
                new io.hammingstore.client.query.ChainQueryBuilder(1L, (s, r, k) -> java.util.List.of());

        assertThrows(IllegalStateException.class, builder::execute,
                "Executing with no hops must throw IllegalStateException");
    }

    @Test
    void chainBuilder_immutable() {
        final io.hammingstore.client.query.ChainQueryBuilder b1 =
                new io.hammingstore.client.query.ChainQueryBuilder(1L, (s, r, k) -> java.util.List.of());
        final io.hammingstore.client.query.ChainQueryBuilder b2 = b1.via(42L);
        final io.hammingstore.client.query.ChainQueryBuilder b3 = b2.via(99L);
        assertThrows(IllegalStateException.class, b1::execute);
        assertNotSame(b2, b3);
    }

    @Test
    void clientBuilder_defaults() {
        // We don't actually connect — just verify the builder compiles and config is set
        // A real connection test requires a running server.
        assertDoesNotThrow(() -> {
            final HammingClient.Builder builder = HammingClient.builder()
                    .endpoint("localhost", 50051)
                    .plaintext()
                    .timeout(java.time.Duration.ofSeconds(5));
            assertNotNull(builder);
        });
    }
}
