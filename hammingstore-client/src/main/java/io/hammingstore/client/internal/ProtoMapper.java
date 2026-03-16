package io.hammingstore.client.internal;

import com.google.protobuf.ByteString;
import io.hammingstore.client.Entity;
import io.hammingstore.client.SearchResult;
import io.hammingstore.grpc.proto.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public final class ProtoMapper {

    private ProtoMapper() {}

    public static StoreFloatRequest toStoreFloatRequest(final long entityId, final float[] embedding) {
        return StoreFloatRequest.newBuilder()
                .setEntityId(entityId)
                .setFloatVec(floatsToBytes(embedding))
                .build();
    }

    public static StoreTypedEdgeRequest toStoreTypedEdgeRequest(final long subjectId,
                                                                final long relationId,
                                                                final long objectId) {
        return StoreTypedEdgeRequest.newBuilder()
                .setSubjectId(subjectId)
                .setRelationId(relationId)
                .setObjectId(objectId)
                .build();
    }

    public static RetractRequest toRetractRequest(final long entityId) {
        return RetractRequest.newBuilder()
                .setEntityId(entityId)
                .build();
    }

    public static QueryHopRequest toQueryHopRequest(final long entityId,
                                                    final long relationId,
                                                    final int k) {
        return QueryHopRequest.newBuilder()
                .setEntityId(entityId)
                .setRelationId(relationId)
                .setK(k)
                .build();
    }

    public static QueryChainRequest toQueryChainRequest(final long startEntityId,
                                                        final long[] relationIds,
                                                        final int k) {
        final QueryChainRequest.Builder builder = QueryChainRequest.newBuilder()
                .setStartEntityId(startEntityId)
                .setK(k);
        for (final long rid : relationIds) {
            builder.addRelationIds(rid);
        }
        return builder.build();
    }

    public static QueryAnalogyRequest toQueryAnalogyRequest(final long subjectAId,
                                                            final long objectAId,
                                                            final long subjectBId,
                                                            final int k) {
        return QueryAnalogyRequest.newBuilder()
                .setSubjectAId(subjectAId)
                .setObjectAId(objectAId)
                .setSubjectBId(subjectBId)
                .setK(k)
                .build();
    }

    public static QuerySetRequest toQuerySetRequest(final long[] entityIds, final int k) {
        final QuerySetRequest.Builder builder = QuerySetRequest.newBuilder().setK(k);
        for (final long id : entityIds) {
            builder.addEntityIds(id);
        }
        return builder.build();
    }

    public static QueryRelationRequest toQueryRelationRequest(final long relationId,
                                                              final long objectId,
                                                              final int  k) {
        return QueryRelationRequest.newBuilder()
                .setRelationId(relationId)
                .setObjectId(objectId)
                .setK(k)
                .build();
    }

    public static SearchFloatRequest toSearchFloatRequest(final float[] embedding, final int k) {
        return SearchFloatRequest.newBuilder()
                .setFloatVec(floatsToBytes(embedding))
                .setK(k)
                .build();
    }

    public static CheckpointRequest toCheckpointRequest(final boolean durable) {
        return CheckpointRequest.newBuilder().setDurable(durable).build();
    }

    public static List<Entity> toEntities(
            final Iterable<io.hammingstore.grpc.proto.SearchResult> protoResults) {
        final List<Entity> results = new ArrayList<>();
        for (final io.hammingstore.grpc.proto.SearchResult r : protoResults) {
            results.add(Entity.of(r.getEntityId()));
        }
        return results;
    }

    public static List<io.hammingstore.client.SearchResult> toSearchResults(
            final Iterable<io.hammingstore.grpc.proto.SearchResult> protoResults) {
        final List<SearchResult> results = new ArrayList<>();
        for (final io.hammingstore.grpc.proto.SearchResult r : protoResults) {
            results.add(io.hammingstore.client.SearchResult.of(
                    Entity.of(r.getEntityId()),
                    r.getSimilarity(),
                    r.getDistance()
            ));
        }
        return results;
    }

    private static ByteString floatsToBytes(final float[] floats) {
        final ByteBuffer buf = ByteBuffer
                .allocate(floats.length * Float.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);

        for (final float f : floats) buf.putFloat(f);

        assert buf.position() == buf.capacity() : "Buffer not fully written";
        return ByteString.copyFrom(buf.array());
    }
}
