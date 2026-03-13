package io.hammingstore.grpc;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import io.hammingstore.graph.VectorGraphRepository;
import io.hammingstore.grpc.proto.*;
import io.hammingstore.hnsw.HNSWIndex;
import io.hammingstore.memory.BinaryVector;
import io.hammingstore.vsa.SymbolicReasoner;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * gRPC service implementation for HammingStore.
 *
 * <p>Each RPC method follows the same pattern:
 * <ol>
 *   <li>Validate inputs; report {@code INVALID_ARGUMENT} on failure.</li>
 *   <li>Delegate to {@link VectorGraphRepository} or {@link SymbolicReasoner}.</li>
 *   <li>Map the result to a proto response and complete the observer.</li>
 *   <li>Catch any unexpected exception and report {@code INTERNAL}.</li>
 * </ol>
 *
 * <p>Thread safety is entirely delegated to {@link VectorGraphRepository}, which is
 * internally guarded by a {@link java.util.concurrent.locks.StampedLock}.
 */
public final class HammingGrpcService extends HammingStoreGrpc.HammingStoreImplBase {

    private static final int VECTOR_BYTES = (int) BinaryVector.VECTOR_BYTES;

    private final VectorGraphRepository repository;
    private final SymbolicReasoner reasoner;

    public HammingGrpcService(
            final VectorGraphRepository repository,
            final SymbolicReasoner reasoner) {
        this.repository = Objects.requireNonNull(repository, "repository");
        this.reasoner   = Objects.requireNonNull(reasoner,   "reasoner");
    }

    @Override
    public void storeBinary(
            final StoreBinaryRequest request,
            final StreamObserver<StoreBinaryResponse> responseObserver) {
        try {
            validateVectorBytes(request.getBinaryVec(), "binary_vec");

            try (Arena arena = Arena.ofConfined()){
                final MemorySegment vec = arena.allocate(VECTOR_BYTES, Long.BYTES);
                copyBytesInto(request.getBinaryVec(), vec);
                repository.storeBinary(request.getEntityId(), vec);
            }

            respond(responseObserver, StoreBinaryResponse.newBuilder().setSuccess(true).build());
        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void storeFloat(
            final StoreFloatRequest request,
            final StreamObserver<StoreFloatResponse> responseObserver) {
        try {
            final float[] embedding = bytesToFloats(request.getFloatVec());
            repository.store(request.getEntityId(), embedding);
            respond(responseObserver, StoreFloatResponse.newBuilder().setSuccess(true).build());
        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void bindEdge(
            final BindEdgeRequest request,
            final StreamObserver<BindEdgeResponse> responseObserver) {
        try {
            repository.bindRelationalEdge(
                    request.getSubjectId(),
                    request.getObjectId(),
                    request.getRelationshipId());

            respond(responseObserver, BindEdgeResponse.newBuilder().setSuccess(true).build());

        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void retract(
            final RetractRequest request,
            final StreamObserver<RetractResponse> responseObserver) {
        try {
            repository.retract(request.getEntityId());
            respond(responseObserver, RetractResponse.newBuilder().setSuccess(true).build());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void checkpoint(
            final CheckpointRequest request,
            final StreamObserver<CheckpointResponse> responseObserver) {
        try {
            repository.checkpoint(request.getDurable());
            respond(responseObserver, CheckpointResponse.newBuilder().setSuccess(true).build());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void queryAnalogy(
            final QueryAnalogyRequest request,
            final StreamObserver<QueryAnalogyResponse> responseObserver) {
        try {
            final HNSWIndex.SearchResults results = reasoner.queryAnalogy(
                    request.getSubjectAId(),
                    request.getObjectAId(),
                    request.getSubjectBId(),
                    request.getK());

            respond(responseObserver, QueryAnalogyResponse.newBuilder()
                    .addAllResults(toProto(results)).build());

        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void querySet(
            final QuerySetRequest request,
            final StreamObserver<QuerySetResponse> responseObserver) {
        try {
            final List<Long> idList = request.getEntityIdsList();
            final long[] entityIds  = new long[idList.size()];
            for (int i = 0; i < entityIds.length; i++) entityIds[i] = idList.get(i);

            final HNSWIndex.SearchResults results = reasoner.querySet(entityIds, request.getK());

            respond(responseObserver, QuerySetResponse.newBuilder()
                    .addAllResults(toProto(results)).build());

        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void queryRelation(
            final QueryRelationRequest request,
            final StreamObserver<QueryRelationResponse> responseObserver) {
        try {
            final HNSWIndex.SearchResults results = reasoner.queryRelation(
                    request.getRelationId(),
                    request.getObjectId(),
                    request.getK());

            respond(responseObserver, QueryRelationResponse.newBuilder()
                    .addAllResults(toProto(results)).build());

        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void searchBinary(
            final SearchBinaryRequest request,
            final StreamObserver<SearchBinaryResponse> responseObserver) {
        try {
            validateVectorBytes(request.getBinaryVec(), "binary_vec");

            final HNSWIndex.SearchResults results;
            try (Arena arena = Arena.ofConfined()) {
                final MemorySegment vec = arena.allocate(VECTOR_BYTES, Long.BYTES);
                copyBytesInto(request.getBinaryVec(), vec);
                results = repository.searchHNSWBinary(vec, request.getK());
            }

            respond(responseObserver, SearchBinaryResponse.newBuilder()
                    .addAllResults(toProto(results)).build());

        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void searchFloat(
            final SearchFloatRequest request,
            final StreamObserver<SearchFloatResponse> responseObserver) {
        try {
            final float[] embedding = bytesToFloats(request.getFloatVec());
            final HNSWIndex.SearchResults results =
                    repository.searchHNSW(embedding, request.getK());
            respond(responseObserver, SearchFloatResponse.newBuilder()
                    .addAllResults(toProto(results)).build());
        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void getProjectionConfig(
            final GetProjectionConfigRequest request,
            final StreamObserver<ProjectionConfig> responseObserver) {
        try {
            final io.hammingstore.vsa.ProjectionConfig cfg = repository.projectionConfig();

            respond(responseObserver, ProjectionConfig.newBuilder()
                    .setSeed(cfg.seed())
                    .setInputDimensions(cfg.inputDimensions())
                    .setOutputBits(cfg.outputBits())
                    .build());

        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void storeTypedEdge(
            final StoreTypedEdgeRequest request,
            final StreamObserver<StoreTypedEdgeResponse> responseObserver) {
        try {
            repository.storeTypedEdge(
                    request.getSubjectId(),
                    request.getRelationId(),
                    request.getObjectId());
            respond(responseObserver,
                    StoreTypedEdgeResponse.newBuilder().setSuccess(true).build());
        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void queryHop(
            final QueryHopRequest request,
            final StreamObserver<QueryHopResponse> responseObserver) {
        try {
            final HNSWIndex.SearchResults results =
                    reasoner.queryHop(request.getEntityId(), request.getRelationId(), request.getK());
            respond(responseObserver, QueryHopResponse.newBuilder()
                    .addAllResults(toProto(results)).build());
        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void queryChain(
            final QueryChainRequest request,
            final StreamObserver<QueryChainResponse> responseObserver) {
        try {
            final long[] relationIds = request.getRelationIdsList()
                    .stream().mapToLong(Long::longValue).toArray();
            final HNSWIndex.SearchResults results =
                    reasoner.queryChain(request.getStartEntityId(), relationIds, request.getK());
            respond(responseObserver, QueryChainResponse.newBuilder()
                    .addAllResults(toProto(results)).build());
        } catch (IllegalArgumentException e) {
            responseObserver.onError(Status.INVALID_ARGUMENT
                    .withDescription(e.getMessage()).asRuntimeException());
        } catch (Exception e) {
            responseObserver.onError(Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    /**
     * Converts {@link HNSWIndex.SearchResults} to a list of proto {@link SearchResult} messages.
     */
    private static List<SearchResult> toProto(final HNSWIndex.SearchResults results) {
        final List<SearchResult> list = new ArrayList<>(results.count());
        for (int i = 0; i < results.count(); i++) {
            list.add(SearchResult.newBuilder()
                    .setEntityId(results.entityId(i))
                    .setDistance(results.distance(i))
                    .setSimilarity(results.similarity(i))
                    .build());
        }
        return list;
    }

    /**
     * Asserts that {@code bytes} is exactly {@link #VECTOR_BYTES} bytes long.
     *
     * @throws IllegalArgumentException if the size does not match
     */
    private static void validateVectorBytes(final ByteString bytes, final String field) {
        if (bytes.size() != VECTOR_BYTES) {
            throw new IllegalArgumentException(
                    field + " must be exactly " + VECTOR_BYTES + " bytes, got " + bytes.size());
        }
    }

    /**
     * Copies the bytes of {@code src} into {@code dest} starting at offset 0.
     * Allocates one intermediate {@code byte[]} via {@link ByteString#toByteArray()}.
     */
    private static void copyBytesInto(final ByteString src, final MemorySegment dest) {
        final byte[] tmp = src.toByteArray();
        MemorySegment.copy(MemorySegment.ofArray(tmp), 0L, dest, 0L, tmp.length);
    }

    /** Completes a unary RPC by sending {@code response} and calling {@code onCompleted}. */
    private static <T> void respond(
            final StreamObserver<T> observer, final T response) {
        observer.onNext(response);
        observer.onCompleted();
    }

    /**
     * Decodes {@code bytes} as a little-endian IEEE-754 float array.
     *
     * @throws IllegalArgumentException if {@code bytes.size()} is not divisible by 4
     */
    private static float[] bytesToFloats(final ByteString bytes) {
        final byte[] raw = bytes.toByteArray();
        if (raw.length % Float.BYTES != 0) {
            throw new IllegalArgumentException(
                    "float_vec byte length must be divisible by 4, got " + raw.length);
        }
        final int count = raw.length / Float.BYTES;
        final float[] result = new float[count];
        final ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) result[i] = buf.getFloat();
        return result;
    }
}
