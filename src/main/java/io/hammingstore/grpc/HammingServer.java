package io.hammingstore.grpc;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.hammingstore.graph.VectorGraphRepository;
import io.hammingstore.vsa.ProjectionConfig;
import io.hammingstore.vsa.SymbolicReasoner;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Entry point for the HammingStore gRPC server.
 *
 * <p>Constructs the service stack, wires the shutdown hook, and owns the server
 * lifecycle. The constructor blocks only until the server has bound its port —
 * callers must invoke {@link #awaitTermination()} to park the main thread.
 *
 * <h2>Shutdown</h2>
 * <p>A JVM shutdown hook checkpoints the repository ({@code durable=true}) and
 * drains in-flight RPCs with a 30-second grace period before closing the port.
 * If {@link #stop()} is called explicitly before JVM exit, the hook will attempt
 * to call it a second time. Both {@code server.shutdown()} (idempotent by gRPC
 * contract) and {@link VectorGraphRepository#close()} (idempotent for mmap close)
 * tolerate this safely.
 */
public final class HammingServer {

    private static final Logger LOG = Logger.getLogger(HammingServer.class.getName());

    private final Server server;
    private final VectorGraphRepository repository;

    /**
     * Starts the gRPC server on {@code port} backed by {@code repository}.
     *
     * @param port TCP port to listen on
     * @param repository the vector store; must be open and not yet closed
     * @throws IOException if the port cannot be bound
     */
    public HammingServer(final int port, final VectorGraphRepository repository) throws IOException {

        this.repository = repository;

        final SymbolicReasoner reasoner = new SymbolicReasoner(repository);
        final HammingGrpcService service = new HammingGrpcService(repository, reasoner);

        this.server = ServerBuilder
                .forPort(port)
                .addService(service)
                .build()
                .start();

        LOG.info("HammingStore started"
                + " | port="       + port
                + " | mode="       + (repository.isDiskBacked() ? "DISK" : "RAM")
                + " | dims="       + repository.projectionConfig().inputDimensions()
                + " | maxVectors=" + repository.hnswIndex().size());

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOG.info("Shutdown signal received - checkpointing...");
            try {
                repository.checkpoint(true);
            } catch (Exception e) {
                LOG.warning("Checkpoint on shutdown failed: " + e.getMessage());
            }
            try {
                stop();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            LOG.info("HammingStore stopped.");
        }, "hammingstore-shutdown"));
    }

    public void awaitTermination() throws InterruptedException {
        server.awaitTermination();
    }

    public void stop() throws InterruptedException {
        server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        repository.close();
    }

    /**
     * Starts a HammingStore server from the command line.
     *
     * <p>Recognised flags:
     * <pre>
     *   --port=&lt;int&gt;          TCP port (default: 50051)
     *   --max-vectors=&lt;long&gt;  Maximum vector capacity (default: 1 000 000)
     *   --dims=&lt;int&gt;          Embedding input dimensions (default: 384, MiniLM)
     *   --seed=&lt;long&gt;         Projection matrix PRNG seed (default: ProjectionConfig.DEFAULT_SEED)
     *   --data-dir=&lt;path&gt;     Directory for disk-backed persistence (RAM mode if omitted)
     * </pre>
     */
    public static void main(final String[] args) throws IOException, InterruptedException {

        int port = 50051;
        long maxVectors = 1_000_000L;
        int dims = ProjectionConfig.DIMS_MINILM;
        long seed = ProjectionConfig.DEFAULT_SEED;
        String dataDir = null;

        for (final String arg : args) {
            if      (arg.startsWith("--port="))         port       = Integer.parseInt(arg.substring(7));
            else if (arg.startsWith("--max-vectors="))  maxVectors = Long.parseLong(arg.substring(14));
            else if (arg.startsWith("--dims="))         dims       = Integer.parseInt(arg.substring(7));
            else if (arg.startsWith("--seed="))         seed       = Long.parseLong(arg.substring(7));
            else if (arg.startsWith("--data-dir="))     dataDir    = arg.substring(11);
        }

        final ProjectionConfig config = ProjectionConfig.of(seed, dims);

        final VectorGraphRepository repository;
        if (dataDir != null) {
            LOG.info("Opening disk-backed store at: " + dataDir);
            repository = VectorGraphRepository.openFromDisk(Path.of(dataDir), config, maxVectors);
        } else {
            LOG.info("Starting in RAM mode (data will not survive restart)");
            repository = new VectorGraphRepository(maxVectors, config);
        }

        new HammingServer(port, repository).awaitTermination();
    }
}
