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


public final class HammingServer {

    private static final Logger LOG = Logger.getLogger(HammingServer.class.getName());

    private final Server server;
    private final VectorGraphRepository repository;

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
            LOG.info("Shutdown signal received — checkpointing...");
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

    public static void main(final String[] args) throws IOException, InterruptedException {

        int port = 50051;
        long maxVectors = 1_000_000L;
        int dims = ProjectionConfig.DIMS_MINILM;   // 384
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
