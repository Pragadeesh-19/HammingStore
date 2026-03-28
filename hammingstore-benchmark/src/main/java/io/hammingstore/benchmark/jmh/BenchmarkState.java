package io.hammingstore.benchmark.jmh;

import io.hammingstore.client.Entity;
import io.hammingstore.client.HammingClient;
import io.hammingstore.client.ServerStatus;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.SplittableRandom;

/**
 * JMH shared state for all HammingStore benchmarks.
 *
 * <p>This state class is shared across all benchmark methods at
 * {@link Scope#Benchmark} scope - one instance per JVM fork, one
 * {@link #setup} call per trial. It holds:
 * <ul>
 *   <li>A connected {@link HammingClient}</li>
 *   <li>Pre-sampled query arrays for the search benchmark</li>
 *   <li>Pre-sampled chain query seeds for the chain benchmark</li>
 * </ul>
 *
 * <h2>Pre-loaded data assumption</h2>
 * <p>This state does NOT load data - it assumes a running HammingStore server
 * already has the dataset loaded via {@link DBpediaLoader} or
 * {@link io.hammingstore.benchmark.dataset.SiftLoader}. This is the correct
 * production benchmark pattern: measure the query path, not the load path.
 *
 * <h2>Query rotation</h2>
 * <p>Search and chain queries are drawn from a fixed pre-sampled pool of
 * {@value #QUERY_POOL_SIZE} queries. The {@code @Benchmark} methods use an
 * atomic index to rotate through the pool, ensuring the JVM does not optimize
 * away repeated identical queries and that the benchmark exercises diverse
 * graph traversal paths.
 */
@State(Scope.Benchmark)
public class BenchmarkState {

    public static final int QUERY_POOL_SIZE = 1_000;

    private static final String DEFAULT_HOST = System.getProperty("benchmark.host","localhost");

    private static final int DEFAULT_PORT =
            Integer.parseInt(System.getProperty("benchmark.port","50051"));

    private static final Path CACHE_DIR =
            Paths.get(System.getProperty("user.home"),".hammingstore-benchmark");

    @Param({"localhost"})
    public String serverHost = DEFAULT_HOST;

    @Param({"50051"})
    public String serverPort = String.valueOf(DEFAULT_PORT);

    public HammingClient client;

    public float[][] searchQueryPool;

    public long[][] chainQueryPool;

    public volatile int searchIndex = 0;

    public volatile int chainIndex = 0;

    @Setup(Level.Trial)
    public void setup() throws IOException, InterruptedException {
        final int port = Integer.parseInt(serverPort);

        System.out.printf("%n[BenchmarkState] Connecting to HammingStore at %s:%d%n",
                serverHost, port);

        client = HammingClient.builder()
                .endpoint(serverHost, port)
                .plaintext()
                .timeout(Duration.ofHours(1))
                .poolSize(Runtime.getRuntime().availableProcessors())
                .build();

        if (!client.ping()) {
            throw new IllegalStateException(
                    "HammingStore server is not reachable at " + serverHost + ":" + port + ".\n"
                            + "Load data first:\n"
                            + "  java -cp benchmarks.jar "
                            + "io.hammingstore.benchmark.dataset.DBpediaLoader 1000000\n"
                            + "  java -cp benchmarks.jar "
                            + "io.hammingstore.benchmark.dataset.SiftLoader");
        }

        final ServerStatus status = client.status();
        final int serverDims = status.inputDimensions();
        System.out.printf("[BenchmarkState] Server dims=%d%n", serverDims);

        searchQueryPool = buildSearchQueryPool(serverDims);
        chainQueryPool  = loadChainQueryPool();

        System.out.printf("[BenchmarkState] Ready. Search pool: %d  Chain pool: %d%n",
                searchQueryPool.length, chainQueryPool.length);
    }

    @TearDown(Level.Trial)
    public void teardown() {
        if (client != null) {
            client.close();
            client = null;
        }
    }

    private static float[][] buildSearchQueryPool(final int dims) {
        final SplittableRandom rng  = new SplittableRandom(1234L);
        final float[][] pool = new float[QUERY_POOL_SIZE][dims];

        for (int q = 0; q < QUERY_POOL_SIZE; q++) {
            double sumSq = 0.0;
            for (int d = 0; d < dims; d++) {
                final double u1 = rng.nextDouble();
                final double u2 = rng.nextDouble();
                final double z = Math.sqrt(-2.0 * Math.log(u1 == 0.0 ? 1e-300 : u1))
                        * Math.cos(2.0 * Math.PI * u2);
                pool[q][d] = (float) z;
                sumSq += z * z;
            }
            final float norm = (float) Math.sqrt(sumSq);
            if (norm > 0f) {
                for (int d = 0; d < dims; d++) pool[q][d] /= norm;
            }
        }
        return pool;
    }

    private static long[][] loadChainQueryPool() throws IOException {
        final Path seedFile = CACHE_DIR.resolve("dbpedia_chain_seeds.txt");

        if (Files.exists(seedFile)) {
            final List<String> lines = Files.readAllLines(seedFile);
            final long[][] pool = new long[Math.min(lines.size(), QUERY_POOL_SIZE)][];
            int count = 0;
            for (final String line : lines) {
                if (count >= QUERY_POOL_SIZE) break;
                final String[] parts = line.split(",");
                if (parts.length < 3) continue;
                pool[count++] = new long[]{
                        Long.parseLong(parts[0].trim()),
                        Long.parseLong(parts[1].trim()),
                        Long.parseLong(parts[2].trim())
                };
            }
            if (count > 0) {
                System.out.printf("[BenchmarkState] Loaded %d chain seeds from %s%n",
                        count, seedFile);
                return count == pool.length ? pool : java.util.Arrays.copyOf(pool, count);
            }
        }

        System.out.println("[BenchmarkState] No chain seed cache found. Generating synthetic seeds.");
        System.out.println("[BenchmarkState] For real seeds, run DBpediaLoader first.");
        return buildSyntheticChainSeeds();
    }

    private static long[][] buildSyntheticChainSeeds() {
        final long[] knownQids = {
                31L, 142L, 183L, 17L, 30L, 145L, 38L, 39L, 40L, 55L,
                16L, 408L, 96L, 45L, 20L, 668L, 159L, 148L, 28L, 36L
        };

        final String[] relations = {
                "P17", "P131", "P31", "P27", "P279", "P361", "P106", "P19", "P20"
        };

        final long[][] pool = new long[QUERY_POOL_SIZE][3];
        final SplittableRandom rng = new SplittableRandom(99L);

        for (int i = 0; i < QUERY_POOL_SIZE; i++) {
            final long startId = knownQids[rng.nextInt(knownQids.length)];
            final long rel1 = Entity.defaultId(
                    "REL:" + relations[rng.nextInt(relations.length)]);
            final long rel2 = Entity.defaultId(
                    "REL:" + relations[rng.nextInt(relations.length)]);
            pool[i] = new long[]{startId, rel1, rel2};
        }
        return pool;
    }

    public float[] nextSearchQuery() {
        final int idx = searchIndex;
        searchIndex = (idx + 1) % searchQueryPool.length;
        return searchQueryPool[idx];
    }

    public long[] nextChainSeed() {
        final int idx = chainIndex;
        chainIndex = (idx + 1) % chainQueryPool.length;
        return chainQueryPool[idx];
    }
}
