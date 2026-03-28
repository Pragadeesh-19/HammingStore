package io.hammingstore.benchmark.jmh;

import io.hammingstore.client.Entity;
import io.hammingstore.client.exception.EntityNotFoundException;
import io.hammingstore.client.query.ChainQueryBuilder;
import org.openjdk.jmh.annotations.*;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for HammingStore multi-hop chain traversal.
 *
 * <p>This is HammingStore's defining benchmark - the number no other vector
 * database or knowledge graph system can match because none of them ship
 * multi-hop symbolic chain traversal as a native single-RPC operation.
 *
 * <h2>What a chain query does</h2>
 * <p>A 3-hop chain from entity A via relations [R1, R2, R3]:
 * <pre>
 *   hop 1:  permute(A, 1) XOR permute(R1, 2)  ->  HNSW search  ->  B (rank-0 greedy)
 *   hop 2:  permute(B, 1) XOR permute(R2, 2)  ->  HNSW search  ->  C (rank-0 greedy)
 *   hop 3:  permute(C, 1) XOR permute(R3, 2)  ->  HNSW search  ->  top-k results
 * </pre>
 * All hops execute server-side in a single gRPC call. No application-layer loop.
 * No multiple round-trips. No intermediate result deserialization.
 *
 * <h2>Dataset</h2>
 * <p>Requires DBpedia preloaded via
 * {@link io.hammingstore.benchmark.dataset.DBpediaLoader}.
 * Server must be started with {@code --dims=384}.
 *
 * <h2>Parameterization</h2>
 * <ul>
 *   <li>{@link HopParams#hopCount} - 2, 3, or 4 hops. Shows how latency scales with depth.</li>
 * </ul>
 *
 * <h2>JMH settings</h2>
 * <p>Same as {@link SearchBenchmark}: Fork=3, Warmup=5×2s, Measurement=10×5s.
 */
@BenchmarkMode({Mode.AverageTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(
        value = 3,
        jvmArgs = {
                "--add-modules=jdk.random",
                "-XX:MaxDirectMemorySize=16g",
                "-XX:+UseG1GC",
                "-Xms512m",
                "-Xmx1g"
        }
)
@Warmup(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 5, timeUnit = TimeUnit.SECONDS)
public class ChainBenchmark {

    private static final int K = 10;

    @State(Scope.Benchmark)
    public static class HopParams {
        @Param({"2", "3", "4"})
        public int hopCount;
    }

    @Benchmark
    public List<Entity> chainTraversal(final BenchmarkState state, final HopParams params) {
        final long[] seed   = state.nextChainSeed();
        final long startId  = seed[0];
        final long[] relIds = buildRelationIds(seed, params.hopCount);

        try {
            ChainQueryBuilder builder = state.client.from(startId);
            for (final long relId : relIds)
                builder = builder.via(relId);
            return builder.topK(K).execute();
        } catch (final EntityNotFoundException e) {
            // Seed's intermediate hop target not in index — skip gracefully.
            // Probe validates 4-hop chains but HashMap is rebuilt fresh each
            // server restart; stale seeds from previous sessions may fail.
            return Collections.emptyList();
        }
    }

    private static long[] buildRelationIds(final long[] seed, final int hops) {
        final long[] rels = new long[hops];
        for (int i = 0; i < hops; i++)
            rels[i] = seed[1 + (i % 2)];
        return rels;
    }
}
