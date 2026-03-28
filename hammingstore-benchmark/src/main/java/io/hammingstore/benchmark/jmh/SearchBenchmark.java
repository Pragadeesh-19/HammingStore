package io.hammingstore.benchmark.jmh;

import io.hammingstore.client.Entity;
import io.hammingstore.client.SearchResult;
import org.openjdk.jmh.annotations.*;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for HammingStore ANN (Approximate Nearest Neighbour) search.
 *
 * <h2>What this measures</h2>
 * <p>Wall-clock latency of a single {@code searchFloat} query call from the
 * Java client to HammingStore and back, including:
 * <ul>
 *   <li>gRPC serialization and loopback network</li>
 *   <li>Random projection encoding (float → binary)</li>
 *   <li>HNSW graph search (efSearch = max(k, 64))</li>
 *   <li>Result deserialization</li>
 * </ul>
 *
 * <h2>Dataset</h2>
 * <p>Requires SIFT-1M pre-loaded via
 * {@link io.hammingstore.benchmark.dataset.SiftLoader}.
 * Server must be started with {@code --dims=128}.
 *
 * <h2>Recall@10 evaluation</h2>
 * <p>Run {@link RecallEvaluator} separately after loading SIFT-1M.
 * JMH measures latency; Recall@10 is computed in a dedicated pass.
 *
 * <h2>JMH settings</h2>
 * <ul>
 *   <li>Fork = 3: eliminates JIT variance across JVM instances</li>
 *   <li>Warmup 5 × 2s: ensures JIT is fully compiled before measurement</li>
 *   <li>Measurement 10 × 5s: 50s of stable data per fork</li>
 * </ul>
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
public class SearchBenchmark {

    private static final int K = 10;

    @Benchmark
    public List<SearchResult> searchTopK(final BenchmarkState state) {
        return state.client
                .search(state.nextSearchQuery())
                .topK(K)
                .execute();
    }

    @Benchmark
    public List<SearchResult> searchTop1(final BenchmarkState state) {
        return state.client
                .search(state.nextSearchQuery())
                .topK(1)
                .execute();
    }
}
