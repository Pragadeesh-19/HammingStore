package io.hammingstore.benchmark.jmh;

import io.hammingstore.client.Entity;
import io.hammingstore.client.HammingClient;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.*;

public final class ChainRecallEvaluator {

    private static final int    TOP_K     = 10;
    private static final int    EVAL_SIZE = 200;

    private static final Path CACHE_DIR = Paths.get(
            System.getProperty("user.home"), ".hammingstore-benchmark");

    public static void main(final String[] args) throws Exception {
        final String host = args.length > 0 ? args[0] : "localhost";
        final int    port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;

        System.out.printf("[RecallEval] Connecting to %s:%d%n", host, port);

        final Path edgeFile = CACHE_DIR.resolve("sparql_stored_edges.tsv");
        if (!Files.exists(edgeFile)) {
            System.err.println("ERROR: sparql_stored_edges.tsv not found.");
            System.err.println("Run WikidataEdgeStorer first (it generates this file).");
            System.err.println("If you ran an older version, delete the old edges from the");
            System.err.println("server and re-run EdgeStorer to regenerate the file.");
            System.exit(1);
        }

        System.out.printf("[RecallEval] Loading stored edges from %s%n", edgeFile);
        final List<Triple> allEdges = new ArrayList<>();
        for (final String line : Files.readAllLines(edgeFile, StandardCharsets.UTF_8)) {
            final String[] parts = line.split("\t");
            if (parts.length != 3) continue;
            try {
                allEdges.add(new Triple(
                        Long.parseLong(parts[0]),
                        parts[1],
                        Long.parseLong(parts[2])));
            } catch (final NumberFormatException ignored) {}
        }
        System.out.printf("[RecallEval] %,d stored edges loaded%n", allEdges.size());

        // Sample EVAL_SIZE edges evenly
        Collections.shuffle(allEdges, new java.util.Random(42));
        final List<Triple> sample = allEdges.subList(0, Math.min(EVAL_SIZE, allEdges.size()));
        System.out.printf("[RecallEval] Evaluating %,d sampled edges%n", sample.size());

        try (final HammingClient client = HammingClient.builder()
                .endpoint(host, port).plaintext()
                .timeout(Duration.ofMinutes(30)).build()) {

            if (!client.ping()) {
                System.err.println("ERROR: Server not reachable."); System.exit(1);
            }

            int hit1 = 0, hit10 = 0, skipped = 0, evaluated = 0;

            for (final Triple t : sample) {
                final long relId = Entity.defaultId("REL:" + t.property);
                try {
                    final List<Entity> results =
                            client.from(t.startQid).via(relId).topK(TOP_K).execute();

                    evaluated++;
                    boolean foundAt1 = false, foundAt10 = false;
                    for (int i = 0; i < results.size(); i++) {
                        // Decode: compositeId ^ subjectId ^ relationId = objectId
                        final long decoded = results.get(i).id() ^ t.startQid ^ relId;
                        if (results.get(0).id() == t.expectedQid) {
                            if (i == 0) foundAt1 = true;
                            foundAt10 = true;
                            break;
                        }
                    }
                    if (foundAt1)  hit1++;
                    if (foundAt10) hit10++;

                    if (evaluated <= 20) {
                        final long firstDecoded = results.isEmpty() ? -1L : results.get(0).id();;
                        System.out.printf("[RecallEval]   Q%d -[%s]-> Q%d : top1=%s top10=%s (got Q%s)%n",
                                t.startQid, t.property, t.expectedQid,
                                foundAt1  ? "HIT " : "MISS",
                                foundAt10 ? "HIT " : "MISS",
                                firstDecoded == -1L ? "none" : String.valueOf(firstDecoded));
                    }

                } catch (final Exception e) {
                    skipped++;
                }
            }

            System.out.println();
            System.out.println("=== Chain Recall Results (on STORED edges) ===");
            System.out.printf("Stored edges sampled : %,d%n", sample.size());
            System.out.printf("Evaluated            : %,d%n", evaluated);
            System.out.printf("Skipped (error)      : %,d%n", skipped);
            System.out.println();

            if (evaluated > 0) {
                final double r1  = (double) hit1  / evaluated;
                final double r10 = (double) hit10 / evaluated;
                System.out.printf("Recall@1  : %.4f  (%.1f%%)%n", r1,  r1  * 100);
                System.out.printf("Recall@10 : %.4f  (%.1f%%)%n", r10, r10 * 100);
                System.out.println();
                System.out.println("Interpretation (on stored edges — this is correctness, not coverage):");
                System.out.println("  ~100%  — VSA decode is correct, storeTypedEdge fix is working");
                System.out.println("  > 70%  — Working well, some HNSW approximation noise");
                System.out.println("  < 50%  — Decode bug still present or old broken edges dominating");
            }
        }
    }

    private record Triple(long startQid, String property, long expectedQid) {}
}
