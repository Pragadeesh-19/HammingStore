package io.hammingstore.benchmark.dataset;

import io.hammingstore.client.Entity;
import io.hammingstore.client.HammingClient;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class WikidataChainProbe {

    private static final Path CACHE_DIR =
            Paths.get(System.getProperty("user.home"), ".hammingstore-benchmark");

    private static final String SEED_FILE = "dbpedia_chain_seeds.txt";

    private static final String[] INDEXED_PROPERTIES = {
            "P17", "P131", "P31", "P27", "P279",
            "P361", "P150", "P106", "P19", "P20"
    };

    public static void main(final String[] args) throws Exception {
        final String host        = args.length > 0 ? args[0] : "localhost";
        final int    port        = args.length > 1 ? Integer.parseInt(args[1]) : 50051;
        final int    targetSeeds = 1_000;

        // Pre-compute relation IDs once
        final long[] relIds = new long[INDEXED_PROPERTIES.length];
        for (int i = 0; i < INDEXED_PROPERTIES.length; i++)
            relIds[i] = Entity.defaultId("REL:" + INDEXED_PROPERTIES[i]);

        // Load ALL stored QIDs from label cache — covers high-QID entities too
        final Path labelCache = CACHE_DIR.resolve("sparql_labels.tsv");
        final List<Long> storedQids = new ArrayList<>();
        for (final String line : Files.readAllLines(labelCache)) {
            final int tab = line.indexOf('\t');
            if (tab > 0) {
                try { storedQids.add(Long.parseLong(line.substring(0, tab))); }
                catch (final NumberFormatException ignored) {}
            }
        }
        Collections.shuffle(storedQids, new java.util.Random(42));

        System.out.printf("[Probe] %s:%d%n", host, port);
        System.out.printf("[Probe] Probing %,d stored QIDs from label cache%n", storedQids.size());

        try (final HammingClient client = HammingClient.builder()
                .endpoint(host, port)
                .plaintext()
                .timeout(Duration.ofMinutes(30))
                .build()) {

            if (!client.ping()) {
                System.err.println("[Probe] ERROR: Server not reachable.");
                System.exit(1);
            }

            final List<long[]> validSeeds = new ArrayList<>(targetSeeds);
            int probed = 0;

            System.out.println("[Probe] Probing entity IDs...");

            for (final long qid : storedQids) {
                if (validSeeds.size() >= targetSeeds) break;
                probed++;

                boolean found = false;
                for (int r1 = 0; r1 < relIds.length && !found; r1++) {
                    for (int r2 = r1 + 1; r2 < relIds.length && !found; r2++) {
                        try {
                            // Validate full 4-hop chain: [r1, r2, r1, r2]
                            client.from(qid)
                                    .via(relIds[r1])
                                    .via(relIds[r2])
                                    .via(relIds[r1])
                                    .via(relIds[r2])
                                    .topK(1)
                                    .execute();

                            validSeeds.add(new long[]{qid, relIds[r1], relIds[r2]});
                            found = true;

                        } catch (final Exception e) {
                            // EntityNotFoundException or empty result — skip
                        }
                    }
                }

                if (probed % 1_000 == 0)
                    System.out.printf("[Probe]   probed=%,d  valid seeds=%,d%n",
                            probed, validSeeds.size());
            }

            System.out.printf("[Probe] Found %,d valid seeds from %,d probed QIDs%n",
                    validSeeds.size(), probed);

            if (validSeeds.isEmpty()) {
                System.err.println("[Probe] ERROR: No valid seeds found.");
                System.exit(1);
            }

            Files.createDirectories(CACHE_DIR);
            final Path seedFile = CACHE_DIR.resolve(SEED_FILE);
            final List<String> lines = new ArrayList<>(validSeeds.size());
            for (final long[] seed : validSeeds)
                lines.add(seed[0] + "," + seed[1] + "," + seed[2]);
            Files.write(seedFile, lines);

            System.out.printf("[Probe] Wrote %,d seeds to %s%n", lines.size(), seedFile);
            System.out.println("[Probe] Run ChainBenchmark now — seeds will be loaded automatically.");
        }
    }
}
