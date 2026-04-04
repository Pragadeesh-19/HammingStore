package io.hammingstore.benchmark.dataset;

import io.hammingstore.client.Edge;
import io.hammingstore.client.Entity;
import io.hammingstore.client.HammingClient;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class EdgeReplayTool {

    private static final int BATCH_SIZE = 500;

    public static void main(final String[] args) throws Exception {
        final Path csvPath = Paths.get(args.length > 0
                ? args[0]
                : "C:\\Users\\anony\\Downloads\\full-oldb.LATEST\\relationships.csv");
        final String host    = args.length > 1 ? args[1] : "localhost";
        final int    port    = args.length > 2 ? Integer.parseInt(args[2]) : 50051;

        System.out.printf("[EdgeReplay] File  : %s (%.1f MB)%n",
                csvPath, Files.size(csvPath) / 1e6);
        System.out.printf("[EdgeReplay] Server: %s:%d%n", host, port);
        System.out.printf("[EdgeReplay] Batch : %d edges per call%n", BATCH_SIZE);
        System.out.println("[EdgeReplay] No encoding — entity vectors already on disk.");
        System.out.println();

        try (final HammingClient client = HammingClient.builder()
                .endpoint(host, port)
                .plaintext()
                .timeout(Duration.ofHours(2))
                .build()) {

            if (!client.ping()) {
                System.err.println("[EdgeReplay] ERROR: Server not reachable at "
                        + host + ":" + port);
                System.exit(1);
            }
            System.out.println("[EdgeReplay] Server reachable. Starting edge ingestion...");

            final AtomicLong stored  = new AtomicLong(0);
            final AtomicLong skipped = new AtomicLong(0);
            final AtomicLong lines   = new AtomicLong(0);
            final long startMs = System.currentTimeMillis();

            final List<Edge> batch = new ArrayList<>(BATCH_SIZE);

            try (final BufferedReader reader = new BufferedReader(
                    new InputStreamReader(
                            Files.newInputStream(csvPath), StandardCharsets.UTF_8),
                    1 << 20)) {

                // Parse header to find column indices
                final String header = reader.readLine();
                if (header == null) {
                    System.err.println("[EdgeReplay] ERROR: Empty CSV file.");
                    System.exit(1);
                }

                final String[] cols    = header.split(",");
                final int startIdx     = indexOf(cols, "node_id_start");
                final int endIdx       = indexOf(cols, "node_id_end");
                final int relTypeIdx   = indexOf(cols, "rel_type");

                if (startIdx < 0 || endIdx < 0 || relTypeIdx < 0) {
                    System.err.printf(
                            "[EdgeReplay] ERROR: Could not find required columns.%n"
                                    + "  node_id_start: col %d%n"
                                    + "  node_id_end:   col %d%n"
                                    + "  rel_type:      col %d%n"
                                    + "  Header: %s%n",
                            startIdx, endIdx, relTypeIdx, header);
                    System.exit(1);
                }

                System.out.printf("[EdgeReplay] Columns: start=%d end=%d rel_type=%d%n%n",
                        startIdx, endIdx, relTypeIdx);

                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isBlank()) continue;
                    lines.incrementAndGet();

                    final String[] fields = line.split(",", -1);
                    if (fields.length <= Math.max(startIdx, Math.max(endIdx, relTypeIdx)))
                        continue;

                    try {
                        final long   startId  = Long.parseLong(fields[startIdx].trim());
                        final long   endId    = Long.parseLong(fields[endIdx].trim());
                        final String relType  = fields[relTypeIdx].trim().toLowerCase();

                        final String relId = relNameFor(relType);
                        batch.add(new Edge(Entity.of(startId), relId, Entity.of(endId)));

                    } catch (final NumberFormatException ignored) {
                        skipped.incrementAndGet();
                        continue;
                    }

                    if (batch.size() >= BATCH_SIZE) {
                        flushBatch(client, batch, stored, skipped);
                        printProgress(stored.get(), skipped.get(),
                                lines.get(), startMs);
                    }
                }
            }

            // Flush remaining
            if (!batch.isEmpty()) {
                flushBatch(client, batch, stored, skipped);
            }

            final long elapsedSec = (System.currentTimeMillis() - startMs) / 1000;
            System.out.println();
            System.out.println("[EdgeReplay] === COMPLETE ===");
            System.out.printf("[EdgeReplay] Lines read : %,d%n",  lines.get());
            System.out.printf("[EdgeReplay] Stored     : %,d%n",  stored.get());
            System.out.printf("[EdgeReplay] Skipped    : %,d%n",  skipped.get());
            System.out.printf("[EdgeReplay] Time       : %dm %ds%n",
                    elapsedSec / 60, elapsedSec % 60);
            System.out.println();
            System.out.println("[EdgeReplay] NOW: trigger a checkpoint so edge_log.dat is committed.");
            System.out.println("[EdgeReplay] Call: POST http://localhost:8081/actuator/shutdown");
            System.out.println("[EdgeReplay] Or stop/restart the HammingStore server — it checkpoints on shutdown.");
            System.out.println("[EdgeReplay] After next restart you will see:");
            System.out.printf("[EdgeReplay]   Replaying %,d edges from edge_log.dat...%n",
                    stored.get());
            System.out.println("[EdgeReplay]   Edge replay complete. Chain traversal is fully operational.");
        }
    }

    private static void flushBatch(
            final HammingClient client,
            final List<Edge> batch,
            final AtomicLong stored,
            final AtomicLong skipped) {
        try {
            final var result = client.storeEdges(batch);
            stored.addAndGet(batch.size() - result.failureCount());
            skipped.addAndGet(result.failureCount());
        } catch (final Exception e) {
            // Entities not yet stored (e.g. relation vectors missing) — skip batch
            skipped.addAndGet(batch.size());
            System.err.printf("[EdgeReplay] Batch failed (%d edges): %s%n",
                    batch.size(), e.getMessage());
        }
        batch.clear();
    }

    private static void printProgress(
            final long stored, final long skipped,
            final long lines, final long startMs) {
        if ((stored + skipped) % 50_000 == 0) {
            final long elapsedSec = (System.currentTimeMillis() - startMs) / 1000;
            final double rate = elapsedSec > 0 ? (double) stored / elapsedSec : 0;
            System.out.printf("[EdgeReplay] %,7d stored  %,6d skipped  %,7d lines  %.0f/sec%n",
                    stored, skipped, lines, rate);
        }
    }

    private static String relNameFor(final String relType) {
        return switch (relType) {
            case "officer_of"         -> "officer_of";
            case "registered_address" -> "registered_address";
            case "intermediary_of"    -> "intermediary_of";
            case "same_name_as"       -> "same_name_as";
            case "same_id_as"         -> "same_id_as";
            case "similar"            -> "similar";
            default                   -> "connected_to";
        };
    }

    private static int indexOf(final String[] arr, final String name) {
        for (int i = 0; i < arr.length; i++)
            if (name.equalsIgnoreCase(arr[i].trim())) return i;
        return -1;
    }
}
