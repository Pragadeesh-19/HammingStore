package io.hammingstore.benchmark.dataset;

import io.hammingstore.client.Edge;
import io.hammingstore.client.Entity;
import io.hammingstore.client.HammingClient;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

public final class WikidataEdgeStorer {

    private static final String SPARQL_ENDPOINT = "https://query.wikidata.org/sparql";
    private static final int    BATCH_SIZE      = 50;
    private static final int    CONCURRENCY     = 5;
    private static final long   DELAY_MS        = 1_100;

    private static final String[] PROPERTIES = {
            "P17", "P131", "P31", "P27", "P279", "P361", "P106", "P19", "P20"
    };

    private static final Path CACHE_DIR = Paths.get(
            System.getProperty("user.home"), ".hammingstore-benchmark");

    public static void main(final String[] args) throws Exception {
        final String host = args.length > 0 ? args[0] : "localhost";
        final int    port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;

        System.out.printf("[EdgeStorer] %s:%d  concurrency=%d%n", host, port, CONCURRENCY);

        final Path labelCache = CACHE_DIR.resolve("sparql_labels.tsv");
        if (!Files.exists(labelCache)) {
            System.err.println("ERROR: sparql_labels.tsv not found. Run WikidataSparqlLoader first.");
            System.exit(1);
        }

        System.out.println("[EdgeStorer] Loading stored QIDs...");
        final List<Long> qids = new ArrayList<>();
        for (final String line : Files.readAllLines(labelCache)) {
            final int tab = line.indexOf('\t');
            if (tab > 0) {
                try { qids.add(Long.parseLong(line.substring(0, tab))); }
                catch (final NumberFormatException ignored) {}
            }
        }
        System.out.printf("[EdgeStorer] %,d QIDs loaded%n", qids.size());

        final Set<Long> qidSet = new HashSet<>(qids);

        final Path progressFile = CACHE_DIR.resolve("sparql_edge_progress.txt");
        int startBatch = 0;
        if (Files.exists(progressFile)) {
            try {
                startBatch = Integer.parseInt(
                        Files.readString(progressFile, StandardCharsets.UTF_8).trim());
                System.out.printf("[EdgeStorer] Resuming from batch %,d%n", startBatch);
            } catch (final Exception ignored) {}
        }

        final int totalBatches = (int) Math.ceil((double) qids.size() / BATCH_SIZE);
        System.out.printf("[EdgeStorer] %,d batches total%n", totalBatches);

        final HttpClient http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        final Semaphore sem            = new Semaphore(CONCURRENCY);
        final AtomicInteger totalEdges = new AtomicInteger(0);
        final AtomicInteger completed  = new AtomicInteger(startBatch);
        final Object progressLock      = new Object();

        try (final HammingClient client = HammingClient.builder()
                .endpoint(host, port).plaintext()
                .timeout(Duration.ofHours(6)).build()) {

            if (!client.ping()) {
                System.err.println("ERROR: Server not reachable."); System.exit(1);
            }

            final Path edgeCacheFile = CACHE_DIR.resolve("sparql_stored_edges.tsv");
            // Open edge log in append mode so resume adds to existing file
            final java.io.BufferedWriter edgeWriter = Files.newBufferedWriter(
                    edgeCacheFile,
                    StandardCharsets.UTF_8,
                    startBatch == 0
                            ? new java.nio.file.OpenOption[]{
                            java.nio.file.StandardOpenOption.CREATE,
                            java.nio.file.StandardOpenOption.WRITE,
                            java.nio.file.StandardOpenOption.TRUNCATE_EXISTING}
                            : new java.nio.file.OpenOption[]{
                            java.nio.file.StandardOpenOption.CREATE,
                            java.nio.file.StandardOpenOption.WRITE,
                            java.nio.file.StandardOpenOption.APPEND});

            final List<Thread> threads = new ArrayList<>();

            for (int b = startBatch; b < totalBatches; b++) {
                final int from = b * BATCH_SIZE;
                final int to   = Math.min(from + BATCH_SIZE, qids.size());
                final List<Long> batch = new ArrayList<>(qids.subList(from, to));

                sem.acquire();
                final Thread t = Thread.ofVirtual().start(() -> {
                    try {
                        final List<long[]> edges = queryAllProperties(http, batch);

                        int batchEdges = 0;
                        final List<String> edgeLines = new ArrayList<>();
                        for (final long[] edge : edges) {
                            if (!qidSet.contains(edge[1])) continue;
                            try {
                                client.storeEdge(new Edge(
                                        Entity.of(edge[0]),
                                        PROPERTIES[(int) edge[2]],
                                        Entity.of(edge[1])));
                                batchEdges++;
                                // Record: subjectId TAB property TAB objectId
                                edgeLines.add(edge[0] + "\t" + PROPERTIES[(int) edge[2]] + "\t" + edge[1]);
                            } catch (final Exception ignored) {}
                        }

                        final int total = totalEdges.addAndGet(batchEdges);
                        final int done  = completed.incrementAndGet();

                        synchronized (progressLock) {
                            try {
                                Files.writeString(progressFile,
                                        String.valueOf(done), StandardCharsets.UTF_8);
                                for (final String line : edgeLines) {
                                    edgeWriter.write(line);
                                    edgeWriter.newLine();
                                }
                                edgeWriter.flush();
                            } catch (final IOException ignored) {}
                        }

                        if (done % 100 == 0)
                            System.out.printf("[EdgeStorer] batch=%,d/%,d  edges=%,d%n",
                                    done, totalBatches, total);

                        Thread.sleep(DELAY_MS);
                    } catch (final InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        sem.release();
                    }
                });
                threads.add(t);
            }

            for (final Thread t : threads) t.join();

            edgeWriter.close();
            System.out.printf("[EdgeStorer] Done. %,d edges stored.%n", totalEdges.get());
            System.out.printf("[EdgeStorer] Edge list saved to: %s%n", edgeCacheFile);
            Files.deleteIfExists(progressFile);
        }
    }

    private static List<long[]> queryAllProperties(
            final HttpClient http, final List<Long> qids)
            throws InterruptedException {

        final StringBuilder values = new StringBuilder();
        for (final long qid : qids) values.append("wd:Q").append(qid).append(" ");

        // Wikidata does not support VALUES for property URIs.
        // Use UNION across all 9 properties in one query instead.
        final StringBuilder union = new StringBuilder();
        for (int i = 0; i < PROPERTIES.length; i++) {
            if (i > 0) union.append(" UNION ");
            union.append("{ ?item wdt:").append(PROPERTIES[i])
                    .append(" ?target . BIND(\"").append(i).append("\" AS ?pi) }");
        }

        final String sparql =
                "SELECT ?item ?pi ?target WHERE { " +
                        "VALUES ?item { " + values + "} " +
                        union +
                        " FILTER(STRSTARTS(STR(?target),\"http://www.wikidata.org/entity/Q\")) }";

        final String url = SPARQL_ENDPOINT + "?format=json&query="
                + URLEncoder.encode(sparql, StandardCharsets.UTF_8);

        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                final HttpResponse<String> resp = http.send(
                        HttpRequest.newBuilder()
                                .uri(URI.create(url))
                                .timeout(Duration.ofSeconds(55))
                                .header("Accept", "application/sparql-results+json")
                                .header("User-Agent", "HammingStore-EdgeStorer/1.0")
                                .GET().build(),
                        HttpResponse.BodyHandlers.ofString());

                if (resp.statusCode() == 429) { Thread.sleep(60_000); continue; }
                if (resp.statusCode() != 200)  return List.of();
                return parseEdges(resp.body());

            } catch (final Exception e) {
                if (attempt < 2) Thread.sleep(3_000);
            }
        }
        return List.of();
    }

    private static List<long[]> parseEdges(final String json) {
        final List<long[]> edges = new ArrayList<>();
        int pos = 0;

        while (true) {
            final int itemStart = json.indexOf("\"item\"", pos);
            if (itemStart < 0) break;

            final int iv  = json.indexOf("\"value\"", itemStart);
            final int iq1 = json.indexOf("/Q", iv);
            final int iq2 = json.indexOf("\"", iq1 + 1);
            if (iq1 < 0 || iq2 < 0) { pos = itemStart + 1; continue; }
            final long subjectId;
            try { subjectId = Long.parseLong(json.substring(iq1 + 2, iq2)); }
            catch (final NumberFormatException e) { pos = itemStart + 1; continue; }

            // Read ?pi — the property index as a string literal
            final int piStart = json.indexOf("\"pi\"", iq2);
            if (piStart < 0) break;
            final int pv1 = json.indexOf("\"value\"", piStart);
            final int pq1 = json.indexOf("\"", pv1 + 8);
            final int pq2 = json.indexOf("\"", pq1 + 1);
            if (pq1 < 0 || pq2 < 0) { pos = piStart + 1; continue; }
            final int propIndex;
            try { propIndex = Integer.parseInt(json.substring(pq1 + 1, pq2)); }
            catch (final NumberFormatException e) { pos = piStart + 1; continue; }
            if (propIndex < 0 || propIndex >= PROPERTIES.length) {
                pos = piStart + 1; continue;
            }

            final int targetStart = json.indexOf("\"target\"", pq2);
            if (targetStart < 0) break;
            final int tv  = json.indexOf("\"value\"", targetStart);
            final int tq1 = json.indexOf("/Q", tv);
            final int tq2 = json.indexOf("\"", tq1 + 1);
            if (tq1 < 0 || tq2 < 0) { pos = targetStart + 1; continue; }
            final long objectId;
            try { objectId = Long.parseLong(json.substring(tq1 + 2, tq2)); }
            catch (final NumberFormatException e) { pos = targetStart + 1; continue; }

            edges.add(new long[]{subjectId, objectId, propIndex});
            pos = tq2 + 1;
        }
        return edges;
    }

    private static int propertyIndex(final String prop) {
        for (int i = 0; i < PROPERTIES.length; i++)
            if (PROPERTIES[i].equals(prop)) return i;
        return -1;
    }
}
