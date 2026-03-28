package io.hammingstore.benchmark.dataset;

import io.hammingstore.client.Edge;
import io.hammingstore.client.Entity;
import io.hammingstore.client.HammingClient;
import io.hammingstore.embeddings.SentenceEncoder;

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
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

public final class WikidataGraphExpander {

    private static final String SPARQL_ENDPOINT = "https://query.wikidata.org/sparql";
    private static final int    BATCH_SIZE      = 50;
    private static final int    CONCURRENCY     = 5;
    private static final int    MAX_ROUNDS      = 4;
    private static final long   DELAY_MS        = 1_100;
    private static final String[] EXPAND_PROPS  = {"P131", "P17"};

    private static final Path CACHE_DIR = Paths.get(
            System.getProperty("user.home"), ".hammingstore-benchmark");

    public static void main(final String[] args) throws Exception {
        final String host = args.length > 0 ? args[0] : "localhost";
        final int    port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;

        System.out.printf("[Expander] %s:%d  concurrency=%d%n", host, port, CONCURRENCY);

        final Path labelCache = CACHE_DIR.resolve("sparql_labels.tsv");
        final Map<Long, String> labels = new HashMap<>();
        for (final String line : Files.readAllLines(labelCache)) {
            final int tab = line.indexOf('\t');
            if (tab > 0) {
                try { labels.put(Long.parseLong(line.substring(0, tab)), line.substring(tab + 1)); }
                catch (final NumberFormatException ignored) {}
            }
        }
        System.out.printf("[Expander] %,d entities in label cache%n", labels.size());

        final Set<Long> storedIds = ConcurrentHashMap.newKeySet();
        storedIds.addAll(labels.keySet());

        final Path expandedTsv = CACHE_DIR.resolve("sparql_stored_edges.tsv");

        final HttpClient http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        try (final SentenceEncoder encoder = new SentenceEncoder();
             final HammingClient client = HammingClient.builder()
                     .endpoint(host, port).plaintext()
                     .timeout(Duration.ofHours(6)).build()) {

            if (!client.ping()) { System.err.println("ERROR: Server not reachable."); System.exit(1); }

            Queue<Long> frontier = new LinkedList<>(storedIds);
            int totalNewEntities = 0, totalNewEdges = 0;

            for (int round = 1; round <= MAX_ROUNDS; round++) {
                System.out.printf("%n[Expander] === Round %d — frontier: %,d ===%n",
                        round, frontier.size());

                final List<Long> frontierList = new ArrayList<>(frontier);
                final Set<Long> nextFrontierSet = ConcurrentHashMap.newKeySet();
                final AtomicInteger roundEntities = new AtomicInteger(0);
                final AtomicInteger roundEdges    = new AtomicInteger(0);
                final AtomicInteger completedBatches = new AtomicInteger(0);
                final Object writeLock = new Object();
                final Semaphore sem = new Semaphore(CONCURRENCY);
                final List<Thread> threads = new ArrayList<>();

                for (int b = 0; b < frontierList.size(); b += BATCH_SIZE) {
                    final int from = b;
                    final int to   = Math.min(b + BATCH_SIZE, frontierList.size());
                    final List<Long> batch = new ArrayList<>(frontierList.subList(from, to));

                    sem.acquire();
                    final Thread t = Thread.ofVirtual().start(() -> {
                        try {
                            // Collect all new target entities from this batch
                            final List<long[]> allEdges = new ArrayList<>();
                            for (final String prop : EXPAND_PROPS) {
                                final List<long[]> edges = queryEdges(http, batch, prop);
                                for (final long[] edge : edges)
                                    allEdges.add(new long[]{edge[0], edge[1],
                                            prop.equals("P131") ? 131 : 17});
                            }

                            // Collect unknown targets for batch label fetch
                            final List<Long> unknown = new ArrayList<>();
                            for (final long[] edge : allEdges)
                                if (!storedIds.contains(edge[1]) && !unknown.contains(edge[1]))
                                    unknown.add(edge[1]);

                            // Batch fetch labels for unknown entities
                            final Map<Long, String> newLabels = fetchLabels(http, unknown);

                            // Encode + store new entities
                            for (final Map.Entry<Long, String> e : newLabels.entrySet()) {
                                try {
                                    final float[] vec = encoder.encode(e.getValue());
                                    client.storeFloat(e.getKey(), vec);
                                    storedIds.add(e.getKey());
                                    nextFrontierSet.add(e.getKey());
                                    roundEntities.incrementAndGet();
                                    synchronized (writeLock) {
                                        Files.writeString(labelCache,
                                                e.getKey() + "\t" + e.getValue() + "\n",
                                                StandardCharsets.UTF_8, StandardOpenOption.APPEND);
                                    }
                                } catch (final Exception ignored) {}
                            }

                            // Store edges where both ends are now stored
                            final StringBuilder edgeLines = new StringBuilder();
                            for (final long[] edge : allEdges) {
                                if (!storedIds.contains(edge[1])) continue;
                                final String prop = edge[2] == 131 ? "P131" : "P17";
                                try {
                                    client.storeEdge(new Edge(Entity.of(edge[0]), prop, Entity.of(edge[1])));
                                    roundEdges.incrementAndGet();
                                    edgeLines.append(edge[0]).append('\t')
                                            .append(prop).append('\t')
                                            .append(edge[1]).append('\n');
                                } catch (final Exception ignored) {}
                            }
                            if (edgeLines.length() > 0) {
                                synchronized (writeLock) {
                                    Files.writeString(expandedTsv, edgeLines.toString(),
                                            StandardCharsets.UTF_8, StandardOpenOption.APPEND);
                                }
                            }

                            Thread.sleep(DELAY_MS);
                            final int done = completedBatches.incrementAndGet();
                            if (done % 10 == 0)
                                System.out.printf("[Expander]   batch %,d/%,d  entities=%,d  edges=%,d%n",
                                        done, frontierList.size() / BATCH_SIZE,
                                        roundEntities.get(), roundEdges.get());
                        } catch (final InterruptedException e) {
                            Thread.currentThread().interrupt();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        } finally {
                            sem.release();
                        }
                    });
                    threads.add(t);
                }

                for (final Thread t : threads) t.join();

                totalNewEntities += roundEntities.get();
                totalNewEdges    += roundEdges.get();

                System.out.printf("[Expander] Round %d: +%,d entities, +%,d edges  (total stored: %,d)%n",
                        round, roundEntities.get(), roundEdges.get(), storedIds.size());

                frontier = new LinkedList<>(nextFrontierSet);
                if (frontier.isEmpty()) {
                    System.out.println("[Expander] Frontier empty — fully expanded.");
                    break;
                }
            }

            System.out.printf("%n[Expander] Done. +%,d entities, +%,d edges total.%n",
                    totalNewEntities, totalNewEdges);
        }
    }

    private static Map<Long, String> fetchLabels(
            final HttpClient http, final List<Long> qids)
            throws IOException, InterruptedException {

        final Map<Long, String> result = new HashMap<>();
        if (qids.isEmpty()) return result;

        // Batch fetch — up to 50 labels per query
        for (int i = 0; i < qids.size(); i += 50) {
            final List<Long> batch = qids.subList(i, Math.min(i + 50, qids.size()));
            final StringBuilder values = new StringBuilder();
            for (final long qid : batch) values.append("wd:Q").append(qid).append(" ");

            final String sparql =
                    "SELECT ?item ?label WHERE { " +
                            "VALUES ?item { " + values + "} " +
                            "?item rdfs:label ?label . " +
                            "FILTER(LANG(?label)=\"en\") }";

            final String url = SPARQL_ENDPOINT + "?format=json&query="
                    + URLEncoder.encode(sparql, StandardCharsets.UTF_8);

            for (int attempt = 0; attempt < 3; attempt++) {
                try {
                    final HttpResponse<String> resp = http.send(
                            HttpRequest.newBuilder()
                                    .uri(URI.create(url))
                                    .timeout(Duration.ofSeconds(55))
                                    .header("Accept", "application/sparql-results+json")
                                    .header("User-Agent", "HammingStore-Expander/1.0")
                                    .GET().build(),
                            HttpResponse.BodyHandlers.ofString());

                    if (resp.statusCode() == 429) { Thread.sleep(60_000); continue; }
                    if (resp.statusCode() != 200) break;

                    // Parse item+label pairs
                    final String json = resp.body();
                    int pos = 0;
                    while (true) {
                        final int itemStart = json.indexOf("\"item\"", pos);
                        if (itemStart < 0) break;
                        final int iv  = json.indexOf("\"value\"", itemStart);
                        final int iq1 = json.indexOf("/Q", iv);
                        final int iq2 = json.indexOf("\"", iq1 + 1);
                        if (iq1 < 0 || iq2 < 0) { pos = itemStart + 1; continue; }
                        final long qid;
                        try { qid = Long.parseLong(json.substring(iq1 + 2, iq2)); }
                        catch (final NumberFormatException e) { pos = itemStart + 1; continue; }

                        final int lStart = json.indexOf("\"label\"", iq2);
                        if (lStart < 0) break;
                        final int lv = json.indexOf("\"value\"", lStart);
                        final int lq1 = json.indexOf("\"", lv + 8);
                        final int lq2 = json.indexOf("\"", lq1 + 1);
                        if (lq1 < 0 || lq2 < 0) { pos = lStart + 1; continue; }
                        result.put(qid, json.substring(lq1 + 1, lq2));
                        pos = lq2 + 1;
                    }
                    break;
                } catch (final Exception e) {
                    if (attempt < 2) Thread.sleep(3_000);
                }
            }
            Thread.sleep(DELAY_MS);
        }
        return result;
    }

    private static List<long[]> queryEdges(
            final HttpClient http, final List<Long> qids, final String prop)
            throws IOException, InterruptedException {

        final StringBuilder values = new StringBuilder();
        for (final long qid : qids) values.append("wd:Q").append(qid).append(" ");

        final String sparql =
                "SELECT ?item ?target WHERE { " +
                        "VALUES ?item { " + values + "} " +
                        "?item wdt:" + prop + " ?target . " +
                        "FILTER(STRSTARTS(STR(?target),\"http://www.wikidata.org/entity/Q\")) }";

        final String url = SPARQL_ENDPOINT + "?format=json&query="
                + URLEncoder.encode(sparql, StandardCharsets.UTF_8);

        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                final HttpResponse<String> resp = http.send(
                        HttpRequest.newBuilder()
                                .uri(URI.create(url))
                                .timeout(Duration.ofSeconds(55))
                                .header("Accept", "application/sparql-results+json")
                                .header("User-Agent", "HammingStore-Expander/1.0")
                                .GET().build(),
                        HttpResponse.BodyHandlers.ofString());

                if (resp.statusCode() == 429) { Thread.sleep(60_000); continue; }
                if (resp.statusCode() != 200) return List.of();
                return parsePairs(resp.body());
            } catch (final Exception e) {
                if (attempt < 2) Thread.sleep(3_000);
            }
        }
        return List.of();
    }

    private static List<long[]> parsePairs(final String json) {
        final List<long[]> pairs = new ArrayList<>();
        int pos = 0;
        while (true) {
            final int is = json.indexOf("\"item\"", pos);
            if (is < 0) break;
            final int iv  = json.indexOf("\"value\"", is);
            final int iq1 = json.indexOf("/Q", iv);
            final int iq2 = json.indexOf("\"", iq1 + 1);
            if (iq1 < 0 || iq2 < 0) { pos = is + 1; continue; }
            final long src;
            try { src = Long.parseLong(json.substring(iq1 + 2, iq2)); }
            catch (final NumberFormatException e) { pos = is + 1; continue; }
            final int ts = json.indexOf("\"target\"", iq2);
            if (ts < 0) break;
            final int tv  = json.indexOf("\"value\"", ts);
            final int tq1 = json.indexOf("/Q", tv);
            final int tq2 = json.indexOf("\"", tq1 + 1);
            if (tq1 < 0 || tq2 < 0) { pos = ts + 1; continue; }
            final long tgt;
            try { tgt = Long.parseLong(json.substring(tq1 + 2, tq2)); }
            catch (final NumberFormatException e) { pos = ts + 1; continue; }
            pairs.add(new long[]{src, tgt});
            pos = tq2 + 1;
        }
        return pairs;
    }
}
