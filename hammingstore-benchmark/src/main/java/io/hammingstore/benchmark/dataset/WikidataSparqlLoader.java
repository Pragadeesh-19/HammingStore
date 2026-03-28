package io.hammingstore.benchmark.dataset;

import ai.onnxruntime.OrtException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.time.Duration;
import java.util.*;

public final class WikidataSparqlLoader {

    private static final String SPARQL_ENDPOINT  = "https://query.wikidata.org/sparql";
    private static final int    BATCH_SIZE        = 1_000;
    private static final long   REQUEST_DELAY_MS  = 1_100;

    public static final int DIMS = 384;

    private static final String[] PROPERTIES = {
            "P17", "P131", "P31", "P27", "P279", "P361", "P150", "P106", "P19", "P20"
    };

    private static final String[][] ENTITY_TYPES = {
            {"Q5",        "human"},
            {"Q515",      "city"},
            {"Q6256",     "country"},
            {"Q4022",     "river"},
            {"Q8502",     "mountain"},
            {"Q43229",    "organization"},
            {"Q11424",    "film"},
            {"Q571",      "book"},
            {"Q7397",     "software"},
            {"Q3918",     "university"},
            {"Q532",      "village"},
            {"Q486972",   "human settlement"},
            {"Q747074",   "commune"},
            {"Q3957",     "town"},
            {"Q16521",    "taxon"},
            {"Q7366",     "song"},
            {"Q482994",   "album"},
            {"Q215380",   "musical group"},
            {"Q639669",   "musician"},
            {"Q33999",    "actor"},
            {"Q82955",    "politician"},
            {"Q1650915",  "researcher"},
            {"Q901",      "scientist"},
            {"Q36180",    "writer"},
            {"Q3455803",  "director"},
            {"Q2066131",  "athlete"},
            {"Q11513337", "sports season"},
            {"Q4438121",  "sports organisation"},
            {"Q2095",     "food"},
            {"Q11173",    "chemical compound"},
            {"Q12136",    "disease"},
            {"Q16970",    "church building"},
            {"Q811979",   "architectural structure"},
            {"Q1107656",  "comic series"},
            {"Q25107",    "planet"},
            {"Q523",      "star"},
            {"Q318",      "galaxy"},
            {"Q2221906",  "geographic location"},
            {"Q23397",    "lake"},
            {"Q35509",    "cave"},
            {"Q131681",   "island group"},
            {"Q23442",    "island"},
            {"Q8502",     "mountain"},
            {"Q46831",    "mountain range"},
            {"Q355304",   "watercourse"},
            {"Q12280",    "bridge"},
            {"Q1248784",  "airport"},
            {"Q12323",    "dam"},
            {"Q55488",    "railway station"},
            {"Q34442",    "road"},
            {"Q15618652", "national park"},
    };

    private final HammingClient client;
    private final int           maxEntities;
    private final Path          progressFile;
    private final Path phase2ProgressFile;
    private final Path labelCacheFile;
    private final HttpClient    http;
    private final ObjectMapper  json;
    private final String serverHost;
    private final int serverPort;

    public WikidataSparqlLoader(final HammingClient client, final int maxEntities, final String serverHost, final int serverPort) {
        this.client       = client;
        this.maxEntities  = maxEntities;
        this.serverHost = serverHost;
        this.serverPort = serverPort;
        this.phase2ProgressFile = Paths.get(System.getProperty("user.home"),
                ".hammingstore-benchmark", "sparql_phase2_progress.txt");
        this.progressFile = Paths.get(System.getProperty("user.home"),
                ".hammingstore-benchmark", "sparql_progress.txt");
        this.labelCacheFile = Paths.get(System.getProperty("user.home"),
                ".hammingstore-benchmark", "sparql_labels.tsv");
        this.http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        this.json = new ObjectMapper();
    }

    public WikidataDataset load(final SentenceEncoder encoder)
            throws IOException, InterruptedException, OrtException {

        Files.createDirectories(progressFile.getParent());
        final long startMs = System.currentTimeMillis();

        // --- Phase 1: collect entity labels via SPARQL ---
        System.out.printf("[SPARQL] Phase 1: collecting %,d entity labels%n", maxEntities);
        final Map<Long, String> entityLabels = new HashMap<>(maxEntities * 2);

        if (Files.exists(labelCacheFile)) {
            System.out.printf("[SPARQL] Loading labels from cache: %s%n", labelCacheFile);
            for (final String line : Files.readAllLines(labelCacheFile)) {
                final int tab = line.indexOf('\t');
                if (tab > 0) {
                    try {
                        entityLabels.put(Long.parseLong(line.substring(0, tab)),
                                line.substring(tab + 1));
                    } catch (final NumberFormatException ignored) {}
                }
            }
            System.out.printf("[SPARQL] Loaded %,d labels from cache%n", entityLabels.size());
        } else {

            for (final String[] type : ENTITY_TYPES) {
                if (entityLabels.size() >= maxEntities) break;
                final String typeQid = type[0];
                final String name = type[1];
                int offset = loadProgress(typeQid);

                while (entityLabels.size() < maxEntities) {
                    final JsonNode result = sparqlQuery(
                            "SELECT ?item ?itemLabel WHERE { " +
                                    "?item wdt:P31 wd:" + typeQid + " . " +
                                    "SERVICE wikibase:label { " +
                                    "bd:serviceParam wikibase:language \"en\" . } } " +
                                    "LIMIT " + BATCH_SIZE + " OFFSET " + offset);

                    if (result == null) break;
                    final JsonNode bindings = result.path("results").path("bindings");
                    if (bindings.isEmpty()) break;

                    for (final JsonNode row : bindings) {
                        if (entityLabels.size() >= maxEntities) break;
                        final long id = qidToLong(row.path("item").path("value").asText());
                        final String label = row.path("itemLabel").path("value").asText();
                        if (id > 0 && !label.isBlank() && !label.startsWith("Q"))
                            entityLabels.put(id, label);
                    }

                    offset += BATCH_SIZE;
                    saveProgress(typeQid, offset);
                    System.out.printf("[SPARQL]   %s offset=%,d  total=%,d%n",
                            name, offset, entityLabels.size());
                    Thread.sleep(REQUEST_DELAY_MS);
                    if (bindings.size() < BATCH_SIZE) break;
                }
            }

            // Save label cache so Phase 1 never re-runs after a crash
            System.out.printf("[SPARQL] Saving %,d labels to cache...%n", entityLabels.size());
            final List<String> cacheLines = new ArrayList<>(entityLabels.size());
            for (final Map.Entry<Long, String> e : entityLabels.entrySet())
                cacheLines.add(e.getKey() + "\t" + e.getValue());
            Files.write(labelCacheFile, cacheLines);
            System.out.println("[SPARQL] Label cache saved.");
        }

        System.out.printf("[SPARQL] Phase 1 done: %,d labels in %,ds%n",
                entityLabels.size(), (System.currentTimeMillis() - startMs) / 1000);

        // --- Phase 2: encode + store entity vectors ---
        System.out.println("[SPARQL] Phase 2: encoding and storing entity vectors...");
        final int phase2Start = loadPhase2Progress();
        if (phase2Start > 0)
            System.out.printf("[SPARQL] Resuming Phase 2 from entity %,d%n", phase2Start);

        final Set<Long> storedIds = new HashSet<>(entityLabels.size() * 2);
        int stored = 0;

        try (final HammingClient phase2Client = HammingClient.builder()
                .endpoint(serverHost, serverPort).plaintext()
                .timeout(Duration.ofHours(24)).build()) {

            final List<Map.Entry<Long, String>> entries =
                    new ArrayList<>(entityLabels.entrySet());

            for (int i = phase2Start; i < entries.size(); i++) {
                final Map.Entry<Long, String> e = entries.get(i);
                phase2Client.storeFloat(e.getKey(), encoder.encode(e.getValue()));
                storedIds.add(e.getKey());
                stored++;
                if (stored % 10_000 == 0) {
                    savePhase2Progress(phase2Start + stored);
                    System.out.printf("[SPARQL]   stored %,d / %,d  (%,ds)%n",
                            phase2Start + stored, entityLabels.size(),
                            (System.currentTimeMillis() - startMs) / 1000);
                }
            }
        }

        System.out.printf("[SPARQL] Phase 2 done: %,d vectors%n", stored);
        Files.deleteIfExists(phase2ProgressFile);

        try (final HammingClient phase34Client = HammingClient.builder()
                .endpoint(serverHost, serverPort).plaintext()
                .timeout(Duration.ofHours(6)).build()) {

            System.out.println("[SPARQL] Phase 3: storing relation vectors...");
            final String[] propLabels = {
                    "country",
                    "located in administrative territorial entity",
                    "instance of",
                    "country of citizenship",
                    "subclass of",
                    "part of",
                    "contains administrative territorial entity",
                    "occupation",
                    "place of birth",
                    "place of death"
            };
            for (int i = 0; i < PROPERTIES.length; i++) {
                final long relId = Entity.defaultId("REL:" + PROPERTIES[i]);
                phase34Client.storeFloat(relId, encoder.encode(propLabels[i]));
                System.out.printf("[SPARQL]   relation: %s%n", PROPERTIES[i]);
            }

            System.out.println("[SPARQL] Phase 4: storing edges...");
            final Map<Long, List<Long>> entityRels = new HashMap<>();
            int edgesStored = 0;

            outer:
            for (final String prop : PROPERTIES) {
                int offset = 0;
                while (true) {
                    final JsonNode result = sparqlQuery(
                            "SELECT ?item ?target WHERE { " +
                                    "?item wdt:" + prop + " ?target . " +
                                    "FILTER(STRSTARTS(STR(?target)," +
                                    "\"http://www.wikidata.org/entity/Q\")) } " +
                                    "LIMIT " + BATCH_SIZE + " OFFSET " + offset);

                    if (result == null) break;
                    final JsonNode bindings = result.path("results").path("bindings");
                    if (bindings.isEmpty()) break;

                    for (final JsonNode row : bindings) {
                        final long sub = qidToLong(row.path("item").path("value").asText());
                        final long obj = qidToLong(row.path("target").path("value").asText());
                        if (!storedIds.contains(sub) || !storedIds.contains(obj)) continue;

                        client.storeEdge(new Edge(Entity.of(sub), prop, Entity.of(obj)));
                        edgesStored++;

                        entityRels.computeIfAbsent(sub, k -> new ArrayList<>())
                                .add(Entity.defaultId("REL:" + prop));
                    }

                    offset += BATCH_SIZE;
                    Thread.sleep(REQUEST_DELAY_MS);
                    if (edgesStored % 10_000 == 0 && edgesStored > 0)
                        System.out.printf("[SPARQL]   edges: %,d%n", edgesStored);
                    if (bindings.size() < BATCH_SIZE) break;
                    if (edgesStored >= (long) maxEntities * 3) break outer;
                }
            }

            System.out.printf("[SPARQL] Phase 4 done: %,d edges%n", edgesStored);

            // --- Build chain seeds ---
            final List<long[]> chainSeeds = new ArrayList<>(10_000);
            final SplittableRandom rng = new SplittableRandom(43L);
            for (final Map.Entry<Long, List<Long>> e : entityRels.entrySet()) {
                if (chainSeeds.size() >= 10_000) break;
                final List<Long> rels = e.getValue();
                if (rels.size() < 2) continue;
                final int i1 = rng.nextInt(rels.size());
                int i2 = rng.nextInt(rels.size() - 1);
                if (i2 >= i1) i2++;
                chainSeeds.add(new long[]{e.getKey(), rels.get(i1), rels.get(i2)});
            }

            final long elapsed = System.currentTimeMillis() - startMs;
            System.out.printf("[SPARQL] Complete: %,d entities, %,d edges, %,d seeds in %,ds%n",
                    stored, edgesStored, chainSeeds.size(), elapsed / 1000);

            return new WikidataDataset(new ArrayList<>(storedIds), chainSeeds);
        }
    }

    private int loadPhase2Progress() {
        try {
            if (Files.exists(phase2ProgressFile))
                return Integer.parseInt(Files.readString(phase2ProgressFile).trim());
        } catch (final Exception ignored) {}
        return 0;
    }

    private void savePhase2Progress(final int count) {
        try { Files.writeString(phase2ProgressFile, String.valueOf(count)); }
        catch (final IOException ignored) {}
    }

    private JsonNode sparqlQuery(final String sparql)
            throws IOException, InterruptedException {
        final String url = SPARQL_ENDPOINT + "?format=json&query="
                + URLEncoder.encode(sparql, StandardCharsets.UTF_8);
        try {
            final HttpResponse<String> resp = http.send(
                    HttpRequest.newBuilder()
                            .uri(URI.create(url))
                            .timeout(Duration.ofSeconds(55))
                            .header("Accept", "application/sparql-results+json")
                            .header("User-Agent", "HammingStore-Benchmark/1.0")
                            .GET().build(),
                    HttpResponse.BodyHandlers.ofString());

            if (resp.statusCode() == 429) {
                System.out.println("[SPARQL] Rate limited — waiting 60s");
                Thread.sleep(60_000);
                return sparqlQuery(sparql);
            }
            if (resp.statusCode() != 200) {
                System.err.printf("[SPARQL] HTTP %d — skipping%n", resp.statusCode());
                return null;
            }
            return json.readTree(resp.body());
        } catch (final Exception e) {
            System.err.printf("[SPARQL] Error: %s — retrying in 5s%n", e.getMessage());
            Thread.sleep(5_000);
            return null;
        }
    }

    private static long qidToLong(final String uri) {
        try {
            final int q = uri.lastIndexOf('Q');
            return q < 0 ? -1L : Long.parseLong(uri.substring(q + 1));
        } catch (final NumberFormatException e) {
            return -1L;
        }
    }

    private int loadProgress(final String key) throws IOException {
        if (!Files.exists(progressFile)) return 0;
        for (final String line : Files.readAllLines(progressFile)) {
            final String[] p = line.split("=");
            if (p.length == 2 && p[0].equals(key))
                return Integer.parseInt(p[1].trim());
        }
        return 0;
    }

    private void saveProgress(final String key, final int offset) throws IOException {
        final List<String> lines = Files.exists(progressFile)
                ? new ArrayList<>(Files.readAllLines(progressFile)) : new ArrayList<>();
        lines.removeIf(l -> l.startsWith(key + "="));
        lines.add(key + "=" + offset);
        Files.write(progressFile, lines);
    }

    public static void main(final String[] args) throws Exception {
        final int    max  = args.length > 0 ? Integer.parseInt(args[0]) : 1_000_000;
        final String host = args.length > 1 ? args[1] : "localhost";
        final int    port = args.length > 2 ? Integer.parseInt(args[2]) : 50051;

        System.out.printf("[SPARQL] %s:%d  max=%,d%n", host, port, max);

        try (final SentenceEncoder encoder = new SentenceEncoder();
             final HammingClient client = HammingClient.builder()
                     .endpoint(host, port).plaintext()
                     .timeout(Duration.ofHours(24)).build()) {

            if (!client.ping()) {
                System.err.println("ERROR: Server not reachable."); System.exit(1);
            }
            final WikidataDataset ds =
                    new WikidataSparqlLoader(client, max, host, port).load(encoder);
            System.out.printf("[SPARQL] Done. Entities=%,d Seeds=%,d%n",
                    ds.entityIds().size(), ds.chainSeeds().size());
        }
    }

    public record WikidataDataset(List<Long> entityIds, List<long[]> chainSeeds) {}
}
