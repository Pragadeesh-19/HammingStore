# HammingStore

> **2.07ms 2-hop · 2.01ms 3-hop · 3.65ms 4-hop · 100% Recall@1**
> Multi-hop knowledge graph traversal in a single gRPC call. No Python. No Neo4j. Pure JVM

HammingStore is a binary hypervector engine that stores any float embedding as a 10,048 binary vector and traverses knowledge graph at sub-millisecond-per-hop latency. It ships as a stand along gRPC server, a java client, and a spring boot autoconfiguration. 

---

## Benchmark Results

Measured on Wikidata: **646,777 entities · 1,298,419 edges · MiniLM-L6-v2 semantic embeddings**
JMH 1.37 · Fork=3 · Warmup=5×2s · Measurement=10×5s · JDK 22.0.2 · Consumer laptop (Windows 11)

```
Benchmark                   hopCount   Mode   Cnt    Score    Error   Units
ChainBenchmark.chain           2       avgt    30    2.067  ± 0.078   ms/op
ChainBenchmark.chain           3       avgt    30    2.012  ± 0.075   ms/op
ChainBenchmark.chain           4       avgt    30    3.645  ± 0.782   ms/op
```

**Recall evaluation (200 randomly sampled stored edges):**
```
Recall@1  : 100.0%
Recall@10 : 100.0%
```

All hops execute server-side in a single gRPC call - no application-layer loop, no multiple round trips. 

Raw JMH JSON: [`benchmark/results/jmh_chain_wikidata_semantic.json`](benchmark/results/jmh_chain_wikidata_semantic.json)

---

## AML Demo

A working demo using the ICIJ Offshore Leaks dataset (1.87M nodes, Panama Papers + Paradise Papers + Pandora Papers).

Five endpoints that are not possible in Neo4j:
- Fuzzy cross-leak entity screening
- Shell company risk scoring via vector similarity
- Beneficial ownership chain traversal (0ms, in-memory BFS)
- Semantic address clustering for shell factory detection
- Analogous structure queries

**Repo:** [hammingstore-aml-demo](https://github.com/Pragadeesh-19/hammingstore-aml-demo)

---

## What Makes it Fast

**Hybrid edge lookup:** Stored edges use an in-memory `HashMap<subjectId XOR relationID, objectId>` - exact lookup that bypasses HNSW entirely. HNSW is the fall approach for approximate reasoning on unseen edges. This is why Recall@1 = 100% on stored edges.

**Binary Hypervectors:** 10,048 bits per vector (1,256 bytes). Hamming distance = `Long.bitCount(a XOR b)` accross 157 longs - no SIMD, no floating-point, no allocation on the hot path. 

**Role-encoded VSA binding:** `storeTypedEdge` stores `bind(permute(S,1), permute(R,2))`. Each role gets a distinct cyclic bit shift before XOR binding, eliminating cross-relation interference. Decode: `objectId = compositeId XOR subjectId XOR relationId`. 

**Zero GC after warmup:** All vectors live in off-heap `MemorySegment` via the Java 22 FFM API. HNSW graph, entity index, and vector store are memory-mapped files. No `byte[]`, no `float[]` on the hot query path.

---

## Quickstart

### 1. Start the server

```bash
java --add-modules=jdk.random -XX:MaxDirectMemorySize=8g \
  -jar hammingstore-core/target/hammingstore-core-1.0.0-SNAPSHOT.jar \
  --port=50051 --dims=384 --max-vectors=1000000
```

### 2. Add the client dependency

```xml
<dependency>
  <groupId>io.hammingstore</groupId>
  <artifactId>hammingstore-client</artifactId>
  <version>1.0.0-SNAPSHOT</version>
</dependency>
```

### 3. Store and traverse
 
```java
try (HammingClient client = HammingClient.builder()
        .endpoint("localhost", 50051)
        .plaintext()
        .build()) {
 
    client.storeFloat(PARIS_ID, encode("Paris"));
    client.storeFloat(FRANCE_ID, encode("France"));
    client.storeFloat(EUROPE_ID, encode("Europe"));

    client.storeEdge(new Edge(Entity.of(PARIS_ID), "locatedIn", Entity.of(FRANCE_ID)));
    client.storeEdge(new Edge(Entity.of(FRANCE_ID), "locatedIn", Entity.of(EUROPE_ID)));
 
    // 2-hop chain
    List<Entity> result = client
        .from(PARIS_ID)
        .via("locatedIn")
        .via("locatedIn")
        .topK(5)
        .execute();
    // result[0] = Europe
}
```

---

## Spring Boot Integration

```xml
<dependency>
  <groupId>io.hammingstore</groupId>
  <artifactId>hammingstore-spring-boot-starter</artifactId>
  <version>1.0.0-SNAPSHOT</version>
</dependency>
```

```yaml
hammingstore:
  endpoint: localhost:50051
  tls: false
  pool-size: 4
  timeout: 30s
  verify-on-startup: true
  expected-dims: 384
```

```java
@HammingEntity
public class Location {
    @HammingId private long    id;
    @HammingName private String  name;
    @HammingEmbedding private float[] embedding;
}
 
public interface LocationRepository extends HammingRepository<Location, Long> {
    @Chain(relations = {"locatedIn", "locatedIn"}, topK = 5)
    List<Entity> findRegionOf(long cityId);
}
```

When `io.micrometer:micrometer-core` is on the classpath, all operations are timed automatically via `InstrumentedHammingClient` and visible at:
 
```
GET /actuator/metrics/hammingstore.operation.duration?tag=operation:chain
```

---

## Architecture

Diagram: Working on it

**Component summary:**

| Component | Responsibility |
|-----------|---------------|
| `HammingGrpcService` | Protobuf serialisation, gRPC endpoint |
| `VectorGraphRepository` | Coordinates HNSW, HashMap, vector store, entity index |
| `SymbolicReasoner` | VSA chain traversal, analogy, bundle queries |
| `RandomProjectionEncoder` | Float→binary binarisation (10,048-bit) |
| `HNSWIndex` | ANN search — fallback for unseen edges |
| `HashMap<S^R, O>` | O(1) exact edge lookup for stored edges |
| `OffHeapVectorStore` | Off-heap `MemorySegment` vector slab (mmap) |
| `SparseEntityIndex` | Open-addressing hash map: entityId→slot offset |
| `MappedFileAllocator` | Disk persistence via memory-mapped files |

---

## Query Types

```java
// Multi-hop chain traversal
client.from(entityId).via("relation1").via("relation2").topK(10).execute();
 
// Single hop
client.hop(entityId, relationId).topK(10).execute();
 
// Analogical reasoning — A:B :: C:?
client.analogy("London", "England").isTo("Paris").topK(1).execute();
 
// Set prototype — nearest concept to a bundle of entities
client.bundle(id1, id2, id3).topK(5).execute();
 
// Nearest-neighbour semantic search
client.search(embedding).topK(10).execute();
```

## Modules
 
| Module | Description |
|--------|-------------|
| `hammingstore-core` | gRPC server, HNSW index, VSA reasoning engine, persistence |
| `hammingstore-client` | Java gRPC client with fluent builder API |
| `hammingstore-embeddings` | MiniLM-L6-v2 ONNX encoder, 384-dim, pure Java |
| `hammingstore-spring-boot-starter` | Spring Boot autoconfiguration, health indicator, Micrometer metrics |
| `hammingstore-spring-data` | Repository pattern, `@HammingEntity`, `@Chain`, `HammingTemplate` |
| `hammingstore-bom` | Bill of materials for dependency management |
| `hammingstore-benchmark` | JMH benchmarks, Wikidata SPARQL loader, graph expander |
 
---

## Building from source

**Requirements:** JDK 22+, Maven 3.9+
 
```bash
# Build all modules
mvn package -DskipTests
 
# Build server only
mvn package -DskipTests -pl hammingstore-core
 
# Run benchmarks (requires running server on localhost:50051)
java --add-modules=jdk.random -XX:MaxDirectMemorySize=16g \
  -jar hammingstore-benchmark/target/benchmarks.jar \
  ".*ChainBenchmark.*" -rf json -rff results.json
```

## Server Options
 
```
--port=<int>          gRPC port (default: 50051)
--max-vectors=<long>  Vector capacity (default: 1,000,000)
--dims=<int>          Input embedding dimensions (default: 384)
--seed=<long>         Projection matrix seed
--data-dir=<path>     Disk persistence directory (RAM-only if omitted)
```
 
---

## Persistence
 
Data survives server restarts when `--data-dir` is set. All vectors and HNSW graph layers are memory-mapped to disk. Checkpoint is written atomically on shutdown via the JVM shutdown hook.

---

## Benchmark Methodology
 
- JMH 1.37, JDK 22.0.2 (Java HotSpot 64-bit Server VM)
- Dataset: Wikidata geographic entities via SPARQL, encoded with `all-MiniLM-L6-v2` (ONNX Runtime)
- Hardware: consumer laptop, Windows 11 - not a cloud VM
- Recall: 200 randomly sampled stored edges, both ends confirmed present in server
- Chain seeds validated for full 4-hop traversal before benchmark run
- Edge HashMap rebuilt from stored edges after each server restart
 
---

## License
 
[Apache License 2.0](LICENSE)
