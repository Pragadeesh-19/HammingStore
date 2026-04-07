# HammingStore

An embeddable graph database for the JVM that detects hidden patterns in financial crime networks.

Most AML systems stitch together a graph database and a vector database - two separate processes,
two query paths, two points of failure - to do what HammingStore does in one call.

```java
// Screen an entity against 1.87M Panama Papers nodes in one call
client.search(encoder.encode("Mossack Fonseca")).topK(10).execute();

// Trace 4-hop beneficial ownership chain — single gRPC call, ~3.65ms
client.from(companyId)
      .via(REL_OFFICER_OF)
      .via(REL_OFFICER_OF)
      .via(REL_OFFICER_OF)
      .via(REL_OFFICER_OF)
      .topK(5).execute();

// Risk score against statistical centroid of 500k real Panama Papers shells
amlService.shellRiskScore(entityId); // returns 0.0–1.0
```

---

## What it does

Entities are stored as 10,048-bit binary hypervectors. Typed edges use Vector Symbolic Architecture - XOR binding with cyclic permutation per role - so subject, relation, and object get encoded into a single binary vector. HNSW handles fuzzy search. A memory-mapped hash table handles O(1) exact edge lookup.

The result: graph traversal and semantic similarity search share the same data structure. You don't need Pinecone alongside Neo4j. You don't need a Python service to handle the embeddings. You add one Maven dependency.

---

## Benchmarks

**Wikidata - 646k entities - JMH - Windows 11**

| Operation | Avg | p99 |
|-----------|-----|-----|
| 2-hop chain traversal | 2.07ms | 2.46ms |
| 3-hop chain traversal | 2.01ms | 2.38ms |
| 4-hop chain traversal | 3.65ms | 4.12ms |
| HNSW search (topK=10) | 1.84ms | - |

**ICIJ Offshore Leaks - 1.87M entities - Panama + Paradise + Pandora Papers**

| Endpoint | Latency |
|----------|---------|
| `/api/screen` - fuzzy name match across all three leaks | 886ms |
| `/api/shell-risk/{id}` - Hamming distance to shell prototype | 290–1748ms |
| `/api/analogous/{id}` - VSA analogy query | 439ms |
| `/api/address-cluster` - semantic address search | 1141ms |
| `/api/ownership/{id}` - 4-hop beneficial ownership chain | 0ms* |

*The ownership chain uses an in-memory BFS index built at load time, not a live gRPC traversal.
I'm flagging it because 0ms on a 4-hop query sounds implausible without that context.

All numbers are on a consumer laptop on Windows. Linux benchmarks are on the roadmap.

---

## Quickstart

**Run the AML demo (requires pre-ingested data):**

```bash
# Clone both repos as siblings
git clone https://github.com/Pragadeesh-19/HammingStore
git clone https://github.com/Pragadeesh-19/hammingstore-aml-demo

# Configure data paths
cd HammingStore
cp env.example .env
# Edit .env: set HAMMINGSTORE_DATA_DIR and OFFSHORE_LEAKS_DIR

# Start everything
docker-compose up
```

Then:
```bash
curl -s http://localhost:8081/api/screen \
  -H "Content-Type: application/json" \
  -d '{"name":"Mossack Fonseca","topK":5}' | jq .
```

**Add to an existing Spring Boot project:**

```xml
<dependency>
    <groupId>io.hammingstore</groupId>
    <artifactId>hammingstore-spring-boot-starter</artifactId>
    <version>1.0.0</version>
</dependency>
```

```yaml
# application.yml
hammingstore:
  endpoint: localhost:50051
```

```java
@Autowired
private InstrumentedHammingClient client;

// That's it. Micrometer timers, health indicator, and graceful shutdown included.
```

---

## Architecture

```
1. Generate float embedding (MiniLM-L6-v2, 384 dims)
2. Apply random projection encoding
3. Produce a 10,048-bit binary hypervector

This hypervector is stored in:
- OffHeapVectorStore (mmap FFM) - for raw vectors (zero GC)
- SparseEntityIndex (mmap) - for entity → slot mapping
- HNSWIndex (mmap, off-heap) - for approximate nearest neighbor search
- EdgeLog WAL (mmap) - for persistent typed edges
```

Typed edges are stored as `bind(permute(subject, 1), bind(permute(relation, 2), permute(object, 3)))`.
Chain traversal decodes the binding step-by-step using XOR and inverse permutation.
No Cypher. No query planner. No JVM heap pressure — the hot path is entirely off-heap via the Foreign Function and Memory API.

---

## Use cases

**AML / KYC**
Trace beneficial ownership chains through layered corporate structures (FATF Recommendation 24 compliance). Screen entity names against sanctions lists with fuzzy matching that handles transliterations and aliases. Score new entities against a VSA prototype of known shell companies.

**Fraud detection**
Find structurally similar transaction networks to known fraud patterns using VSA analogy queries. No ML training required - the prototype is built from observed fraud cases using majority-vote bundling.

**Knowledge graphs**
Multi-hop reasoning across typed relationships in a single JVM process. No external graph database, no Cypher, no network round-trips between query hops.

---

## Honest limitations

- No Python client. Java gRPC or the Spring Boot starter only.
- No multi-tenancy. One namespace per server instance.
- Benchmarks are from a consumer laptop on Windows. Reproduce them yourself before trusting them.
- Zero production deployments. This is an open-source project, not a deployed system.
- The Watch API fires on every `storeFloat` write. High-volume ingestion may need rate limiting.

---

## Modules

| Module | Purpose |
|--------|---------|
| `hammingstore-core` | gRPC server, HNSW engine, VSA encoding, persistence |
| `hammingstore-client` | Java gRPC client |
| `hammingstore-embeddings` | MiniLM-L6-v2 via ONNX Runtime |
| `hammingstore-spring-boot-starter` | Spring Boot autoconfiguration, Micrometer, Watch API |
| `hammingstore-spring-data` | Repository abstraction |
| `hammingstore-benchmark` | JMH benchmarks, dataset loaders |

---

## Contributing

The project needs Linux benchmark numbers, a Python client, and a proper comparison against Neo4j and JanusGraph on a standardized fraud detection workload. If you work on graph databases, AML systems, or HNSW implementations and want to collaborate, open an issue or send a connection request on LinkedIn.

---

## License
 
[Apache License 2.0](LICENSE)
