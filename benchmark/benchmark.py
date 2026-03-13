#!/usr/bin/env python3
"""
Benchmark HammingStore on FB15k-237.

Run embed_and_load.py first to populate the server and cache embeddings.

Usage:
    python benchmark.py --data-dir ./data/fb15k-237 --encoder minilm --dims 384
    python benchmark.py --data-dir ./data/fb15k-237 --encoder gemini --dims 1536

Outputs:
    benchmark_results.json
    benchmark_report.md
"""

import argparse
import datetime
import json
import hashlib
import statistics
import time
from pathlib import Path

import numpy as np
import grpc

try:
    import hammingstore_pb2 as pb
    import hammingstore_pb2_grpc as pb_grpc
except ImportError:
    print("gRPC stubs not found. Run: python -m grpc_tools.protoc ...")
    raise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


def relation_id(name: str) -> int:
    return int(hashlib.sha256(("REL:" + name).encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF


def stub(host, port):
    return pb_grpc.HammingStoreStub(grpc.insecure_channel(f"{host}:{port}"))


def percentile(data, p):
    return float(np.percentile(data, p))


def latency_stats(times_ns):
    ms = [t / 1e6 for t in times_ns]
    return {
        "p50_ms":  round(statistics.median(ms), 3),
        "p99_ms":  round(percentile(ms, 99), 3),
        "mean_ms": round(statistics.mean(ms), 3),
        "n":       len(ms),
    }


def bench_memory(n_entities, dims):
    VECTOR_LONGS = 157
    binary_bytes = n_entities * (VECTOR_LONGS * 8)
    float_bytes  = n_entities * (VECTOR_LONGS * 64) * 4
    return {
        "n_entities":      n_entities,
        "dims":            dims,
        "hammingstore_mb": round(binary_bytes / 1e6, 1),
        "float32_mb":      round(float_bytes / 1e6, 1),
        "ratio_x":         round(float_bytes / binary_bytes, 1),
    }


def bench_search(client, embeddings, ids, n_queries, k):
    rng     = np.random.default_rng(42)
    sample  = rng.choice(len(embeddings), size=min(n_queries, len(embeddings)), replace=False)
    times   = []
    for i in sample:
        t0 = time.perf_counter_ns()
        client.SearchFloat(pb.SearchFloatRequest(
            float_vec=embeddings[i].astype(np.float32).tobytes(), k=k,
        ))
        times.append(time.perf_counter_ns() - t0)
    return {"k": k, **latency_stats(times)}


def bench_multihop(client, ids, rel_ids, n_queries, k):
    rng    = np.random.default_rng(99)
    result = {}
    for n_hops in (2, 3):
        times = []
        for _ in range(n_queries):
            eid  = int(rng.choice(ids))
            rels = [int(rng.choice(rel_ids)) for _ in range(n_hops)]
            t0   = time.perf_counter_ns()
            try:
                client.QueryChain(pb.QueryChainRequest(
                    start_entity_id=eid, relation_ids=rels, k=k,
                ))
            except grpc.RpcError:
                pass
            times.append(time.perf_counter_ns() - t0)
        result[f"{n_hops}hop"] = {"n_hops": n_hops, **latency_stats(times)}
    return result


def bench_link_prediction(client, test_triples, rel_ids_set, k, max_triples):
    """
    Link prediction via typed-relation query (queryHop).
    For each test triple (s, r, o): call QueryHop(s, r, k) and check if o appears.
    This tests the VSA reasoning layer, not just nearest-neighbour search.
    """
    hits1 = hits10 = n = errors = 0
    rr = 0.0
    first_error = None

    for subj_id, rel_id, obj_id in tqdm(test_triples[:max_triples], desc="link prediction"):
        if rel_id not in rel_ids_set:
            continue
        try:
            resp = client.QueryHop(pb.QueryHopRequest(
                entity_id=subj_id, relation_id=rel_id, k=k,
            ))
            # QueryHop returns composite IDs: stored as s^r^o.
            # Decode by XOR-ing out the known subject and relation.
            for rank, r in enumerate(resp.results):
                decoded_id = r.entity_id ^ subj_id ^ rel_id
                if decoded_id == obj_id:
                    rr += 1.0 / (rank + 1)
                    if rank == 0: hits1  += 1
                    if rank < 10: hits10 += 1
                    break
        except grpc.RpcError as e:
            errors += 1
            if first_error is None:
                first_error = str(e.details())
        n += 1

    if first_error:
        print(f"  First RpcError: {first_error} ({errors}/{n} calls failed)")
    if n == 0:
        return {"error": "no valid triples"}
    return {
        "n": n,
        "errors": errors,
        "hit@1":  round(hits1  / n, 4),
        "hit@10": round(hits10 / n, 4),
        "mrr":    round(rr / n, 4),
    }


def render_report(r):
    m  = r["memory"]
    s  = r.get("search", {})
    mh = r.get("multihop", {})
    lp = r.get("link_prediction", {})

    lines = [
        "# HammingStore Benchmark — FB15k-237",
        "",
        f"**Encoder:** {r['encoder']} | **Dims:** {r['dims']} | "
        f"**Entities:** {m['n_entities']:,} | **Date:** {r['date']}",
        "",
        "## Memory",
        "",
        "| | MB |",
        "|--|--|",
        f"| HammingStore (binary) | **{m['hammingstore_mb']}** |",
        f"| Float32 baseline | {m['float32_mb']} |",
        f"| Compression | **{m['ratio_x']}×** |",
        "",
    ]

    if s:
        lines += [
            "## Search latency (single-hop)",
            "",
            f"P50 **{s['p50_ms']} ms** | P99 {s['p99_ms']} ms | "
            f"k={s['k']} | n={s['n']}",
            "",
        ]

    if mh:
        lines += [
            "## Chain traversal latency",
            "",
            "| Hops | P50 ms | P99 ms | n |",
            "|------|--------|--------|---|",
        ]
        for label, h in mh.items():
            lines.append(f"| {h['n_hops']} | **{h['p50_ms']}** | {h['p99_ms']} | {h['n']} |")
        lines.append("")

    if lp and "error" not in lp:
        lines += [
            "## Link prediction (tail, FB15k-237 test set)",
            "",
            f"Hit@1 {lp['hit@1']} | Hit@10 {lp['hit@10']} | MRR {lp['mrr']} | n={lp['n']}",
            "",
        ]

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   required=True, type=Path)
    ap.add_argument("--cache-dir",  default="./embedding-cache", type=Path)
    ap.add_argument("--encoder",    default="minilm", choices=["minilm", "gemini"])
    ap.add_argument("--dims",       default=384, type=int)
    ap.add_argument("--host",       default="localhost")
    ap.add_argument("--port",       default=50051, type=int)
    ap.add_argument("--n-queries",  default=500, type=int)
    ap.add_argument("--n-triples",  default=500, type=int)
    ap.add_argument("--k",          default=10, type=int)
    args = ap.parse_args()

    cache_emb = args.cache_dir / f"fb15k237_{args.encoder}_{args.dims}.npy"
    cache_ids = args.cache_dir / f"fb15k237_{args.encoder}_{args.dims}_ids.json"

    if not cache_emb.exists():
        print(f"Cache not found: {cache_emb}\nRun embed_and_load.py first.")
        return

    embeddings = np.load(cache_emb)
    ids        = json.load(open(cache_ids))

    def eid(name):
        return int(hashlib.sha256(name.encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF

    rel_ids      = []
    test_triples = []
    for split, is_test in (("train.txt", False), ("test.txt", True)):
        p = args.data_dir / split
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                s, r, o = parts
                rid = relation_id(r)
                if rid not in rel_ids:
                    rel_ids.append(rid)
                if is_test:
                    test_triples.append((eid(s), rid, eid(o)))

    client  = stub(args.host, args.port)
    results = {
        "encoder": args.encoder,
        "dims":    args.dims,
        "date":    datetime.date.today().isoformat(),
    }

    print("Memory...")
    results["memory"] = bench_memory(len(ids), args.dims)
    m = results["memory"]
    print(f"  {m['hammingstore_mb']} MB vs {m['float32_mb']} MB float32 ({m['ratio_x']}×)")

    print("Search latency...")
    results["search"] = bench_search(client, embeddings, ids, args.n_queries, args.k)
    s = results["search"]
    print(f"  P50={s['p50_ms']}ms P99={s['p99_ms']}ms")

    if rel_ids:
        print("Multi-hop latency...")
        results["multihop"] = bench_multihop(client, ids, rel_ids, args.n_queries, args.k)
        for label, h in results["multihop"].items():
            print(f"  {h['n_hops']}-hop P50={h['p50_ms']}ms P99={h['p99_ms']}ms")

    # --- Diagnostic: verify server actually has entities ---
    print("Verifying server state...")
    try:
        # SearchFloat should return results if entities are loaded
        probe_vec = embeddings[0].astype(np.float32).tobytes()
        probe_resp = client.SearchFloat(pb.SearchFloatRequest(float_vec=probe_vec, k=3))
        print(f"  SearchFloat returned {len(probe_resp.results)} results")
        if probe_resp.results:
            print(f"  First result entity_id={probe_resp.results[0].entity_id}")
            # Now try QueryHop with a known stored entity ID
            known_id = ids[0]
            known_rel = rel_ids[0] if rel_ids else None
            if known_rel:
                try:
                    hop_resp = client.QueryHop(pb.QueryHopRequest(
                        entity_id=known_id, relation_id=known_rel, k=3))
                    print(f"  QueryHop(ids[0], rel_ids[0]) returned {len(hop_resp.results)} results (no error)")
                except grpc.RpcError as e:
                    print(f"  QueryHop(ids[0], rel_ids[0]) FAILED: {e.details()}")
        else:
            print("  WARNING: SearchFloat returned 0 results — server may be empty")
    except grpc.RpcError as e:
        print(f"  SearchFloat FAILED: {e.details()}")

    if test_triples:
        print("Link prediction...")
        results["link_prediction"] = bench_link_prediction(
            client, test_triples, set(rel_ids), args.k, args.n_triples,
        )
        lp = results["link_prediction"]
        if "error" not in lp:
            print(f"  Hit@1={lp['hit@1']} Hit@10={lp['hit@10']} MRR={lp['mrr']}")

    Path("benchmark_results.json").write_text(json.dumps(results, indent=2))
    Path("benchmark_report.md").write_text(render_report(results))
    print("\nWrote benchmark_results.json and benchmark_report.md")


if __name__ == "__main__":
    main()