#!/usr/bin/env python3
"""
Embed FB15k-237 entities and load them into HammingStore.

Usage:
    python embed_and_load.py --data-dir ./data/fb15k-237 --encoder minilm --dims 384
    python embed_and_load.py --data-dir ./data/fb15k-237 --encoder gemini --dims 1536
    python embed_and_load.py --data-dir ./data/fb15k-237 --encoder minilm --dims 384 --load-triples
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import grpc

try:
    import hammingstore_pb2 as pb
    import hammingstore_pb2_grpc as pb_grpc
except ImportError:
    print("gRPC stubs not found. Run: python -m grpc_tools.protoc ...")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


def entity_id(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF

def relation_id(name: str) -> int:
    return int(hashlib.sha256(("REL:" + name).encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF

def load_fb15k237(data_dir: Path):
    entities  = {}
    relations = {}
    triples   = []
    for split in ("train.txt", "valid.txt", "test.txt"):
        path = data_dir / split
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                s, r, o = parts
                if s not in entities:  entities[s]  = entity_id(s)
                if o not in entities:  entities[o]  = entity_id(o)
                if r not in relations: relations[r] = relation_id(r)
                triples.append((entities[s], relations[r], entities[o]))
    print(f"FB15k-237: {len(entities)} entities, {len(relations)} relations, {len(triples)} triples")
    return entities, relations, triples


class MiniLMEncoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dims  = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 batch_size=256, show_progress_bar=True)


class GeminiEncoder:
    BATCH = 100

    def __init__(self):
        from google import genai
        from google.genai import types
        self._types  = types
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.dims   = 1536

    def encode(self, texts: list[str]) -> np.ndarray:
        results = []
        for i in tqdm(range(0, len(texts), self.BATCH), desc="Gemini batches"):
            batch = texts[i : i + self.BATCH]
            resp  = self.client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=batch,
                config=self._types.EmbedContentConfig(output_dimensionality=1536),
            )
            for emb in resp.embeddings:
                results.append(emb.values)
        arr   = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms > 0, norms, 1.0)


def stub(host, port):
    return pb_grpc.HammingStoreStub(grpc.insecure_channel(f"{host}:{port}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     required=True, type=Path)
    ap.add_argument("--encoder",      default="minilm", choices=["minilm", "gemini"])
    ap.add_argument("--dims",         default=384, type=int)
    ap.add_argument("--host",         default="localhost")
    ap.add_argument("--port",         default=50051, type=int)
    ap.add_argument("--cache-dir",    default="./embedding-cache", type=Path)
    ap.add_argument("--load-triples", action="store_true")
    args = ap.parse_args()

    entities, relations, triples = load_fb15k237(args.data_dir)

    enc = GeminiEncoder() if args.encoder == "gemini" else MiniLMEncoder()

    args.cache_dir.mkdir(exist_ok=True)
    cache_emb = args.cache_dir / f"fb15k237_{args.encoder}_{enc.dims}.npy"
    cache_ids = args.cache_dir / f"fb15k237_{args.encoder}_{enc.dims}_ids.json"

    names = list(entities.keys())
    ids   = [entities[n] for n in names]

    if cache_emb.exists():
        print(f"Loading cached embeddings from {cache_emb}")
        embeddings = np.load(cache_emb)
    else:
        print(f"Embedding {len(names)} entities via {args.encoder}...")
        t0 = time.time()
        embeddings = enc.encode(names)
        print(f"Done in {time.time()-t0:.1f}s ({len(names)/(time.time()-t0):.0f}/sec)")
        np.save(cache_emb, embeddings)
        json.dump(ids, open(cache_ids, "w"))

    client = stub(args.host, args.port)

    print(f"Loading {len(names)} entities into HammingStore...")
    errors = 0
    t0 = time.time()
    for eid, emb in tqdm(zip(ids, embeddings), total=len(ids)):
        resp = client.StoreFloat(pb.StoreFloatRequest(
            entity_id=eid,
            float_vec=emb.astype(np.float32).tobytes(),
        ))
        if not resp.success:
            errors += 1
    elapsed = time.time() - t0
    rate = f"{len(names)/elapsed:.0f}/sec" if elapsed > 0 else "n/a"
    print(f"Entities: {len(names)-errors} stored, {errors} errors, {rate}")

    print(f"Loading {len(relations)} relation vectors...")
    rng = np.random.default_rng(seed=0xC0FFEE)
    for rel_name, rel_id_val in tqdm(relations.items()):
        v = rng.standard_normal(enc.dims).astype(np.float32)
        v /= np.linalg.norm(v)
        client.StoreFloat(pb.StoreFloatRequest(
            entity_id=rel_id_val,
            float_vec=v.tobytes(),
        ))

    if args.load_triples:
        print(f"Loading {len(triples)} typed edges...")
        stored = 0
        skipped = 0
        for s, r, o in tqdm(triples):
            try:
                resp = client.StoreTypedEdge(pb.StoreTypedEdgeRequest(
                    subject_id=s, relation_id=r, object_id=o,
                ))
                if resp.success:
                    stored += 1
                else:
                    skipped += 1
            except grpc.RpcError as e:
                skipped += 1
        print(f"Triples: {stored} stored, {skipped} skipped")


if __name__ == "__main__":
    main()