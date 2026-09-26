#!/usr/bin/env python3
"""
EntityMatch AI — Embedding Pre-computation & Caching
Run ONCE to encode all texts with MiniLM and cache to disk.
Subsequent inference loads cached embeddings for instant FAISS search.

Usage: python scripts/prepare_embeddings.py
Output: output/cache/s1_embeddings.npy, output/cache/target_embeddings.npy, etc.
"""
import os
import sys
import time
import gc
import json
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl

CACHE_DIR = Path("output/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEST_DIR = "student_resource/dataset/test"


def find_model_path():
    """Find MiniLM model — local copy first, then kagglehub."""
    local = Path("code/models/minilm")
    if local.exists() and (local / "config.json").exists():
        return str(local)
    try:
        import kagglehub
        return kagglehub.dataset_download("shinomoriaoshi/sentencetransformersallminilml6v2")
    except Exception:
        raise RuntimeError("Model not found! Run kagglehub download first.")


def make_text(name: str, addr: str) -> str:
    """Combine name + address for embedding. Safe null handling."""
    n = str(name).strip() if name else ""
    a = str(addr).strip() if addr else ""
    if n and a:
        return f"{n} [SEP] {a}"
    return n or a or ""


def main():
    start = time.time()
    print("=" * 70)
    print("  EMBEDDING PRE-COMPUTATION (one-time)")
    print("=" * 70)

    # Check if already cached
    if all((CACHE_DIR / f).exists() for f in [
        "s1_embeddings.npy", "s1_ids.json",
        "target_embeddings.npy", "target_ids.json"
    ]):
        print("\n✅ Embeddings already cached! Delete output/cache/ to recompute.")
        return

    # Load model
    print("\n[1/4] Loading model...")
    from sentence_transformers import SentenceTransformer
    model_path = find_model_path()
    model = SentenceTransformer(model_path)
    print(f"  Model loaded from: {model_path}")
    print(f"  Embedding dim: {model.get_embedding_dimension()}")

    # Load data
    print("\n[2/4] Loading data...")
    t0 = time.time()
    
    s1 = pl.read_csv(
        os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t",
        schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                          "business_address": pl.Utf8, "country": pl.Utf8}
    ).with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    ])

    targets = pl.concat([
        pl.read_csv(
            os.path.join(TEST_DIR, fn), separator="\t",
            schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                              "business_address": pl.Utf8, "country": pl.Utf8}
        ).with_columns([
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
            pl.col("country").fill_null(""),
        ])
        for fn in ["test_source2.tsv", "test_source3.tsv"]
    ])

    print(f"  S1: {len(s1):,} | Targets: {len(targets):,} | Load: {time.time()-t0:.1f}s")

    # Prepare texts
    print("\n[3/4] Preparing texts...")
    s1_ids = s1["entity_id"].to_list()
    s1_texts = [make_text(n, a) for n, a in zip(
        s1["business_name"].to_list(), s1["business_address"].to_list()
    )]

    target_ids = targets["entity_id"].to_list()
    target_countries = targets["country"].to_list()
    target_texts = [make_text(n, a) for n, a in zip(
        targets["business_name"].to_list(), targets["business_address"].to_list()
    )]

    # Save country info for per-country FAISS
    s1_countries = s1["country"].to_list()
    with open(CACHE_DIR / "s1_countries.json", "w") as f:
        json.dump(s1_countries, f)
    with open(CACHE_DIR / "target_countries.json", "w") as f:
        json.dump(target_countries, f)

    del s1, targets
    gc.collect()

    # Encode S1
    print(f"\n[4/4] Encoding texts...")
    print(f"  Encoding S1 ({len(s1_texts):,} texts)...")
    t0 = time.time()
    s1_emb = model.encode(s1_texts, batch_size=512, show_progress_bar=True,
                          normalize_embeddings=True)
    print(f"  S1 encoded in {time.time()-t0:.1f}s | Shape: {s1_emb.shape}")

    # Save S1
    np.save(CACHE_DIR / "s1_embeddings.npy", s1_emb)
    with open(CACHE_DIR / "s1_ids.json", "w") as f:
        json.dump(s1_ids, f)
    print(f"  S1 saved to cache")
    del s1_emb, s1_texts
    gc.collect()

    # Encode targets in chunks to avoid OOM
    print(f"\n  Encoding targets ({len(target_texts):,} texts)...")
    CHUNK = 500_000
    all_target_emb = []
    for start in range(0, len(target_texts), CHUNK):
        end = min(start + CHUNK, len(target_texts))
        t0 = time.time()
        chunk_emb = model.encode(target_texts[start:end], batch_size=512,
                                 show_progress_bar=True, normalize_embeddings=True)
        all_target_emb.append(chunk_emb)
        print(f"  Chunk {start:,}-{end:,} encoded in {time.time()-t0:.1f}s")
        gc.collect()

    target_emb = np.vstack(all_target_emb)
    del all_target_emb
    print(f"  Target shape: {target_emb.shape}")

    np.save(CACHE_DIR / "target_embeddings.npy", target_emb)
    with open(CACHE_DIR / "target_ids.json", "w") as f:
        json.dump(target_ids, f)
    print(f"  Targets saved to cache")

    # Also save target names/addresses for feature engineering
    # Reload since we deleted the DF
    targets = pl.concat([
        pl.read_csv(
            os.path.join(TEST_DIR, fn), separator="\t",
            schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                              "business_address": pl.Utf8, "country": pl.Utf8}
        ).with_columns([
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
        ])
        for fn in ["test_source2.tsv", "test_source3.tsv"]
    ])
    target_names = targets["business_name"].to_list()
    target_addrs = targets["business_address"].to_list()
    with open(CACHE_DIR / "target_names.json", "w") as f:
        json.dump(target_names, f)
    with open(CACHE_DIR / "target_addrs.json", "w") as f:
        json.dump(target_addrs, f)

    # Save S1 names/addresses too
    s1_reload = pl.read_csv(
        os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t",
        schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                          "business_address": pl.Utf8, "country": pl.Utf8}
    ).with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
    ])
    with open(CACHE_DIR / "s1_names.json", "w") as f:
        json.dump(s1_reload["business_name"].to_list(), f)
    with open(CACHE_DIR / "s1_addrs.json", "w") as f:
        json.dump(s1_reload["business_address"].to_list(), f)

    total = time.time() - start
    print(f"\n{'='*70}")
    print(f"  ✅ DONE — Embeddings cached in {total:.1f}s ({total/60:.1f} min)")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
