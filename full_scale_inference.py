#!/usr/bin/env python3
"""
EntityMatch AI — Ultra-Fast Full-Scale Inference with FAISS + LightGBM
Requires: python scripts/prepare_embeddings.py (run once to cache embeddings)

Pipeline:
1. Load cached embeddings from output/cache/ (~2s)
2. Build per-country FAISS index + search top-K neighbors (~5s)
3. Compute RapidFuzz features on candidate pairs (~30s)
4. Train LightGBM on train data + classify test candidates (~10s)
5. Write results with 100% S1 coverage guarantee

Total target: <60s after embeddings are cached.
"""

import os
import sys
import gc
import re
import time
import json
import logging
import unicodedata
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import faiss
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("failed_countries.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EntityMatch")

CACHE_DIR = Path("output/cache")
TMP_DIR = Path("output/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
Path("output").mkdir(exist_ok=True)

TEST_DIR = "student_resource/dataset/test"
TRAIN_DIR = "student_resource/dataset/train"

# Regex patterns
CLEAN_CHARS_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
FIRST_WORD_RE = re.compile(r"^[^\w]*(\w+)")

# Indic transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _itrans
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False


def safe_transliterate(text: str) -> str:
    """NEVER returns None."""
    if not isinstance(text, str) or not text:
        return ""
    if not HAS_INDIC:
        return text
    try:
        has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in text)
        if not has_indic:
            return text
        for c in text:
            code = ord(c)
            ranges = [
                (0x0900, 0x097F, sanscript.DEVANAGARI),
                (0x0980, 0x09FF, sanscript.BENGALI),
                (0x0A00, 0x0A7F, sanscript.GURMUKHI),
                (0x0A80, 0x0AFF, sanscript.GUJARATI),
                (0x0B00, 0x0B7F, sanscript.ORIYA),
                (0x0B80, 0x0BFF, sanscript.TAMIL),
                (0x0C00, 0x0C7F, sanscript.TELUGU),
                (0x0C80, 0x0CFF, sanscript.KANNADA),
                (0x0D00, 0x0D7F, sanscript.MALAYALAM),
            ]
            for lo, hi, script in ranges:
                if lo <= code <= hi:
                    result = _itrans(text, script, sanscript.ITRANS)
                    return result if isinstance(result, str) and result else text
        return text
    except Exception:
        return str(text)


def normalize_text(s: str) -> str:
    """NEVER returns None."""
    if not isinstance(s, str) or not s:
        return ""
    try:
        s = safe_transliterate(s)
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower().strip()
        s = CLEAN_CHARS_RE.sub(" ", s)
        s = MULTI_SPACE_RE.sub(" ", s).strip()
        return s if s else ""
    except Exception:
        return str(s).lower().strip() if s else ""


def find_model_path():
    """Find MiniLM model path."""
    local = Path("code/models/minilm")
    if local.exists() and (local / "config.json").exists():
        return str(local)
    try:
        import kagglehub
        return kagglehub.dataset_download("shinomoriaoshi/sentencetransformersallminilml6v2")
    except Exception:
        raise RuntimeError("Model not found!")


def compute_features_batch(
    s1_names: List[str], s1_addrs: List[str],
    tgt_names: List[str], tgt_addrs: List[str],
    cosine_scores: List[float]
) -> np.ndarray:
    """Compute feature vectors for candidate pairs. Vectorized-style batch."""
    n = len(s1_names)
    features = np.zeros((n, 6), dtype=np.float32)
    
    for i in range(n):
        s1n = (s1_names[i] or "").lower().strip()
        tgtn = (tgt_names[i] or "").lower().strip()
        s1a = (s1_addrs[i] or "").lower().strip()
        tgta = (tgt_addrs[i] or "").lower().strip()
        
        # 1. Embedding cosine similarity
        features[i, 0] = cosine_scores[i]
        # 2. Name token_sort_ratio
        features[i, 1] = fuzz.token_sort_ratio(s1n, tgtn) / 100.0
        # 3. Address token_sort_ratio
        features[i, 2] = fuzz.token_sort_ratio(s1a, tgta) / 100.0
        # 4. Name WRatio
        features[i, 3] = fuzz.WRatio(s1n, tgtn) / 100.0
        # 5. Token overlap (Jaccard)
        s1_tokens = set(s1n.split())
        tgt_tokens = set(tgtn.split())
        union = s1_tokens | tgt_tokens
        features[i, 4] = len(s1_tokens & tgt_tokens) / max(len(union), 1)
        # 6. Address WRatio
        features[i, 5] = fuzz.WRatio(s1a, tgta) / 100.0
    
    return features


# ====================================================================
# TRAINING: Build LightGBM model from train data
# ====================================================================
def train_lgbm_model(model_st=None) -> Tuple:
    """Train LightGBM on train ground truth using embedding + fuzzy features."""
    logger.info("--- Training LightGBM classifier ---")
    t0 = time.time()
    
    # Load train data
    s1_train = pl.read_csv(
        os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t",
        schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                          "business_address": pl.Utf8, "country": pl.Utf8}
    ).with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
    ])
    
    gt = pl.read_csv(
        os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), separator="\t",
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))
    
    # Load targets
    targets_train = pl.concat([
        pl.read_csv(
            os.path.join(TRAIN_DIR, fn), separator="\t",
            schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                              "business_address": pl.Utf8, "country": pl.Utf8}
        ).with_columns([
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
            pl.col("country").fill_null(""),
        ])
        for fn in ["train_source2.tsv", "train_source3.tsv"]
    ])
    
    # Build GT lookup
    gt_map = {}
    for row in gt.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        if matches:
            gt_map[s1_id] = set(matches)
    
    # Build target lookup
    tgt_lookup = {}
    for row in targets_train.select(["entity_id", "business_name", "business_address"]).iter_rows(named=True):
        tgt_lookup[row["entity_id"]] = (row["business_name"], row["business_address"])
    
    # Sample S1 for training (use 20k for speed)
    sample_size = min(20000, len(s1_train))
    s1_sample = s1_train.sample(n=sample_size, seed=42)
    
    logger.info(f"  Train sample: {sample_size:,} S1 entities, {len(tgt_lookup):,} targets")
    
    # Generate training pairs
    X_list = []
    y_list = []
    
    s1_sample_rows = s1_sample.select(["entity_id", "business_name", "business_address"]).rows()
    
    for s1_id, s1_name, s1_addr in s1_sample_rows:
        true_matches = gt_map.get(s1_id, set())
        
        # Positive pairs
        for tgt_id in true_matches:
            if tgt_id in tgt_lookup:
                tgt_name, tgt_addr = tgt_lookup[tgt_id]
                # Compute features without embedding (use 0.9 as placeholder for true matches)
                feats = compute_features_batch(
                    [s1_name], [s1_addr], [tgt_name], [tgt_addr], [0.9]
                )[0]
                X_list.append(feats)
                y_list.append(1)
        
        # Negative pairs (random targets, capped at 3 per S1)
        neg_count = 0
        for tgt_id in list(tgt_lookup.keys())[:50]:
            if tgt_id not in true_matches and neg_count < 3:
                tgt_name, tgt_addr = tgt_lookup[tgt_id]
                feats = compute_features_batch(
                    [s1_name], [s1_addr], [tgt_name], [tgt_addr], [0.3]
                )[0]
                X_list.append(feats)
                y_list.append(0)
                neg_count += 1
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    logger.info(f"  Training data: {len(X):,} pairs ({sum(y):,} positive, {len(y)-sum(y):,} negative)")
    
    # Split and train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Calibrate threshold for F0.5
    probs = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f05 = 0.5, 0.0
    for t in [i / 100 for i in range(40, 96, 2)]:
        preds = (probs >= t).astype(int)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_thresh = f05, t
    
    logger.info(f"  Model trained in {time.time()-t0:.1f}s | Threshold: {best_thresh:.2f} | F0.5: {best_f05:.4f}")
    
    del s1_train, gt, targets_train, X, y
    gc.collect()
    
    return model, best_thresh


# ====================================================================
# MAIN INFERENCE
# ====================================================================
def main():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("   ENTITYMATCH AI — FAISS + LightGBM INFERENCE")
    logger.info("=" * 80)
    
    # -------------------------------------------------------
    # Check if embeddings are cached
    # -------------------------------------------------------
    has_cache = all((CACHE_DIR / f).exists() for f in [
        "s1_embeddings.npy", "s1_ids.json",
        "target_embeddings.npy", "target_ids.json"
    ])
    
    if not has_cache:
        logger.info("⚠️  Embeddings not cached. Running fallback (dict-based blocking)...")
        run_fallback_inference()
        return
    
    # -------------------------------------------------------
    # STEP 1: Load cached data
    # -------------------------------------------------------
    t0 = time.time()
    logger.info("[1/5] Loading cached embeddings and metadata...")
    
    s1_emb = np.load(CACHE_DIR / "s1_embeddings.npy")
    target_emb = np.load(CACHE_DIR / "target_embeddings.npy")
    
    with open(CACHE_DIR / "s1_ids.json") as f:
        s1_ids = json.load(f)
    with open(CACHE_DIR / "target_ids.json") as f:
        target_ids = json.load(f)
    with open(CACHE_DIR / "s1_countries.json") as f:
        s1_countries = json.load(f)
    with open(CACHE_DIR / "target_countries.json") as f:
        target_countries = json.load(f)
    with open(CACHE_DIR / "s1_names.json") as f:
        s1_names = json.load(f)
    with open(CACHE_DIR / "s1_addrs.json") as f:
        s1_addrs = json.load(f)
    with open(CACHE_DIR / "target_names.json") as f:
        target_names = json.load(f)
    with open(CACHE_DIR / "target_addrs.json") as f:
        target_addrs = json.load(f)
    
    logger.info(f"  S1: {len(s1_ids):,} emb={s1_emb.shape} | Targets: {len(target_ids):,} emb={target_emb.shape}")
    logger.info(f"  Load: {time.time()-t0:.2f}s")
    
    # -------------------------------------------------------
    # STEP 2: Train LightGBM
    # -------------------------------------------------------
    lgbm_model, threshold = train_lgbm_model()
    
    # -------------------------------------------------------
    # STEP 3: Per-country FAISS search + scoring
    # -------------------------------------------------------
    t0 = time.time()
    logger.info("[3/5] Per-country FAISS blocking + scoring...")
    
    countries = sorted(set(s1_countries))
    
    # Build country index maps
    s1_by_country = defaultdict(list)
    for i, c in enumerate(s1_countries):
        s1_by_country[c].append(i)
    
    tgt_by_country = defaultdict(list)
    for i, c in enumerate(target_countries):
        tgt_by_country[c].append(i)
    
    all_results: Dict[str, str] = {}
    TOP_K = 20
    
    for country in countries:
        tc = time.time()
        logger.info(f"\n  [{country}] Processing...")
        
        try:
            s1_indices = s1_by_country[country]
            tgt_indices = tgt_by_country[country]
            
            s1_count = len(s1_indices)
            tgt_count = len(tgt_indices)
            logger.info(f"  [{country}] S1: {s1_count:,} | Targets: {tgt_count:,}")
            
            if tgt_count == 0:
                for si in s1_indices:
                    all_results[s1_ids[si]] = ""
                continue
            
            # Build FAISS index for this country's targets
            tgt_emb_c = target_emb[tgt_indices].astype(np.float32)
            dim = tgt_emb_c.shape[1]
            
            # Normalize for cosine similarity (embeddings should already be normalized, but ensure it)
            faiss.normalize_L2(tgt_emb_c)
            
            # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
            index = faiss.IndexFlatIP(dim)
            index.add(tgt_emb_c)
            
            logger.info(f"  [{country}] FAISS index built ({tgt_count:,} vectors)")
            
            # Search in batches
            s1_emb_c = s1_emb[s1_indices].astype(np.float32)
            faiss.normalize_L2(s1_emb_c)
            
            # FAISS search — top K neighbors
            k = min(TOP_K, tgt_count)
            scores, neighbors = index.search(s1_emb_c, k)
            
            logger.info(f"  [{country}] FAISS search done in {time.time()-tc:.2f}s")
            
            # Score candidate pairs with RapidFuzz + LightGBM
            t_score = time.time()
            matched_count = 0
            
            for qi in range(s1_count):
                s1_idx = s1_indices[qi]
                s1_id = s1_ids[s1_idx]
                s1_name = s1_names[s1_idx]
                s1_addr = s1_addrs[s1_idx]
                
                # Get candidate target indices (mapped back to global)
                cand_local_indices = neighbors[qi]
                cand_cosines = scores[qi]
                
                # Filter out -1 indices (FAISS padding)
                valid_mask = cand_local_indices >= 0
                cand_local_indices = cand_local_indices[valid_mask]
                cand_cosines = cand_cosines[valid_mask]
                
                if len(cand_local_indices) == 0:
                    all_results[s1_id] = ""
                    continue
                
                # Map local FAISS indices to global target indices
                cand_global_indices = [tgt_indices[ci] for ci in cand_local_indices]
                
                # Compute features
                cand_names = [target_names[gi] for gi in cand_global_indices]
                cand_addrs = [target_addrs[gi] for gi in cand_global_indices]
                
                features = compute_features_batch(
                    [s1_name] * len(cand_names),
                    [s1_addr] * len(cand_addrs),
                    cand_names, cand_addrs,
                    cand_cosines.tolist()
                )
                
                # LightGBM predict
                probs = lgbm_model.predict_proba(features)[:, 1]
                
                # Apply threshold
                matches = []
                for j, p in enumerate(probs):
                    if p >= threshold:
                        matches.append(target_ids[cand_global_indices[j]])
                
                if matches:
                    all_results[s1_id] = ",".join(matches)
                    matched_count += 1
                else:
                    all_results[s1_id] = ""
                
                # Progress
                if (qi + 1) % 50000 == 0:
                    elapsed = time.time() - t_score
                    rate = (qi + 1) / elapsed
                    logger.info(f"  [{country}] {qi+1:,}/{s1_count:,} scored ({rate:.0f}/s) | Matched: {matched_count:,}")
            
            match_rate = matched_count / s1_count if s1_count > 0 else 0
            logger.info(f"  [{country}] DONE in {time.time()-tc:.1f}s | Matched: {matched_count:,}/{s1_count:,} ({match_rate:.1%})")
            
            # Checkpoint
            ckpt = TMP_DIR / f"matching_{country}.tsv"
            with open(ckpt, "w", encoding="utf-8", newline="\n") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for si in s1_indices:
                    sid = s1_ids[si]
                    f.write(f"{sid}\t{all_results.get(sid, '')}\n")
            
            del tgt_emb_c, s1_emb_c, index
            gc.collect()
            
        except Exception as e:
            logger.error(f"  [{country}] CRITICAL FAILURE: {e}", exc_info=True)
            for si in s1_by_country.get(country, []):
                if s1_ids[si] not in all_results:
                    all_results[s1_ids[si]] = ""
    
    # -------------------------------------------------------
    # STEP 4: Write final output
    # -------------------------------------------------------
    t0 = time.time()
    logger.info("\n[4/5] Writing final output...")
    
    # Backfill missing
    missing_count = 0
    for sid in s1_ids:
        if sid not in all_results:
            all_results[sid] = ""
            missing_count += 1
    if missing_count > 0:
        logger.warning(f"  Backfilled {missing_count:,} missing IDs")
    
    final_path = "output/matching_results.tsv"
    with open(final_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{all_results[sid]}\n")
    
    cand_path = "output/candidate_pairs.tsv"
    with open(cand_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{all_results[sid]}\n")
    
    logger.info(f"  Output written in {time.time()-t0:.2f}s")
    
    # -------------------------------------------------------
    # STEP 5: Assertions
    # -------------------------------------------------------
    logger.info("[5/5] Assertions...")
    
    n_written = sum(1 for _ in open(final_path, encoding="utf-8")) - 1
    assert n_written == len(s1_ids), f"INCOMPLETE: {n_written}/{len(s1_ids)}"
    
    written_ids = set()
    with open(final_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            sid = line.split("\t")[0]
            assert sid not in written_ids, f"DUPLICATE: {sid}"
            written_ids.add(sid)
    assert len(written_ids) == len(s1_ids), f"ID MISMATCH: {len(written_ids)}/{len(s1_ids)}"
    
    empty_count = sum(1 for sid in s1_ids if all_results[sid] == "")
    matched_count = len(s1_ids) - empty_count
    empty_rate = empty_count / len(s1_ids)
    
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"  ✅ {len(s1_ids):,} rows | Matched: {matched_count:,} ({matched_count/len(s1_ids):.1%}) | Singletons: {empty_count:,} ({empty_rate:.1%})")
    logger.info(f"  ✅ Total: {total_time:.1f}s")
    logger.info("=" * 80)


# ====================================================================
# FALLBACK: Dict-based blocking (no embeddings needed)
# ====================================================================
def run_fallback_inference():
    """Fast fallback when embeddings aren't cached — uses first-word blocking + RapidFuzz."""
    logger.info("Running FALLBACK inference (no FAISS, dict-based blocking)...")
    
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
    
    logger.info(f"  Loaded S1: {len(s1):,} | Targets: {len(targets):,} in {time.time()-t0:.1f}s")
    
    countries = s1["country"].unique().sort().to_list()
    all_results: Dict[str, str] = {}
    
    for country in countries:
        tc = time.time()
        logger.info(f"\n  [{country}] Processing...")
        
        try:
            s1_c = s1.filter(pl.col("country") == country)
            tgt_c = targets.filter(pl.col("country") == country)
            
            s1_count = len(s1_c)
            tgt_count = len(tgt_c)
            logger.info(f"  [{country}] S1: {s1_count:,} | Targets: {tgt_count:,}")
            
            if tgt_count == 0:
                for sid in s1_c["entity_id"].to_list():
                    all_results[sid] = ""
                continue
            
            # Build inverted index from column lists
            tgt_ids = tgt_c["entity_id"].to_list()
            tgt_names = tgt_c["business_name"].to_list()
            tgt_addrs = tgt_c["business_address"].to_list()
            
            idx = defaultdict(list)
            for i in range(len(tgt_ids)):
                fw_match = FIRST_WORD_RE.match((tgt_names[i] or "").lower())
                fw = fw_match.group(1) if fw_match else ""
                if fw and len(idx[fw]) < 200:
                    idx[fw].append(i)
            
            logger.info(f"  [{country}] Index: {len(idx):,} keys")
            
            # Score
            s1_ids_c = s1_c["entity_id"].to_list()
            s1_names_c = s1_c["business_name"].to_list()
            s1_addrs_c = s1_c["business_address"].to_list()
            
            matched_count = 0
            for i in range(s1_count):
                s1_id = s1_ids_c[i]
                s1_name = (s1_names_c[i] or "").lower().strip()
                s1_addr = (s1_addrs_c[i] or "").lower().strip()
                
                fw_match = FIRST_WORD_RE.match(s1_name)
                fw = fw_match.group(1) if fw_match else ""
                
                cand_indices = idx.get(fw, []) if fw else []
                if not cand_indices:
                    all_results[s1_id] = ""
                    continue
                
                best = []
                for ci in cand_indices:
                    tn = (tgt_names[ci] or "").lower().strip()
                    ta = (tgt_addrs[ci] or "").lower().strip()
                    ns = fuzz.token_sort_ratio(s1_name, tn)
                    if ns < 55:
                        continue
                    asc = fuzz.token_sort_ratio(s1_addr, ta)
                    if ns > 65 and asc > 35:
                        best.append((tgt_ids[ci], ns * 0.7 + asc * 0.3))
                
                if best:
                    best.sort(key=lambda x: -x[1])
                    all_results[s1_id] = ",".join(m[0] for m in best)
                    matched_count += 1
                else:
                    all_results[s1_id] = ""
                
                if (i + 1) % 50000 == 0:
                    elapsed = time.time() - tc
                    rate = (i + 1) / elapsed
                    logger.info(f"  [{country}] {i+1:,}/{s1_count:,} ({rate:.0f}/s) | Matched: {matched_count:,}")
            
            logger.info(f"  [{country}] Done in {time.time()-tc:.1f}s | Matched: {matched_count:,}/{s1_count:,}")
            
            # Checkpoint
            ckpt = TMP_DIR / f"matching_{country}.tsv"
            with open(ckpt, "w", encoding="utf-8", newline="\n") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for sid in s1_ids_c:
                    f.write(f"{sid}\t{all_results.get(sid, '')}\n")
            
            del tgt_ids, tgt_names, tgt_addrs, s1_ids_c, s1_names_c, s1_addrs_c
            gc.collect()
            
        except Exception as e:
            logger.error(f"  [{country}] FAILURE: {e}", exc_info=True)
            try:
                for sid in s1.filter(pl.col("country") == country)["entity_id"].to_list():
                    if sid not in all_results:
                        all_results[sid] = ""
            except Exception:
                pass
    
    # Write final output
    s1_ids_all = s1["entity_id"].to_list()
    for sid in s1_ids_all:
        if sid not in all_results:
            all_results[sid] = ""
    
    final_path = "output/matching_results.tsv"
    with open(final_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids_all:
            f.write(f"{sid}\t{all_results[sid]}\n")
    
    cand_path = "output/candidate_pairs.tsv"
    with open(cand_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids_all:
            f.write(f"{sid}\t{all_results[sid]}\n")
    
    # Assertions
    n_written = sum(1 for _ in open(final_path, encoding="utf-8")) - 1
    assert n_written == len(s1_ids_all), f"INCOMPLETE: {n_written}/{len(s1_ids_all)}"
    
    empty_count = sum(1 for sid in s1_ids_all if all_results[sid] == "")
    matched_count = len(s1_ids_all) - empty_count
    
    total_time = time.time()
    logger.info("=" * 80)
    logger.info(f"  ✅ {len(s1_ids_all):,} rows | Matched: {matched_count:,} | Singletons: {empty_count:,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
