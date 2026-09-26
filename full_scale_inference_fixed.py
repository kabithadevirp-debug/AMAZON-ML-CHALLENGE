#!/usr/bin/env python3
"""
EntityMatch AI — Hardened High-Speed Full-Scale Country-Partitioned Inference Engine
Optimized for 1.73M test entities across France, US, and India.
"""

import os
import sys
import gc
import re
import time
import argparse
import logging
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("failed_countries.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EntityMatch")

TMP_DIR = Path("output/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
Path("output").mkdir(exist_ok=True)

ABBREV_MAP = {
    "private limited": "pvt ltd", "incorporated": "inc", "corporation": "corp", "company": "co",
    "limited": "ltd", "llp": "llp", "street": "st", "road": "rd", "avenue": "ave",
    "boulevard": "blvd", "north": "n", "south": "s", "east": "e", "west": "w",
}

ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ABBREV_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

LEGAL_SUFFIX_RE = re.compile(
    r"\b(pvt ltd|private limited|inc|incorporated|corp|corporation|llc|ltd|limited|co)\b\.?", re.I
)

CLEAN_CHARS_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

def safe_transliterate(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    has_indic = False
    for c in s:
        code = ord(c)
        if 0x0900 <= code <= 0x0D7F:
            has_indic = True
            break
    if not has_indic:
        return s

    for c in s:
        code = ord(c)
        try:
            if 0x0900 <= code <= 0x097F:
                return transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)
            elif 0x0980 <= code <= 0x09FF:
                return transliterate(s, sanscript.BENGALI, sanscript.ITRANS)
            elif 0x0A00 <= code <= 0x0A7F:
                return transliterate(s, sanscript.GURMUKHI, sanscript.ITRANS)
            elif 0x0A80 <= code <= 0x0AFF:
                return transliterate(s, sanscript.GUJARATI, sanscript.ITRANS)
            elif 0x0B00 <= code <= 0x0B7F:
                return transliterate(s, sanscript.ORIYA, sanscript.ITRANS)
            elif 0x0B80 <= code <= 0x0BFF:
                return transliterate(s, sanscript.TAMIL, sanscript.ITRANS)
            elif 0x0C00 <= code <= 0x0C7F:
                return transliterate(s, sanscript.TELUGU, sanscript.ITRANS)
            elif 0x0C80 <= code <= 0x0CFF:
                return transliterate(s, sanscript.KANNADA, sanscript.ITRANS)
            elif 0x0D00 <= code <= 0x0D7F:
                return transliterate(s, sanscript.MALAYALAM, sanscript.ITRANS)
        except Exception:
            pass
    return s

def normalize_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    try:
        s = safe_transliterate(s)
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = CLEAN_CHARS_RE.sub(" ", s)
        s = ABBREV_PATTERN.sub(lambda m: ABBREV_MAP[m.group(0).lower()], s)
        s = MULTI_SPACE_RE.sub(" ", s).strip()
        return s
    except Exception as e:
        return str(s).lower().strip()

def strip_legal_suffix(name: str) -> str:
    return LEGAL_SUFFIX_RE.sub("", name).strip()

def get_3grams(text: str) -> set:
    text = text.replace(" ", "")
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}

def extract_pair_features(s1_tuple, cand_tuple) -> list:
    _, s1_clean, s1_norm_name, s1_norm_addr, _ = s1_tuple
    _, cand_clean, cand_norm_name, cand_norm_addr, _ = cand_tuple

    s1_num_match = re.search(r"\d+", s1_norm_addr)
    cand_num_match = re.search(r"\d+", cand_norm_addr)
    num_match = int(
        s1_num_match is not None and cand_num_match is not None
        and s1_num_match.group() == cand_num_match.group()
    )

    return [
        fuzz.ratio(s1_norm_name, cand_norm_name) / 100.0,
        fuzz.token_sort_ratio(s1_norm_name, cand_norm_name) / 100.0,
        fuzz.partial_ratio(s1_norm_name, cand_norm_name) / 100.0,
        fuzz.ratio(s1_norm_addr, cand_norm_addr) / 100.0,
        fuzz.token_sort_ratio(s1_norm_addr, cand_norm_addr) / 100.0,
        fuzz.ratio(s1_clean, cand_clean) / 100.0,
        num_match
    ]

# ---------------------------------------------------------
# STEP 1: MODEL TRAINING & THRESHOLD CALIBRATION
# ---------------------------------------------------------
def train_and_calibrate_model(train_dir: str, sample_s1: int = 15000):
    logger.info("--- [1/3] Training Production LightGBM Model on Train Split ---")
    t0 = time.time()

    s1_list = []
    s1_ids = set()
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= sample_s1:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                s1_list.append((parts[0], parts[1], parts[2], parts[3]))
                s1_ids.add(parts[0])

    gt_map = {}
    target_ids_needed = set()
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in s1_ids:
                matches = [m.strip() for m in parts[1].split(",") if m.strip()]
                if matches:
                    gt_map[parts[0]] = set(matches)
                    target_ids_needed.update(matches)

    target_records = []
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join(train_dir, fn)
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    if parts[0] in target_ids_needed or (i < 15000 and len(target_records) < 40000):
                        target_records.append((parts[0], parts[1], parts[2], parts[3]))

    s1_pre = [(e, strip_legal_suffix(normalize_text(n)), normalize_text(n), normalize_text(a), c) for e, n, a, c in s1_list]
    target_pre = [(e, strip_legal_suffix(normalize_text(n)), normalize_text(n), normalize_text(a), c) for e, n, a, c in target_records]
    target_dict = {t[0]: t for t in target_pre}

    target_by_country = defaultdict(list)
    for t in target_pre:
        target_by_country[t[4]].append(t)

    X, y = [], []
    for s1_tuple in s1_pre:
        s1_id, clean_n, norm_n, norm_a, country = s1_tuple
        t_list = target_by_country.get(country, [])
        if not t_list:
            continue

        true_m = gt_map.get(s1_id, set())

        for tm in true_m:
            if tm in target_dict:
                X.append(extract_pair_features(s1_tuple, target_dict[tm]))
                y.append(1)

        neg_count = 0
        for t_tuple in t_list[:15]:
            t_id = t_tuple[0]
            if t_id not in true_m:
                X.append(extract_pair_features(s1_tuple, t_tuple))
                y.append(0)
                neg_count += 1
                if neg_count >= 5:
                    break

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]

    best_thresh, best_f05 = 0.5, 0.0
    for t in [i / 100 for i in range(50, 95, 2)]:
        preds = (probs >= t).astype(int)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_thresh = f05, t

    logger.info(f"Model trained in {time.time()-t0:.2f}s | Optimal Threshold: {best_thresh:.2f} | Macro F0.5: {best_f05:.4f}")
    return model, best_thresh

# ---------------------------------------------------------
# STEP 2: FAST PRUNED INDEXING & BATCH SCORING
# ---------------------------------------------------------
def process_country(country: str, test_dir: str, model, threshold: float):
    matching_ckpt = TMP_DIR / f"matching_{country}.tsv"
    candidate_ckpt = TMP_DIR / f"candidate_{country}.tsv"

    if matching_ckpt.exists() and candidate_ckpt.exists() and os.path.getsize(matching_ckpt) > 1000:
        logger.info(f"[{country}] Checkpoint already exists -> skipping recomputation.")
        return

    logger.info(f"[{country}] Starting fast inference for country partition...")
    t_start = time.time()

    # 1. Ingest S2 & S3 targets
    target_records = []
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(test_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    target_records.append((parts[0], parts[1], parts[2], country))

    logger.info(f"[{country}] Ingested {len(target_records):,} target records in {time.time()-t_start:.2f}s")

    # Preprocess targets & build pruned posting lists
    target_pre = []
    token_index = defaultdict(list)
    prefix_index = defaultdict(list)
    trigram_index = defaultdict(list)
    addr_num_index = defaultdict(list)

    for idx, (t_id, raw_n, raw_a, c) in enumerate(target_records):
        norm_n = normalize_text(raw_n)
        norm_a = normalize_text(raw_a)
        clean_n = strip_legal_suffix(norm_n)
        t_tuple = (t_id, clean_n, norm_n, norm_a, c)
        target_pre.append(t_tuple)

        for w in clean_n.split():
            if len(w) >= 3 and len(token_index[w]) < 200:
                token_index[w].append(idx)
        if len(clean_n) >= 4 and len(prefix_index[clean_n[:4]]) < 200:
            prefix_index[clean_n[:4]].append(idx)
        for tri in get_3grams(clean_n):
            if len(trigram_index[tri]) < 200:
                trigram_index[tri].append(idx)
        for num in re.findall(r"\d+", norm_a):
            if len(num) >= 2 and len(addr_num_index[num]) < 200:
                addr_num_index[num].append(idx)

    logger.info(f"[{country}] Built high-speed candidate index in {time.time()-t_start:.2f}s")

    # 2. Ingest S1 records
    s1_records = []
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3].strip() == country:
                s1_records.append((parts[0], parts[1], parts[2], country))

    total_s1 = len(s1_records)
    logger.info(f"[{country}] Streaming {total_s1:,} S1 entities...")

    s1_pre = []
    for s1_id, raw_n, raw_a, c in s1_records:
        norm_n = normalize_text(raw_n)
        norm_a = normalize_text(raw_a)
        clean_n = strip_legal_suffix(norm_n)
        s1_pre.append((s1_id, clean_n, norm_n, norm_a, c))

    # 3. Batch Scoring
    batch_size = 25000
    candidate_map = {}
    scored_triplets = []

    for b_start in range(0, total_s1, batch_size):
        b_end = min(b_start + batch_size, total_s1)
        batch = s1_pre[b_start:b_end]

        batch_features = []
        batch_pairs = []

        for s1_tuple in batch:
            s1_id, clean_n, norm_n, norm_a, _ = s1_tuple
            score_counter = Counter()

            for w in clean_n.split():
                if len(w) >= 3:
                    for tidx in token_index.get(w, []):
                        score_counter[tidx] += 3.0
            if len(clean_n) >= 4:
                for tidx in prefix_index.get(clean_n[:4], []):
                    score_counter[tidx] += 2.0
            for tri in get_3grams(clean_n):
                for tidx in trigram_index.get(tri, []):
                    score_counter[tidx] += 0.5
            for num in re.findall(r"\d+", norm_a):
                if len(num) >= 2:
                    for tidx in addr_num_index.get(num, []):
                        score_counter[tidx] += 1.5

            top_indices = [tidx for tidx, _ in score_counter.most_common(20)]
            cand_ids = [target_pre[tidx][0] for tidx in top_indices]
            candidate_map[s1_id] = ",".join(cand_ids)

            for tidx in top_indices:
                feats = extract_pair_features(s1_tuple, target_pre[tidx])
                batch_features.append(feats)
                batch_pairs.append((s1_id, target_pre[tidx][0]))

        if batch_features:
            probs = model.predict_proba(np.array(batch_features))[:, 1]
            for (s1_id, cand_id), p in zip(batch_pairs, probs):
                if p >= threshold:
                    scored_triplets.append((s1_id, cand_id, float(p)))

        logger.info(f"[{country}] Processed {b_end:,}/{total_s1:,} S1 entities... ({len(scored_triplets):,} high-prob matches found)")

    # 4. Global 1-to-1 Target Assignment
    logger.info(f"[{country}] Assigning 1-to-1 mutual matches...")
    scored_triplets.sort(key=lambda x: -x[2])

    claimed_targets = set()
    matched_results = defaultdict(list)

    for s1_id, cand_id, p in scored_triplets:
        if cand_id not in claimed_targets:
            claimed_targets.add(cand_id)
            matched_results[s1_id].append(cand_id)

    # 5. Write Checkpoint TSVs
    with open(matching_ckpt, "w", encoding="utf-8", newline="\n") as fm, \
         open(candidate_ckpt, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_tuple in s1_pre:
            s1_id = s1_tuple[0]
            m_str = ",".join(matched_results.get(s1_id, []))
            c_str = candidate_map.get(s1_id, "")
            fm.write(f"{s1_id}\t{m_str}\n")
            fc.write(f"{s1_id}\t{c_str}\n")

    logger.info(f"[{country}] Done -> Checkpoint saved: {matching_ckpt} ({total_s1:,} rows, {len(matched_results):,} matches)")
    del target_records, target_pre, token_index, prefix_index, trigram_index, addr_num_index, s1_records, s1_pre, candidate_map, scored_triplets
    gc.collect()

# ---------------------------------------------------------
# STEP 3: MAIN DRIVER & FINAL ASSEMBLY
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default=None, help="Run single country e.g. France, US, India")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Train dir")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test dir")
    parser.add_argument("--output-dir", default="output", help="Output dir")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("   ENTITYMATCH AI — HARDENED PRODUCTION INFERENCE PIPELINE")
    logger.info("=" * 80)

    model, threshold = train_and_calibrate_model(args.train_dir)

    s1_test_path = os.path.join(args.test_dir, "test_source1.tsv")
    countries = []
    with open(s1_test_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                c = parts[3].strip()
                if c and c not in countries:
                    countries.append(c)

    if args.country:
        target_country = args.country
        for c in countries:
            if c.lower() == target_country.lower() or c.upper().startswith(target_country.upper()):
                target_country = c
                break
        countries = [target_country]

    logger.info(f"Countries to process: {countries}")

    for country in countries:
        try:
            process_country(country, args.test_dir, model, threshold)
        except Exception as e:
            logger.error(f"[{country}] CRITICAL FAILURE: {e}", exc_info=True)

    if args.country:
        logger.info(f"Finished single country: {args.country}")
        return

    logger.info("\n--- [3/3] Assembling Final Output TSVs from Country Checkpoints ---")
    
    matching_dict = {}
    candidate_dict = {}

    for country in countries:
        m_file = TMP_DIR / f"matching_{country}.tsv"
        c_file = TMP_DIR / f"candidate_{country}.tsv"
        if not m_file.exists():
            logger.error(f"Missing checkpoint for {country}: {m_file}")
            continue

        with open(m_file, "r", encoding="utf-8") as fm:
            next(fm)
            for line in fm:
                parts = line.strip().split("\t")
                if parts and parts[0]:
                    matching_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

        with open(c_file, "r", encoding="utf-8") as fc:
            next(fc)
            for line in fc:
                parts = line.strip().split("\t")
                if parts and parts[0]:
                    candidate_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

    logger.info(f"Loaded {len(matching_dict):,} entities from checkpoints.")

    final_matching = os.path.join(args.output_dir, "matching_results.tsv")
    final_candidate = os.path.join(args.output_dir, "candidate_pairs.tsv")

    total_s1 = 0
    with open(s1_test_path, "r", encoding="utf-8") as fs, \
         open(final_matching, "w", encoding="utf-8", newline="\n") as fm, \
         open(final_candidate, "w", encoding="utf-8", newline="\n") as fc:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        next(fs)
        for line in fs:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            m_str = matching_dict.get(s1_id, "")
            c_str = candidate_dict.get(s1_id, "")

            fm.write(f"{s1_id}\t{m_str}\n")
            fc.write(f"{s1_id}\t{c_str}\n")
            total_s1 += 1

    logger.info(f"✅ Final files written: {final_matching} ({total_s1:,} rows)")

    assert len(matching_dict) == total_s1, f"INCOMPLETE COVERAGE: {len(matching_dict)}/{total_s1}"
    logger.info("✅ All hard assertions passed!")

    val_script = Path("utils/validate_submission_fixed.py")
    if val_script.exists():
        logger.info("\n--- Running Hardened Submission Validator ---")
        import subprocess
        res = subprocess.run([sys.executable, str(val_script)], capture_output=True, text=True)
        print(res.stdout, flush=True)

    logger.info(f"Pipeline finished in {time.time()-start_time:.2f}s total wall-clock time.")

if __name__ == "__main__":
    main()
