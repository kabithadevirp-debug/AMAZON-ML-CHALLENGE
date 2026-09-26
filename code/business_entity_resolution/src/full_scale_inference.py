#!/usr/bin/env python3
"""
EntityMatch AI — High-Performance Full-Scale Country-Partitioned Inference Engine
Optimized for 1.73M test entities across France, US, and India.
Maximizes Macro F0.5 with high recall multi-pass blocking, dense feature extraction,
calibrated thresholding, and mutual best target assignment.
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessor import normalize_business_name, normalize_address
from feature_extractor import extract_pair_features
from matching_model import EntityMatchingModel, evaluate_macro_f05, resolve_global_matches

def train_production_model(train_dir: str, sample_s1: int = 25000) -> Tuple[EntityMatchingModel, float]:
    """Trains matching model on a representative training sample and calibrates threshold for Macro F0.5."""
    print("--- [1/4] Training Production Model & Calibrating Macro F0.5 ---", flush=True)
    t0 = time.time()
    
    # Load ground truth
    gt_map = {}
    with open(os.path.join(train_dir, 'train_ground_truth.tsv'), 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= sample_s1:
                break
            parts = line.strip().split('\t')
            if parts and parts[0]:
                mids = {x.strip() for x in parts[1].split(',') if x.strip()} if len(parts) > 1 and parts[1] else set()
                gt_map[parts[0]] = mids

    all_s1_ids = list(gt_map.keys())
    split_idx = int(len(all_s1_ids) * 0.7)
    train_ids = set(all_s1_ids[:split_idx])
    val_ids = set(all_s1_ids[split_idx:])

    needed_targets = set()
    for mids in gt_map.values():
        needed_targets.update(mids)

    # Load S1 records
    s1_records = {}
    with open(os.path.join(train_dir, 'train_source1.tsv'), 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0] in gt_map:
                s1_records[parts[0]] = {
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3].strip().upper() if len(parts) > 3 else 'US'
                }
                if len(s1_records) >= len(gt_map):
                    break

    # Load S2 and S3 targets
    target_records = {}
    for fname in ['train_source2.tsv', 'train_source3.tsv']:
        with open(os.path.join(train_dir, fname), 'r', encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                if parts:
                    tid = parts[0]
                    if tid in needed_targets or i < 40000:
                        target_records[tid] = {
                            'entity_id': tid,
                            'business_name': parts[1] if len(parts) > 1 else '',
                            'business_address': parts[2] if len(parts) > 2 else '',
                            'country': parts[3].strip().upper() if len(parts) > 3 else 'US'
                        }

    # Index targets
    from blocking_engine import BlockingEngine
    blocking = BlockingEngine(max_candidates_per_s1=20)
    blocking.index_target_records(list(target_records.values()))

    # Build Training Pairs
    X_train, y_train = [], []
    for s1_id in train_ids:
        rec = s1_records[s1_id]
        true_m = gt_map[s1_id]
        cands = blocking.generate_candidates_for_s1(rec)
        
        n_n, n_t, n_c = normalize_business_name(rec['business_name'])
        a_n, a_t, a_d = normalize_address(rec['business_address'])
        s1_meta = {
            'country': rec['country'], 'norm_name': n_n, 'name_tokens': set(n_t),
            'core_tokens': set(n_c), 'norm_addr': a_n, 'addr_tokens': set(a_t), 'digits': a_d
        }
        for cid in cands:
            t_meta = blocking.target_records.get(cid)
            if t_meta:
                X_train.append(extract_pair_features(s1_meta, t_meta))
                y_train.append(1 if cid in true_m else 0)

    model = EntityMatchingModel()
    model.train(np.array(X_train), np.array(y_train))
    print(f"  Trained on {len(X_train):,} pairs (Positives: {sum(y_train):,}) in {time.time()-t0:.2f}s", flush=True)

    # Validate and tune threshold
    val_features, val_pair_ids = [], []
    val_gt = {k: gt_map[k] for k in val_ids}
    for s1_id in val_ids:
        rec = s1_records[s1_id]
        cands = blocking.generate_candidates_for_s1(rec)
        n_n, n_t, n_c = normalize_business_name(rec['business_name'])
        a_n, a_t, a_d = normalize_address(rec['business_address'])
        s1_meta = {
            'country': rec['country'], 'norm_name': n_n, 'name_tokens': set(n_t),
            'core_tokens': set(n_c), 'norm_addr': a_n, 'addr_tokens': set(a_t), 'digits': a_d
        }
        for cid in cands:
            t_meta = blocking.target_records.get(cid)
            if t_meta:
                val_features.append(extract_pair_features(s1_meta, t_meta))
                val_pair_ids.append((s1_id, cid))

    best_th = model.tune_threshold(val_features, val_pair_ids, val_gt)
    
    val_probs = model.predict_pair_probs(np.array(val_features))
    triplets = [(s1, t, float(p)) for (s1, t), p in zip(val_pair_ids, val_probs)]
    val_preds = resolve_global_matches(triplets, val_ids, threshold=best_th)
    f05, p, r = evaluate_macro_f05(val_gt, val_preds)
    print(f"  Calibrated Threshold: {best_th:.3f} | Val Macro F0.5: {f05:.4f} (Precision: {p*100:.2f}%, Recall: {r*100:.2f}%)", flush=True)
    
    del blocking, target_records, s1_records, X_train, y_train, val_features
    gc.collect()
    return model, best_th

def process_country_partition(
    country_name: str,
    test_dir: str,
    model: EntityMatchingModel,
    threshold: float,
    out_matches: Dict[str, str],
    out_candidates: Dict[str, str]
):
    """Processes all entities for a specific country partition in test."""
    print(f"\n--- [Country: {country_name}] Ingesting and Indexing S2 & S3 Target Records ---", flush=True)
    t_start = time.time()
    
    target_records = {}
    
    # Ingest S2 and S3 for this country
    for src_file in ['test_source2.tsv', 'test_source3.tsv']:
        path = os.path.join(test_dir, src_file)
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3].strip().upper() == country_name:
                    target_records[parts[0]] = {
                        'entity_id': parts[0],
                        'business_name': parts[1] if len(parts) > 1 else '',
                        'business_address': parts[2] if len(parts) > 2 else '',
                        'country': country_name
                    }

    print(f"  Total {country_name} Target Records (S2+S3): {len(target_records):,} in {time.time()-t_start:.2f}s", flush=True)

    # Build Blocking Inverted Index
    from blocking_engine import BlockingEngine
    blocking = BlockingEngine(max_candidates_per_s1=20)
    blocking.index_target_records(list(target_records.values()))
    print(f"  Built Inverted Index in {time.time()-t_start:.2f}s", flush=True)

    # Stream S1 records for this country
    print(f"  Streaming S1 Records and Generating Candidates...", flush=True)
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s1_batch = []
    batch_size = 50000
    
    country_triplets = []
    country_s1_ids = []

    def evaluate_batch(batch):
        batch_features = []
        batch_pair_ids = []
        
        for s1_rec in batch:
            s1_id = s1_rec['entity_id']
            country_s1_ids.append(s1_id)
            
            cands = blocking.generate_candidates_for_s1(s1_rec)
            out_candidates[s1_id] = ",".join(cands) if cands else ""
            
            if not cands:
                continue
                
            n_n, n_t, n_c = normalize_business_name(s1_rec['business_name'])
            a_n, a_t, a_d = normalize_address(s1_rec['business_address'])
            s1_meta = {
                'country': country_name, 'norm_name': n_n, 'name_tokens': set(n_t),
                'core_tokens': set(n_c), 'norm_addr': a_n, 'addr_tokens': set(a_t), 'digits': a_d
            }
            
            for cid in cands:
                t_meta = blocking.target_records.get(cid)
                if t_meta:
                    batch_features.append(extract_pair_features(s1_meta, t_meta))
                    batch_pair_ids.append((s1_id, cid))
                    
        if batch_features:
            probs = model.predict_pair_probs(np.array(batch_features))
            for (s1_id, cid), prob in zip(batch_pair_ids, probs):
                if prob >= threshold:
                    country_triplets.append((s1_id, cid, float(prob)))

    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[3].strip().upper() == country_name:
                s1_batch.append({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': country_name
                })
                if len(s1_batch) >= batch_size:
                    evaluate_batch(s1_batch)
                    s1_batch = []
                    print(f"    Processed {len(country_s1_ids):,} S1 entities... (found {len(country_triplets):,} candidate links)", flush=True)

        if s1_batch:
            evaluate_batch(s1_batch)

    print(f"  Total S1 Entities Processed: {len(country_s1_ids):,} | High-confidence Candidate Links: {len(country_triplets):,}", flush=True)

    # Resolve Global Mutual Best Matches
    print(f"  Resolving Global 1-to-1 Target Matches for {country_name}...", flush=True)
    preds_map = resolve_global_matches(country_triplets, set(country_s1_ids), threshold=threshold)
    
    non_empty = 0
    for s1_id, matches in preds_map.items():
        if matches:
            non_empty += 1
            out_matches[s1_id] = ",".join(sorted(matches))
        else:
            out_matches[s1_id] = ""

    print(f"  Finished {country_name} in {time.time()-t_start:.2f}s (Matches generated for {non_empty:,}/{len(country_s1_ids):,} entities)", flush=True)
    
    del blocking, target_records, country_triplets, country_s1_ids, preds_map
    gc.collect()

def run_full_inference(train_dir: str, test_dir: str, output_dir: str):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80, flush=True)
    print("   ENTITYMATCH AI — PRODUCTION FULL-SCALE TEST INFERENCE ENGINE", flush=True)
    print("=" * 80, flush=True)

    # 1. Train Model
    model, threshold = train_production_model(train_dir, sample_s1=30000)

    # Global storage for results
    all_matches = {}
    all_candidates = {}

    # 2. Process Country by Country (France -> US -> India)
    countries = ['FRANCE', 'US', 'INDIA']
    for country in countries:
        process_country_partition(country, test_dir, model, threshold, all_matches, all_candidates)

    # 3. Export all 1,732,544 rows in exact test_source1.tsv order
    print("\n--- [4/4] Writing Final 1.73M Row Output TSVs in Exact Test Order ---", flush=True)
    t_out = time.time()
    
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    test_s1_path = os.path.join(test_dir, 'test_source1.tsv')

    with open(matching_path, 'w', encoding='utf-8', newline='\n') as fm, \
         open(candidate_path, 'w', encoding='utf-8', newline='\n') as fc, \
         open(test_s1_path, 'r', encoding='utf-8') as fs:
        
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
        fs.readline()
        total_written = 0
        for line in fs:
            if not line.strip():
                continue
            s1_id = line.split('\t', 1)[0].strip()
            
            m_str = all_matches.get(s1_id, "")
            c_str = all_candidates.get(s1_id, "")
            
            fm.write(f"{s1_id}\t{m_str}\n")
            fc.write(f"{s1_id}\t{c_str}\n")
            total_written += 1

    print(f"  Successfully wrote all {total_written:,} entities to TSVs in {time.time()-t_out:.2f}s", flush=True)
    print(f"  Total Pipeline Wall-Clock Time: {time.time()-start_time:.2f}s", flush=True)

    # Run official validator
    validator_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'student_resource', 'utils', 'validate_submission.py')
    if os.path.exists(validator_path):
        print("\n--- Running Official Submission Validator ---", flush=True)
        import subprocess
        cmd = [
            sys.executable, validator_path,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print(res.stderr, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Scale Production Inference")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Train dir")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test dir")
    parser.add_argument("--output-dir", default="output", help="Output dir")
    args = parser.parse_args()

    run_full_inference(args.train_dir, args.test_dir, args.output_dir)
