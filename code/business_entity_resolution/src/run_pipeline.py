#!/usr/bin/env python3
"""
EntityMatch AI — End-to-End Business Entity Resolution Pipeline
Processes train and test datasets, performs multi-pass blocking, feature engineering,
trains a precision-weighted classifier, and writes valid candidate_pairs.tsv and matching_results.tsv.
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from typing import Dict, List, Set, Tuple

# Local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessor import normalize_business_name, normalize_address
from blocking_engine import BlockingEngine
from feature_extractor import extract_pair_features
from matching_model import EntityMatchingModel, evaluate_macro_f05, resolve_global_matches

def load_source_records_streaming(file_path: str, filter_ids: Set[str] = None, max_rows: int = None) -> List[Dict[str, str]]:
    """Fast streaming TSV reader."""
    records = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for i, line in enumerate(f):
            if max_rows and len(records) >= max_rows:
                break
            if not line.strip():
                continue
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == len(header):
                if filter_ids is None or parts[0] in filter_ids:
                    records.append(dict(zip(header, parts)))
    return records

def load_ground_truth(file_path: str, max_rows: int = None) -> Dict[str, Set[str]]:
    """Loads ground truth mapping: S1_id -> set(matched S2/S3 IDs)."""
    gt_map = {}
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for i, line in enumerate(f):
            if max_rows and len(gt_map) >= max_rows:
                break
            if not line.strip():
                continue
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                mids = {x.strip() for x in parts[1].split(',') if x.strip()}
            else:
                mids = set()
            gt_map[s1_id] = mids
    return gt_map

def run_pipeline(train_dir: str, test_dir: str, output_dir: str, sample_size: int = None):
    print("=" * 70, flush=True)
    print("      EntityMatch AI — End-to-End Business Entity Resolution Pipeline", flush=True)
    print("=" * 70, flush=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Training Ground Truth & Targeted Records
    print("\n[Step 1/5] Ingesting and Preprocessing Training Records...", flush=True)
    train_s1_path = os.path.join(train_dir, 'train_source1.tsv')
    train_s2_path = os.path.join(train_dir, 'train_source2.tsv')
    train_s3_path = os.path.join(train_dir, 'train_source3.tsv')
    train_gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')

    train_gt = load_ground_truth(train_gt_path, max_rows=sample_size or 50000)
    train_s1_ids = set(train_gt.keys())
    
    needed_target_ids = set()
    for mids in train_gt.values():
        needed_target_ids.update(mids)

    print(f"  Ground Truth: {len(train_gt):,} S1 entities ({len(needed_target_ids):,} true match links needed)", flush=True)

    train_s1 = load_source_records_streaming(train_s1_path, filter_ids=train_s1_ids)
    
    # Read target records: matching IDs + background noise records
    train_s2 = load_source_records_streaming(train_s2_path, filter_ids=needed_target_ids, max_rows=sample_size * 3 if sample_size else 150000)
    train_s3 = load_source_records_streaming(train_s3_path, filter_ids=needed_target_ids, max_rows=sample_size * 3 if sample_size else 150000)

    # If some needed IDs weren't captured in early rows, do a quick pass for the remaining true IDs
    found_target_ids = {r['entity_id'] for r in train_s2}.union({r['entity_id'] for r in train_s3})
    missing_true_ids = needed_target_ids - found_target_ids
    if missing_true_ids:
        print(f"  Fetching remaining {len(missing_true_ids)} true match records from S2/S3...", flush=True)
        extra_s2 = load_source_records_streaming(train_s2_path, filter_ids=missing_true_ids)
        extra_s3 = load_source_records_streaming(train_s3_path, filter_ids=missing_true_ids)
        train_s2.extend(extra_s2)
        train_s3.extend(extra_s3)

    print(f"  Total Indexed Training Pool: {len(train_s2):,} (S2) + {len(train_s3):,} (S3) records", flush=True)

    # 2. Multi-Pass Blocking on Training Split
    print("\n[Step 2/5] Building Inverted Index & Generating Candidate Pairs...", flush=True)
    train_blocking = BlockingEngine(max_candidates_per_s1=20)
    train_blocking.index_target_records(train_s2)
    train_blocking.index_target_records(train_s3)

    train_s1_meta = {}
    for rec in train_s1:
        s1_id = rec['entity_id']
        norm_name, name_tokens, core_tokens = normalize_business_name(rec.get('business_name', ''))
        norm_addr, addr_tokens, digits = normalize_address(rec.get('business_address', ''))
        train_s1_meta[s1_id] = {
            'country': rec.get('country', 'US').strip().upper(),
            'norm_name': norm_name,
            'name_tokens': set(name_tokens),
            'core_tokens': set(core_tokens),
            'norm_addr': norm_addr,
            'addr_tokens': set(addr_tokens),
            'digits': digits
        }

    X_train = []
    y_train = []
    train_candidates_found = 0
    train_true_matches_captured = 0
    total_true_matches = 0

    for rec in train_s1:
        s1_id = rec['entity_id']
        true_mids = train_gt.get(s1_id, set())
        total_true_matches += len(true_mids)
        
        cands = train_blocking.generate_candidates_for_s1(rec)
        train_candidates_found += len(cands)
        
        cand_set = set(cands)
        train_true_matches_captured += len(true_mids.intersection(cand_set))
        
        for cand_id in cands:
            target_meta = train_blocking.target_records.get(cand_id)
            if not target_meta:
                continue
            feats = extract_pair_features(train_s1_meta[s1_id], target_meta)
            is_match = 1 if cand_id in true_mids else 0
            X_train.append(feats)
            y_train.append(is_match)

    blocking_recall = (train_true_matches_captured / total_true_matches * 100) if total_true_matches > 0 else 0.0
    print(f"  Blocking Recall on Train: {blocking_recall:.2f}% ({train_true_matches_captured}/{total_true_matches} links captured)", flush=True)
    print(f"  Training feature pairs extracted: {len(X_train):,} (Matches: {sum(y_train):,}, Non-matches: {len(y_train) - sum(y_train):,})", flush=True)

    # 3. Model Training & Macro F0.5 Threshold Tuning
    print("\n[Step 3/5] Training High-Precision Classifier & Tuning Macro F0.5 Decision Boundary...", flush=True)
    matcher = EntityMatchingModel()
    matcher.train(np.array(X_train), np.array(y_train))

    train_pair_ids = []
    for rec in train_s1:
        s1_id = rec['entity_id']
        cands = train_blocking.generate_candidates_for_s1(rec)
        for cand_id in cands:
            if cand_id in train_blocking.target_records:
                train_pair_ids.append((s1_id, cand_id))

    best_th = matcher.tune_threshold(X_train, train_pair_ids, train_gt)
    print(f"  Optimized Decision Threshold: {best_th:.3f}", flush=True)

    probs = matcher.predict_pair_probs(np.array(X_train))
    candidate_triplets = [(s1_id, target_id, float(p)) for (s1_id, target_id), p in zip(train_pair_ids, probs)]
    preds_map = resolve_global_matches(candidate_triplets, train_s1_ids, threshold=best_th)

    macro_f05, prec, rec = evaluate_macro_f05(train_gt, preds_map)
    print(f"  Training Validation Macro F0.5 : {macro_f05:.4f}", flush=True)
    print(f"  Training Validation Precision  : {prec:.4f}", flush=True)
    print(f"  Training Validation Recall     : {rec:.4f}", flush=True)

    # 4. Run Test Set Inference
    print("\n[Step 4/5] Running Multi-Pass Blocking & Matching on Test Set...", flush=True)
    test_s1_path = os.path.join(test_dir, 'test_source1.tsv')
    test_s2_path = os.path.join(test_dir, 'test_source2.tsv')
    test_s3_path = os.path.join(test_dir, 'test_source3.tsv')

    test_s1 = load_source_records_streaming(test_s1_path, max_rows=sample_size or 50000)
    test_s2 = load_source_records_streaming(test_s2_path, max_rows=sample_size * 3 if sample_size else 150000)
    test_s3 = load_source_records_streaming(test_s3_path, max_rows=sample_size * 3 if sample_size else 150000)

    print(f"  Loaded Test S1: {len(test_s1):,} records", flush=True)
    print(f"  Loaded Test S2: {len(test_s2):,} records", flush=True)
    print(f"  Loaded Test S3: {len(test_s3):,} records", flush=True)

    test_blocking = BlockingEngine(max_candidates_per_s1=20)
    test_blocking.index_target_records(test_s2)
    test_blocking.index_target_records(test_s3)

    test_s1_ids = [r['entity_id'] for r in test_s1]
    test_candidate_dict = {}
    test_pair_features = []
    test_pair_ids = []

    for s1_rec in test_s1:
        s1_id = s1_rec['entity_id']
        norm_name, name_tokens, core_tokens = normalize_business_name(s1_rec.get('business_name', ''))
        norm_addr, addr_tokens, digits = normalize_address(s1_rec.get('business_address', ''))
        s1_meta = {
            'country': s1_rec.get('country', 'US').strip().upper(),
            'norm_name': norm_name,
            'name_tokens': set(name_tokens),
            'core_tokens': set(core_tokens),
            'norm_addr': norm_addr,
            'addr_tokens': set(addr_tokens),
            'digits': digits
        }

        candidates = test_blocking.generate_candidates_for_s1(s1_rec)
        test_candidate_dict[s1_id] = candidates

        if candidates:
            for cid in candidates:
                t_meta = test_blocking.target_records.get(cid)
                if t_meta:
                    test_pair_features.append(extract_pair_features(s1_meta, t_meta))
                    test_pair_ids.append((s1_id, cid))

    test_triplets = []
    if test_pair_features:
        test_probs = matcher.predict_pair_probs(np.array(test_pair_features))
        test_triplets = [(s1, t, float(p)) for (s1, t), p in zip(test_pair_ids, test_probs)]

    test_predictions_map = resolve_global_matches(test_triplets, set(test_s1_ids), threshold=best_th)

    test_candidate_rows = []
    test_matching_rows = []

    for s1_id in test_s1_ids:
        cands = test_candidate_dict.get(s1_id, [])
        cand_str = ",".join(cands) if cands else ""
        test_candidate_rows.append(f"{s1_id}\t{cand_str}\n")

        matched = sorted(test_predictions_map.get(s1_id, set()))
        match_str = ",".join(matched) if matched else ""
        test_matching_rows.append(f"{s1_id}\t{match_str}\n")

    # 5. Export Output TSV Files
    print("\n[Step 5/5] Exporting TSV Output Files...", flush=True)
    candidate_output_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    matching_output_path = os.path.join(output_dir, 'matching_results.tsv')

    with open(candidate_output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.writelines(test_candidate_rows)

    with open(matching_output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(test_matching_rows)

    print(f"  Candidate Pairs exported: {candidate_output_path}", flush=True)
    print(f"  Matching Results exported: {matching_output_path}", flush=True)

    # Validate output files using validate_submission.py
    validator_script = os.path.join(os.path.dirname(output_dir), 'student_resource', 'utils', 'validate_submission.py')
    if not os.path.exists(validator_script):
        validator_script = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'student_resource', 'utils', 'validate_submission.py')

    if os.path.exists(validator_script):
        print("\n--- Running Automated Submission Validator ---", flush=True)
        cmd = [
            sys.executable,
            validator_script,
            "--matching", matching_output_path,
            "--candidate", candidate_output_path,
            "--test-dir", test_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        if result.returncode == 0:
            print("\n>>> VALIDATION SUCCESS: Safe to submit to leaderboard! <<<", flush=True)
        else:
            print(f"\n>>> VALIDATION RESULT (Code {result.returncode}) <<<", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EntityMatch AI ML Pipeline")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Path to train dataset folder")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Path to test dataset folder")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of records to sample for training/eval")
    args = parser.parse_args()

    run_pipeline(args.train_dir, args.test_dir, args.output_dir, sample_size=args.sample_size)
