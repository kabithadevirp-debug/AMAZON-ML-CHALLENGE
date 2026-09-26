import os
import threading
import subprocess
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from backend.services.storage import dataset_manager, OUTPUT_DIR, BASE_DIR
from backend.services.preprocessor import normalize_business_name, normalize_address
from backend.services.blocking_engine import BlockingEngine
from backend.services.feature_extractor import extract_pair_features
from backend.services.matching_model import EntityMatchingModel, evaluate_macro_f05

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

class RunPipelineRequest(BaseModel):
    max_s1_records: int = 20000
    candidate_cap: int = 20
    confidence_threshold: float = 0.75

# Global in-memory pipeline state
pipeline_state = {
    "status": "idle", # idle, normalizing, blocking, features, matching, done, error
    "progress_percent": 0,
    "current_step": "Ready to execute pipeline",
    "metrics": {
        "s1_processed": 0,
        "candidate_pairs": 0,
        "confirmed_matches": 0,
        "singletons": 0,
        "blocking_recall": 0.0,
        "macro_f05": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": 0.75
    },
    "logs": []
}

# In-memory store of inference results for interactive review
inference_results = {
    "records": [],
    "threshold": 0.75
}

def log_msg(msg: str):
    pipeline_state["logs"].append(msg)
    pipeline_state["current_step"] = msg

def execute_pipeline_task(req: RunPipelineRequest):
    try:
        pipeline_state["status"] = "normalizing"
        pipeline_state["progress_percent"] = 10
        log_msg("Step 1: Normalizing reference and target source records...")

        s1_file = dataset_manager.sources["source1"]["file_path"]
        s2_file = dataset_manager.sources["source2"]["file_path"]
        s3_file = dataset_manager.sources["source3"]["file_path"]
        gt_file = dataset_manager.sources["ground_truth"]["file_path"]

        if not s1_file or not s2_file:
            log_msg("Error: Source 1 and Source 2 files must be loaded.")
            pipeline_state["status"] = "error"
            return

        # Load S1 records
        from code.business_entity_resolution.src.run_pipeline import load_source_records, load_ground_truth
        s1_records = load_source_records(s1_file, max_rows=req.max_s1_records)
        s2_records = load_source_records(s2_file, max_rows=req.max_s1_records * 2)
        s3_records = load_source_records(s3_file, max_rows=req.max_s1_records * 2) if s3_file and os.path.exists(s3_file) else []
        gt_map = load_ground_truth(gt_file, max_rows=req.max_s1_records) if gt_file and os.path.exists(gt_file) else {}

        # Step 2: Blocking
        pipeline_state["status"] = "blocking"
        pipeline_state["progress_percent"] = 30
        log_msg(f"Step 2: Building multi-pass candidate blocking index over {len(s2_records)+len(s3_records):,} target records...")

        blocking = BlockingEngine(max_candidates_per_s1=req.candidate_cap)
        blocking.index_target_records(s2_records)
        if s3_records:
            blocking.index_target_records(s3_records)

        # Step 3: Feature extraction
        pipeline_state["status"] = "features"
        pipeline_state["progress_percent"] = 60
        log_msg("Step 3: Generating candidate pairs and computing similarity feature vectors...")

        s1_meta_map = {}
        candidate_rows = []
        X_pairs = []
        pair_ids = []
        y_labels = []
        total_candidates = 0

        for s1 in s1_records:
            s1_id = s1['entity_id']
            norm_name, name_tokens, core_tokens = normalize_business_name(s1.get('business_name', ''))
            norm_addr, addr_tokens, digits = normalize_address(s1.get('business_address', ''))
            s1_meta = {
                'country': s1.get('country', 'US').strip().upper(),
                'norm_name': norm_name,
                'name_tokens': set(name_tokens),
                'core_tokens': set(core_tokens),
                'norm_addr': norm_addr,
                'addr_tokens': set(addr_tokens),
                'digits': digits
            }
            s1_meta_map[s1_id] = s1_meta

            cands = blocking.generate_candidates_for_s1(s1)
            total_candidates += len(cands)
            candidate_rows.append(f"{s1_id}\t{','.join(cands)}\n")

            true_mids = gt_map.get(s1_id, set())

            for cid in cands:
                t_meta = blocking.target_records.get(cid)
                if t_meta:
                    feats = extract_pair_features(s1_meta, t_meta)
                    X_pairs.append(feats)
                    pair_ids.append((s1_id, cid))
                    if gt_map:
                        y_labels.append(1 if cid in true_mids else 0)

        pipeline_state["metrics"]["candidate_pairs"] = total_candidates
        dataset_manager.pipeline_stats["candidate_pairs"] = total_candidates

        # Step 4: Matching & Threshold Optimization
        pipeline_state["status"] = "matching"
        pipeline_state["progress_percent"] = 80
        log_msg("Step 4: Training precision classifier and tuning Macro F0.5 decision boundary...")

        matcher = EntityMatchingModel()
        if X_pairs and y_labels and sum(y_labels) > 0:
            matcher.train(np.array(X_pairs), np.array(y_labels))
            best_th = matcher.tune_threshold(X_pairs, pair_ids, gt_map)
        else:
            best_th = req.confidence_threshold
            matcher.decision_threshold = best_th

        log_msg(f"Optimal decision threshold calibrated to: {best_th:.3f}")
        pipeline_state["metrics"]["threshold"] = float(best_th)

        # Infer probabilities
        matching_rows = []
        detailed_records = []
        confirmed_matches_count = 0
        singleton_count = 0

        if X_pairs:
            probs = matcher.predict_pair_probs(np.array(X_pairs))
        else:
            probs = []

        preds_by_s1: Dict[str, List[Dict[str, Any]]] = {s1['entity_id']: [] for s1 in s1_records}
        for (s1_id, cid), prob in zip(pair_ids, probs):
            target_meta = blocking.target_records.get(cid, {})
            preds_by_s1[s1_id].append({
                "target_id": cid,
                "confidence": float(prob),
                "norm_name": target_meta.get("norm_name", ""),
                "norm_addr": target_meta.get("norm_addr", ""),
                "is_match": bool(prob >= best_th)
            })

        for s1 in s1_records:
            s1_id = s1['entity_id']
            cand_matches = preds_by_s1.get(s1_id, [])
            matched_cids = [m['target_id'] for m in cand_matches if m['is_match']]
            
            matching_rows.append(f"{s1_id}\t{','.join(matched_cids)}\n")
            
            if matched_cids:
                confirmed_matches_count += len(matched_cids)
            else:
                singleton_count += 1

            s1_meta = s1_meta_map.get(s1_id, {})
            detailed_records.append({
                "s1_id": s1_id,
                "business_name": s1.get("business_name", ""),
                "business_address": s1.get("business_address", ""),
                "country": s1.get("country", ""),
                "matched_ids": matched_cids,
                "status": "matched" if matched_cids else "singleton",
                "candidates": sorted(cand_matches, key=lambda x: x['confidence'], reverse=True)
            })

        inference_results["records"] = detailed_records
        inference_results["threshold"] = float(best_th)

        # Step 5: Export TSVs
        log_msg("Step 5: Writing candidate_pairs.tsv and matching_results.tsv to output/...")
        cand_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")

        with open(cand_out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.writelines(candidate_rows)

        with open(match_out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.writelines(matching_rows)

        # Update stats
        pipeline_state["metrics"]["s1_processed"] = len(s1_records)
        pipeline_state["metrics"]["confirmed_matches"] = confirmed_matches_count
        pipeline_state["metrics"]["singletons"] = singleton_count
        pipeline_state["metrics"]["avg_confidence"] = float(np.mean(probs)) if len(probs) > 0 else 0.0

        dataset_manager.pipeline_stats["s1_count"] = len(s1_records)
        dataset_manager.pipeline_stats["candidate_pairs"] = total_candidates
        dataset_manager.pipeline_stats["confirmed_matches"] = confirmed_matches_count
        dataset_manager.pipeline_stats["singletons"] = singleton_count
        dataset_manager.pipeline_stats["avg_confidence"] = pipeline_state["metrics"]["avg_confidence"]

        if gt_map:
            preds_set_map = {r['s1_id']: set(r['matched_ids']) for r in detailed_records}
            f05, p, r = evaluate_macro_f05(gt_map, preds_set_map)
            pipeline_state["metrics"]["macro_f05"] = f05
            pipeline_state["metrics"]["precision"] = p
            pipeline_state["metrics"]["recall"] = r
            dataset_manager.pipeline_stats["macro_f05"] = f05

        pipeline_state["status"] = "done"
        pipeline_state["progress_percent"] = 100
        dataset_manager.pipeline_stats["pipeline_stage"] = "done"
        log_msg("Pipeline execution finished successfully. Output TSVs ready for submission.")

    except Exception as e:
        pipeline_state["status"] = "error"
        log_msg(f"Pipeline error: {str(e)}")

@router.post("/run")
def start_pipeline_run(req: RunPipelineRequest, background_tasks: BackgroundTasks):
    if pipeline_state["status"] in ["normalizing", "blocking", "features", "matching"]:
        raise HTTPException(status_code=400, detail="A pipeline run is already in progress.")
    
    pipeline_state["status"] = "normalizing"
    pipeline_state["progress_percent"] = 5
    pipeline_state["logs"] = []
    dataset_manager.pipeline_stats["pipeline_stage"] = "normalizing"

    background_tasks.add_task(execute_pipeline_task, req)
    return {"status": "started", "message": "Pipeline run dispatched."}

@router.get("/status")
def get_pipeline_status():
    return pipeline_state
