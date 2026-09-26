import os
import io
import zipfile
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from backend.services.storage import OUTPUT_DIR, BASE_DIR
from backend.routers.pipeline import inference_results, pipeline_state

router = APIRouter(prefix="/api/results", tags=["results"])

class ReThresholdRequest(BaseModel):
    threshold: float

class ManualDecisionRequest(BaseModel):
    s1_id: str
    target_id: str
    action: str # "accept" or "reject"

audit_log: List[Dict[str, Any]] = []

@router.get("/overview")
def get_results_overview():
    records = inference_results.get("records", [])
    threshold = inference_results.get("threshold", 0.75)
    
    total_s1 = len(records)
    matched_count = sum(1 for r in records if r["status"] == "matched")
    singleton_count = total_s1 - matched_count
    
    # Borderline items (40% to 70% confidence)
    borderline_records = []
    for r in records:
        for c in r.get("candidates", []):
            if 0.40 <= c["confidence"] < threshold:
                borderline_records.append({
                    "s1_id": r["s1_id"],
                    "business_name": r["business_name"],
                    "business_address": r["business_address"],
                    "country": r["country"],
                    "candidate": c
                })
                break
                
    return {
        "total_s1": total_s1,
        "matched_count": matched_count,
        "singleton_count": singleton_count,
        "current_threshold": threshold,
        "borderline_count": len(borderline_records),
        "borderline_records": borderline_records[:50],
        "sample_records": records[:50],
        "audit_log_count": len(audit_log)
    }

@router.post("/re-threshold")
def re_threshold_results(req: ReThresholdRequest):
    """
    Live lightweight re-thresholding without retraining.
    Updates match and singleton assignments instantly.
    """
    new_th = req.threshold
    inference_results["threshold"] = new_th
    records = inference_results.get("records", [])
    
    candidate_rows = []
    matching_rows = []
    confirmed_matches = 0
    singletons = 0
    
    for r in records:
        s1_id = r["s1_id"]
        cands = r.get("candidates", [])
        cand_ids = [c["target_id"] for c in cands]
        candidate_rows.append(f"{s1_id}\t{','.join(cand_ids)}\n")
        
        matched_cids = []
        for c in cands:
            c["is_match"] = bool(c["confidence"] >= new_th)
            if c["is_match"]:
                matched_cids.append(c["target_id"])
                
        r["matched_ids"] = matched_cids
        r["status"] = "matched" if matched_cids else "singleton"
        
        if matched_cids:
            confirmed_matches += len(matched_cids)
        else:
            singletons += 1
            
        matching_rows.append(f"{s1_id}\t{','.join(matched_cids)}\n")
        
    pipeline_state["metrics"]["confirmed_matches"] = confirmed_matches
    pipeline_state["metrics"]["singletons"] = singletons
    pipeline_state["metrics"]["threshold"] = new_th
    
    # Re-save TSVs
    cand_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    
    with open(cand_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.writelines(candidate_rows)
        
    with open(match_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(matching_rows)
        
    return {
        "status": "updated",
        "threshold": new_th,
        "confirmed_matches": confirmed_matches,
        "singletons": singletons
    }

@router.post("/decision")
def record_manual_decision(req: ManualDecisionRequest):
    audit_log.append({
        "s1_id": req.s1_id,
        "target_id": req.target_id,
        "action": req.action,
        "timestamp": "now"
    })
    
    # Update inference record
    records = inference_results.get("records", [])
    for r in records:
        if r["s1_id"] == req.s1_id:
            for c in r.get("candidates", []):
                if c["target_id"] == req.target_id:
                    c["is_match"] = (req.action == "accept")
            matched_cids = [c["target_id"] for c in r.get("candidates", []) if c["is_match"]]
            r["matched_ids"] = matched_cids
            r["status"] = "matched" if matched_cids else "singleton"
            break
            
    return {"status": "recorded", "audit_log": audit_log[-10:]}

@router.get("/export/download-zip")
def download_submission_zip():
    """Builds and returns the complete final submission zip."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Output files
        cand_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")
        doc_path = os.path.join(BASE_DIR, "Documentation_template.md")
        
        if os.path.exists(match_out):
            zip_file.write(match_out, arcname="output/matching_results.tsv")
        if os.path.exists(cand_out):
            zip_file.write(cand_out, arcname="output/candidate_pairs.tsv")
        if os.path.exists(doc_path):
            zip_file.write(doc_path, arcname="Documentation_template.md")
            
        # 2. Code package
        src_dir = os.path.join(BASE_DIR, "code", "business_entity_resolution")
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, BASE_DIR)
                zip_file.write(full_p, arcname=rel_p)
                
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=EntityMatch_AI_submission.zip"}
    )
