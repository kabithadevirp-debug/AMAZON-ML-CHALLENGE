import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.services.storage import dataset_manager, UPLOAD_DIR, PRESET_DATASET_DIR
from backend.services.validator import validate_tsv_file

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

class LoadPresetRequest(BaseModel):
    preset_type: str # "train" or "test"

@router.get("/summary")
def get_datasets_summary():
    """Returns total row counts, validation status, and pipeline stat cards."""
    return dataset_manager.get_summary()

@router.get("/preview/{source_key}")
def get_source_preview(source_key: str):
    """Returns the first 20 sample rows and schema info for the specified source."""
    if source_key not in ["source1", "source2", "source3", "ground_truth"]:
        raise HTTPException(status_code=400, detail="Invalid source key")
    return dataset_manager.get_preview(source_key)

@router.post("/upload")
async def upload_dataset_file(
    file: UploadFile = File(...),
    source_type: str = Form(...) # "source1", "source2", "source3", "ground_truth"
):
    """
    Uploads and strictly validates a .tsv file.
    Validates delimiter, expected headers, and non-empty records.
    Returns preview + schema verification.
    """
    if source_type not in ["source1", "source2", "source3", "ground_truth"]:
        raise HTTPException(status_code=400, detail="Invalid source_type parameter.")

    filename = file.filename or f"{source_type}.tsv"
    if not filename.endswith(".tsv") and not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .tsv files are allowed.")

    saved_path = os.path.join(UPLOAD_DIR, f"uploaded_{source_type}_{filename}")
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    validation_result = validate_tsv_file(saved_path, expected_source_type=source_type)
    dataset_manager.register_source(source_type, saved_path, validation_result)
    dataset_manager.current_preset = "custom_upload"

    return {
        "source_key": source_type,
        "filename": filename,
        "validation": validation_result,
        "summary": dataset_manager.get_summary()
    }

@router.post("/load-preset")
def load_preset_dataset(request: LoadPresetRequest):
    """
    Loads pre-bundled dataset split ('train' or 'test') with instant streaming preview and schema validation.
    """
    split = request.preset_type.lower()
    if split not in ["train", "test"]:
        raise HTTPException(status_code=400, detail="Invalid preset type. Choose 'train' or 'test'.")

    target_dir = os.path.join(PRESET_DATASET_DIR, split)
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail=f"Preset directory not found at {target_dir}")

    s1_file = os.path.join(target_dir, f"{split}_source1.tsv")
    s2_file = os.path.join(target_dir, f"{split}_source2.tsv")
    s3_file = os.path.join(target_dir, f"{split}_source3.tsv")
    gt_file = os.path.join(target_dir, f"{split}_ground_truth.tsv")

    results = {}
    if os.path.exists(s1_file):
        v1 = validate_tsv_file(s1_file, "source1")
        dataset_manager.register_source("source1", s1_file, v1)
        results["source1"] = v1

    if os.path.exists(s2_file):
        v2 = validate_tsv_file(s2_file, "source2")
        dataset_manager.register_source("source2", s2_file, v2)
        results["source2"] = v2

    if os.path.exists(s3_file):
        v3 = validate_tsv_file(s3_file, "source3")
        dataset_manager.register_source("source3", s3_file, v3)
        results["source3"] = v3

    if split == "train" and os.path.exists(gt_file):
        v_gt = validate_tsv_file(gt_file, "ground_truth")
        dataset_manager.register_source("ground_truth", gt_file, v_gt)
        results["ground_truth"] = v_gt
    else:
        dataset_manager.sources["ground_truth"] = {
            "file_path": None, "total_rows": 0, "status": "none", "country_counts": {}, "sample_rows": []
        }

    dataset_manager.current_preset = split
    return {
        "status": "loaded",
        "preset": split,
        "results": results,
        "summary": dataset_manager.get_summary()
    }
