import os
import shutil
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PRESET_DATASET_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DatasetManager:
    def __init__(self):
        self.sources: Dict[str, Dict[str, Any]] = {
            "source1": {"file_path": None, "total_rows": 0, "status": "missing", "country_counts": {}, "sample_rows": []},
            "source2": {"file_path": None, "total_rows": 0, "status": "missing", "country_counts": {}, "sample_rows": []},
            "source3": {"file_path": None, "total_rows": 0, "status": "missing", "country_counts": {}, "sample_rows": []},
            "ground_truth": {"file_path": None, "total_rows": 0, "status": "missing", "country_counts": {}, "sample_rows": []}
        }
        self.current_preset: Optional[str] = None
        self.pipeline_stats = {
            "s1_count": 0,
            "s2_count": 0,
            "s3_count": 0,
            "candidate_pairs": 0,
            "confirmed_matches": 0,
            "singletons": 0,
            "avg_confidence": 0.0,
            "macro_f05": 0.0,
            "blocking_recall": 0.0,
            "pipeline_stage": "idle" # idle | normalizing | blocking | features | matching | done
        }

    def register_source(self, source_key: str, file_path: str, validation_meta: Dict[str, Any]):
        self.sources[source_key] = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "total_rows": validation_meta.get("total_rows", 0),
            "status": "ready" if validation_meta.get("valid", False) else "error",
            "country_counts": validation_meta.get("country_counts", {}),
            "sample_rows": validation_meta.get("sample_rows", []),
            "warnings": validation_meta.get("warnings", []),
            "errors": validation_meta.get("errors", [])
        }
        self._update_stats()

    def _update_stats(self):
        self.pipeline_stats["s1_count"] = self.sources["source1"]["total_rows"]
        self.pipeline_stats["s2_count"] = self.sources["source2"]["total_rows"]
        self.pipeline_stats["s3_count"] = self.sources["source3"]["total_rows"]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "sources": self.sources,
            "stats": self.pipeline_stats,
            "current_preset": self.current_preset
        }

    def get_preview(self, source_key: str) -> Dict[str, Any]:
        if source_key not in self.sources:
            return {"error": f"Source {source_key} not found"}
        return {
            "source_key": source_key,
            "meta": self.sources[source_key]
        }

dataset_manager = DatasetManager()
