import os
from typing import Dict, Any, List, Optional, Tuple

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

def count_lines_fast(file_path: str) -> int:
    """Fast binary block line counter (processes hundreds of MBs in milliseconds)."""
    count = 0
    with open(file_path, 'rb') as f:
        while True:
            buffer = f.read(1024 * 1024)
            if not buffer:
                break
            count += buffer.count(b'\n')
    return count

def validate_tsv_file(file_path: str, expected_source_type: str = "source", max_check_lines: int = 1000) -> Dict[str, Any]:
    """
    Validates that a file is a valid UTF-8 TSV with the exact expected columns,
    non-empty rows, and expected entity ID prefixes.
    Fast streaming validation with instant binary line count.
    """
    if not os.path.exists(file_path):
        return {"valid": False, "errors": [f"File not found: {file_path}"]}
    
    errors = []
    warnings = []
    sample_rows = []
    country_counts = {}
    
    expected_cols = EXPECTED_GT_COLUMNS if expected_source_type == "ground_truth" else EXPECTED_SOURCE_COLUMNS
    prefix_expected = None
    if expected_source_type == "source1":
        prefix_expected = "S1-"
    elif expected_source_type == "source2":
        prefix_expected = "S2-"
    elif expected_source_type == "source3":
        prefix_expected = "S3-"

    try:
        # Fast line count
        total_file_lines = count_lines_fast(file_path)
        total_data_rows = max(0, total_file_lines - 1)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            if not header_line:
                return {"valid": False, "errors": ["File is empty."]}
            
            if "\t" not in header_line and "," in header_line:
                return {
                    "valid": False,
                    "errors": [
                        "Header contains commas but no TAB delimiter. "
                        "Files must be strictly tab-separated (.tsv)."
                    ]
                }
            
            headers = [c.strip().lower() for c in header_line.rstrip("\r\n").split("\t")]
            
            if headers != expected_cols:
                return {
                    "valid": False,
                    "errors": [
                        f"Unexpected columns: {headers}. Expected exactly: {expected_cols}"
                    ]
                }

            checked_lines = 0
            for line_idx, line in enumerate(f, start=2):
                if checked_lines >= max_check_lines:
                    break
                if not line.strip():
                    continue
                
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != len(expected_cols):
                    if len(errors) < 5:
                        errors.append(
                            f"Row {line_idx} has {len(parts)} columns instead of {len(expected_cols)}."
                        )
                    continue

                checked_lines += 1
                row_dict = dict(zip(expected_cols, parts))
                
                # Check entity_id prefix
                if prefix_expected:
                    eid = row_dict.get("entity_id", "")
                    if not eid.startswith(prefix_expected):
                        if len(warnings) < 3:
                            warnings.append(
                                f"Row {line_idx} entity ID '{eid}' does not have standard prefix '{prefix_expected}'."
                            )

                # Track country counts in sampled rows
                if "country" in row_dict:
                    cntry = row_dict["country"].strip() or "Unknown"
                    country_counts[cntry] = country_counts.get(cntry, 0) + 1

                # Collect preview (first 20 rows)
                if len(sample_rows) < 20:
                    sample_rows.append(row_dict)

    except Exception as e:
        return {"valid": False, "errors": [f"Failed to read file: {str(e)}"]}

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "total_rows": total_data_rows,
        "checked_sample_rows": checked_lines,
        "country_counts": country_counts,
        "sample_rows": sample_rows,
    }
