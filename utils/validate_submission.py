#!/usr/bin/env python3
"""
validate_submission.py — Polars-based submission validator
Checks: row count, uniqueness, tab-separation, coverage, no None values.
Run: python utils/validate_submission.py
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
from pathlib import Path

S1_PATH = "student_resource/dataset/test/test_source1.tsv"
RESULT_PATH = "output/matching_results.tsv"
CANDIDATE_PATH = "output/candidate_pairs.tsv"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[PASS] {msg}")


def main():
    print("=" * 60)
    print("=== SUBMISSION VALIDATION ===")
    print("=" * 60)

    # --- Load S1 ---
    if not Path(S1_PATH).exists():
        fail(f"{S1_PATH} not found")
    s1 = pl.read_csv(S1_PATH, separator="\t", schema_overrides={"entity_id": pl.Utf8})

    # --- Load results ---
    if not Path(RESULT_PATH).exists():
        fail(f"{RESULT_PATH} not found")
    res = pl.read_csv(RESULT_PATH, separator="\t",
                      schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    res = res.with_columns(pl.col("matched_entity_ids").fill_null(""))

    # 1. Row count
    if len(res) != len(s1):
        fail(f"Row count mismatch: S1={len(s1):,}, results={len(res):,}. Missing {len(s1)-len(res):,} rows.")
    ok(f"Row count matches: {len(res):,}")

    # 2. Column check
    expected_cols = ["source1_entity_id", "matched_entity_ids"]
    if list(res.columns) != expected_cols:
        fail(f"Wrong columns: got {list(res.columns)}, expected {expected_cols}")
    ok(f"Columns correct: {expected_cols}")

    # 3. Tab-separated check
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" not in first_line:
            fail("File is not tab-separated")
    ok("Tab-separated format confirmed")

    # 4. Uniqueness — one row per S1
    n_unique = res["source1_entity_id"].n_unique()
    if n_unique != len(res):
        fail(f"Duplicate S1 IDs: {len(res) - n_unique:,} duplicates found")
    ok(f"All S1 IDs unique ({n_unique:,})")

    # 5. Coverage — every S1 ID present
    s1_ids = set(s1["entity_id"].to_list())
    res_ids = set(res["source1_entity_id"].to_list())
    missing = s1_ids - res_ids
    if missing:
        fail(f"{len(missing):,} S1 IDs missing from output. Sample: {list(missing)[:5]}")
    ok("All S1 IDs present in output")

    # 6. No None/null values (only "" for empty matches)
    null_count = res["matched_entity_ids"].null_count()
    if null_count > 0:
        fail(f"{null_count:,} null values found in matched_entity_ids")
    # Check for literal "None" strings
    none_count = res.filter(pl.col("matched_entity_ids") == "None").height
    if none_count > 0:
        fail(f'{none_count:,} literal "None" strings found in matched_entity_ids')
    ok("No null or None values")

    # 7. Empty rate sanity check
    empty_count = res.filter(pl.col("matched_entity_ids").str.strip_chars() == "").height
    empty_rate = empty_count / len(res)
    matched_count = len(res) - empty_count
    print(f"\n--- Statistics ---")
    print(f"  Matched:    {matched_count:,} ({matched_count/len(res):.1%})")
    print(f"  Singletons: {empty_count:,} ({empty_rate:.1%})")
    if empty_rate > 0.50:
        print(f"  [WARNING] Empty rate {empty_rate:.1%} is very high (expected ~5-8%)")
    elif empty_rate > 0.15:
        print(f"  [WARNING] Empty rate {empty_rate:.1%} higher than train GT (5.58%)")
    else:
        print(f"  [OK] Empty rate within expected range")

    # 8. Candidate pairs check
    if Path(CANDIDATE_PATH).exists():
        cand = pl.read_csv(CANDIDATE_PATH, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8})
        ok(f"candidate_pairs.tsv exists ({len(cand):,} rows)")
    else:
        print(f"  [INFO] {CANDIDATE_PATH} not found (optional)")

    print(f"\n{'='*60}")
    print("=== ✅ ALL VALIDATION CHECKS PASSED — Ready to submit ===")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
