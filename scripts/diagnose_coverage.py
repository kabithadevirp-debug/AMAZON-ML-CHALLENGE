#!/usr/bin/env python3
"""
Diagnose incomplete matching_results.tsv
Run: python scripts/diagnose_coverage.py
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
from pathlib import Path

S1_PATH = "student_resource/dataset/test/test_source1.tsv"
RESULT_PATH = "output/matching_results.tsv"


def main():
    if not os.path.exists(S1_PATH):
        print(f"[MISSING] {S1_PATH}")
        sys.exit(1)
    if not os.path.exists(RESULT_PATH):
        print(f"[MISSING] {RESULT_PATH}")
        sys.exit(1)

    s1 = pl.read_csv(S1_PATH, separator="\t",
                     schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                                       "business_address": pl.Utf8, "country": pl.Utf8})
    res = pl.read_csv(RESULT_PATH, separator="\t",
                      schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    res = res.with_columns(pl.col("matched_entity_ids").fill_null(""))

    print(f"\n{'='*60}")
    print(f"=== ROW COUNT CHECK ===")
    print(f"S1 rows:     {len(s1):,}")
    print(f"Result rows: {len(res):,}")
    if len(s1) != len(res):
        print(f"[ERROR] INCOMPLETE: {len(s1) - len(res):,} rows missing")
    else:
        print("[PASS] Row counts match")

    s1_ids = set(s1["entity_id"].to_list())
    res_ids = set(res["source1_entity_id"].to_list())
    missing = s1_ids - res_ids
    extra = res_ids - s1_ids

    print(f"\n{'='*60}")
    print(f"=== ID COVERAGE ===")
    print(f"Missing S1 IDs: {len(missing):,}")
    print(f"Extra IDs:      {len(extra):,}")
    if missing:
        print(f"  Sample missing: {list(missing)[:10]}")

    n_unique = res["source1_entity_id"].n_unique()
    if n_unique != len(res):
        print(f"[ERROR] Duplicates: {len(res) - n_unique:,}")
    else:
        print("[PASS] All S1 IDs unique")

    # Empty analysis
    res = res.with_columns(
        (pl.col("matched_entity_ids").str.strip_chars().eq("")).alias("is_empty")
    )
    empty_rate = res["is_empty"].sum() / len(res)

    print(f"\n{'='*60}")
    print(f"=== EMPTY ANALYSIS ===")
    print(f"Overall empty rate: {empty_rate:.2%} (train GT singleton ~5.58%)")
    if empty_rate > 0.15:
        print("[ALERT] >15% empty")

    # By country
    merged = res.join(
        s1.select(["entity_id", "country"]),
        left_on="source1_entity_id", right_on="entity_id", how="left"
    )
    stats = merged.group_by("country").agg([
        pl.len().alias("count"),
        pl.col("is_empty").sum().alias("empty"),
        pl.col("is_empty").mean().alias("rate"),
    ]).sort("count", descending=True)

    print(f"\n{'='*60}")
    print(f"=== EMPTY RATE BY COUNTRY ===")
    for row in stats.iter_rows(named=True):
        marker = " <--- CRASHED?" if row["rate"] > 0.95 else ""
        print(f"  {row['country']:10s}: {row['count']:>10,} rows, {row['rate']:.2%} empty{marker}")

    # By position
    print(f"\n{'='*60}")
    print(f"=== EMPTY BY POSITION ===")
    chunk_size = 50_000
    for start in range(0, len(res), chunk_size):
        chunk = res.slice(start, chunk_size)
        rate = chunk["is_empty"].sum() / len(chunk)
        marker = " <--- CRASH" if rate > 0.95 and start > 0 else ""
        print(f"  {start:>9,} - {start+len(chunk):>9,}: {rate:.2%} empty{marker}")

    print(f"\n{'='*60}")
    if empty_rate > 0.50:
        print("[FAIL] Pipeline catastrophically broken")
    elif empty_rate > 0.15:
        print("[WARN] Empty rate too high")
    else:
        print("[PASS] Empty rate within expected range")
    print("Done.")


if __name__ == "__main__":
    main()
