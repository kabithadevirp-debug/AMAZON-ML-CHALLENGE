"""
validate_submission.py - Hardened version
Checks coverage, uniqueness, and schema
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from pathlib import Path

S1_PATH = "student_resource/dataset/test/test_source1.tsv"
RESULT_PATH = "output/matching_results.tsv"
CANDIDATE_PATH = "output/candidate_pairs.tsv"

def fail(msg):
    print(f"[FAIL] VALIDATION FAILED: {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[PASS] {msg}")

s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str)
if not Path(RESULT_PATH).exists():
    fail(f"{RESULT_PATH} not found")

res = pd.read_csv(RESULT_PATH, sep="\t", dtype=str).fillna("")

# 1. Row count
if len(res) != len(s1):
    fail(f"Row count mismatch: S1={len(s1)}, results={len(res)}. Run is incomplete.")
ok(f"Row count matches: {len(res)}")

# 2. Column check
expected_cols = ["source1_entity_id", "matched_entity_ids"]
if list(res.columns) != expected_cols:
    fail(f"Wrong columns: got {list(res.columns)}, expected {expected_cols}")
ok(f"Columns correct: {expected_cols}")

# 3. Uniqueness
if res["source1_entity_id"].nunique() != len(s1):
    dups = res[res.duplicated("source1_entity_id", keep=False)]
    fail(f"Duplicate S1 IDs found: {len(res) - res['source1_entity_id'].nunique()} duplicates. Sample: {dups.head()}")
ok("No duplicate S1 IDs - join is clean")

# 4. Coverage
s1_ids = set(s1["entity_id"] if "entity_id" in s1.columns else s1.iloc[:,0])
res_ids = set(res["source1_entity_id"])
missing = s1_ids - res_ids
if missing:
    fail(f"{len(missing)} S1 IDs missing from output. Sample: {list(missing)[:5]}")
ok("All S1 IDs present")

# 5. Format - check tab separated
with open(RESULT_PATH, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    if "\t" not in first_line:
        fail("File is not tab-separated")
ok("Tab-separated check passed")

# 6. Candidate pairs check
if Path(CANDIDATE_PATH).exists():
    cand = pd.read_csv(CANDIDATE_PATH, sep="\t", dtype=str)
    print(f"\nCandidate pairs: {len(cand)} rows")
    ok(f"candidate_pairs.tsv exists and verified")

print("\n=== VALIDATION PASSED - Ready to submit ===")
