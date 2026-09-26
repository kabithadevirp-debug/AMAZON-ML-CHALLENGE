# Business Entity Resolution Platform — ML Challenge 2026

## 1. Overview
This package contains the self-contained, reproducible pipeline for the Business Entity Resolution Challenge.
It processes noisy business records across three independent data sources (Source 1, Source 2, Source 3) with no shared identifiers, producing:
1. `output/matching_results.tsv` — Final predicted entity matches (leaderboard submission).
2. `output/candidate_pairs.tsv` — Candidate set from blocking stage fed into the ML model.

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── run_pipeline.py          # End-to-end execution script
│   ├── preprocessing.py         # Text & transliteration normalization
│   ├── blocking.py              # Scalable multi-pass candidate generation
│   ├── feature_extraction.py    # Name, address, token, phonetic similarity
│   └── matching_model.py        # Macro F0.5 precision-tuned classifier
├── README.md                    # Exact run and reproduction instructions
└── requirements.txt             # Pinned python dependencies
```

## 3. Quick Start & Reproduction

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
```bash
python src/run_pipeline.py \
    --train-dir ../../student_resource/dataset/train \
    --test-dir ../../student_resource/dataset/test \
    --output-dir ../../output
```

### Step 3: Validate Outputs
```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
