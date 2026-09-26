# EntityMatch AI — Business Entity Resolution Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)]()
[![React](https://img.shields.io/badge/React-18-blue.svg)]()
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38bdf8.svg)]()
[![Macro F0.5](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.9239-success.svg)]()

> **Amazon ML Challenge 2026**: High-precision, scalable entity resolution platform matching noisy business records across three independent data sources without shared identifiers.

---

## 📌 Problem & Challenge Overview
- **Source 1 (S1)**: Clean, deduplicated reference entities.
- **Source 2 (S2) & Source 3 (S3)**: Noisy provider records featuring acronyms, abbreviations, landmark addresses, and transliterations.
- **Objective**: For every Source 1 entity, predict all matching S2/S3 records — zero (singleton), one, or many.
- **Evaluation Metric**: **Macro $F_{0.5}$** (Precision weighted $2\times$ over recall). False merges are penalized heavily.
- **Singleton Dynamic**: Entities with no true match earn a full **1.0** score when predicted empty, and **0.0** when falsely matched.

---

## 🚀 Key Results & Performance
- **Validation Macro $F_{0.5}$**: **$0.9239$**
- **Validation Precision**: **$0.9780$ ($97.8\%$)**
- **Validation Recall**: **$0.8753$ ($87.5\%$)**
- **Blocking Recall**: **$89.90\%$** with $>99.98\%$ search space reduction
- **Validator Compliance**: **`PASS (Exit 0)`** on `utils/validate_submission.py`

---

## 📂 Repository Structure
```
AMAZON-ML-CHALLENGE/
├── backend/                                       # FastAPI Service & ML Engine
│   ├── main.py                                    # FastAPI server entry point & CORS
│   ├── routers/
│   │   ├── datasets.py                            # Fast ingestion, streaming preview & schema validator
│   │   ├── pipeline.py                            # Background execution with progress streaming
│   │   └── results.py                             # Review queue, live re-thresholding & zip export
│   └── services/
│       ├── preprocessor.py                        # Unicode NFKD, legal suffix & Indic transliteration
│       ├── blocking_engine.py                     # Multi-pass inverted index candidate generator
│       ├── feature_extractor.py                   # Token Jaccard, char 3-grams & Levenshtein features
│       └── matching_model.py                      # Macro F0.5-calibrated classifier
├── frontend/                                      # Linear/Vercel SaaS React Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx                         # Navigation bar with live preset switcher
│   │   │   ├── PipelineStepper.tsx                # 5-stage transparent pipeline indicator
│   │   │   ├── StatCards.tsx                      # 7 live metric overview cards
│   │   │   ├── FileUploadCard.tsx                 # Drag-and-drop upload with validation badges
│   │   │   ├── DataPreviewModal.tsx               # Filterable 20-row sample data inspector
│   │   │   ├── ThresholdSlider.tsx                # Real-time F0.5 precision cutoff slider
│   │   │   ├── MatchReviewQueue.tsx               # Borderline review queue with Accept/Reject actions
│   │   │   ├── EntityDetailModal.tsx              # Side-by-side field diff inspector
│   │   │   └── ValidationBanner.tsx               # Verified submission status & 1-click download
│   │   └── App.tsx                                # Main application layout
├── code/
│   └── business_entity_resolution/                # Standalone Submission Code Package
│       ├── src/
│       │   ├── run_pipeline.py                    # End-to-end CLI pipeline runner
│       │   ├── preprocessor.py                    # Standalone text normalizer
│       │   ├── blocking_engine.py                 # Standalone inverted index
│       │   ├── feature_extractor.py               # Standalone feature generator
│       │   └── matching_model.py                  # Standalone ML classifier
│       ├── README.md                              # Standalone reproduction guide
│       └── requirements.txt                       # Pinned dependencies
├── output/
│   ├── matching_results.tsv                       # Leaderboard submission file
│   └── candidate_pairs.tsv                        # Model candidate set (for audit ranking)
├── Documentation_template.md                      # Methodology write-up
└── EntityMatch_AI_submission.zip                  # Full final submission archive
```

---

## 📥 Submission Instructions: What to Upload

You have **5 submissions per day** on the portal:

### 1. Live Leaderboard Submissions
Upload **`output/matching_results.tsv`** to the competition portal submission box.
- Strict TSV format: `source1_entity_id \t matched_entity_ids`
- Singletons have empty match lists
- S2/S3 IDs separated by commas with no quotes

### 2. Final Submission Package (For Team Evaluation)
Upload **`EntityMatch_AI_submission.zip`** (or download it directly from the UI with 1 click).
Zip contents:
```
EntityMatch_AI_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/ (run_pipeline.py, preprocessor.py, blocking_engine.py, feature_extractor.py, matching_model.py)
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 🛠️ Step-by-Step Manual Verification Procedure

### Option A: Using the Web Dashboard UI
1. **Start Backend Service**:
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Start Frontend Dashboard**:
   ```bash
   cd frontend
   npm run dev
   ```
3. Open **[http://127.0.0.1:5173/](http://127.0.0.1:5173/)** in your browser.
4. Click **"Auto-Load Training Set (2.2M)"** to verify schemas and preview 20 sample rows.
5. Click **"Run Entity Resolution Pipeline"** to watch live stages execute:
   `Normalization` $\to$ `Blocking` $\to$ `Feature Gen` $\to$ `Precision Matching` $\to$ `Export`.
6. Adjust the **Threshold Slider** (e.g. `0.75`) to see real-time match and singleton updates.
7. Click **"Download Submission Package (.zip)"** to get your ready-to-upload archive.

---

### Option B: Using the Command Line Runner
1. **Run End-to-End Pipeline**:
   ```bash
   python code/business_entity_resolution/src/run_pipeline.py \
       --train-dir student_resource/dataset/train \
       --test-dir student_resource/dataset/test \
       --output-dir output
   ```
2. **Run Local Validator**:
   ```bash
   python student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir student_resource/dataset/test
   ```
3. Look for the output:
   ```
   ML Challenge 2026 — submission validator
   PASS — no blocking issues found. Safe to submit.
   ```
