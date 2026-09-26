# EntityMatch AI — Production Entity Resolution Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)]()
[![React](https://img.shields.io/badge/React-18-blue.svg)]()
[![Polars](https://img.shields.io/badge/Polars-0.20%2B-navy.svg)]()
[![FAISS](https://img.shields.io/badge/FAISS-1.13-purple.svg)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-orange.svg)]()

> **Amazon ML Challenge 2026**: High-precision, ultra-fast offline business entity resolution matching noisy records across independent data sources without shared identifiers.

---

## 📌 Problem & Challenge Overview
- **Source 1 (S1)**: 1,732,544 clean reference entities in test set.
- **Source 2 (S2) & Source 3 (S3)**: ~10 Million noisy provider records featuring transliterations, legal acronym shifts, and varied addresses.
- **Objective**: For every Source 1 entity, predict matching S2/S3 IDs (0, 1, or many).
- **Evaluation Metric**: **Macro $F_{0.5}$** (Precision weighted $2\times$ over recall). False merges incur a severe penalty.
- **Singleton Dynamic**: Unmatched entities must output an empty string to earn full 1.0 credit.

---

## 🚀 Completed Milestones & What Has Been Done

### ✅ Milestone 1: Offline Kaggle Model Loading
- Downloaded and bundled `sentence-transformers/all-MiniLM-L6-v2` (80MB) locally under `code/models/minilm/` from Kaggle dataset `shinomoriaoshi/sentencetransformersallminilml6v2`.
- Confirmed 100% offline inference capability without internet access or external APIs.
- Built test script [scripts/test_model_load.py](file:///scripts/test_model_load.py) verifying embedding dimension (384) and similarity computation.

### ✅ Milestone 2: Ultra-Fast Blocking with FAISS & Embeddings
- Batch encoded normalized texts (`name [SEP] address`) into 384-dimensional dense vectors.
- Built **FAISS** index on target entities for top-20 nearest neighbor candidate retrieval via cosine similarity.
- Reduced search space from $17.3\text{ Trillion}$ pairs to top 20 candidates per entity.
- Saved candidate pairs to [output/candidate_pairs.tsv](file:///output/candidate_pairs.tsv).

### ✅ Milestone 3: Vectorized Feature Engineering & LightGBM Classifier
- Extracted similarity features using **Polars** and **RapidFuzz** (C++ backend):
  1. `embedding_cosine`: Dense FAISS similarity score
  2. `name_score`: `fuzz.token_sort_ratio`
  3. `address_score`: `fuzz.WRatio`
  4. `city_exact_match`: Normalized municipality match
  5. `token_overlap`: Token Jaccard index
- Trained a **LightGBM** classifier with hard negatives from FAISS top-20 search.
- Tuned decision threshold specifically for **Macro $F_{0.5}$** ($\theta \approx 0.82 - 0.88$).

### ✅ Milestone 4: Hardened Checkpointed Inference Engine
- Rewrote [full_scale_inference.py](file:///full_scale_inference.py) with:
  - Safe text normalization and encoding.
  - Per-country checkpointing (`US`, `India`, `France`) with isolated error recovery.
  - Strict singleton preservation: unmerged S1 entities are retained as empty match strings `""`.
  - Invariant assertions verifying that exactly 1,732,544 rows are produced.

### ✅ Milestone 5: Submission Validation & Packaging
- Executed [utils/validate_submission.py](file:///utils/validate_submission.py):
  - **1,732,544 rows** (100% match with `test_source1.tsv`)
  - **1,732,544 unique S1 IDs** (Zero duplicates)
  - **Zero NULL/NaN values**
  - **Strict TSV tab-separated structure** confirmed
- Generated final submission archive [EntityMatch_AI_submission.zip](file:///EntityMatch_AI_submission.zip).

---

## 📂 Repository Structure
```
AMAZON-ML-CHALLENGE/
├── code/
│   ├── models/
│   │   └── minilm/                                # Offline MiniLM-L6-v2 model weights & tokenizer
│   └── business_entity_resolution/                # Modular entity resolution library
├── scripts/
│   ├── test_model_load.py                         # Test script proving offline model loading
│   └── prepare_embeddings.py                      # Precomputed embedding generator
├── utils/
│   ├── validate_submission.py                     # Competition submission validator
│   └── validate_submission_fixed.py               # Enhanced invariant checker
├── output/
│   ├── matching_results.tsv                       # Portal submission file (1,732,544 rows)
│   └── candidate_pairs.tsv                        # Top candidate pairs (1,732,544 rows)
├── full_scale_inference.py                        # Ultra-fast end-to-end inference engine
├── Documentation_template.md                      # Detailed technical documentation
├── EntityMatch_AI_submission.zip                  # Packaged submission archive
└── README.md                                      # Project overview and reproduction guide
```

---

## 📥 Submission Files: What to Upload

1. **Leaderboard Submission**:
   - File: [output/matching_results.tsv](file:///output/matching_results.tsv)
   - Format: `source1_entity_id \t matched_entity_ids` (Tab-separated)
   - Upload directly to the competition portal submission portal.

2. **Code & Documentation Package**:
   - File: [EntityMatch_AI_submission.zip](file:///EntityMatch_AI_submission.zip)
   - Contains offline code, candidate pairs, matching results, and technical documentation.

---

## 🛠️ Reproduction & Offline Execution Guide

```bash
# 1. Install dependencies
pip install polars rapidfuzz faiss-cpu sentence-transformers lightgbm kagglehub

# 2. Verify offline model load
python scripts/test_model_load.py

# 3. Run full-scale inference
python full_scale_inference.py

# 4. Validate output integrity
python utils/validate_submission.py
```

---

## 🔮 Further Steps & Future Improvements

1. **GPU Acceleration for Offline Embeddings**:
   - Utilize CUDA/TensorRT for sub-second batch encoding across 10M records when GPU resources are available.
2. **Multi-Model Ensembling**:
   - Blend MiniLM-L6-v2 embeddings with character-level byte embeddings (e.g. ByT5 or Canine) for enhanced robustness against rare OCR and phonetic corruption.
3. **Adaptive Thresholding by Country**:
   - Fine-tune country-specific decision thresholds ($\theta_{\text{US}}$, $\theta_{\text{India}}$, $\theta_{\text{France}}$) based on empirical validation splits.
4. **Graph-Based Transitive Closure**:
   - Apply connected-component clustering across (S1, S2, S3) candidate bipartite graphs to enforce multi-source consistency constraints.
