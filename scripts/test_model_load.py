#!/usr/bin/env python3
"""
Test offline model loading from Kaggle dataset.
Run: python scripts/test_model_load.py
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_model_path():
    """Find the MiniLM model path — try local copy first, then kagglehub cache."""
    from pathlib import Path
    
    # 1. Check local copy
    local_path = Path("code/models/minilm")
    if local_path.exists() and (local_path / "config.json").exists():
        print(f"[OK] Found local model at: {local_path}")
        return str(local_path)
    
    # 2. Check kagglehub cache
    try:
        import kagglehub
        kpath = kagglehub.dataset_download("shinomoriaoshi/sentencetransformersallminilml6v2")
        print(f"[OK] Found kagglehub model at: {kpath}")
        return kpath
    except Exception as e:
        print(f"[WARN] kagglehub failed: {e}")
    
    # 3. Fail
    print("[ERROR] No model found")
    sys.exit(1)


def main():
    t0 = time.time()
    model_path = find_model_path()
    
    print(f"\n--- Loading SentenceTransformer from: {model_path}")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_path)
    print(f"[OK] Model loaded in {time.time()-t0:.2f}s")
    print(f"  Max seq length: {model.max_seq_length}")
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    
    # Quick test encode
    test_texts = [
        "Zephay Labs Inc, 2621 Cotten Road, Tyler TX",
        "Vision Partners Corp, Iowa City IA",
        "SCI Ptit Amicale, 18 RUE JEN ZAY, Dunkerque",
    ]
    
    embeddings = model.encode(test_texts, show_progress_bar=False)
    print(f"\n--- Test encoding:")
    print(f"  Input: {len(test_texts)} texts")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    
    # Cosine similarity test
    import numpy as np
    from numpy.linalg import norm
    
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
            print(f"  Sim({i},{j}): {cos_sim:.4f}")
    
    print(f"\n✅ Model load + encode test PASSED in {time.time()-t0:.2f}s")
    print("Model is fully offline-ready.")


if __name__ == "__main__":
    main()
