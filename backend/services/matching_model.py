import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Set, Tuple, Any

def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F0.5 for a single S1 entity.
    - If true is empty and pred is empty (singleton correct): (1.0, 1.0, 1.0)
    - If true is empty and pred is non-empty (singleton false positive): (0.0, 0.0, 0.0)
    - If true is non-empty and pred is empty (false negative): (0.0, 0.0, 0.0)
    - If true and pred are non-empty: standard F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not true_matches and not pred_matches:
        return 1.0, 1.0, 1.0
    if not true_matches and pred_matches:
        return 0.0, 0.0, 0.0
    if true_matches and not pred_matches:
        return 0.0, 0.0, 0.0
        
    tp = len(true_matches.intersection(pred_matches))
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0.0 or recall == 0.0:
        return precision, recall, 0.0
        
    num = 1.25 * precision * recall
    denom = 0.25 * precision + recall
    f05 = num / denom if denom > 0 else 0.0
    return precision, recall, f05

def evaluate_macro_f05(ground_truth_map: Dict[str, Set[str]], predictions_map: Dict[str, Set[str]]) -> Tuple[float, float, float]:
    """
    Computes Macro F0.5, Macro Precision, and Macro Recall across all S1 entities.
    """
    all_s1_ids = set(ground_truth_map.keys())
    if not all_s1_ids:
        return 0.0, 0.0, 0.0
        
    f05_list = []
    prec_list = []
    rec_list = []
    
    for s1_id in all_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())
        
        p, r, f05 = compute_entity_f05(true_set, pred_set)
        prec_list.append(p)
        rec_list.append(r)
        f05_list.append(f05)
            
    return float(np.mean(f05_list)), float(np.mean(prec_list)), float(np.mean(rec_list))

def resolve_global_matches(
    candidate_scores: List[Tuple[str, str, float]],
    all_s1_ids: Set[str],
    threshold: float
) -> Dict[str, Set[str]]:
    """
    Enforces target uniqueness constraint: an S2 or S3 entity can ONLY match
    to at most ONE S1 reference entity.
    Resolves collisions greedily by assigning to the highest confidence S1 entity first.
    """
    sorted_pairs = sorted(candidate_scores, key=lambda x: x[2], reverse=True)
    
    assigned_targets: Set[str] = set()
    predictions: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    
    for s1_id, target_id, prob in sorted_pairs:
        if prob < threshold:
            continue
        if target_id in assigned_targets:
            continue
            
        predictions[s1_id].add(target_id)
        assigned_targets.add(target_id)
        
    return predictions

class EntityMatchingModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
        self.decision_threshold = 0.75

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def tune_threshold(
        self,
        val_pairs_features: List[List[float]],
        val_pair_ids: List[Tuple[str, str]],
        val_ground_truth: Dict[str, Set[str]]
    ) -> float:
        if not val_pairs_features:
            return self.decision_threshold
            
        probs = self.model.predict_proba(np.array(val_pairs_features))[:, 1]
        all_s1_ids = set(val_ground_truth.keys())
        
        candidate_triplets = [
            (s1_id, target_id, float(prob))
            for (s1_id, target_id), prob in zip(val_pair_ids, probs)
        ]
        
        best_threshold = 0.75
        best_macro_f05 = -1.0
        
        for th in np.linspace(0.50, 0.95, 19):
            preds_map = resolve_global_matches(candidate_triplets, all_s1_ids, threshold=th)
            macro_f05, _, _ = evaluate_macro_f05(val_ground_truth, preds_map)
            if macro_f05 > best_macro_f05:
                best_macro_f05 = macro_f05
                best_threshold = th
                
        self.decision_threshold = best_threshold
        return best_threshold

    def predict_pair_probs(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
