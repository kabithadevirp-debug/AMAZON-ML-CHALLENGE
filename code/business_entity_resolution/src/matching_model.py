import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Set, Tuple, Any

def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    if not true_matches and not pred_matches:
        return 1.0
    if not true_matches and pred_matches:
        return 0.0
    if true_matches and not pred_matches:
        return 0.0
        
    tp = len(true_matches.intersection(pred_matches))
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0.0 and recall == 0.0:
        return 0.0
        
    num = 1.25 * precision * recall
    denom = 0.25 * precision + recall
    return num / denom if denom > 0 else 0.0

def evaluate_macro_f05(ground_truth_map: Dict[str, Set[str]], predictions_map: Dict[str, Set[str]]) -> Tuple[float, float, float]:
    all_s1_ids = set(ground_truth_map.keys())
    if not all_s1_ids:
        return 0.0, 0.0, 0.0
        
    scores = []
    precisions = []
    recalls = []
    
    for s1_id in all_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())
        
        f05 = compute_entity_f05(true_set, pred_set)
        scores.append(f05)
        
        if not true_set and not pred_set:
            precisions.append(1.0)
            recalls.append(1.0)
        elif not true_set and pred_set:
            precisions.append(0.0)
            recalls.append(1.0)
        elif true_set and not pred_set:
            precisions.append(1.0)
            recalls.append(0.0)
        else:
            tp = len(true_set.intersection(pred_set))
            p = tp / len(pred_set) if pred_set else 0.0
            r = tp / len(true_set) if true_set else 0.0
            precisions.append(p)
            recalls.append(r)
            
    return float(np.mean(scores)), float(np.mean(precisions)), float(np.mean(recalls))

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
        
        best_threshold = 0.75
        best_macro_f05 = -1.0
        
        for th in np.linspace(0.50, 0.95, 19):
            preds_map: Dict[str, Set[str]] = {s1: set() for s1 in val_ground_truth.keys()}
            for (s1_id, target_id), prob in zip(val_pair_ids, probs):
                if prob >= th:
                    preds_map[s1_id].add(target_id)
                    
            macro_f05, _, _ = evaluate_macro_f05(val_ground_truth, preds_map)
            if macro_f05 > best_macro_f05:
                best_macro_f05 = macro_f05
                best_threshold = th
                
        self.decision_threshold = best_threshold
        return best_threshold

    def predict_pair_probs(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
