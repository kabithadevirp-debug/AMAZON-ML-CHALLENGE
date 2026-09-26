import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
from preprocessor import normalize_business_name, normalize_address

class BlockingEngine:
    def __init__(self, max_candidates_per_s1: int = 25):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.inverted_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.target_records: Dict[str, Dict[str, Any]] = {}

    def index_target_records(self, records: List[Dict[str, str]]):
        for rec in records:
            eid = rec['entity_id']
            cntry = rec.get('country', 'US').strip().upper()
            raw_name = rec.get('business_name', '')
            raw_addr = rec.get('business_address', '')
            
            norm_name, name_tokens, core_tokens = normalize_business_name(raw_name)
            norm_addr, addr_tokens, digits = normalize_address(raw_addr)
            
            self.target_records[eid] = {
                'country': cntry,
                'norm_name': norm_name,
                'name_tokens': set(name_tokens),
                'core_tokens': set(core_tokens),
                'norm_addr': norm_addr,
                'addr_tokens': set(addr_tokens),
                'digits': digits
            }
            
            for token in core_tokens:
                if len(token) >= 3:
                    self.inverted_index[(cntry, f"TOK:{token}")].append(eid)
                    
            if core_tokens and len(core_tokens[0]) >= 4:
                prefix = core_tokens[0][:4]
                self.inverted_index[(cntry, f"PRE:{prefix}")].append(eid)
                
            if digits and core_tokens:
                initial = core_tokens[0][:1]
                for d in digits:
                    if len(d) >= 2:
                        self.inverted_index[(cntry, f"DIG:{d}_{initial}")].append(eid)

    def generate_candidates_for_s1(self, s1_record: Dict[str, str]) -> List[str]:
        cntry = s1_record.get('country', 'US').strip().upper()
        raw_name = s1_record.get('business_name', '')
        raw_addr = s1_record.get('business_address', '')
        
        norm_name, name_tokens, core_tokens = normalize_business_name(raw_name)
        norm_addr, addr_tokens, digits = normalize_address(raw_addr)
        
        s1_name_set = set(name_tokens)
        s1_core_set = set(core_tokens)
        s1_addr_set = set(addr_tokens)
        
        candidate_scores: Dict[str, float] = defaultdict(float)
        
        for token in core_tokens:
            if len(token) >= 3:
                for target_id in self.inverted_index.get((cntry, f"TOK:{token}"), []):
                    candidate_scores[target_id] += 3.0
                    
        if core_tokens and len(core_tokens[0]) >= 4:
            prefix = core_tokens[0][:4]
            for target_id in self.inverted_index.get((cntry, f"PRE:{prefix}"), []):
                candidate_scores[target_id] += 1.5
                
        if digits and core_tokens:
            initial = core_tokens[0][:1]
            for d in digits:
                if len(d) >= 2:
                    for target_id in self.inverted_index.get((cntry, f"DIG:{d}_{initial}"), []):
                        candidate_scores[target_id] += 2.0

        if not candidate_scores:
            return []

        ranked_candidates = []
        for target_id, base_score in candidate_scores.items():
            target_meta = self.target_records.get(target_id)
            if not target_meta:
                continue
            
            name_overlap = len(s1_name_set.intersection(target_meta['name_tokens']))
            core_overlap = len(s1_core_set.intersection(target_meta['core_tokens']))
            addr_overlap = len(s1_addr_set.intersection(target_meta['addr_tokens']))
            digit_overlap = len(digits.intersection(target_meta['digits']))
            
            total_score = base_score + (name_overlap * 2.0) + (core_overlap * 3.0) + (addr_overlap * 1.5) + (digit_overlap * 2.5)
            ranked_candidates.append((target_id, total_score))

        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in ranked_candidates[:self.max_candidates_per_s1]]
