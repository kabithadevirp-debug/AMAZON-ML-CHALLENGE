from typing import Dict, Any, List, Set

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0

def quick_levenshtein_ratio(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if s1.startswith(s2) or s2.startswith(s1):
        return min(len1, len2) / max(len1, len2)
    if max(len1, len2) > 80:
        return jaccard_similarity(get_char_ngrams(s1, 3), get_char_ngrams(s2, 3))
        
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
            
    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_len))

def extract_pair_features(s1_meta: Dict[str, Any], target_meta: Dict[str, Any]) -> List[float]:
    s1_name = s1_meta.get('norm_name', '')
    t_name = target_meta.get('norm_name', '')
    
    s1_tokens = s1_meta.get('name_tokens', set())
    t_tokens = target_meta.get('name_tokens', set())
    
    s1_core = s1_meta.get('core_tokens', set())
    t_core = target_meta.get('core_tokens', set())
    
    s1_core_str = " ".join(sorted(s1_core))
    t_core_str = " ".join(sorted(t_core))
    
    s1_addr = s1_meta.get('norm_addr', '')
    t_addr = target_meta.get('norm_addr', '')
    
    s1_addr_tokens = s1_meta.get('addr_tokens', set())
    t_addr_tokens = target_meta.get('addr_tokens', set())
    
    s1_digits = s1_meta.get('digits', set())
    t_digits = target_meta.get('digits', set())
    
    # 1. Name Features
    f_name_jaccard = jaccard_similarity(s1_tokens, t_tokens)
    f_core_jaccard = jaccard_similarity(s1_core, t_core)
    f_name_char3 = jaccard_similarity(get_char_ngrams(s1_name, 3), get_char_ngrams(t_name, 3))
    f_name_lev = quick_levenshtein_ratio(s1_name, t_name)
    f_core_lev = quick_levenshtein_ratio(s1_core_str, t_core_str)
    
    f_prefix_match = 1.0 if (s1_name and t_name and s1_name.split()[0] == t_name.split()[0]) else 0.0
    f_first_core_match = 1.0 if (s1_core and t_core and next(iter(s1_core)) == next(iter(t_core))) else 0.0
    
    f_containment = 0.0
    if s1_name and t_name:
        if s1_name in t_name or t_name in s1_name:
            f_containment = 1.0
            
    # 2. Address Features (STRICT: no 0.5 default imputation for missing addresses)
    both_have_addr = 1.0 if (s1_addr and t_addr) else 0.0
    f_addr_jaccard = jaccard_similarity(s1_addr_tokens, t_addr_tokens) if both_have_addr else 0.0
    f_addr_char3 = jaccard_similarity(get_char_ngrams(s1_addr, 3), get_char_ngrams(t_addr, 3)) if both_have_addr else 0.0
    f_addr_lev = quick_levenshtein_ratio(s1_addr, t_addr) if both_have_addr else 0.0
    
    # 3. Numeric / PIN / Door number Features
    both_have_digits = 1.0 if (s1_digits and t_digits) else 0.0
    f_digit_match = jaccard_similarity(s1_digits, t_digits) if both_have_digits else 0.0
    f_digit_overlap = 1.0 if (both_have_digits and len(s1_digits.intersection(t_digits)) > 0) else 0.0
    
    # 4. Strict Country Match
    f_country_match = 1.0 if s1_meta.get('country') == target_meta.get('country') else 0.0

    return [
        f_name_jaccard,
        f_core_jaccard,
        f_name_char3,
        f_name_lev,
        f_core_lev,
        f_first_core_match,
        f_prefix_match,
        f_containment,
        both_have_addr,
        f_addr_jaccard,
        f_addr_char3,
        f_addr_lev,
        both_have_digits,
        f_digit_match,
        f_digit_overlap,
        f_country_match
    ]
