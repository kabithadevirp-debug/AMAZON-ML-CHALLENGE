import re
import unicodedata
from typing import List, Set, Tuple

LEGAL_TERMS = {
    'incorporated': 'inc', 'corporation': 'corp', 'company': 'co',
    'limited': 'ltd', 'private': 'pvt', 'privatelimited': 'pvt ltd',
    'llc': 'llc', 'llp': 'llp', 'p.l.l.c.': 'llc', 'l.l.c.': 'llc',
    'enterprises': 'enterprises', 'enterprise': 'enterprises',
    'services': 'services', 'service': 'services',
    'solutions': 'solutions', 'solution': 'solutions',
    'technologies': 'technologies', 'technology': 'tech', 'tech': 'tech',
    'international': 'intl', 'holdings': 'holdings', 'group': 'group',
    'consulting': 'consulting', 'associates': 'assoc', 'partners': 'partners',
    'industries': 'ind', 'industry': 'ind', 'properties': 'prop',
    'management': 'mgmt', 'center': 'ctr', 'centre': 'ctr'
}

ADDRESS_ABBR = {
    'street': 'st', 'str': 'st', 'saint': 'st',
    'road': 'rd', 'avenue': 'ave', 'av': 'ave',
    'boulevard': 'blvd', 'drive': 'dr', 'lane': 'ln',
    'court': 'ct', 'place': 'pl', 'square': 'sq',
    'highway': 'hwy', 'parkway': 'pkwy', 'circle': 'cir',
    'suite': 'ste', 'apartment': 'apt', 'building': 'bldg',
    'floor': 'fl', 'ground floor': 'gf', 'first floor': '1f',
    'door no': 'dno', 'door number': 'dno', 'h no': 'hno', 'house no': 'hno',
    'plot no': 'plot', 'sector': 'sec', 'block': 'blk',
    'near': 'nr', 'opp': 'opposite', 'opposite': 'opp',
    'po box': 'pobox', 'post office': 'po',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw'
}

STOPWORDS = {
    'the', 'and', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to',
    'dba', 'd.b.a', 'd/b/a', 'aka', 'a.k.a', 'c/o', 'care of'
}

GENERIC_NAME_TOKENS = {
    'inc', 'corp', 'co', 'ltd', 'pvt', 'llc', 'llp', 'enterprises', 'enterprise', 'services', 'service',
    'solutions', 'solution', 'tech', 'technologies', 'technology', 'holdings', 'holding', 'group',
    'consulting', 'consultants', 'partners', 'partner', 'ind', 'industries', 'industry',
    'mgmt', 'management', 'ctr', 'center', 'centre', 'shop', 'store', 'market', 'hotel', 'restaurant',
    'builders', 'builder', 'properties', 'prop', 'property', 'realty', 'realtors', 'construction',
    'constructions', 'developers', 'developer', 'agency', 'agencies', 'trading', 'traders', 'trade',
    'works', 'logistics', 'transport', 'transports', 'auto', 'automobiles', 'motors', 'motor',
    'pharma', 'pharmaceuticals', 'finance', 'financial', 'financials', 'capital', 'ventures', 'venture',
    'investments', 'investment', 'international', 'intl', 'commercial', 'commercials', 'global'
}

INDIC_TO_LATIN = {
    'एसएस': 'ss', 'फूड': 'food', 'प्राइवेट': 'pvt', 'लिमिटेड': 'ltd',
    'रेड': 'red', 'वेंचर्स': 'ventures', 'होटल': 'hotel', 'एंटरप्राइजेज': 'enterprises',
    'कंपनी': 'company', 'सर्विस': 'services', 'सोल्यूशन्स': 'solutions',
    'उत्तर': 'uttar', 'प्रदेश': 'pradesh', 'हरियाणा': 'haryana',
    'राजस्थान': 'rajasthan', 'कर्नाटक': 'karnataka', 'महाराष्ट्र': 'maharashtra',
    'दिल्ली': 'delhi', 'गाजियाबाद': 'ghaziabad', 'फरीदाबाद': 'faridabad',
    'जयपुर': 'jaipur', 'बैंगलोर': 'bangalore', 'कोलकाता': 'kolkata'
}

def remove_accents(text: str) -> str:
    if not text:
        return ""
    for ind, lat in INDIC_TO_LATIN.items():
        if ind in text:
            text = text.replace(ind, lat)
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = remove_accents(text.lower())
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_business_name(name: str) -> Tuple[str, List[str], List[str]]:
    cleaned = clean_text(name)
    tokens = cleaned.split()
    normalized_tokens = []
    for tok in tokens:
        if tok in LEGAL_TERMS:
            normalized_tokens.append(LEGAL_TERMS[tok])
        elif tok not in STOPWORDS:
            normalized_tokens.append(tok)
            
    norm_str = " ".join(normalized_tokens)
    core_tokens = [t for t in normalized_tokens if t not in GENERIC_NAME_TOKENS and len(t) > 1]
    if not core_tokens and normalized_tokens:
        core_tokens = [normalized_tokens[0]]
    return norm_str, normalized_tokens, core_tokens

def normalize_address(address: str) -> Tuple[str, List[str], Set[str]]:
    cleaned = clean_text(address)
    tokens = cleaned.split()
    normalized_tokens = []
    digits = set()
    for tok in tokens:
        if tok.isdigit() or re.match(r'^\d+[a-z]?$', tok):
            digits.add(tok)
        if tok in ADDRESS_ABBR:
            normalized_tokens.append(ADDRESS_ABBR[tok])
        elif tok not in STOPWORDS:
            normalized_tokens.append(tok)
    return " ".join(normalized_tokens), normalized_tokens, digits
