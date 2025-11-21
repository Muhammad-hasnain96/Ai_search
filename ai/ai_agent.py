import re
from typing import Tuple, Dict, Any, List

# Lazy import for sentence-transformers (only when needed)
_embedding_model = None
def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # small, fast, and Railway-friendly
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

class MedFinderAI:
    """
    Lightweight, semantic-aware intent parser using MiniLM + robust rule fallback.
    Returns dict:
      { "query": "...", "is_medical": True/False, "max_price": float|None, "currency": "USD"|None }
    """

    # medical phrases (kept short and relevant)
    MEDICAL_KEYWORDS = [
        "urine bag", "catheter", "blood pressure monitor", "bp monitor",
        "thermometer", "pulse oximeter", "glucometer", "stethoscope",
        "surgical gloves", "wheelchair", "bandage", "nebulizer",
        "oxygen concentrator", "hearing aid", "walker", "syringe",
        "iv set", "face mask", "first aid kit"
    ]

    # phrases to remove from free-form query
    REMOVE_PHRASES = [
        "recommend", "suggest", "give me", "show me", "find", "buy", "get me",
        "cheap", "best", "i need", "i want", "for my", "for me", "please",
        "can you", "something for"
    ]

    CURRENCY_SYMBOLS = {
        '$': 'USD', 'usd': 'USD',
        '€': 'EUR', 'eur': 'EUR',
        '£': 'GBP', 'gbp': 'GBP',
        '₹': 'INR', 'inr': 'INR',
        'rs': 'PKR', 'pkr': 'PKR', 'pk': 'PKR'
    }

    # price regex: captures common phrasings like "under 5,000 PKR", "$20", "1000 rs"
    PRICE_RE = re.compile(
        r'(?P<prefix>\b(?:under|below|less than|up to|upto|<=|≤|<)\b)?\s*'
        r'(?P<symbol>[$€£₹])?\s*'
        r'(?P<amount>\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?)\s*'
        r'(?P<code>\b(?:[A-Za-z]{2,4}|rs|pkr)\b)?',
        flags=re.IGNORECASE
    )

    STOPWORDS = set([
        "for","my","the","a","an","to","and","of","in","on","with","by","at",
        "from","is","it","this","that","these","those"
    ])

    # semantic similarity threshold for labeling as medical (tweakable)
    SEMANTIC_MEDICAL_THRESH = 0.60

    def __init__(self):
        # precompute embeddings for medical keywords (lazy)
        self._med_emb = None

    def _clean_text(self, q: str) -> str:
        if not q:
            return ""
        q = q.strip()
        q = q.lower()
        for p in self.REMOVE_PHRASES:
            q = q.replace(p, "")
        q = re.sub(r'[\?\!]', '', q)
        q = " ".join(q.split())
        return q.strip()

    def _extract_price(self, q: str) -> Tuple[float, str]:
        """
        Returns (amount_or_None, currency_or_None)
        """
        if not q:
            return None, None
        m = self.PRICE_RE.search(q)
        if not m:
            return None, None
        amt = m.group("amount")
        sym = m.group("symbol")
        code = m.group("code")
        amount = None
        try:
            amount = float(str(amt).replace(",", "")) if amt else None
        except Exception:
            amount = None
        currency = None
        if sym:
            currency = self.CURRENCY_SYMBOLS.get(sym.lower(), None)
        if code:
            code_norm = code.lower()
            currency = self.CURRENCY_SYMBOLS.get(code_norm, currency) or code_norm.upper()
        return amount, currency

    def _semantic_is_medical(self, query: str) -> bool:
        """
        Use embedding similarity between query and medical keywords.
        Returns True if query semantically close to any medical keyword.
        """
        if not query:
            return False
        model = _get_embedding_model()
        if self._med_emb is None:
            self._med_emb = model.encode(self.MEDICAL_KEYWORDS, normalize_embeddings=True)
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        # compute cosine similarity (dot product for normalized)
        import numpy as np
        sims = np.dot(self._med_emb, q_emb)
        max_sim = float(sims.max()) if sims.size > 0 else 0.0
        return max_sim >= self.SEMANTIC_MEDICAL_THRESH

    def _best_med_keyword(self, query: str) -> str:
        """
        Return the exact medical phrase if present or most similar medical keyword (semantic),
        else empty string.
        """
        # literal check first
        lq = (query or "").lower()
        for k in self.MEDICAL_KEYWORDS:
            if k in lq:
                return k
        # semantic fallback
        try:
            model = _get_embedding_model()
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            med_embs = model.encode(self.MEDICAL_KEYWORDS, normalize_embeddings=True)
            import numpy as np
            sims = np.dot(med_embs, q_emb)
            idx = int(sims.argmax())
            if float(sims[idx]) >= self.SEMANTIC_MEDICAL_THRESH:
                return self.MEDICAL_KEYWORDS[idx]
        except Exception:
            pass
        return ""

    def _general_product_from_query(self, q: str) -> str:
        # remove stopwords and keep last up to 4 meaningful tokens
        parts = [p for p in q.split() if p not in self.STOPWORDS]
        if not parts:
            return q.strip()
        return " ".join(parts[-4:]).strip()

    def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Main API.
        Returns:
          {
            "query": "thermometer",
            "is_medical": True,
            "max_price": 1500.0 or None,
            "currency": "PKR" or None
          }
        """
        raw = query or ""
        cleaned = self._clean_text(raw)

        # extract price from raw original (keeps "under/upto" semantics)
        amount, currency = self._extract_price(raw)

        # remove extracted price text from cleaned for intent detection
        cleaned_no_price = re.sub(self.PRICE_RE, "", cleaned).strip()

        # semantic + keyword medical detection
        is_med_keyword = any(k in cleaned_no_price.lower() for k in self.MEDICAL_KEYWORDS)
        is_med_semantic = False
        try:
            is_med_semantic = self._semantic_is_medical(cleaned_no_price)
        except Exception:
            is_med_semantic = False

        is_med = bool(is_med_keyword or is_med_semantic)

        if is_med:
            med = self._best_med_keyword(cleaned_no_price)
            q_text = med or cleaned_no_price
        else:
            q_text = self._general_product_from_query(cleaned_no_price)

        # final sanitization
        q_text = (q_text or "").strip()

        return {
            "query": q_text or cleaned_no_price or cleaned or raw,
            "is_medical": bool(is_med),
            "max_price": float(amount) if amount is not None else None,
            "currency": (currency or None)
        }
