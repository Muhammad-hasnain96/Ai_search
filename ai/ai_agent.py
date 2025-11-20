import re

class MedFinderAI:
    """
    Lightweight hybrid agent:
    - extracts intent (medical vs general)
    - normalizes query
    - extracts price and currency if present
    - returns a structured dict:
    {
        "query": "thermometer",
        "is_medical": True,
        "max_price": 15000.0,  # float or None
        "currency": "PKR"      # ISO-like code or None
    }
    """

    MEDICAL_KEYWORDS = [
        "urine bag", "catheter", "blood pressure monitor", "bp monitor",
        "thermometer", "pulse oximeter", "glucometer", "stethoscope",
        "surgical gloves", "wheelchair", "bandage", "nebulizer",
        "oxygen concentrator", "hearing aid", "walker", "syringe",
        "iv set", "face mask", "first aid kit"
    ]

    REMOVE_PHRASES = [
        "recommend", "recommend me", "suggest", "suggest me", "give me",
        "show me", "find", "buy", "get me", "cheap", "best", "i need",
        "i want", "for my", "for me", "please", "can you", "something for",
        "under", "below", "less than", "less than or equal to", "upto", "up to"
    ]

    CURRENCY_SYMBOLS = {
        '$': 'USD', 'usd': 'USD', '€': 'EUR', 'eur': 'EUR',
        '£': 'GBP', 'gbp': 'GBP', '₹': 'INR', 'inr': 'INR',
        'rs': 'PKR', 'pkr': 'PKR', 'pk': 'PKR'
    }

    PRICE_RE = re.compile(
        r'(?:under|below|less than|upto|up to|<=|≤|<)?\s*'
        r'(?P<symbol>[$€£₹])?\s*'
        r'(?P<amount>\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?)\s*'
        r'(?P<code>[A-Za-z]{2,4}|rs|pkr)?',
        flags=re.IGNORECASE
    )

    def normalize_query(self, q: str) -> str:
        q = q.lower().strip()
        for phrase in self.REMOVE_PHRASES:
            q = q.replace(phrase, "")
        q = re.sub(r'[\?\!]', '', q)
        q = " ".join(q.split())
        return q.strip()

    def detect_medical_intent(self, q: str) -> bool:
        lq = q.lower()
        for k in self.MEDICAL_KEYWORDS:
            if k in lq:
                return True
        # also check general medical words
        for w in ["medical", "hospital", "clinic", "health", "surgical"]:
            if w in lq:
                return True
        return False

    def extract_price(self, q: str):
        """returns (max_price: float|None, currency: str|None)"""
        m = self.PRICE_RE.search(q)
        if not m:
            return None, None
        amt = m.group("amount")
        symbol = m.group("symbol")
        code = m.group("code")
        try:
            amount = float(amt.replace(",", ""))
        except Exception:
            amount = None
        cur = None
        if symbol:
            cur = self.CURRENCY_SYMBOLS.get(symbol, None)
        if code:
            code_norm = code.lower()
            cur = self.CURRENCY_SYMBOLS.get(code_norm, cur) or code_norm.upper()
        return amount, cur

    def detect_general_intent(self, q: str) -> str:
        parts = q.split()
        if len(parts) <= 1:
            return q
        return " ".join(parts[-3:]).strip()

    def optimize_query(self, query: str):
        """Returns structured dict"""
        if not isinstance(query, str):
            query = str(query)
        cleaned = self.normalize_query(query)
        max_price, currency = self.extract_price(query)
        cleaned = re.sub(self.PRICE_RE, "", cleaned).strip()
        is_med = self.detect_medical_intent(cleaned)
        if is_med:
            found_med = None
            l = cleaned.lower()
            for k in self.MEDICAL_KEYWORDS:
                if k in l:
                    found_med = k
                    break
            q_text = found_med or cleaned
        else:
            q_text = self.detect_general_intent(cleaned)
        return {
            "query": q_text.strip() or cleaned,
            "is_medical": bool(is_med),
            "max_price": float(max_price) if max_price else None,
            "currency": currency
        }
