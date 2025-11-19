class MedFinderAI:
    def __init__(self):
        print("🧠 Lightweight hybrid AI agent loaded — supports medical + general products.")

    # Medical keywords
    MEDICAL_KEYWORDS = [
        "urine bag", "catheter", "blood pressure monitor", "bp monitor",
        "thermometer", "pulse oximeter", "glucometer", "stethoscope",
        "surgical gloves", "wheelchair", "bandage", "nebulizer",
        "oxygen concentrator", "hearing aid", "walker", "syringe",
        "iv set", "face mask", "first aid kit"
    ]

    # Remove conversational text
    REMOVE_PHRASES = [
        "recommend", "recommend me", "suggest", "suggest me", "give me",
        "show me", "find", "buy", "get me", "cheap", "best", "i need",
        "i want", "for my", "for me", "please", "can you", "something for"
    ]

    # Clean natural language into keyword
    def normalize_query(self, q: str) -> str:
        q = q.lower().strip()
        for phrase in self.REMOVE_PHRASES:
            q = q.replace(phrase, "")
        return q.strip()

    # Detect medical intent
    def detect_medical_intent(self, query: str) -> str:
        q = query.lower()
        for k in self.MEDICAL_KEYWORDS:
            if k in q:
                return k
        return ""

    # Fallback for general products
    def detect_general_intent(self, query: str) -> str:
        parts = query.split()
        if len(parts) <= 1:
            return query
        return " ".join(parts[-3:]).strip()

    # Final optimized search keyword
    def optimize_query(self, query: str) -> str:
        cleaned = self.normalize_query(query)
        med = self.detect_medical_intent(cleaned)
        if med:
            return med
        return self.detect_general_intent(cleaned)
