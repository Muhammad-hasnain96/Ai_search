class MedFinderAI:
    def __init__(self):
        print("🧠 Lightweight AI agent loaded (no heavy transformer models).")

    # infer intent (simple + fast)
    def infer_intent(self, query: str) -> str:
        intents = [
            "urine bag", "catheter", "blood pressure monitor", "thermometer",
            "pulse oximeter", "glucometer", "stethoscope", "surgical gloves",
            "wheelchair", "bandage", "nebulizer", "oxygen concentrator"
        ]

        q = query.lower()
        for word in intents:
            if word in q:
                return word

        return query

    # enhance keywords (optional lightweight version)
    def enhance_query_keywords(self, query: str) -> str:
        return query

    def optimize_query(self, query: str) -> str:
        inferred = self.infer_intent(query)
        return inferred
