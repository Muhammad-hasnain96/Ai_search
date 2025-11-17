from transformers import pipeline
from sentence_transformers import SentenceTransformer

class MedFinderAI:
    def __init__(self):
        print("🧠 Loading AI models...")
        try:
            self.intent_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except:
            self.intent_model = None

        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except:
            self.embedding_model = None

    # Infer intent, returns category if recognized
    def infer_intent(self, query):
        medical_categories = [
            "urine bag","catheter","blood pressure monitor","thermometer",
            "pulse oximeter","glucometer","stethoscope","surgical gloves",
            "wheelchair","hearing aid","bandage","nebulizer",
            "oxygen concentrator","rehabilitation equipment","hospital bed",
            "first aid kit","dental tool"
        ]
        if not self.intent_model:
            return query
        res = self.intent_model(query, medical_categories)
        label, score = res["labels"][0], res["scores"][0]
        return label if score > 0.5 else query

    # Enhance query only for medical
    def enhance_query_keywords(self, query):
        keywords = {
            "urine":["urine bag","urinary drainage bag","catheter bag","urinal"],
            "blood pressure":["bp monitor","sphygmomanometer"],
            "temperature":["thermometer"],
            "oxygen":["pulse oximeter","oxygen concentrator"],
            "sugar":["glucometer","blood glucose meter"],
            "bandage":["first aid","wound care"]
        }
        q = query.lower()
        for k, syn in keywords.items():
            if k in q:
                return f"{query}, {', '.join(syn)}"
        return query

    # Optimize query (medical enhancement optional)
    def optimize_query(self, q):
        inferred = self.infer_intent(q)
        if inferred.lower() in q.lower():  # only enhance if medical
            enhanced = self.enhance_query_keywords(inferred)
            return f"{q} related to {enhanced}"
        return q  # general product

    # Format output for logs
    def format_response(self, results):
        if not results:
            return "❌ No products found."
        return "\n".join([f"• {r['title']} (${r['price']})\n  {r['url']}" for r in results])
