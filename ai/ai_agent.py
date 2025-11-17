from transformers import pipeline
from sentence_transformers import SentenceTransformer

class MedFinderAI:
    def __init__(self):
        print("Loading AI models...")
        try:
            self.intent_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except:
            self.intent_model = None

        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except:
            self.embedding_model = None

    def infer_intent(self, query):
        categories = [
            "urine bag","catheter","blood pressure monitor","thermometer",
            "pulse oximeter","glucometer","stethoscope","surgical gloves",
            "wheelchair","hearing aid","bandage","nebulizer",
            "oxygen concentrator","rehabilitation equipment","hospital bed",
            "first aid kit","dental tool"
        ]
        if not self.intent_model:
            return query

        res = self.intent_model(query, categories)
        label = res["labels"][0]
        score = res["scores"][0]
        if score > 0.5:
            return label
        return query

    def enhance_query_keywords(self, query):
        kw = {
            "urine":["urine bag","urinary drainage bag","catheter bag","urinal"],
            "blood pressure":["bp monitor","sphygmomanometer"],
            "temperature":["thermometer"],
            "oxygen":["pulse oximeter","oxygen concentrator"],
            "sugar":["glucometer","blood glucose meter"],
            "bandage":["first aid","wound care"]
        }
        q = query.lower()
        for k, syn in kw.items():
            if k in q:
                return f"{query}, {', '.join(syn)}"
        return query

    def optimize_query(self, q):
        inferred = self.infer_intent(q)
        enhanced = self.enhance_query_keywords(inferred)
        return f"{q} related to {enhanced}"

    def format_response(self, results):
        if not results:
            return "No medical products found."
        out=[]
        for r in results:
            out.append(f"• {r.get('title')} (${r.get('price')})\n  {r.get('url')}")
        return "\n".join(out)
