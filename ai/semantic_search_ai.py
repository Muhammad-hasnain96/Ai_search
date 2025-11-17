import os, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from .ai_agent import MedFinderAI
from .ebay_api import search_ebay, get_valid_token

# Load embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except:
    embedding_model = None

ai_agent = MedFinderAI()

BASE = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE, "..", "embeddings", "metadata.json")

index = None
metadata = []

# Load FAISS index & metadata if exists
try:
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH,"r",encoding="utf-8") as f:
            metadata=json.load(f)
except:
    index = None
    metadata = []

def is_medical_product(title, query):
    t, q = title.lower(), query.lower()
    terms = [w for w in q.split() if len(w)>2]
    must = any(w in t for w in terms)
    med_terms = ["medical","health","surgical","hospital","glove","bp",
                 "blood pressure","stethoscope","mask","thermometer","bandage",
                 "rehab","wheelchair","clinic","oxygen","pulse","nebulizer",
                 "walker","hearing","care","sanitizer","brace","pill","medicine",
                 "drug","first aid","iv","infusion","orthopedic","dental",
                 "urine","urinal","catheter"]
    return must and any(m in t for m in med_terms)

def semantic_search(query, threshold=0.65, k=15):
    if not index or not metadata or not embedding_model:
        return [], 0.0
    q_emb = np.array(embedding_model.encode([query], normalize_embeddings=True), dtype=np.float32)
    scores, ids = index.search(q_emb, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx].copy()
            if score >= threshold and is_medical_product(item.get("title",""), query):
                item["score"] = float(score)
                out.append(item)
    return out, float(np.max(scores[0])) if len(scores[0]) else 0.0

def enhanced_search(user_query, limit=10):
    opt = ai_agent.optimize_query(user_query)
    local, sc = semantic_search(opt)
    res = local[:limit]

    if not res or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt, tok, limit)
            res.extend(items)
        except:
            pass

    seen, uniq = set(), []
    for it in res:
        t = it.get("title","").lower().strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(it)
    return uniq[:limit]
