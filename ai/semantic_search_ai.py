import os, json, numpy as np, faiss
from ai.ebay_api import search_ebay, get_valid_token
from ai.ai_agent import MedFinderAI

ai_agent = MedFinderAI()

# FAISS + metadata
BASE = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE, "..", "embeddings", "metadata.json")

index = None
metadata = []
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

try:
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
except:
    index, metadata = None, []

MEDICAL_TERMS = [
    "medical","health","surgical","hospital","glove","monitor","bp",
    "blood pressure","stethoscope","mask","thermometer","bandage",
    "rehab","wheelchair","clinic","oxygen","pulse","nebulizer",
    "walker","hearing","care","sanitizer","brace","pill","medicine",
    "drug","first aid","iv","infusion","orthopedic","dental",
    "urine","urinal","catheter"
]

def is_medical_product(title, query_text):
    t, q = (title or "").lower(), (query_text or "").lower()
    terms = [w for w in q.split() if len(w) > 2]
    must = any(w in t for w in terms) if terms else True
    has = any(m in t for m in MEDICAL_TERMS)
    if any(m in q for m in MEDICAL_TERMS):
        return must and has
    return True

def semantic_search(query, threshold=0.65, k=15):
    if index is None or not metadata:
        return [], 0.0
    model = get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    scores, ids = index.search(q_emb, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        if score >= threshold and is_medical_product(item.get("title",""), query):
            item["score"] = float(score)
            out.append(item)
    return out, float(np.max(scores[0])) if scores.size else 0.0

def enhanced_search(user_query, limit=10):
    """Context-aware search for medical + general products with price filter"""
    q_struct = user_query if isinstance(user_query, dict) else ai_agent.optimize_query(user_query)
    opt_query = q_struct.get("query")
    is_med = q_struct.get("is_medical", False)
    max_price = q_struct.get("max_price", None)
    currency = q_struct.get("currency", None)

    # Semantic local search
    local, sc = semantic_search(opt_query)
    results = local[:limit]

    # eBay fallback
    if len(results) < 1 or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt_query, tok, limit)
            filtered = []
            for it in items:
                title = it.get("title") or ""
                if not title or not it.get("url"):
                    continue
                if max_price is not None:
                    price_val = it.get("price")
                    currency_item = it.get("currency")
                    if currency and currency_item and currency_item.upper() != currency.upper():
                        continue
                    try:
                        if price_val is None or float(price_val) > float(max_price):
                            continue
                    except:
                        continue
                if is_med and not is_medical_product(title, opt_query):
                    continue
                filtered.append(it)
            results.extend(filtered)
        except:
            pass

    # Dedupe
    seen = set()
    uniq = []
    for it in results:
        title = (it.get("title") or "").strip()
        url = it.get("url") or it.get("itemWebUrl") or ""
        if not title or not url: continue
        key = (title.lower(), url)
        if key in seen: continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= limit:
            break
    return uniq
