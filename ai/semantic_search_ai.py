import os, json, faiss, numpy as np
from ai.ebay_api import search_ebay, get_valid_token
from ai.ai_agent import MedFinderAI

print("Loading FAISS + metadata...")

# lazy embedding model
embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

ai_agent = MedFinderAI()

BASE = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE, "..", "embeddings", "metadata.json")

index = None
metadata = []
try:
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("FAISS loaded:", len(metadata))
except Exception:
    index = None
    metadata = []

# medical term list (for stricter filtering)
MEDICAL_TERMS = [
    "medical", "health", "surgical", "hospital", "glove", "monitor", "bp",
    "blood pressure", "stethoscope", "mask", "thermometer", "bandage",
    "rehab", "wheelchair", "clinic", "oxygen", "pulse", "nebulizer",
    "walker", "hearing", "care", "sanitizer", "brace", "pill", "medicine",
    "drug", "first aid", "iv", "infusion", "orthopedic", "dental",
    "urine", "urinal", "catheter"
]

def is_medical_product(title, query_text):
    t = (title or "").lower()
    q = (query_text or "").lower()
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
        title = item.get("title", "")
        if score >= threshold and is_medical_product(title, query):
            item["score"] = float(score)
            out.append(item)
    maxs = float(np.max(scores[0])) if len(scores[0]) else 0.0
    return out, maxs

def enhanced_search(user_query, limit=10):
    """
    user_query: can be a raw string OR dict returned from ai_agent.optimize_query
    returns list of product dicts (no hallucinated items)
    """
    # accept structured or raw
    if isinstance(user_query, dict):
        q_struct = user_query
    else:
        q_struct = ai_agent.optimize_query(user_query)

    opt_query = q_struct.get("query")
    is_med = q_struct.get("is_medical", False)
    max_price = q_struct.get("max_price", None)
    currency = q_struct.get("currency", None)

    # 1) semantic local search
    local, sc = semantic_search(opt_query)
    results = local[:limit]

    # 2) if not enough or low score, fallback to eBay live search
    if len(results) < 1 or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt_query, tok, limit)
            # filter items: strict for medical, looser for general
            filtered = []
            for it in items:
                title = it.get("title", "")
                # ensure we only include items with real data
                if not title or not it.get("url"):
                    continue
                # price filtering (if user asked a price)
                if max_price is not None:
                    price_val = it.get("price")
                    currency_item = it.get("currency")
                    # If currency specified and matches, compare numeric
                    if currency and currency_item and currency_item.upper() != currency.upper():
                        # skip item if currency mismatch (avoid wrong filtering)
                        # We keep it if currency not provided by user or item -- but here user asked a specific currency so skip
                        continue
                    try:
                        if price_val is not None and float(price_val) <= float(max_price):
                            filtered.append(it)
                        else:
                            # if price unknown, skip (avoid hallucination)
                            continue
                    except Exception:
                        continue
                else:
                    filtered.append(it)
            results.extend(filtered)
        except Exception:
            pass

    # 3) dedupe and final validation (avoid hallucination)
    seen = set()
    uniq = []
    for it in results:
        title = (it.get("title") or "").strip()
        url = it.get("url") or it.get("itemWebUrl") or ""
        if not title or not url:
            continue
        key = (title.lower(), url)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= limit:
            break

    return uniq
