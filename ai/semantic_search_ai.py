import os, json, faiss, numpy as np
from ai.ai_agent import MedFinderAI
from ai.ebay_api import search_ebay, get_valid_token

print("Loading FAISS + metadata...")
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

MEDICAL_TERMS = [
    "medical","health","surgical","hospital","glove","monitor","bp",
    "blood pressure","stethoscope","mask","thermometer","bandage",
    "rehab","wheelchair","clinic","oxygen","pulse","nebulizer",
    "walker","hearing","care","sanitizer","brace","pill","medicine",
    "drug","first aid","iv","infusion","orthopedic","dental",
    "urine","urinal","catheter"
]

CURRENCY_RATES = {"USD":1.0,"PKR":0.0055,"EUR":1.1,"GBP":1.25,"INR":0.012}

def is_medical_product(title, query_text):
    t = (title or "").lower()
    q = (query_text or "").lower()
    terms = [w for w in q.split() if len(w) > 2]
    must = any(w in t for w in terms) if terms else True
    has = any(m in t for m in MEDICAL_TERMS)
    if any(m in q for m in MEDICAL_TERMS):
        return must and has
    return True

def is_price_ok(item_price, item_currency, user_price, user_currency):
    if item_price is None or user_price is None:
        return True
    try:
        item_cur = item_currency.upper() if item_currency else "USD"
        user_cur = user_currency.upper() if user_currency else "USD"
        item_usd = float(item_price) * CURRENCY_RATES.get(item_cur, 1)
        user_usd = float(user_price) * CURRENCY_RATES.get(user_cur, 1)
        return item_usd <= user_usd
    except:
        return False

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
    if isinstance(user_query, dict):
        q_struct = user_query
    else:
        q_struct = ai_agent.optimize_query(user_query)

    opt_query = q_struct.get("query")
    is_med = q_struct.get("is_medical", False)
    max_price = q_struct.get("max_price", None)
    currency = q_struct.get("currency", None)

    local, sc = semantic_search(opt_query)
    results = local[:limit]

    if len(results) < 1 or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt_query, tok, limit)
            filtered = []
            for it in items:
                title = it.get("title", "")
                if not title or not it.get("url"):
                    continue
                if max_price is not None:
                    if not is_price_ok(it.get("price"), it.get("currency"), max_price, currency):
                        continue
                filtered.append(it)
            results.extend(filtered)
        except:
            pass

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
