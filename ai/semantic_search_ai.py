import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from ai.ai_agent import MedFinderAI
from ai.ebay_api import search_ebay, get_valid_token

ai_agent = MedFinderAI()

BASE = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE, "..", "embeddings", "metadata.json")

index = None
metadata = []
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

# Load FAISS index & metadata
try:
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
except:
    index, metadata = None, []

CURRENCY_RATES = {"USD":1.0,"PKR":0.0055,"EUR":1.1,"GBP":1.25,"INR":0.012}

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
        item["score"] = float(score)
        out.append(item)
    maxs = float(np.max(scores[0])) if len(scores[0]) else 0.0
    return out, maxs

def enhanced_search(user_query, limit=10):
    # user_query can be dict or string
    if isinstance(user_query, str):
        q_struct = ai_agent.parse_query(user_query)
    else:
        q_struct = user_query

    opt_query = q_struct.get("query")
    max_price = q_struct.get("max_price")
    currency = q_struct.get("currency")

    local, sc = semantic_search(opt_query)
    results = [r for r in local if is_price_ok(r.get("price"), r.get("currency"), max_price, currency)][:limit]

    # Fallback to eBay if local results insufficient
    if len(results) < 1 or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt_query, tok, limit)
            for it in items:
                if is_price_ok(it.get("price"), it.get("currency"), max_price, currency):
                    results.append(it)
        except:
            pass

    # Deduplicate
    seen = set()
    uniq = []
    for it in results:
        key = (it.get("title","").lower(), it.get("url",""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= limit:
            break

    return uniq
