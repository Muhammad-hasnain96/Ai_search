import os, json, faiss, numpy as np
from ai.ebay_api import search_ebay, get_valid_token
from ai.ai_agent import MedFinderAI

print("Loading FAISS + metadata...")

# ---------- LAZY LOAD EMBEDDING MODEL ----------
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


# ---------- AI AGENT ----------
ai_agent = MedFinderAI()


# ---------- FAISS + METADATA ----------
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
except:
    index = None
    metadata = []


# ---------- FILTER ----------
def is_medical_product(title, query):
    t = title.lower()
    q = query.lower()
    terms = [w for w in q.split() if len(w) > 2]

    must = any(w in t for w in terms)

    med = [
        "medical", "health", "surgical", "hospital", "glove", "monitor", "bp",
        "blood pressure", "stethoscope", "mask", "thermometer", "bandage",
        "rehab", "wheelchair", "clinic", "oxygen", "pulse", "nebulizer",
        "walker", "hearing", "care", "sanitizer", "brace", "pill", "medicine",
        "drug", "first aid", "iv", "infusion", "orthopedic", "dental",
        "urine", "urinal", "catheter"
    ]
    has = any(m in t for m in med)

    return must and has


# ---------- SEMANTIC SEARCH ----------
def semantic_search(query, threshold=0.65, k=15):
    if index is None or not metadata:
        return [], 0.0

    model = get_embedding_model()  # lazy load
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


# ---------- ENHANCED SEARCH ----------
def enhanced_search(user_query, limit=10):
    opt = ai_agent.optimize_query(user_query)

    local, sc = semantic_search(opt)
    res = local[:limit]

    # fallback: eBay search API
    if not res or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt, tok, limit)
            filt = [i for i in items if is_medical_product(i.get("title", ""), opt)]
            res.extend(filt)
        except:
            pass

    # remove duplicates
    seen = set()
    uniq = []
    for it in res:
        title = it.get("title", "").lower().strip()
        if title and title not in seen:
            seen.add(title)
            uniq.append(it)

    return uniq
