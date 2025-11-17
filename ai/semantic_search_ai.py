import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from ai.ebay_api import search_ebay, get_valid_token
from ai.ai_agent import MedFinderAI

print("Loading FAISS & metadata...")

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

try:
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("FAISS loaded:", len(metadata))
except:
    index = None
    metadata = []

# ------------------------
# Multi-field category keywords
CATEGORY_KEYWORDS = {
    "medical": ["medical","health","surgical","bp","mask","glove","thermometer","stethoscope"],
    "electronics": ["laptop","phone","camera","headphones","smartwatch"],
    "books": ["book","novel","textbook","guide","manual"],
    "clothing": ["shirt","pants","dress","jacket","shoes"]
}

def matches_category(title, inferred_label):
    if not title:
        return False
    title = title.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if inferred_label.lower() in " ".join(keywords):
            return any(k in title for k in keywords)
    return True  # fallback: include all

def semantic_search(query, threshold=0.65, k=15):
    if index is None or not metadata or embedding_model is None:
        return [], 0.0
    q_emb = embedding_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    scores, ids = index.search(q_emb, k)

    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        if score >= threshold:
            out.append(item)

    maxs = float(np.max(scores[0])) if len(scores[0]) else 0.0
    return out, maxs

def enhanced_search(user_query, limit=10):
    inferred_label = ai_agent.infer_intent(user_query)
    opt = ai_agent.optimize_query(user_query)

    local, sc = semantic_search(opt)
    res = [item for item in local if matches_category(item.get("title", ""), inferred_label)][:limit]

    if not res or sc < 0.65:
        try:
            tok = get_valid_token()
            items = search_ebay(opt, tok, limit)
            res.extend([i for i in items if matches_category(i.get("title", ""), inferred_label)])
        except:
            pass

    seen = set()
    uniq = []
    for it in res:
        t = it.get("title", "").lower().strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(it)

    return uniq
