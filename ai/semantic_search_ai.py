import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from ai.ebay_api import search_ebay, get_valid_token
from ai.ai_agent import MedFinderAI

embedding_model = None
ai_agent = MedFinderAI()
index = None
metadata = []

BASE = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE, "..", "embeddings", "metadata.json")

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except:
    embedding_model = None

try:
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH,"r",encoding="utf-8") as f:
            metadata=json.load(f)
except:
    index=None
    metadata=[]

def semantic_search(query, threshold=0.65, k=15):
    if index is None or not metadata or embedding_model is None:
        return [], 0.0
    q_emb = embedding_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    scores, ids = index.search(q_emb, k)

    out=[]
    for score, idx in zip(scores[0], ids[0]):
        if idx<0 or idx>=len(metadata):
            continue
        item=metadata[idx].copy()
        title=item.get("title","")
        if score>=threshold:
            item["score"]=float(score)
            out.append(item)

    maxs=float(np.max(scores[0])) if len(scores[0]) else 0.0
    return out, maxs

def enhanced_search(user_query, limit=10, allow_general=True):
    opt = ai_agent.optimize_query(user_query)
    local, sc = semantic_search(opt)
    res = local[:limit]

    if (not res or sc < 0.65) and allow_general:
        try:
            tok = get_valid_token()
            items = search_ebay(user_query, tok, limit)
            res.extend(items)
        except:
            pass

    seen=set()
    uniq=[]
    for it in res:
        t=it.get("title","").lower().strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(it)

    return uniq[:limit]
