import requests, json, base64, os, re
from ai import config

MEDICAL_CATEGORIES = ["11815","177646","40943","18412","11818","10968"]

def get_access_token(force_refresh=False):
    token_file = config.TOKEN_FILE
    if not force_refresh and os.path.exists(token_file):
        try:
            with open(token_file,"r") as f:
                token_data = json.load(f)
                if token_data.get("access_token"):
                    return token_data["access_token"]
        except:
            pass

    if not (config.CLIENT_ID and config.CLIENT_SECRET and config.REFRESH_TOKEN):
        raise RuntimeError("eBay credentials missing in config")

    auth = base64.b64encode(f"{config.CLIENT_ID}:{config.CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type":"application/x-www-form-urlencoded"}
    data = {"grant_type":"refresh_token","refresh_token":config.REFRESH_TOKEN,"scope":"https://api.ebay.com/oauth/api_scope"}
    response = requests.post(config.OAUTH_URL, headers=headers, data=data, timeout=10)
    response.raise_for_status()
    token_data = response.json()
    try:
        with open(token_file,"w") as f:
            json.dump(token_data,f)
    except:
        pass
    return token_data["access_token"]

def get_valid_token():
    try:
        return get_access_token(False)
    except:
        return get_access_token(True)

# clean query used for eBay Browse endpoint
def clean_query(query: str) -> str:
    if not query:
        return ""
    q = query.lower()
    q = re.sub(r"\b(give me|suggest|show|find|best|recommend|buy|cheap|under|below|less than|up to|upto)\b", "", q)
    q = " ".join(q.split())
    return q.strip()

def _to_float(v):
    try:
        return float(v)
    except:
        return None

def search_ebay(query: str, token: str, limit: int = 5):
    """
    Search eBay Browse API. Returns list of normalized items:
    { title, price (float|None), currency, url, condition, image }
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    q = clean_query(query)
    all_results = []

    # try medical categories first (better precision)
    for cid in MEDICAL_CATEGORIES:
        try:
            r = requests.get(config.BUY_BROWSE_URL, headers=headers,
                             params={"q": q, "limit": limit, "category_ids": cid}, timeout=8)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        for it in data.get("itemSummaries", []):
            price_val = it.get("price", {}).get("value", None)
            price_val = _to_float(price_val)
            all_results.append({
                "title": it.get("title",""),
                "price": price_val,
                "currency": it.get("price",{}).get("currency", None),
                "url": it.get("itemWebUrl","#"),
                "condition": it.get("condition","N/A"),
                "image": it.get("image",{}).get("imageUrl","")
            })

    # fallback to general search if nothing found
    if not all_results:
        try:
            r = requests.get(config.BUY_BROWSE_URL, headers=headers, params={"q": q, "limit": limit}, timeout=8)
            if r.status_code == 200:
                try:
                    data = r.json()
                except:
                    data = {}
                for it in data.get("itemSummaries", []):
                    price_val = it.get("price", {}).get("value", None)
                    price_val = _to_float(price_val)
                    all_results.append({
                        "title": it.get("title",""),
                        "price": price_val,
                        "currency": it.get("price",{}).get("currency", None),
                        "url": it.get("itemWebUrl","#"),
                        "condition": it.get("condition","N/A"),
                        "image": it.get("image",{}).get("imageUrl","")
                    })
        except Exception:
            pass

    # dedupe by (title,url)
    seen = set()
    uniq = []
    for it in all_results:
        key = (((it.get("title") or "").lower().strip()), it.get("url") or "")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= limit:
            break

    return uniq
