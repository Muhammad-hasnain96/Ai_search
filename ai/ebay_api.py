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

    auth = base64.b64encode(f"{config.CLIENT_ID}:{config.CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type":"application/x-www-form-urlencoded"}
    data = {"grant_type":"refresh_token","refresh_token":config.REFRESH_TOKEN,"scope":"https://api.ebay.com/oauth/api_scope"}
    response = requests.post(config.OAUTH_URL, headers=headers, data=data)
    response.raise_for_status()
    token_data = response.json()
    try:
        with open(token_file,"w") as f:
            json.dump(token_data,f)
    except: pass
    return token_data["access_token"]

def get_valid_token():
    try: return get_access_token(False)
    except: return get_access_token(True)

def clean_query(query):
    query = query.lower()
    query = re.sub(r"\b(give me|suggest|show|find|best|recommend|buy|cheap|under|below|less than|up to|upto)\b", "", query)
    return query.strip()

def search_ebay(query, token, limit=5):
    headers = {"Authorization": f"Bearer {token}","Content-Type":"application/json"}
    query = clean_query(query)
    all_results = []

    for cid in MEDICAL_CATEGORIES:
        try:
            r = requests.get(config.BUY_BROWSE_URL, headers=headers, params={"q":query,"limit":limit,"category_ids":cid}, timeout=8)
        except: continue
        if r.status_code != 200: continue
        data = r.json()
        for it in data.get("itemSummaries", []):
            all_results.append({
                "title": it.get("title",""),
                "price": it.get("price",{}).get("value",None),
                "currency": it.get("price",{}).get("currency",None),
                "url": it.get("itemWebUrl","#"),
                "condition": it.get("condition","N/A"),
                "image": it.get("image",{}).get("imageUrl","")
            })

    if not all_results:
        try:
            r = requests.get(config.BUY_BROWSE_URL, headers=headers, params={"q":query,"limit":limit}, timeout=8)
            if r.status_code==200:
                data = r.json()
                for it in data.get("itemSummaries", []):
                    all_results.append({
                        "title": it.get("title",""),
                        "price": it.get("price",{}).get("value",None),
                        "currency": it.get("price",{}).get("currency",None),
                        "url": it.get("itemWebUrl","#"),
                        "condition": it.get("condition","N/A"),
                        "image": it.get("image",{}).get("imageUrl","")
                    })
        except: pass

    seen=set(); uniq=[]
    for it in all_results:
        key = ((it.get("title") or "").lower().strip(), it.get("url") or "")
        if key in seen: continue
        seen.add(key); uniq.append(it)
        if len(uniq)>=limit: break
    return uniq
