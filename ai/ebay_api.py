import requests, json, base64, os, re
from ai import config

MEDICAL_CATEGORIES = ["11815","177646","40943","18412","11818","10968"]

def get_access_token(force_refresh=False):
    token_file = config.TOKEN_FILE
    if not force_refresh and os.path.exists(token_file):
        with open(token_file,"r") as f:
            data = json.load(f)
            if data.get("access_token"):
                return data["access_token"]

    auth = base64.b64encode(f"{config.CLIENT_ID}:{config.CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type":"refresh_token","refresh_token":config.REFRESH_TOKEN,"scope":"https://api.ebay.com/oauth/api_scope"}

    r = requests.post(config.OAUTH_URL, headers=headers, data=data)
    r.raise_for_status()
    token_data = r.json()

    try:
        with open(token_file,"w") as f:
            json.dump(token_data,f)
    except: pass

    return token_data["access_token"]

def get_valid_token():
    try:
        return get_access_token(False)
    except:
        return get_access_token(True)

def clean_query(query):
    query = query.lower()
    return re.sub(r"\b(give me|suggest|show|find|best|recommend|buy|cheap)\b","",query).strip()

def search_ebay(query, token, limit=10):
    headers = {"Authorization": f"Bearer {token}","Content-Type":"application/json"}
    query = clean_query(query)
    all_results=[]

    # Search both medical and general categories
    for cid in MEDICAL_CATEGORIES + ["0"]:  # "0" = general
        params = {"q": query, "limit": limit, "category_ids": cid} if cid != "0" else {"q": query, "limit": limit}
        r = requests.get(config.BUY_BROWSE_URL, headers=headers, params=params)
        if r.status_code != 200: continue
        data = r.json()
        for it in data.get("itemSummaries", []):
            all_results.append({
                "title": it.get("title",""),
                "price": it.get("price",{}).get("value","N/A"),
                "currency": it.get("price",{}).get("currency","USD"),
                "url": it.get("itemWebUrl","#"),
                "condition": it.get("condition","N/A"),
                "image": it.get("image",{}).get("imageUrl","")
            })

    seen=set()
    uniq=[]
    for it in all_results:
        t = it["title"].lower().strip()
        if t not in seen:
            seen.add(t)
            uniq.append(it)

    return uniq
