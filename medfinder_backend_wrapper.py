import os, importlib, traceback, inspect
from flask import Flask, request, jsonify
from ai.ai_agent import MedFinderAI
from ai.ebay_api import search_ebay, get_access_token

app = Flask(__name__)
agent = MedFinderAI()

@app.route("/")
def home():
    return jsonify({"message":"MedFinder AI backend running!","routes":["/api/search","/api/live"],"status":"OK"})

@app.route("/api/search")
def api_search():
    q = request.args.get("q","").strip()
    limit = int(request.args.get("limit",20))
    if not q: return jsonify({"error":"no query provided"}),400
    try:
        q_struct = agent.parse_query(q)
        mod = importlib.import_module("ai.semantic_search_ai")
        func = getattr(mod,"enhanced_search",None)
        if func is None: return jsonify({"error":"enhanced_search missing"}),500
        sig = inspect.signature(func)
        res = func(q_struct, limit=limit) if "limit" in sig.parameters else func(q_struct)
        return jsonify({"query":q,"structured":q_struct,"limit":limit,"results":res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"search failed","detail":str(e)}),500

@app.route("/api/live")
def api_live():
    q = request.args.get("q","").strip()
    limit = int(request.args.get("limit",10))
    if not q: return jsonify({"error":"no query provided"}),400
    try:
        tok = get_access_token()
        res = search_ebay(q,tok,limit)
        return jsonify({"query":q,"limit":limit,"results":res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"live ebay search failed","detail":str(e)}),500

if __name__=="__main__":
    port = int(os.getenv("PORT",8501))
    app.run(host="0.0.0.0", port=port)
