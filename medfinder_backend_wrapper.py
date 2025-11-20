import os
from flask import Flask, request, jsonify
import importlib, traceback, inspect
from ai.ai_agent import MedFinderAI
from ai.ebay_api import search_ebay, get_access_token

app = Flask(__name__)
agent = MedFinderAI()

@app.route("/")
def home():
    return jsonify({
        "message": "MedFinder AI backend running successfully!",
        "routes": ["/api/search", "/api/live"],
        "status": "OK"
    })

def parse_query_struct(q_raw):
    """
    Use the AI agent to parse the raw user query into a structured dict.
    """
    try:
        return agent.optimize_query(q_raw)
    except Exception:
        # fallback minimal structure
        return {"query": q_raw, "is_medical": False, "max_price": None, "currency": None}

@app.route('/api/search')
def api_search():
    q = request.args.get('q','')
    limit = int(request.args.get('limit',20))
    if not q.strip():
        return jsonify({'error':'no query provided'}),400

    try:
        # structured query
        q_struct = parse_query_struct(q)

        # import semantic search module and call enhanced_search with structured dict
        mod = importlib.import_module('ai.semantic_search_ai')
        func = getattr(mod,'enhanced_search',None)
        if func is None:
            return jsonify({'error':'enhanced_search missing'}),500

        # call
        sig = inspect.signature(func)
        if 'limit' in sig.parameters:
            res = func(q_struct, limit=limit)
        else:
            res = func(q_struct)
        return jsonify({'query':q,'structured':q_struct,'limit':limit,'results':res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'search failed','detail':str(e)}),500

@app.route('/api/live')
def api_live():
    q = request.args.get('q','')
    limit = int(request.args.get('limit',10))
    if not q.strip():
        return jsonify({'error':'no query provided'}),400
    try:
        tok = get_access_token()
        res = search_ebay(q, tok, limit)
        return jsonify({'query':q,'limit':limit,'results':res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'live ebay search failed','detail':str(e)}),500

if __name__=='__main__':
    port = int(os.getenv('PORT',8501))
    app.run(host='0.0.0.0', port=port)
