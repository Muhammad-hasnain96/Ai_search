from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

class MedFinderAI:
    """
    AI-powered query understanding using Flan-T5.
    Returns structured query: text, is_medical, max_price, currency
    """
    CURRENCY_SYMBOLS = {
        '$':'USD','usd':'USD','€':'EUR','eur':'EUR','£':'GBP','gbp':'GBP','₹':'INR','inr':'INR',
        'rs':'PKR','pkr':'PKR','pk':'PKR'
    }

    PRICE_RE = re.compile(
        r'(?:under|below|less than|upto|up to|<=|≤|<)?\s*'
        r'(?P<symbol>[$€£₹])?\s*'
        r'(?P<amount>\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?)\s*'
        r'(?P<code>[A-Za-z]{2,4}|rs|pkr)?', flags=re.IGNORECASE
    )

    def __init__(self):
        # Load Flan-T5 small (lightweight, CPU-friendly)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    def extract_price(self, text):
        m = self.PRICE_RE.search(text)
        if not m:
            return None, None
        try:
            amount = float(m.group("amount").replace(",", ""))
        except:
            amount = None
        cur = None
        symbol = m.group("symbol")
        code = m.group("code")
        if symbol:
            cur = self.CURRENCY_SYMBOLS.get(symbol.lower(), None)
        if code:
            cur = self.CURRENCY_SYMBOLS.get(code.lower(), cur) or code.upper()
        return amount, cur

    def parse_query(self, text):
        """
        Use Flan-T5 to classify query type and extract main text.
        """
        prompt = f"Classify this product query as medical or general, extract main product name:\nQuery: {text}\nRespond in JSON format with fields: query, is_medical"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64)
        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Expecting JSON response
        try:
            import json
            parsed = json.loads(resp)
            q_text = parsed.get("query", text)
            is_med = bool(parsed.get("is_medical", False))
        except:
            q_text, is_med = text, False

        max_price, currency = self.extract_price(text)
        return {
            "query": q_text.strip(),
            "is_medical": is_med,
            "max_price": float(max_price) if max_price else None,
            "currency": currency
        }
