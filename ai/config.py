import os

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')

TOKEN_FILE = os.getenv('TOKEN_FILE', 'ebay_token.json')
OAUTH_URL = os.getenv('OAUTH_URL', 'https://api.ebay.com/identity/v1/oauth2/token')
BUY_BROWSE_URL = os.getenv('BUY_BROWSE_URL', 'https://api.ebay.com/buy/browse/v1/item_summary/search')

REFRESH_INTERVAL = int(os.getenv('REFRESH_INTERVAL', '7200'))
PORT = int(os.getenv('PORT', 8501))
