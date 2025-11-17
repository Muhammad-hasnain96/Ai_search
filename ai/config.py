import os

# eBay credentials (stored in Railway environment variables)
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')

# Token file location (auto token caching)
TOKEN_FILE = os.getenv('TOKEN_FILE', 'ebay_token.json')

# eBay API URLs
OAUTH_URL = os.getenv('OAUTH_URL', 'https://api.ebay.com/identity/v1/oauth2/token')
BUY_BROWSE_URL = os.getenv('BUY_BROWSE_URL', 'https://api.ebay.com/buy/browse/v1/item_summary/search')

# Token refresh interval (seconds)
REFRESH_INTERVAL = int(os.getenv('REFRESH_INTERVAL', '7200'))

# Server port
PORT = int(os.getenv('PORT', 8501))
