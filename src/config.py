import os

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "communications.csv")

# Paramètres Marché
MARKET_TICKER = "^SPX"  # ou "^GSPC" pour S&P 500 via yfinance
START_DATE = "2000-01-01"

# Paramètres NLP
BERT_MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_LEN = 512
CHUNK_SIZE = 512

# Filtres Disours
RELEVANT_TYPES = ['Statement', 'Press Conference', 'Minute']
MIN_TEXT_LENGTH = 200
KEYWORDS = ['interest', 'rate', 'inflation', 'policy', 'employment', 'economy']

# Random Forest
RF_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE = 0.2
