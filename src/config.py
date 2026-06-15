# src/config.py

ASSETS = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "EUR/USD": "EURUSD=X",
}

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

TECHNICAL_WINDOWS = {
    "ma_short": 20,
    "ma_long": 50,
    "rsi": 14,
    "bollinger": 20,
}

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
}
