"""
Configuration settings for S&P 500 Options Scanner
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration - ALPACA (Primary)
ALPACA_API_KEY_ID = os.getenv('ALPACA_API_KEY_ID', 'PKY99PILSOIXLP7953TG')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'n3VUzYUwojjyOzs4k1IYYj1lPNBVCqh9vWXZCL6D')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
ALPACA_DATA_URL = 'https://data.alpaca.markets'  # Market data endpoint

# Legacy Finnhub configuration (kept for backward compatibility)
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'd296qf9r01qhoena9cp0d296qf9r01qhoena9cpg')
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Data settings
INTRADAY_RESOLUTION = '15'  # 15-minute bars
LOOKBACK_DAYS = 3  # Days of historical data to fetch
MIN_REQUIRED_BARS = 30  # Minimum bars needed for calculations

# Technical indicator periods (in bars, not days)
RSI_PERIOD = 14  # 14 bars = 3.5 hours
BOLLINGER_PERIOD = 20  # 20 bars = 5 hours
BOLLINGER_STD = 2  # Standard deviations
MACD_FAST = 12  # 12 bars = 3 hours
MACD_SLOW = 26  # 26 bars = 6.5 hours
MACD_SIGNAL = 9  # 9 bars = 2.25 hours
OBV_SMA_PERIOD = 20  # 20 bars = 5 hours

# Scoring thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Scoring weights
WEIGHT_RSI = 0.30  # 30%
WEIGHT_MACD = 0.30  # 30%
WEIGHT_BOLLINGER = 0.20  # 20%
WEIGHT_OBV = 0.20  # 20%

# Signal thresholds
SIGNAL_STRONG_BUY = 85
SIGNAL_BUY = 70
SIGNAL_HOLD_MIN = 30
SIGNAL_HOLD_MAX = 70

# API rate limiting
API_RATE_LIMIT_DELAY = 1.0  # Seconds between API calls
API_MAX_RETRIES = 3
API_RETRY_DELAY = 5  # Seconds between retries

# Cache settings
CACHE_DIR = 'cache/intraday'
CACHE_EXPIRY_MINUTES = 15  # Cache validity period
SP500_CACHE_EXPIRY_DAYS = 1  # S&P 500 list cache validity

# Output settings
OUTPUT_DIR = 'output/intraday_scans'
LOG_DIR = 'logs'
DECIMAL_PLACES = 1  # For score display

# Dashboard settings
REFRESH_INTERVAL_MINUTES = 30  # Auto-refresh interval
TOP_STOCKS_DISPLAY = 10  # Number of top stocks to show
SHOW_DETAILED_INDICATORS = True  # Show individual indicator values

# Console colors (for rich library)
COLOR_STRONG_BUY = 'bright_green'
COLOR_BUY = 'green'
COLOR_HOLD = 'white'
COLOR_AVOID = 'red'

# Create directories if they don't exist
for directory in [CACHE_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)