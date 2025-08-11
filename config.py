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
LOOKBACK_DAYS = 7  # Days of historical data to fetch (7 days = ~182 15-min bars, enough for 50-period MA)
MIN_REQUIRED_BARS = 30  # Minimum bars needed for calculations

# Technical indicator periods (in bars, not days)
RSI_PERIOD = 14  # 14 bars = 3.5 hours
BOLLINGER_PERIOD = 20  # 20 bars = 5 hours
BOLLINGER_STD = 2  # Standard deviations
MACD_FAST = 12  # 12 bars = 3 hours
MACD_SLOW = 26  # 26 bars = 6.5 hours
MACD_SIGNAL = 9  # 9 bars = 2.25 hours
OBV_SMA_PERIOD = 20  # 20 bars = 5 hours
ATR_PERIOD = 14  # 14 bars for ATR calculation
ATR_SMA_PERIOD = 30  # 30 bars for ATR SMA

# Market regime filter - Three factors
MARKET_REGIME_MA_PERIOD = 50  # 50-day moving average for SPY
MARKET_BREADTH_THRESHOLD = 60  # Minimum % of stocks above their 50MA for bullish
VIX_THRESHOLD = 25  # Maximum VIX level for bullish regime
VIX_WARNING_THRESHOLD = 20  # VIX level for caution alert
MIN_TRADING_DAYS_REQUIRED = 252  # Minimum days of history required

# Market regime behavior
HALT_ON_BEARISH_REGIME = False  # If True, stops scan when regime not bullish
SHOW_REGIME_WARNING = True  # Display warnings when regime not bullish

# Scanner Mode Settings
DEFAULT_SCANNER_MODE = 'adaptive'  # 'adaptive', 'bullish', 'bearish', 'mixed'
BEARISH_BREADTH_THRESHOLD = 40  # Below this = bearish mode
BULLISH_BREADTH_THRESHOLD = 60  # Above this = bullish mode
BEARISH_VIX_THRESHOLD = 25  # Above this = bearish mode
BULLISH_VIX_THRESHOLD = 20  # Below this = bullish mode
MIXED_MODE_MIN_SCORE = 70  # Higher threshold for mixed markets
TREND_CONFIRMATION_THRESHOLD = 0.5  # Min % distance from MA for trend confirmation

# Future defensive mode thresholds (Phase 2)
EXTREME_VIX_THRESHOLD = 30  # High volatility opportunities
DEFENSIVE_MODE_ENABLED = False  # Future feature flag

# Scoring thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Scoring weights (must sum to 1.0)
WEIGHT_RSI = 0.25  # 25%
WEIGHT_MACD = 0.25  # 25%
WEIGHT_BOLLINGER = 0.20  # 20%
WEIGHT_OBV = 0.10  # 10%
WEIGHT_VOLUME = 0.10  # 10% (relative volume)
WEIGHT_ATR = 0.10  # 10% (volatility expansion)

# Signal thresholds (5 categories for nuanced strategies)
SIGNAL_STRONG_BUY = 85  # > 85: Strong bullish
SIGNAL_BUY = 70  # 70-85: Bullish
SIGNAL_NEUTRAL_BULLISH = 50  # 50-70: Neutral bullish
SIGNAL_NEUTRAL_BEARISH = 30  # 30-50: Neutral bearish
SIGNAL_STRONG_SELL = 30  # < 30: Strong bearish
# Legacy compatibility
SIGNAL_HOLD_MIN = 30
SIGNAL_HOLD_MAX = 70

# API rate limiting
API_RATE_LIMIT_DELAY = 1.0  # Seconds between API calls
API_MAX_RETRIES = 3
API_RETRY_DELAY = 5  # Seconds between retries

# Cache settings
CACHE_DIR = 'cache/intraday'
CACHE_EXPIRY_MINUTES = 15  # Cache validity period for intraday data
SP500_CACHE_EXPIRY_DAYS = 1  # S&P 500 list cache validity
DAILY_CACHE_EXPIRY_HOURS = 24  # Daily data cache validity
BREADTH_CACHE_HOURS = 24  # Cache market breadth calculation
VIX_CACHE_MINUTES = 60  # Cache VIX data for 1 hour
SPY_CACHE_MINUTES = 60  # Cache SPY data for 1 hour

# Output settings
OUTPUT_DIR = 'output/intraday_scans'
LOG_DIR = 'logs'
DECIMAL_PLACES = 1  # For score display

# Dashboard settings
REFRESH_INTERVAL_MINUTES = 30  # Auto-refresh interval
TOP_STOCKS_DISPLAY = 20  # Number of top stocks to show (increased to show more opportunities)
SHOW_DETAILED_INDICATORS = True  # Show individual indicator values

# Watchlist settings
WATCHLIST_DIR = 'watchlists'  # Default directory for watchlist files
WATCHLIST_CACHE_ENABLED = True  # Use cache for watchlist stocks
WATCHLIST_OUTPUT_DIR = 'output/watchlist_scans'  # Separate output directory for watchlist scans

# Options Contract Selection
OPTIONS_TARGET_DAYS = 45  # Target expiration (sweet spot)
OPTIONS_MIN_DAYS = 30     # Minimum days to expiration
OPTIONS_MAX_DAYS = 60     # Maximum days to expiration
OPTIONS_CALL_DELTA = 0.70  # Target delta for calls (slightly ITM)
OPTIONS_PUT_DELTA = -0.70  # Target delta for puts (slightly ITM)
OPTIONS_NEUTRAL_DELTA = 0.50  # Target delta for neutral/ATM contracts

# Liquidity Requirements (Configurable)
OPTIONS_MIN_OPEN_INTEREST = 100   # Minimum open interest for liquidity
OPTIONS_MAX_SPREAD_PERCENT = 0.10  # Maximum 10% bid-ask spread

# API Settings
OPTIONS_CACHE_MINUTES = 10  # Cache options data for 10 minutes
OPTIONS_API_DELAY = 0.5     # Delay between API calls in seconds

# Display Settings
OPTIONS_MAX_DISPLAY = 10    # Maximum contracts to display in panel

# Risk Management Settings
PORTFOLIO_VALUE = 30000  # Total portfolio value in USD
RISK_PER_TRADE_PERCENT = 0.015  # 1.5% risk per trade
DAILY_LOSS_LIMIT_PERCENT = 0.05  # 5% max daily loss
DAILY_LOSS_LIMIT = PORTFOLIO_VALUE * DAILY_LOSS_LIMIT_PERCENT  # $1,500
MAX_DOLLAR_RISK_PER_TRADE = PORTFOLIO_VALUE * RISK_PER_TRADE_PERCENT  # $450

# Risk data persistence
RISK_DATA_DIR = 'risk_data'
RISK_DATA_FILE = os.path.join(RISK_DATA_DIR, 'daily_pnl.json')
TRADE_HISTORY_FILE = os.path.join(RISK_DATA_DIR, 'trade_history.csv')

# Console colors (for rich library)
COLOR_STRONG_BUY = 'bright_green'
COLOR_BUY = 'green'
COLOR_HOLD = 'white'
COLOR_AVOID = 'red'

# ============================================================================
# MAXIMUM PROFIT SCANNER CONFIGURATION
# High-gamma, short-dated options scanner for explosive opportunities
# ============================================================================

# Stock Selection Criteria
MAX_PROFIT_BETA_THRESHOLD = 1.2                    # Minimum beta vs SPY
MAX_PROFIT_IV_RANK_THRESHOLD = 70                  # Minimum IV rank (0-100 scale)
MAX_PROFIT_MIN_STOCK_DAILY_VOLUME = 300000        # Lowered for mid-caps
MAX_PROFIT_MIN_STOCK_PRICE = 5.0                  # Avoid penny stocks

# Options Selection - Two-stage filtering
MAX_PROFIT_DELTA_SCAN_MIN = 0.10                  # Wide initial scan range
MAX_PROFIT_DELTA_SCAN_MAX = 0.50
MAX_PROFIT_DELTA_FINAL_MIN = 0.15                 # Narrower final filter
MAX_PROFIT_DELTA_FINAL_MAX = 0.45

# Expiration Window (sweet spot for gamma)
MAX_PROFIT_MIN_EXPIRY_DAYS = 7
MAX_PROFIT_MAX_EXPIRY_DAYS = 21

# Liquidity Requirements (strict filtering)
MAX_PROFIT_MIN_OPTION_OI = 100                    # Minimum open interest
MAX_PROFIT_MIN_OPTION_AVG_VOLUME_5D = 5           # 5-day average volume
MAX_PROFIT_MAX_SPREAD_PCT = 0.15                  # 15% max bid-ask spread
MAX_PROFIT_MIN_BID = 0.05                         # Minimum bid price

# Scoring Reference Constants
MAX_PROFIT_OI_REF = 1000                          # Reference OI for log normalization
MAX_PROFIT_VOL_REF = 50                           # Reference volume for log normalization
MAX_PROFIT_EPSILON = 1e-6                         # Stability guard for division

# Scoring Weights - Main Components (must sum close to 1.0)
MAX_PROFIT_GTR_WEIGHT = 0.50                      # Gamma/Theta ratio weight
MAX_PROFIT_IVR_WEIGHT = 0.30                      # IV Rank weight  
MAX_PROFIT_LIQ_WEIGHT = 0.20                      # Liquidity composite weight

# Liquidity Sub-weights (must sum to 1.0)
MAX_PROFIT_LIQ_OI_WEIGHT = 0.40                   # Open interest weight
MAX_PROFIT_LIQ_VOL_WEIGHT = 0.30                  # Volume weight
MAX_PROFIT_LIQ_SPREAD_WEIGHT = 0.30               # Spread penalty weight

# Price Penalty
MAX_PROFIT_PRICE_PENALTY_ALPHA = 0.15             # Multiplicative penalty factor

# Performance Settings
MAX_PROFIT_MAX_WORKERS = 10                       # Parallel fetching threads
MAX_PROFIT_WINSORIZE_PCT = 0.02                   # Top/bottom 2% winsorization
MAX_PROFIT_TOP_RESULTS = 5                        # Display only top 5

# Rate Limiting Settings
MAX_PROFIT_RATE_LIMIT_DELAY = 0.1                 # 100ms between API calls
MAX_PROFIT_BATCH_SIZE = 50                        # Process stocks in batches
MAX_PROFIT_CONCURRENT_QUOTES = 5                  # Max concurrent quote requests
MAX_PROFIT_CONCURRENT_OPTIONS = 3                 # Max concurrent options requests

# Output Settings
MAX_PROFIT_OUTPUT_DIR = 'output/max_profit'       # Output directory
MAX_PROFIT_LOG_SKIPPED = True                     # Log why contracts were skipped
MAX_PROFIT_CACHE_MINUTES = 5                      # Cache validity for scans

# Risk Management for Max Profit trades
MAX_PROFIT_POSITION_SIZE_MULT = 0.5               # 50% of normal position size
MAX_PROFIT_MAX_DAILY_TRADES = 3                   # Limit number of trades

# Create directories if they don't exist
for directory in [CACHE_DIR, OUTPUT_DIR, LOG_DIR, WATCHLIST_DIR, WATCHLIST_OUTPUT_DIR, 
                  RISK_DATA_DIR, MAX_PROFIT_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)