# S&P 500 Intraday Options Scanner 📈

An advanced Python application that analyzes S&P 500 stocks using intraday technical indicators to identify the top options trading opportunities. The scanner uses 15-minute bars to provide real-time, actionable signals throughout the trading day.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Alpaca](https://img.shields.io/badge/API-Alpaca-green.svg)
![Finnhub](https://img.shields.io/badge/API-Finnhub-orange.svg)

## Features

### Core Scanner Features
- **Intraday Analysis**: Uses 15-minute candles for responsive signals
- **Adaptive Dual-Mode Scanner**: Automatically detects market conditions and adjusts strategy
  - Bullish Mode: Identifies SELL PUT / BUY CALL opportunities
  - Bearish Mode: Identifies BUY PUT / SELL CALL opportunities
  - Mixed Mode: Shows trend-confirmed setups only
- **Custom Watchlist Support**: Scan your personal stock lists instead of full S&P 500
- **Market Regime Filter**: Analyzes SPY trend, VIX level, and market breadth
- **Enhanced Technical Indicators**:
  - RSI (14-period): Momentum indicator - 25% weight
  - MACD (12,26,9): Trend indicator - 25% weight
  - Bollinger Bands (20-period): Volatility indicator - 20% weight
  - OBV with 20-period SMA: Volume confirmation - 10% weight
  - Volume: Relative volume analysis - 10% weight
  - ATR (14-period): Volatility expansion/contraction - 10% weight
- **Weighted Scoring System**: Combines indicators into a single score (0-100)
- **Clear Trading Signals**: Generates specific options trading recommendations
- **Interactive Dashboard**: Rich console UI with real-time updates and volatility trends
- **Smart Caching**: Reduces API calls with aggressive daily data caching
- **Quick Scan Mode**: Use cached data for rapid re-analysis
- **Comprehensive Error Handling**: Continues processing despite individual stock errors

### 🚀 NEW: Maximum Profit Scanner (v2.0 - Adaptive Mode)
- **Adaptive High-Gamma Scanner**: Automatically adjusts thresholds to find opportunities in any market
- **3-Tier Filtering System**: 
  - STRICT: Beta>1.2, IVR>70% (high conviction trades)
  - MODERATE: Beta>1.1, IVR>60% (balanced opportunities)
  - RELAXED: Beta>1.0, IVR>50% (broader search)
- **Enhanced Scoring Algorithm**: GTR (45%) + IVR (25%) + Liquidity (15%) + Momentum (10%) + Earnings (5%)
- **ETF Fallback**: Includes high-volatility ETFs (SPY, QQQ, IWM, etc.) when individual stocks don't qualify
- **Near-Miss Tracking**: Shows contracts that barely missed criteria for manual review
- **Smart Position Sizing**: Adaptive sizing based on filter mode and confidence
- **Never Empty Results**: Guaranteed opportunities through progressive filtering
- **See [MAX_PROFIT_SCANNER.md](MAX_PROFIT_SCANNER.md) for detailed documentation**

## 📅 Recent Updates (v2.1.0)

### Bug Fixes
- **Fixed**: `clean_symbol` undefined error in market regime SPY data fetching
- **Fixed**: Insufficient data for indicator calculations (increased LOOKBACK_DAYS from 3 to 7)
- **Fixed**: Test connection now fetches 3 days instead of 1 for proper validation
- **Enhanced**: Better error messages showing exact bar counts when data is insufficient

### New Features  
- **Adaptive Max-Profit Mode**: Automatically relaxes filters through 3 tiers if no opportunities found
- **Momentum Scoring**: Added technical momentum indicators (RSI, trend, volume) to scoring
- **Near-Miss Tracking**: Tracks and reports contracts that barely missed criteria
- **ETF Support**: Falls back to high-volatility ETFs when individual stocks don't qualify
- **Color-Coded Results**: Visual indicators for filter mode (Green=Strict, Yellow=Moderate, Gray=Relaxed, Blue=ETF)

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## 🚀 Getting Started - Complete Setup Guide

### What You Need Before Starting
- **Python 3.8+** installed on your computer ([Download Python](https://www.python.org/downloads/))
- **Terminal/Command Prompt** access
- **Internet connection** for fetching market data
- **5 minutes** for initial setup

### Step 1: Clone or Download the Repository
```bash
# Clone from GitHub
git clone https://github.com/crajarshi/SP500-Options-Scanner.git

# Navigate to the project directory
cd SP500-Options-Scanner
```

### Step 2: Set Up Python Environment
```bash
# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Configure API Keys (Optional)
The scanner comes with default Alpaca API keys for testing. For your own keys:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your favorite editor
nano .env  # or vim, code, etc.

# Add your Alpaca API credentials:
# ALPACA_API_KEY_ID=your_key_here
# ALPACA_SECRET_KEY=your_secret_here
```

### Step 5: Run the Scanner
```bash
# Run a single scan (recommended for first time)
python sp500_options_scanner.py

# The scanner will:
# 1. Check market regime (SPY vs 50-day MA)
# 2. Test Alpaca connection
# 3. Fetch S&P 500 stock list
# 4. Analyze each stock (~45 seconds)
# 5. Display top 10 trading opportunities with ATR trends
# 6. Save results to CSV
```

### Step 6: Understanding the Output

**What happens when you run the scanner:**

1. **Connection Test** - Verifies Alpaca API access
2. **Fetches S&P 500 List** - Gets current list of ~500 stocks
3. **Progress Bar** - Shows real-time progress:
   ```
   Scanning: 45%|████████████░░░░░░░| 225/503 [00:21<00:26, 10.52it/s]
   ```
4. **Analysis Complete** - Displays results dashboard:
   ```
   TOP OPTIONS TRADING SIGNALS (Last 3.5 hours analysis)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Rank  Ticker  Price    Chg%   Score  Signal
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1     AAPL   $175.32  -2.1%   92.5  🟢 STRONG BUY
   2     MSFT   $382.15  -1.8%   88.3  🟢 STRONG BUY
   3     NVDA   $695.20  -3.2%   85.7  🟢 STRONG BUY
   4     TSLA   $242.18  -2.5%   78.9  🟢 BUY
   5     META   $485.62  -1.9%   75.2  🟢 BUY
   ```
5. **Summary Stats** - Shows signal counts:
   ```
   ✓ Scan complete. Found 489 stocks with valid data.
   Strong Buy signals: 3 | Buy signals: 35
   ```

### Additional Running Options
```bash
# Run in demo mode (no API needed, skips market regime check)
python sp500_options_scanner.py --demo

# Run in quick mode (uses cached data for faster scans)
python sp500_options_scanner.py --quick

# Run continuously (auto-refresh every 30 min)
python sp500_options_scanner.py --continuous

# Use legacy Finnhub API instead of Alpaca
python sp500_options_scanner.py --finnhub

# 🚀 NEW: Run Maximum Profit Scanner (high-gamma opportunities)
python sp500_options_scanner.py --max-profit

# Combine options
python sp500_options_scanner.py --demo --continuous
python sp500_options_scanner.py --quick --continuous
```

### 📁 Output Files
- **CSV Results**: `output/intraday_scans/sp500_scan_YYYY-MM-DD_HHMM.csv`
- **Error Log**: `logs/error_log.txt`
- **Scanner Log**: `logs/scanner.log`

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. "ModuleNotFoundError" when running the scanner**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. "403 Forbidden" or API errors**
```bash
# Check your API keys are correct
# Run in demo mode to test without API
python sp500_options_scanner.py --demo
```

**3. Scanner seems slow**
- Normal scan time: 45-60 seconds for 500+ stocks
- API rate limits require ~1 second between requests
- Use cached data for faster subsequent runs

**4. Missing stocks or "No data" errors**
- Some tickers may have limited data (newly listed, low volume)
- Special characters in tickers (like BRK.B) may need conversion
- This is normal - scanner will skip and continue

## API Setup

### Alpaca (Recommended - Default)
The scanner now uses Alpaca Market Data API by default, which provides reliable historical intraday data.

1. Sign up for a free Alpaca account at https://alpaca.markets
2. Get your API keys from the dashboard
3. Add to `.env` or use the defaults in `config.py`:
```bash
ALPACA_API_KEY_ID=your_api_key_id
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
```

### Finnhub (Legacy)
If you prefer to use Finnhub (requires paid plan for historical data):
```bash
python sp500_options_scanner.py --finnhub
```

## Configuration

Edit `config.py` to customize:
- Technical indicator periods (RSI, MACD, BB, OBV, ATR)
- Scoring weights:
  - `WEIGHT_RSI = 0.30` (30%)
  - `WEIGHT_MACD = 0.30` (30%)
  - `WEIGHT_BOLLINGER = 0.20` (20%)
  - `WEIGHT_OBV = 0.10` (10%)
  - `WEIGHT_ATR = 0.10` (10%)
- Market regime parameters:
  - `MARKET_REGIME_MA_PERIOD = 50` (50-day MA for SPY)
  - `MIN_TRADING_DAYS_REQUIRED = 252` (1 year history)
- Signal generation thresholds
- API rate limits
- Cache settings
- Display preferences

## Usage

### Demo Mode (Recommended for Testing)
Due to Finnhub API limitations on the free tier, use demo mode to test the scanner:
```bash
python sp500_options_scanner.py --demo
```

### Production Mode
If you have a paid Finnhub API key with access to historical candles:
```bash
python sp500_options_scanner.py
```

### Continuous Mode
Run with automatic refresh every 30 minutes:
```bash
python sp500_options_scanner.py --continuous
python sp500_options_scanner.py --demo --continuous  # Demo + continuous
```

### Watchlist Mode (Custom Stock Lists)
Scan your personal watchlist instead of all S&P 500 stocks:
```bash
# Scan stocks from watchlists/ folder
python sp500_options_scanner.py --watchlist tech.txt
python sp500_options_scanner.py --watchlist energy.txt

# Skip market regime check for faster results
python sp500_options_scanner.py --watchlist my_stocks.txt --no-regime

# Get options recommendations for your watchlist
python sp500_options_scanner.py --watchlist my_stocks.txt --options

# Combine with other options
python sp500_options_scanner.py --watchlist tech.txt --mode bearish --top 10
```

Create your watchlist file in `watchlists/` folder:
```
# watchlists/tech.txt
AAPL
MSFT
NVDA
# Add comments with #
AMD
GOOGL
```

### Scanner Modes (Adaptive/Bullish/Bearish)
The scanner can operate in different modes based on market conditions:
```bash
# Adaptive mode (default) - auto-detects market conditions
python sp500_options_scanner.py

# Force bullish mode - find SELL PUT / BUY CALL opportunities
python sp500_options_scanner.py --mode bullish

# Force bearish mode - find BUY PUT / SELL CALL opportunities  
python sp500_options_scanner.py --mode bearish

# Mixed mode - show both bullish and bearish opportunities
python sp500_options_scanner.py --mode mixed
```

### Data Provider Comparison

**Alpaca (Default)**
- ✅ Free historical intraday data (15-min bars)
- ✅ Reliable and fast API
- ✅ No rate limiting issues for S&P 500 scans
- ✅ Extended hours data available
- ✅ Paper trading account included

**Finnhub (Legacy)**
- ✅ Real-time quotes (free tier)
- ✅ Company profiles (free tier)
- ❌ Historical candle data (paid only)
- ❌ Technical analysis endpoints (paid only)

## 🎯 Options Trading Enhancements

### Market Regime Filter
The scanner now checks if the overall market is bullish before scanning:

```
✅ Market Regime is BULLISH
   SPY: $450.25 | 50-day MA: $445.30 (+1.1% above MA)
   Proceeding with bullish scan...
```

If SPY is below its 50-day MA:
```
❌ Market Regime is BEARISH
   SPY: $440.15 | 50-day MA: $445.30 (-1.2% below MA)
   Halting bullish scan. Consider defensive strategies.
```

### ATR Volatility Component
The scanner now includes Average True Range (ATR) analysis:
- **10% weight** in the composite score
- Shows volatility trend: ↑ Rising or ↓ Falling
- Helps identify stocks with expanding volatility (better for options)

## 📊 How to Read the Scanner Report

### Console Output Structure

#### 1. Market Regime Check (First Thing You See)
```
Checking market regime...
✅ Market Regime is BULLISH
   SPY: $450.25 | 50-day MA: $445.30 (+1.1% above MA)
```
- **Green checkmark (✅)**: Market is bullish, scan proceeds
- **Red X (❌)**: Market is bearish, scan halts
- Shows exact SPY price vs 50-day MA for context

#### 2. Progress Bar
```
Scanning: 45%|████████████░░░░░░░| 225/503 [00:21<00:26, 10.52it/s]
```
- Shows real-time progress through S&P 500 stocks
- Displays current ticker being processed
- Estimates time remaining

#### 3. Main Dashboard
```
╔═══════════════════════════════════════════════════════════════╗
║             S&P 500 Intraday Options Scanner                  ║
║                                                               ║
║  Market Status: OPEN    Time: 10:45 AM ET    Next Scan: 14:52 ║
╚═══════════════════════════════════════════════════════════════╝

TOP OPTIONS TRADING SIGNALS (Last 3.5 hours analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank  Ticker  Price    Chg%   Score  ATR    Vol  Signal           Indicators
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1     AAPL   $175.32  -2.1%   92.5  $3.45   ↑   🟢 STRONG BUY    RSI:28 MACD:✓ BB:95% OBV:✓ ATR:✓
2     MSFT   $382.15  -1.8%   88.3  $5.20   ↑   🟢 STRONG BUY    RSI:32 MACD:✓ BB:88% OBV:✓ ATR:✓
3     NVDA   $695.20  -3.2%   78.5  $12.30  ↓   🟢 BUY           RSI:38 MACD:✓ BB:75% OBV:✗ ATR:✗
```

### Understanding Each Column

| Column | Description | What to Look For |
|--------|-------------|------------------|
| **Rank** | Position by composite score | Lower numbers = stronger signals |
| **Ticker** | Stock symbol | Click to research further |
| **Price** | Current stock price | Dollar value |
| **Chg%** | Price change % (from open) | Red = down, Green = up |
| **Score** | Composite score (0-100) | >85 = Strong Buy, 70-85 = Buy |
| **ATR** | Average True Range | Higher = more volatility |
| **Vol** | Volatility trend | ↑ = expanding, ↓ = contracting |
| **Signal** | Trading recommendation | 🟢 = Bullish, ⚪ = Neutral, 🔴 = Avoid |
| **Indicators** | Individual indicator status | ✓ = positive, ✗ = negative |

### Signal Types and Options Strategies

#### 🟢 **STRONG BUY (Score > 85)**
**What it means**: Multiple indicators are strongly bullish with expanding volatility
**Options Strategies**:
- **Buy Call Options**: High conviction directional play
- **Sell Cash-Secured Puts**: Generate income with potential to own stock
- **Bull Call Spread**: Lower cost alternative to buying calls
**Best When**: ATR is rising (↑) indicating expanding volatility

#### 🟢 **BUY (Score 70-85)**
**What it means**: Solid bullish setup with good risk/reward
**Options Strategies**:
- **Sell Put Spreads**: Generate income with defined risk
- **Buy Call Options**: Moderate conviction play
- **Call Calendar Spreads**: If expecting gradual move up
**Consider**: Check if ATR trend aligns with your strategy timeframe

#### ⚪ **HOLD (Score 30-70)**
**What it means**: Mixed signals, no clear directional edge
**Options Strategies**:
- **Avoid new positions**: Wait for clearer signals
- **Iron Condors**: If you expect range-bound movement
- **Monitor only**: Add to watchlist for future opportunities

#### 🔴 **AVOID (Score < 30)**
**What it means**: No bullish edge detected
**Options Strategies**:
- **Skip for bullish strategies**: Look elsewhere
- **Consider bearish strategies**: Only if market regime permits
- **Wait for reversal signals**: Monitor for potential bottoming

### Interpreting ATR for Options
- **↑ Rising ATR**: Volatility expanding - options premiums increasing
  - Good for: Buying options (calls/puts)
  - Caution for: Selling options (higher risk)
  
- **↓ Falling ATR**: Volatility contracting - options premiums decreasing
  - Good for: Selling options (premium collection)
  - Caution for: Buying options (need larger moves)

### Indicator Scoring Details

- **RSI Score**: 100 for oversold (< 30), 0 for overbought (> 70), linear scale between
- **MACD Score**: 100 for bullish crossover (MACD > Signal), 0 otherwise
- **Bollinger Band Score**: Higher when price near lower band (100 at lower, 0 at upper)
- **OBV Score**: 100 when OBV above 20-period SMA, 0 otherwise
- **ATR Score**: 100 when ATR > 30-period SMA (expanding volatility), 0 otherwise

### Weighted Final Score Calculation
```
Composite Score = (RSI × 30%) + (MACD × 30%) + (BB × 20%) + (OBV × 10%) + (ATR × 10%)
```

### Quick Scan Mode
For faster re-analysis using cached data:
```bash
python sp500_options_scanner.py --quick
```
- Uses data cached within last 24 hours
- Ideal for testing different parameters
- Completes in seconds instead of minutes

## Output Files

- **CSV Export**: `output/intraday_scans/sp500_scan_YYYY-MM-DD_HHMM.csv`
- **Error Log**: `logs/error_log.txt`
- **Scanner Log**: `logs/scanner.log`

## API Considerations

- Finnhub free tier: 60 calls/minute
- Scanner implements 1-second delay between calls
- Full S&P 500 scan takes approximately 8-10 minutes
- Caching reduces redundant API calls

## After-Hours Trading Preparation 🌙

The scanner now automatically adapts when markets are closed, allowing you to prepare your trades for the next session:

### How It Works
1. **Market Detection**: Scanner checks Alpaca's Clock API to determine if markets are open
2. **Next-Day Calculation**: If closed, it finds the next trading day (skipping weekends/holidays)
3. **Options Analysis**: All options expiration dates are calculated from the NEXT trading day
4. **Clear Indication**: Dashboard shows "📅 Options Analysis for: [Next Trading Date]"

### Example Usage
```bash
# Run after market close (4 PM ET)
python sp500_options_scanner.py --watchlist my_stocks.txt --options

# Output will show:
# Market Status: AFTER-HOURS
# 📅 Options Analysis for: August 08, 2025
# Options expirations calculated from next trading day
```

### Benefits
- **Prepare Orders**: Set up your options orders for market open
- **Weekend Analysis**: Run scans on weekends to plan Monday's trades
- **Holiday Planning**: Automatically handles market holidays
- **Accurate Expirations**: Options DTE (days to expiration) calculated correctly

## Best Practices

### For Options Trading
1. **Market Regime**: Only trade bullish strategies when SPY > 50-day MA
2. **Volatility Preference**: Focus on stocks with rising ATR (↑) for better options premiums
3. **Score Threshold**: Consider only stocks with scores > 70 for higher probability
4. **Time of Day**: Best signals often appear 30-60 minutes after market open

### Operational Tips
1. **Market Hours**: Run during market hours (9:30 AM - 4:00 PM ET) for best results
2. **Refresh Frequency**: 30-60 minute intervals capture meaningful price movements
3. **Quick Mode**: Use `--quick` for rapid re-scans when fine-tuning
4. **Signal Validation**: Combine with your own analysis and risk management
5. **Position Sizing**: Never risk more than you can afford to lose

### Reading Priority
1. **First**: Check market regime (bullish/bearish)
2. **Second**: Look at top 3-5 stocks by score
3. **Third**: Verify volatility trend (prefer ↑)
4. **Fourth**: Check individual indicators for confirmation

## Troubleshooting

### Common Issues

1. **"Insufficient data" errors**: Stock may be newly listed or halted
2. **API rate limit errors**: Increase delay in config.py
3. **Connection errors**: Check internet connection and API key

### Debug Mode
Enable detailed logging:
```python
# In sp500_options_scanner.py
logging.basicConfig(level=logging.DEBUG)
```

## 🆕 What's New - Options Trading Enhancements

### Latest Updates (v3.0) - After-Hours Options Analysis 🌙
**NEW: Prepare your trades when markets are closed!**

1. **Next-Day Options Chain Analysis**: 
   - Automatically detects when markets are closed
   - Calculates options expiration from the NEXT trading day
   - Shows "📅 Options Analysis for: [Next Trading Date]" in dashboard

2. **Smart Market Detection**:
   - Uses Alpaca v2 Clock API to check real-time market status
   - Uses Alpaca v2 Calendar API to find next trading day
   - Handles weekends and holidays automatically

3. **Single Optimal Contract Selection**:
   - Returns exactly ONE best contract per stock
   - Filters to single best expiration (~45 days out)
   - Targets specific deltas: 0.70 for directional, 0.50 for neutral

4. **Options Recommendations with `--options` Flag**:
   ```bash
   python sp500_options_scanner.py --watchlist my_stocks.txt --options
   ```
   - Shows strike, expiration, delta, bid/ask, spread %, OI, volume, IV
   - Only displays liquid contracts (100+ OI, <10% spread)

### Previous Updates (v2.0)
1. **Market Regime Filter**: Automatically checks if SPY > 50-day MA before scanning
2. **ATR Integration**: 10% weight for volatility expansion/contraction signals  
3. **Enhanced Dashboard**: Shows ATR values and volatility trends (↑/↓)
4. **Quick Scan Mode**: `--quick` flag for rapid re-analysis using cached data
5. **Improved Weights**: Rebalanced for options trading (MACD/RSI 30%, BB 20%, OBV/ATR 10%)

### Why These Changes Matter for Options Traders
- **Market Regime**: Prevents taking bullish positions in bearish markets
- **ATR Trends**: Identifies when volatility is expanding (better for buying options)
- **Quick Mode**: Test different strategies without waiting for data fetches
- **Better Scoring**: More emphasis on momentum (MACD/RSI) for short-term options

## Disclaimer

This tool is for educational and informational purposes only. Options trading involves substantial risk and is not suitable for all investors. Always conduct your own research and consult with a financial advisor before making investment decisions.

## License

This project is provided as-is without any warranty. Use at your own risk.