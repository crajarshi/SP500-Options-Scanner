# S&P 500 Intraday Options Scanner 📈

An advanced Python application that analyzes S&P 500 stocks using intraday technical indicators to identify the top options trading opportunities. The scanner uses 15-minute bars to provide real-time, actionable signals throughout the trading day.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Alpaca](https://img.shields.io/badge/API-Alpaca-green.svg)
![Finnhub](https://img.shields.io/badge/API-Finnhub-orange.svg)

## Features

- **Intraday Analysis**: Uses 15-minute candles for responsive signals
- **Multiple Technical Indicators**:
  - RSI (14-period): Momentum indicator
  - Bollinger Bands (20-period): Volatility indicator
  - MACD (12,26,9): Trend indicator
  - OBV with 20-period SMA: Volume confirmation
- **Weighted Scoring System**: Combines indicators into a single score (0-100)
- **Clear Trading Signals**: Generates actionable options trading recommendations
- **Interactive Dashboard**: Rich console UI with real-time updates
- **Smart Caching**: Reduces API calls and improves performance
- **Comprehensive Error Handling**: Continues processing despite individual stock errors

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- macOS, Linux, or Windows
- Internet connection
- Alpaca account (free) - Sign up at https://alpaca.markets

### Step 1: Clone the Repository
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
# 1. Test Alpaca connection
# 2. Fetch S&P 500 stock list
# 3. Analyze each stock (~45 seconds)
# 4. Display top 10 trading opportunities
# 5. Save results to CSV
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
# Run in demo mode (no API needed)
python sp500_options_scanner.py --demo

# Run continuously (auto-refresh every 30 min)
python sp500_options_scanner.py --continuous

# Combine options
python sp500_options_scanner.py --demo --continuous
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
- Technical indicator periods
- Scoring weights and thresholds
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

## Understanding the Output

### Dashboard Display
```
╔═══════════════════════════════════════════════════════════════╗
║             S&P 500 Intraday Options Scanner                  ║
║                                                               ║
║  Market Status: OPEN    Time: 10:45 AM ET    Next Scan: 14:52 ║
╚═══════════════════════════════════════════════════════════════╝

TOP OPTIONS TRADING SIGNALS (Last 3.5 hours analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank  Ticker  Price    Chg%   Score  Signal              Indicators
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1     AAPL   $175.32  -2.1%   92.5  🟢 STRONG BUY       RSI:28 ✓ MACD:✓ BB:95 OBV:✓
```

### Signal Types

1. **🟢 STRONG BUY (Score > 85)**: High conviction bullish signal
   - Recommended: Sell Put or Buy Call options
   - Multiple indicators strongly aligned

2. **🟢 BUY (Score 70-85)**: Solid bullish setup
   - Recommended: Sell Put or Buy Call options
   - Good risk/reward ratio

3. **⚪ HOLD (Score 30-70)**: Mixed signals
   - No clear edge, wait for better setup

4. **🔴 AVOID (Score < 30)**: No bullish edge
   - Skip for bullish strategies

### Indicator Scoring

- **RSI Score**: 100 for oversold (< 30), 0 for overbought (> 70)
- **Bollinger Band Score**: Higher when price near lower band
- **MACD Score**: 100 for bullish crossover, 0 otherwise
- **OBV Score**: 100 when above 20-period SMA

### Weighted Final Score
```
Score = (RSI × 30%) + (MACD × 30%) + (BB × 20%) + (OBV × 20%)
```

## Output Files

- **CSV Export**: `output/intraday_scans/sp500_scan_YYYY-MM-DD_HHMM.csv`
- **Error Log**: `logs/error_log.txt`
- **Scanner Log**: `logs/scanner.log`

## API Considerations

- Finnhub free tier: 60 calls/minute
- Scanner implements 1-second delay between calls
- Full S&P 500 scan takes approximately 8-10 minutes
- Caching reduces redundant API calls

## Best Practices

1. **Market Hours**: Run during market hours (9:30 AM - 4:00 PM ET) for best results
2. **Refresh Frequency**: 30-60 minute intervals capture meaningful price movements
3. **Signal Validation**: Combine with your own analysis and risk management
4. **Position Sizing**: Never risk more than you can afford to lose

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

## Disclaimer

This tool is for educational and informational purposes only. Options trading involves substantial risk and is not suitable for all investors. Always conduct your own research and consult with a financial advisor before making investment decisions.

## License

This project is provided as-is without any warranty. Use at your own risk.