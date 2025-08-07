# 📋 Quick Reference Guide - S&P 500 Options Scanner

## 🚀 Starting the Scanner

### Basic Commands
```bash
# Standard scan with market regime check
python sp500_options_scanner.py

# Demo mode (no API needed, skips market check)
python sp500_options_scanner.py --demo

# Quick mode (uses cached data)
python sp500_options_scanner.py --quick

# Continuous mode (auto-refresh every 30 min)
python sp500_options_scanner.py --continuous

# Watchlist mode (scan custom stocks)
python sp500_options_scanner.py --watchlist tech.txt
python sp500_options_scanner.py --watchlist my_stocks.txt --no-regime

# Scanner modes
python sp500_options_scanner.py --mode adaptive  # Default
python sp500_options_scanner.py --mode bullish   # Force bullish
python sp500_options_scanner.py --mode bearish   # Force bearish

# Combine modes
python sp500_options_scanner.py --watchlist tech.txt --mode bearish --top 10
```

## 📊 Reading the Console Output

### 1. Market Regime Check (First Output)
```
✅ Market Regime is BULLISH
   SPY: $450.25 | 50-day MA: $445.30 (+1.1% above MA)
```
- ✅ = Bullish (scan continues)
- ❌ = Bearish (scan stops)

### 2. Main Results Table
```
Rank  Ticker  Price    Chg%   Score  ATR    Vol  Signal
1     AAPL   $175.32  -2.1%   92.5  $3.45   ↑   🟢 STRONG BUY
```

### Column Meanings
| Column | What It Shows | What's Good |
|--------|--------------|-------------|
| Score | 0-100 composite | >85 = Strong, 70-85 = Good |
| ATR | Volatility value | Higher = more movement |
| Vol | Trend direction | ↑ = expanding (better for options) |
| Signal | Action to take | 🟢 = Bullish opportunity |

## 📝 Watchlist Files

### Creating a Watchlist
Create text files in `watchlists/` folder:
```
# watchlists/tech.txt
AAPL    # Apple
MSFT    # Microsoft
NVDA    # Nvidia
AMD     # Comments supported
```

### Using Watchlists
```bash
# Basic watchlist scan
python sp500_options_scanner.py --watchlist tech.txt

# Skip market check for speed
python sp500_options_scanner.py --watchlist my_stocks.txt --no-regime

# Export results
python sp500_options_scanner.py --watchlist energy.txt --export
```

## 🔄 Scanner Modes (Adaptive Feature)

### Adaptive Mode (Default)
- Automatically detects market conditions
- Switches between bullish/bearish/mixed based on:
  - SPY trend vs 50MA
  - VIX level (<25 bullish, >25 bearish)
  - Market breadth (>60% bullish)

### Mode Selection
| Mode | When to Use | Finds |
|------|------------|-------|
| **Adaptive** | Most times | Auto-adjusts to market |
| **Bullish** | Uptrending market | SELL PUT, BUY CALL |
| **Bearish** | Downtrending market | BUY PUT, SELL CALL |
| **Mixed** | Uncertain market | Both (trend-confirmed only) |

## 🎯 Signal Interpretations

### In BULLISH Mode:
#### Strong Buy (Score > 85) 🟢
- **Meaning**: Multiple strong bullish signals
- **Best Options**: SELL PUT, BUY CALL
- **Ideal When**: ATR is ↑ (rising)

### In BEARISH Mode:
#### Strong Bearish (Score > 85) 🔴
- **Meaning**: Multiple strong bearish signals
- **Best Options**: BUY PUT, SELL CALL
- **Ideal When**: Stock is overbought

### Buy (Score 70-85) 🟢
- **Meaning**: Good bullish setup
- **Best Options**: Bull spreads, sell put spreads
- **Check**: ATR trend before entering

### Hold (Score 30-70) ⚪
- **Meaning**: Mixed signals
- **Action**: Wait for better setup
- **Alternative**: Iron condors if range-bound

### Avoid (Score < 30) 🔴
- **Meaning**: No bullish edge
- **Action**: Skip or look elsewhere

## 📈 Indicator Breakdown

### Scoring Weights
- **MACD**: 30% - Trend strength
- **RSI**: 30% - Momentum
- **Bollinger**: 20% - Price position
- **OBV**: 10% - Volume confirmation
- **ATR**: 10% - Volatility expansion

### What Each Indicator Tells You
| Indicator | ✓ Means | ✗ Means |
|-----------|---------|---------|
| RSI | Oversold (<30) | Overbought (>70) |
| MACD | Bullish crossover | Bearish crossover |
| BB | Near lower band | Near upper band |
| OBV | Above average | Below average |
| ATR | Expanding volatility | Contracting volatility |

## 🔄 ATR for Options Trading

### ↑ Rising ATR (Volatility Expanding)
✅ **Good for**:
- Buying call options
- Buying put options
- Directional strategies

⚠️ **Caution**:
- Selling naked options
- Premium will be higher

### ↓ Falling ATR (Volatility Contracting)
✅ **Good for**:
- Selling covered calls
- Selling cash-secured puts
- Credit spreads

⚠️ **Caution**:
- Buying options (need bigger moves)
- Directional plays

## ⚡ Quick Decision Framework

1. **Check Market Regime**
   - SPY above 50-MA? ✅ Proceed
   - SPY below 50-MA? ❌ Stop

2. **Look at Top 3 Stocks**
   - Score > 85? Strong opportunity
   - Score 70-85? Good opportunity
   - Score < 70? Keep looking

3. **Verify ATR Trend**
   - ↑ Rising? Good for buying options
   - ↓ Falling? Good for selling options

4. **Check Individual Indicators**
   - 4+ positive? High confidence
   - 3 positive? Moderate confidence
   - <3 positive? Low confidence

## 🛠️ Troubleshooting

### Scanner Won't Start
```bash
# Activate virtual environment first
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### Market Check Fails
```bash
# Use demo mode to skip
python sp500_options_scanner.py --demo
```

### Slow Performance
```bash
# Use quick mode for faster scans
python sp500_options_scanner.py --quick
```

## 📁 Output Files

- **Results**: `output/intraday_scans/sp500_scan_[date]_[time].csv`
- **Errors**: `logs/error_log.txt`
- **Full Log**: `logs/scanner.log`

## 🎓 Best Practices

1. **Run During Market Hours**: 9:30 AM - 4:00 PM ET
2. **Best Times**: 30-60 min after open, 1 hour before close
3. **Frequency**: Every 30-60 minutes for day trading
4. **Validation**: Always verify with your own analysis
5. **Risk Management**: Never risk more than 2% per trade

## ⌨️ Keyboard Shortcuts

During continuous mode:
- **R**: Refresh now
- **Q**: Quit scanner
- **Ctrl+C**: Emergency stop

## 💡 Pro Tips

1. **Morning Scan**: Best opportunities often appear 30-60 min after open
2. **ATR Priority**: Focus on stocks with rising ATR for options
3. **Score Threshold**: Only trade scores > 70 for higher probability
4. **Market First**: Always check market regime before individual stocks
5. **Quick Mode**: Use for rapid strategy testing without API delays

---
*Last Updated: Version 2.0 - Options Trading Enhanced*