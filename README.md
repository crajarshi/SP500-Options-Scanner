# S&P 500 Intraday Options Scanner 📈

An advanced Python application that analyzes S&P 500 stocks using intraday technical indicators to identify the top options trading opportunities. The scanner uses 15-minute bars to provide real-time, actionable signals throughout the trading day.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
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

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TradingAnalytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Finnhub API key:
```bash
cp .env.example .env
# Edit .env and add your Finnhub API key
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

### API Limitations
The Finnhub free tier provides:
- ✅ Real-time quotes
- ✅ Company profiles
- ❌ Historical candle data (required for indicators)
- ❌ Technical analysis endpoints

For production use, you'll need:
1. Upgrade to a paid Finnhub plan, OR
2. Use an alternative data provider (Alpha Vantage, Yahoo Finance, etc.)

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