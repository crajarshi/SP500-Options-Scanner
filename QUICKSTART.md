# 🚀 Quick Start - S&P 500 Options Scanner

## 5-Minute Setup

### 1️⃣ Clone & Navigate
```bash
git clone https://github.com/crajarshi/SP500-Options-Scanner.git
cd SP500-Options-Scanner
```

### 2️⃣ Set Up Python
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Scanner
```bash
python sp500_options_scanner.py
```

That's it! The scanner will analyze all S&P 500 stocks and show you the top options trading opportunities.

## 📊 Sample Output
```
TOP OPTIONS TRADING SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ticker  Price    Score  Signal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AAPL   $175.32   92.5  🟢 STRONG BUY
MSFT   $382.15   88.3  🟢 STRONG BUY
NVDA   $695.20   85.7  🟢 STRONG BUY
```

## 🎯 Trading Signals Explained
- **🟢 STRONG BUY (Score > 85)**: Sell Put or Buy Call - High conviction
- **🟢 BUY (Score 70-85)**: Sell Put or Buy Call - Good opportunity
- **⚪ HOLD (Score 30-70)**: No clear signal - Wait
- **🔴 AVOID (Score < 30)**: No bullish edge - Skip

## 💡 Pro Tips
1. Run during market hours for best results (9:30 AM - 4:00 PM ET)
2. Scan takes ~45 seconds for all 500 stocks
3. Results are saved to `output/` folder as CSV
4. Use `--continuous` for auto-refresh every 30 minutes

## 🆘 Need Help?
- Run in demo mode: `python sp500_options_scanner.py --demo`
- Check full README.md for detailed documentation
- View logs in `logs/` folder for troubleshooting