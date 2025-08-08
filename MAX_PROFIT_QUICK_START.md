# Maximum Profit Scanner - Quick Start Guide

## ⚡ Quick Start (30 seconds)

```bash
# Run the scanner
python sp500_options_scanner.py --max-profit
```

That's it! The scanner will find the top 5 high-gamma options opportunities.

## 🎯 What It Does

Finds **explosive short-term options** with:
- High gamma (maximum leverage)
- 7-21 days to expiration
- High-volatility stocks (beta > 1.2)
- Elevated IV rank (> 70%)

## ⚠️ Risk Warning

**THESE ARE HIGH-RISK TRADES**
- Can lose 100% of premium
- Significant daily theta decay
- Only use risk capital
- Position size = 50% of normal

## 📊 Understanding the Output

### Score (0-100)
- **80+**: Exceptional opportunity
- **60-80**: Strong opportunity
- **40-60**: Moderate opportunity
- **< 40**: Weak (not shown)

### Key Metrics
- **G/T Ratio**: Gamma/Theta - higher = more explosive
- **IVR**: IV Rank % - higher = more volatility expected
- **Delta**: 0.15-0.45 target range
- **Risk**: Maximum loss per contract

## 💰 Trading Strategy

### Entry
1. Buy when score > 70
2. Use 50% normal position size
3. Set stop loss at 50% of premium

### Exit
- **Profit Target**: 2-3x premium paid
- **Stop Loss**: 50% of premium
- **Time Stop**: Exit 2 days before expiry

## 📈 Best Market Conditions

### Ideal For
- High volatility (VIX > 20)
- Before earnings/events
- Technical breakouts
- Market extremes

### Avoid When
- Low volatility (VIX < 15)
- Uncertain direction
- Before long weekends
- Illiquid markets

## 🔧 Configuration

Edit `config.py` to adjust:
```python
MAX_PROFIT_BETA_THRESHOLD = 1.2      # Min stock beta
MAX_PROFIT_IV_RANK_THRESHOLD = 70    # Min IV rank
MAX_PROFIT_DELTA_FINAL_MIN = 0.15    # Min delta
MAX_PROFIT_DELTA_FINAL_MAX = 0.45    # Max delta
```

## 📁 Output Files

Results saved to:
- **Top 5**: `output/max_profit/top_opportunities_*.csv`
- **All Data**: `output/max_profit/all_contracts_*.parquet`

## 🧪 Testing

```bash
# Run unit tests
python -m unittest test_max_profit -v

# Test with demo data
python sp500_options_scanner.py --demo --max-profit
```

## 🚨 Common Issues

### No Results?
- Market may be too calm (low IV)
- No stocks meeting criteria
- Try lowering thresholds in config

### Data Issues?
- Check Alpaca API credentials
- Verify market hours
- Review logs/scanner.log

## 📖 Full Documentation

See [MAX_PROFIT_SCANNER.md](MAX_PROFIT_SCANNER.md) for:
- Detailed scoring algorithm
- Complete configuration options
- Backtesting guidelines
- Risk management rules

## 💡 Pro Tips

1. **Best Time**: First 30 min after open
2. **Avoid**: Last hour of trading
3. **Sweet Spot**: 10-14 DTE options
4. **Max Positions**: 3 at once
5. **Review**: Check score breakdown

## ⚖️ Remember

> "High reward comes with high risk. Never trade more than you can afford to lose."

---
*Quick Start Guide v1.0 | Last Updated: August 2025*