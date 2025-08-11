# Maximum Profit Scanner Documentation

## Overview

The Maximum Profit Scanner (v2.0) is an adaptive high-risk, high-reward options scanner designed to identify explosive short-term opportunities through high-gamma, short-dated options. The scanner now features intelligent filtering that automatically adjusts criteria to find opportunities in any market condition.

**⚠️ WARNING**: This scanner identifies highly speculative trades with significant risk of total loss. Use only with capital you can afford to lose.

### Key Improvements in v2.0
- **Adaptive Filtering**: Automatically tries multiple threshold levels
- **Never Empty Results**: Guaranteed to find opportunities through progressive relaxation
- **Momentum Integration**: Includes technical momentum in scoring
- **ETF Coverage**: Falls back to high-volatility ETFs when needed
- **Near-Miss Tracking**: Shows contracts that barely missed criteria

## Strategy Philosophy

The scanner targets:
- **High-gamma options**: Maximum rate of change in delta (explosive potential)
- **Short expiration**: 7-21 days to expiration (sweet spot for gamma)
- **High volatility stocks**: Beta > 1.2, IV Rank > 70%
- **Liquid contracts**: Strict filtering for tradeable opportunities

## Usage

### Basic Command
```bash
python sp500_options_scanner.py --max-profit
```

### With Additional Options
```bash
# Use with demo mode for testing
python sp500_options_scanner.py --demo --max-profit

# Use with Finnhub instead of Alpaca
python sp500_options_scanner.py --finnhub --max-profit
```

## Scoring Algorithm

### Enhanced Formula (v2.0)
The scanner now uses an adaptive scoring formula that includes momentum:

#### Standard Mode:
```
Final Score = (0.5 × GTR_norm + 0.3 × IVR + 0.2 × LIQ) × (1 - 0.15 × price_penalty)
```

#### Enhanced Mode (with momentum data):
```
Final Score = (0.45 × GTR_norm + 0.25 × IVR + 0.15 × LIQ + 0.10 × MOM + 0.05 × EARN) × (1 - 0.15 × price_penalty)
```

Where:
- **GTR_norm**: Normalized Gamma/Theta Ratio (45-50% weight)
- **IVR**: IV Rank percentile (25-30% weight)
- **LIQ**: Liquidity composite score (15-20% weight)
- **MOM**: Momentum score from technical indicators (10% weight)
- **EARN**: Earnings proximity boost (5% weight)
- **price_penalty**: Reduces score for expensive options (15% impact)

### Components Breakdown

#### 1. Gamma/Theta Ratio (GTR)
- Measures explosive potential vs time decay
- Formula: `gamma / max(abs(theta), epsilon)`
- Normalized using min-max scaling with winsorization
- Higher ratio = more leverage potential

#### 2. IV Rank (IVR)
- Percentile rank of current IV vs 1-year history
- Range: 0-100 (higher = more volatility expected)
- Phase 1: Approximated based on IV levels
- Phase 2: Will use actual historical percentiles

#### 3. Liquidity Score
Composite of three factors:
- **Open Interest** (40%): `log(1 + OI) / log(1 + 1000)`
- **Volume** (30%): `log(1 + avg_vol_5d) / log(1 + 50)`
- **Spread** (30%): `max(0, 1 - spread_pct / 0.15)`

#### 4. Price Penalty
- Prevents expensive outliers from dominating
- Formula: `1 / (1 + log(1 + mid_price))`
- Applied as multiplicative factor

## Adaptive Filtering System (NEW)

The scanner now uses a 3-tier adaptive filtering system that automatically relaxes criteria if no opportunities are found:

### Tier 1: STRICT Mode (Default)
| Criteria | Stock Threshold | Options Threshold | Description |
|----------|----------------|-------------------|-------------|
| Beta | > 1.2 | - | High volatility stocks |
| IV Rank | > 70% | > 70% | Very high implied volatility |
| Daily Volume | > 300,000 | - | Good liquidity |
| Stock Price | > $5 | - | No penny stocks |
| Delta | - | 0.15-0.45 | Slightly OTM |
| Days to Expiry | - | 7-21 | Optimal gamma window |

### Tier 2: MODERATE Mode (Auto-fallback)
| Criteria | Stock Threshold | Options Threshold | Description |
|----------|----------------|-------------------|-------------|
| Beta | > 1.1 | - | Moderate volatility |
| IV Rank | > 60% | > 60% | Above-average volatility |
| Daily Volume | > 250,000 | - | Decent liquidity |
| Delta | - | 0.12-0.48 | Wider range |
| Days to Expiry | - | 5-23 | Slightly wider window |

### Tier 3: RELAXED Mode (Final fallback)
| Criteria | Stock Threshold | Options Threshold | Description |
|----------|----------------|-------------------|-------------|
| Beta | > 1.0 | - | Market beta or higher |
| IV Rank | > 50% | > 50% | Median volatility |
| Daily Volume | > 200,000 | - | Minimum liquidity |
| Delta | - | 0.10-0.50 | Full OTM range |
| Days to Expiry | - | 5-25 | Maximum flexibility |

### ETF Fallback
If no individual stocks qualify, the scanner includes high-volatility ETFs:
- SPY, QQQ, IWM (Major indices)
- XLF, SMH, XLE (Sector ETFs)
- ARKK, GDX, TLT, VXX (Specialty ETFs)

## Risk Management

### Position Sizing
- **Recommended**: 50% of normal position size
- **Max per trade**: 1.5% of portfolio (vs 3% normal)
- **Daily limit**: Maximum 3 trades per day
- **Portfolio allocation**: No more than 5% total in max profit trades

### Risk Warnings
1. **Total Loss Risk**: Options can expire worthless
2. **Theta Decay**: Significant daily time decay
3. **Liquidity Risk**: May be difficult to exit
4. **Volatility Risk**: Sharp moves can work against you
5. **Assignment Risk**: For short positions

## Output Format

### Display Table
The scanner displays the top 5 opportunities with:
- **Symbol**: Stock ticker
- **Contract**: Strike, type (C/P), expiration
- **Score**: Overall score (0-100)
- **G/T**: Gamma/Theta ratio
- **IVR**: IV Rank percentage
- **Liq**: Liquidity score
- **Delta**: Option delta
- **Price**: Mid price
- **Breakdown**: Score component percentages
- **Risk**: Maximum loss per contract

### Data Export
Results are saved in two formats:
1. **CSV**: Top opportunities with key metrics
   - Location: `output/max_profit/top_opportunities_YYYYMMDD_HHMM.csv`
2. **Parquet**: All evaluated contracts with raw components
   - Location: `output/max_profit/all_contracts_YYYYMMDD_HHMM.parquet`

## Configuration

All settings are in `config.py` under the Maximum Profit Scanner section:

```python
# Stock Selection
MAX_PROFIT_BETA_THRESHOLD = 1.2
MAX_PROFIT_IV_RANK_THRESHOLD = 70
MAX_PROFIT_MIN_STOCK_DAILY_VOLUME = 300000
MAX_PROFIT_MIN_STOCK_PRICE = 5.0

# Options Selection
MAX_PROFIT_DELTA_SCAN_MIN = 0.10
MAX_PROFIT_DELTA_SCAN_MAX = 0.50
MAX_PROFIT_DELTA_FINAL_MIN = 0.15
MAX_PROFIT_DELTA_FINAL_MAX = 0.45

# Scoring Weights
MAX_PROFIT_GTR_WEIGHT = 0.50
MAX_PROFIT_IVR_WEIGHT = 0.30
MAX_PROFIT_LIQ_WEIGHT = 0.20
```

## Performance Optimization

### Parallel Processing
- Uses ThreadPoolExecutor with 10 workers
- Parallel stock pre-filtering
- Concurrent options chain fetching

### Caching
- Stock beta and IV rank cached per run
- Options chains cached for 5 minutes
- Skip reasons logged for analysis

### Efficiency Features
- Two-stage delta filtering
- Pre-filtering by cheap operations (beta, volume)
- Winsorization for outlier handling
- Batch normalization for GTR

## Interpreting Results

### Color-Coded Symbols
The scanner color-codes stock symbols to indicate which filter tier found them:
- 🟢 **Green**: STRICT mode (highest conviction)
- 🟡 **Yellow**: MODERATE mode (good opportunities)
- ⬜ **Gray**: RELAXED mode (broader criteria)
- 🔵 **Blue**: ETF (fallback opportunities)

### Score Interpretation
| Score | Quality | Action |
|-------|---------|--------|
| 80-100 | Excellent | Strong opportunity, consider full position |
| 60-79 | Good | Solid setup, consider reduced position |
| 40-59 | Fair | Marginal opportunity, extra caution |
| < 40 | Poor | Usually filtered out |

### Near-Miss Contracts
When enabled, the scanner shows contracts that failed 1-2 criteria:
- Review these for manual override decisions
- Often just miss IV rank or delta thresholds
- Can be valuable in trending markets

## Troubleshooting

### No Results Found (Resolved in v2.0)
The adaptive system now prevents empty results by:
1. Automatically relaxing criteria through 3 tiers
2. Including ETF fallback options
3. Tracking near-misses for review

### Still Getting Empty Results?
Only happens if:
1. API rate limits preventing data fetch
2. No options data available (pre/post market)
3. System error (check logs)
3. Options lack liquidity (spread too wide, low OI)
4. All contracts outside delta range

### Data Provider Issues
- Ensure Alpaca API credentials are configured
- Check rate limits (30-second cooldown on 429 errors)
- Verify market hours for live data

### Score Calculation Issues
- Check for missing Greeks (gamma, theta)
- Verify IV data availability
- Review skip logs in `logs/scanner.log`

## Testing

Run unit tests:
```bash
python -m unittest test_max_profit -v
```

Test coverage includes:
- Scoring calculations
- Filtering logic
- Edge cases (zero theta, outliers)
- Integration testing

## Future Enhancements (Phase 2)

### Planned Features
1. **Local Greeks Calculation**
   - Using py_vollib or mibian
   - Fallback when API Greeks unavailable

2. **Historical IV Rank**
   - Store 252 days of IV data
   - Calculate true percentile ranks

3. **Backtesting Framework**
   - Historical performance analysis
   - Parameter optimization
   - Win/loss ratio tracking

4. **Machine Learning**
   - Feature engineering
   - Pattern recognition
   - Entry/exit optimization

## Example Workflow

1. **Morning Scan** (9:45 AM ET)
   ```bash
   python sp500_options_scanner.py --max-profit
   ```

2. **Review Results**
   - Check score breakdown
   - Verify liquidity metrics
   - Assess risk per trade

3. **Position Entry**
   - Use 50% normal position size
   - Set stop loss at 50% of premium
   - Target 2-3x profit

4. **Management**
   - Monitor gamma changes
   - Watch for IV crush
   - Exit before theta accelerates

## Best Practices

### When to Use
- High volatility environments (VIX > 20)
- Before major events (earnings, Fed)
- Technical breakout setups
- Market extremes (oversold/overbought)

### When to Avoid
- Low volatility periods
- Uncertain market direction
- Before long weekends (theta decay)
- Illiquid underlyings

### Risk Controls
1. Never risk more than you can afford to lose
2. Limit to 5% of portfolio maximum
3. Use stop losses religiously
4. Take profits at 2-3x
5. Don't average down on losers

## Support

For issues or questions:
- Check logs: `logs/scanner.log`
- Review skip reasons in output files
- Verify configuration in `config.py`
- Run unit tests for validation

## Disclaimer

**This scanner identifies highly speculative trades. Options trading involves significant risk and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.**

---

*Last Updated: August 2025*
*Version: 1.0.0*