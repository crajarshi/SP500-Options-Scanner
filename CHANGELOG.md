# Changelog

All notable changes to the S&P 500 Intraday Options Scanner will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-01-11 (Ultra Mode Update)

### Added
- **Ultra-Relaxed Mode**: 4th tier (STRICT → MODERATE → RELAXED → ULTRA) that guarantees results
- **Best Available Logic**: Always returns minimum results even if no contracts meet thresholds
- **Smart Retry Mechanism**: Exponential backoff with special handling for rate limits (429 errors)
- **Absolute Quality Floor**: Configurable minimum standards below which contracts are never shown
- **Dynamic Penalty System**: Reduces scores in ultra mode based on how far metrics fall below thresholds
- **Quality Warnings**: Clear warnings when results are below normal quality thresholds
- **Force Results Mode**: Configuration to always show minimum number of results

### Enhanced
- **Retry Logic**: All API calls now use `fetch_with_retry()` with exponential backoff
- **Contract Tracking**: `last_scan_contracts` stores all seen contracts for best_available fallback
- **ETF Scanning**: Fixed method calls and properly integrated with adaptive system
- **Error Recovery**: Better handling of rate limits and transient API failures

### Configuration Changes
- Added `MAX_PROFIT_ULTRA_*` thresholds for ultra-relaxed mode
- Added `MAX_PROFIT_ABSOLUTE_FLOOR` dictionary with minimum quality standards
- Added `MAX_PROFIT_FORCE_RESULTS` flag (default: False)
- Added `MAX_PROFIT_MIN_RESULTS` setting (default: 5)

## [2.1.0] - 2025-01-11

### Added
- **Adaptive Max-Profit Mode**: Scanner now automatically relaxes filters through 3 tiers (STRICT → MODERATE → RELAXED)
- **ETF Fallback Support**: Includes high-volatility ETFs (SPY, QQQ, IWM, XLF, SMH, ARKK, etc.) when stocks don't qualify
- **Momentum Scoring**: Technical momentum indicators (RSI, trend, volume) now contribute 10% to scoring
- **Earnings Proximity Boost**: 5% score boost for options near earnings dates
- **Near-Miss Tracking**: Tracks and can display contracts that failed only 1-2 criteria
- **Color-Coded Results**: Visual indicators showing filter mode (Green=Strict, Yellow=Moderate, Gray=Relaxed, Blue=ETF)
- **Enhanced Error Messages**: More detailed error reporting with exact bar counts

### Changed
- **Scoring Formula**: Updated to GTR (45%) + IVR (25%) + LIQ (15%) + MOM (10%) + EARN (5%) when momentum data available
- **LOOKBACK_DAYS**: Increased from 3 to 7 days for sufficient indicator calculation (now ~182 15-min bars)
- **Test Connection**: Now fetches 3 days instead of 1 for more robust validation
- **Max-Profit Scanner**: Now uses `run_adaptive_scan()` method instead of single-pass `run_scan()`

### Fixed
- **Critical Bug**: Fixed `clean_symbol` undefined error in `fetch_daily_bars()` method (line 217)
- **Data Insufficiency**: Fixed "Failed to calculate indicators" error by ensuring adequate bar count
- **Market Regime Check**: SPY data now fetches correctly for regime analysis
- **Indicator Calculations**: Added defensive checks ensuring minimum required bars (50 for SMA50)

### Configuration Changes
- Added `MAX_PROFIT_MODERATE_*` thresholds for moderate mode
- Added `MAX_PROFIT_RELAXED_*` thresholds for relaxed mode
- Added `MAX_PROFIT_ETFS` list for fallback opportunities
- Added `MAX_PROFIT_AUTO_ADAPT` flag (default: True)
- Added `MAX_PROFIT_SHOW_NEAR_MISSES` flag (default: True)
- Added enhanced weight configurations for momentum/earnings scoring

## [2.0.0] - 2025-01-10

### Added
- **Maximum Profit Scanner**: New high-gamma, short-dated options scanner
- **Risk Management System**: Portfolio-based position sizing and daily loss limits
- **Options Contract Recommendations**: Detailed contract selection with Greeks
- **Watchlist Support**: Scan custom stock lists instead of full S&P 500
- **Market Regime Filter**: SPY trend, VIX level, and market breadth analysis

### Changed
- Migrated from Finnhub to Alpaca as primary data provider
- Updated scoring weights for better signal quality
- Enhanced dashboard with volatility trends and risk status

### Fixed
- API rate limiting issues
- Cache invalidation problems
- Symbol sanitization for special tickers (BRK.B, etc.)

## [1.5.0] - 2024-12-15

### Added
- Adaptive dual-mode scanner (Bullish/Bearish/Mixed)
- ATR volatility expansion/contraction indicator
- Volume relative strength analysis
- Quick scan mode using cached data

### Changed
- Switched to 15-minute bars for more responsive signals
- Improved caching strategy for reduced API calls
- Enhanced error handling and recovery

## [1.0.0] - 2024-11-01

### Initial Release
- Core scanning functionality for S&P 500 stocks
- Technical indicators: RSI, MACD, Bollinger Bands, OBV
- Weighted scoring system
- Console dashboard with Rich library
- Finnhub API integration
- Basic caching system