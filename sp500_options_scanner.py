"""
S&P 500 Intraday Options Scanner
Main script that coordinates data fetching, analysis, and display
"""
import os
import sys
import time
import json
import pickle
import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

import config
from indicators import calculate_all_indicators
from signals import analyze_stock, rank_stocks
from dashboard import OptionsScannnerDashboard
from demo_data_generator import generate_demo_intraday_data, get_demo_sp500_tickers
from alpaca_data_provider import AlpacaDataProvider
from options_chain import OptionsChainAnalyzer
from risk_manager import RiskManager
from max_profit_scanner import MaxProfitScanner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'scanner.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SP500OptionsScanner:
    """Main scanner class"""
    
    def __init__(self, demo_mode=False, use_alpaca=True, quick_mode=False, scanner_mode='adaptive', 
                 watchlist_file=None, skip_regime=False, fetch_options=False):
        # Initialize risk manager
        self.risk_manager = RiskManager(
            portfolio_value=config.PORTFOLIO_VALUE,
            risk_per_trade=config.RISK_PER_TRADE_PERCENT,
            daily_loss_limit=config.DAILY_LOSS_LIMIT,
            data_file=config.RISK_DATA_FILE
        )
        
        # Initialize dashboard with risk manager
        self.dashboard = OptionsScannnerDashboard(risk_manager=self.risk_manager)
        self.session = requests.Session()
        self.errors = []
        self.running = True
        self.demo_mode = demo_mode
        self.use_alpaca = use_alpaca
        self.quick_mode = quick_mode
        self.scanner_mode = scanner_mode
        self.market_regime = None
        self.effective_mode = None
        self.watchlist_file = watchlist_file
        self.skip_regime = skip_regime
        self.invalid_tickers = []  # Track invalid tickers from watchlist
        self.fetch_options = fetch_options
        
        # Initialize data provider
        if self.use_alpaca and not self.demo_mode:
            self.data_provider = AlpacaDataProvider()
            logger.info("Using Alpaca Market Data API")
            
            # Initialize options analyzer with risk manager if requested
            if self.fetch_options:
                self.options_analyzer = OptionsChainAnalyzer(self.data_provider, self.risk_manager)
                logger.info("Options contract recommendations enabled with risk management")
            else:
                self.options_analyzer = None
        else:
            self.data_provider = None
            self.options_analyzer = None
            self.api_key = config.FINNHUB_API_KEY
            
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        if self.demo_mode:
            logger.info("Running in DEMO MODE - using simulated data")
        if self.quick_mode:
            logger.info("Running in QUICK MODE - using cached data when available")
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("Shutdown signal received...")
        self.running = False
        sys.exit(0)
    
    def build_scan_context(self, scan_mode: str = 'adaptive') -> Dict:
        """
        Build context object with all market timing info
        
        Args:
            scan_mode: Scanning mode ('bullish', 'bearish', 'mixed', 'adaptive')
            
        Returns:
            Dictionary containing all scan context information
        """
        # Get market clock information
        clock = self.data_provider.get_market_clock()
        
        if clock and clock['is_open']:
            # Market is open - use current time
            reference_date = datetime.now()
            is_realtime = True
            data_timestamp = datetime.now()
        else:
            # Market is closed - use next trading day
            reference_date = self.data_provider.get_next_trading_day()
            is_realtime = False
            # Data timestamp is last market close
            data_timestamp = clock['next_close'] if clock and clock['next_close'] else datetime.now()
        
        # Build the context dictionary
        scan_context = {
            "reference_date": reference_date,
            "is_market_open": clock['is_open'] if clock else False,
            "next_market_open": clock['next_open'] if clock else None,
            "last_market_close": clock['next_close'] if clock else None,
            "scan_mode": scan_mode,
            "data_timestamp": data_timestamp,
            "is_realtime": is_realtime
        }
        
        # Log the context for debugging
        logger.info(f"Scan context built - Market {'OPEN' if scan_context['is_market_open'] else 'CLOSED'}")
        logger.info(f"Reference date: {reference_date.strftime('%Y-%m-%d %H:%M')}")
        if not scan_context['is_market_open']:
            logger.info(f"Next market open: {scan_context['next_market_open']}")
        
        return scan_context
    
    def get_cache_path(self, cache_type: str, identifier: str = "") -> str:
        """Get cache file path"""
        filename = f"{cache_type}_{identifier}.pkl" if identifier else f"{cache_type}.pkl"
        return os.path.join(config.CACHE_DIR, filename)
    
    def load_cache(self, cache_type: str, identifier: str = "", 
                  max_age_minutes: int = None) -> Optional[any]:
        """Load data from cache if valid"""
        cache_path = self.get_cache_path(cache_type, identifier)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(cache_path)
            )
            
            if max_age_minutes:
                if cache_age > timedelta(minutes=max_age_minutes):
                    return None
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
            return None
    
    def save_cache(self, data: any, cache_type: str, identifier: str = ""):
        """Save data to cache"""
        cache_path = self.get_cache_path(cache_type, identifier)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def determine_effective_mode(self) -> str:
        """Determine the effective scanning mode based on market conditions"""
        if self.scanner_mode != 'adaptive':
            return self.scanner_mode
        
        # Use market regime to determine mode
        if not self.market_regime:
            return 'bullish'  # Safe default
        
        breadth = self.market_regime.get('breadth_pct', 50)
        vix = self.market_regime.get('vix_level', 20)
        
        if breadth < config.BEARISH_BREADTH_THRESHOLD or vix > config.BEARISH_VIX_THRESHOLD:
            logger.info("📉 Market regime is BEARISH - scanning for bearish setups")
            return 'bearish'
        elif breadth > config.BULLISH_BREADTH_THRESHOLD and vix < config.BULLISH_VIX_THRESHOLD:
            logger.info("📈 Market regime is BULLISH - scanning for bullish setups")
            return 'bullish'
        else:
            logger.info("🔄 Market regime is MIXED - showing trend-confirmed setups")
            return 'mixed'
    
    def fetch_sp500_tickers(self) -> List[str]:
        """Fetch S&P 500 constituents"""
        if self.demo_mode:
            tickers = get_demo_sp500_tickers()
            logger.info(f"Using {len(tickers)} demo S&P 500 tickers")
            return tickers
            
        # Try cache first
        cached_tickers = self.load_cache(
            'sp500_tickers', 
            max_age_minutes=60 * 24 * config.SP500_CACHE_EXPIRY_DAYS
        )
        if cached_tickers:
            logger.info(f"Loaded {len(cached_tickers)} S&P 500 tickers from cache")
            return cached_tickers
        
        # Use Alpaca data provider if available
        if self.use_alpaca and self.data_provider:
            tickers = self.data_provider.get_sp500_tickers()
            if tickers:
                self.save_cache(tickers, 'sp500_tickers')
                return tickers
        
        # Try to fetch from Wikipedia
        try:
            import pandas as pd
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean tickers (some have dots that need to be converted)
            tickers = [t.replace('.', '-') for t in tickers]
            self.save_cache(tickers, 'sp500_tickers')
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers
        except Exception as e:
            logger.warning(f"Could not fetch from Wikipedia: {e}")
            
        # Fallback to demo tickers
        logger.warning("Using demo ticker list")
        return get_demo_sp500_tickers()
    
    def fetch_watchlist_tickers(self, filename: str) -> List[str]:
        """Fetch tickers from a watchlist file"""
        import os
        
        # Check if it's a full path or just a filename
        if os.path.isabs(filename):
            filepath = filename
        else:
            # Look in the watchlists directory
            filepath = os.path.join(config.WATCHLIST_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"Watchlist file not found: {filepath}")
            self.dashboard.display_error(f"Watchlist file not found: {filepath}")
            sys.exit(1)
        
        tickers = []
        self.invalid_tickers = []
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Strip whitespace
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Extract ticker (handle potential inline comments)
                    ticker = line.split('#')[0].strip().upper()
                    
                    if ticker:
                        # Basic validation - ticker should be alphanumeric with possible dots/dashes
                        if ticker.replace('-', '').replace('.', '').isalnum():
                            # Convert dots to dashes for consistency
                            ticker = ticker.replace('.', '-')
                            tickers.append(ticker)
                        else:
                            logger.warning(f"Invalid ticker format on line {line_num}: {ticker}")
                            self.invalid_tickers.append(ticker)
            
            if not tickers:
                logger.error(f"No valid tickers found in watchlist: {filepath}")
                self.dashboard.display_error(f"Watchlist file is empty or contains no valid tickers: {filepath}")
                sys.exit(1)
            
            logger.info(f"Loaded {len(tickers)} tickers from watchlist: {filename}")
            if self.invalid_tickers:
                logger.warning(f"Skipped {len(self.invalid_tickers)} invalid tickers: {', '.join(self.invalid_tickers)}")
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error reading watchlist file: {e}")
            self.dashboard.display_error(f"Error reading watchlist file: {e}")
            sys.exit(1)
    
    def fetch_intraday_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch intraday candle data for a ticker"""
        if self.demo_mode:
            # Generate demo data
            df = generate_demo_intraday_data(ticker, days=config.LOOKBACK_DAYS)
            return df
            
        # Check cache first
        cache_identifier = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        # In quick mode, use cache even if slightly stale
        cache_max_age = config.CACHE_EXPIRY_MINUTES
        if self.quick_mode:
            cache_max_age = config.DAILY_CACHE_EXPIRY_HOURS * 60  # Use daily cache in quick mode
        
        cached_data = self.load_cache(
            'intraday_data', 
            cache_identifier,
            max_age_minutes=cache_max_age
        )
        if cached_data is not None:
            return cached_data
        
        # Use Alpaca if available
        if self.use_alpaca and self.data_provider:
            df = self.data_provider.fetch_bars(
                symbol=ticker,
                timeframe='15Min',  # 15-minute bars
                days_back=config.LOOKBACK_DAYS
            )
            if df is not None and not df.empty:
                # Save to cache
                self.save_cache(df, 'intraday_data', cache_identifier)
                return df
            else:
                logger.warning(f"No data from Alpaca for {ticker}")
                # Don't fall through to Finnhub, return None
                return None
        
        # Calculate time range - FIXED to use proper UNIX timestamp calculation
        today = datetime.now()
        days_ago = today - timedelta(days=config.LOOKBACK_DAYS)
        
        # Convert to UNIX timestamps as integers
        to_timestamp = int(time.mktime(today.timetuple()))
        from_timestamp = int(time.mktime(days_ago.timetuple()))
        
        url = f"{config.FINNHUB_BASE_URL}/stock/candle"
        params = {
            'symbol': ticker,
            'resolution': config.INTRADAY_RESOLUTION,
            'from': from_timestamp,
            'to': to_timestamp,
            'token': self.api_key
        }
        
        try:
            time.sleep(config.API_RATE_LIMIT_DELAY)  # Rate limiting
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') == 'ok' and 'c' in data:
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                
                # Save to cache
                self.save_cache(df, 'intraday_data', cache_identifier)
                return df
            else:
                # If API fails, use demo data
                logger.warning(f"API returned no data for {ticker}, using demo data")
                return generate_demo_intraday_data(ticker, days=config.LOOKBACK_DAYS)
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            # Fall back to demo data
            logger.info(f"Using demo data for {ticker}")
            return generate_demo_intraday_data(ticker, days=config.LOOKBACK_DAYS)
    
    def process_stock(self, ticker: str) -> Optional[Dict]:
        """Process a single stock"""
        try:
            # Fetch intraday data
            df = self.fetch_intraday_data(ticker)
            if df is None:
                self.errors.append({
                    'ticker': ticker,
                    'error': 'Failed to fetch data',
                    'timestamp': datetime.now()
                })
                return None
            
            if len(df) < config.MIN_REQUIRED_BARS:
                self.errors.append({
                    'ticker': ticker,
                    'error': f'Insufficient data: {len(df)} bars, need {config.MIN_REQUIRED_BARS}',
                    'timestamp': datetime.now()
                })
                logger.warning(f"{ticker}: Only {len(df)} bars available, need at least {config.MIN_REQUIRED_BARS}")
                return None
            
            # Calculate indicators
            indicators = calculate_all_indicators(df)
            if indicators is None:
                # More detailed error - check minimum requirements
                required_bars = max(50, config.ATR_SMA_PERIOD, config.MACD_SLOW)
                self.errors.append({
                    'ticker': ticker,
                    'error': f'Failed to calculate indicators: {len(df)} bars, need {required_bars} for all indicators',
                    'timestamp': datetime.now()
                })
                logger.warning(f"{ticker}: Cannot calculate indicators with {len(df)} bars (need {required_bars})")
                return None
            
            # Analyze with mode awareness
            analysis = analyze_stock(
                ticker, 
                indicators, 
                mode=self.effective_mode if self.effective_mode else 'adaptive',
                market_regime=self.market_regime
            )
            return analysis
            
        except Exception as e:
            self.errors.append({
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now()
            })
            logger.error(f"Error processing {ticker}: {e}")
            return None
    
    def save_results(self, analyses: List[Dict], scan_time: datetime, scan_type='sp500'):
        """Save scan results to CSV"""
        if not analyses:
            return
        
        # Create DataFrame
        results_data = []
        for analysis in analyses:
            results_data.append({
                'ticker': analysis['ticker'],
                'current_price': analysis['current_price'],
                'price_change_pct': analysis['price_change_pct'],
                'composite_score': analysis['scores']['composite_score'],
                'rsi_score': analysis['scores']['rsi_score'],
                'macd_score': analysis['scores']['macd_score'],
                'bollinger_score': analysis['scores']['bollinger_score'],
                'obv_score': analysis['scores']['obv_score'],
                'atr_score': analysis['scores'].get('atr_score', 0),
                'signal': analysis['signal']['text'],
                'rsi_value': analysis['indicators']['rsi'],
                'macd_bullish': analysis['indicators']['macd_bullish'],
                'atr_value': analysis['indicators'].get('atr_value', 0),
                'atr_trend': analysis['indicators'].get('atr_trend', 'Unknown'),
                'scan_time': scan_time
            })
        
        df = pd.DataFrame(results_data)
        
        # Save to CSV with appropriate naming
        if scan_type == 'watchlist' and self.watchlist_file:
            # Extract watchlist name without extension
            watchlist_name = os.path.splitext(os.path.basename(self.watchlist_file))[0]
            filename = f"{watchlist_name}_scan_{scan_time.strftime('%Y-%m-%d_%H%M')}.csv"
            output_dir = config.WATCHLIST_OUTPUT_DIR
        else:
            filename = f"sp500_scan_{scan_time.strftime('%Y-%m-%d_%H%M')}.csv"
            output_dir = config.OUTPUT_DIR
        
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    
    def calculate_market_breadth(self, show_progress=True) -> Tuple[float, Dict]:
        """
        Calculate percentage of S&P 500 stocks above their 50-day MA
        This is an intensive operation that should be cached
        
        Returns:
            Tuple of (breadth_percentage, details_dict)
        """
        try:
            # Check cache first (24-hour validity)
            cache_key = f"market_breadth_{datetime.now().strftime('%Y%m%d')}"
            cached_breadth = self.load_cache('market_breadth', cache_key,
                                           max_age_minutes=config.BREADTH_CACHE_HOURS * 60)
            
            if cached_breadth is not None:
                logger.info(f"Using cached market breadth: {cached_breadth['breadth_pct']:.1f}%")
                return cached_breadth['breadth_pct'], cached_breadth
            
            # Calculate breadth - this is intensive!
            self.dashboard.console.print("[yellow]Calculating market breadth (this may take a few minutes)...[/yellow]")
            
            # Get S&P 500 tickers
            tickers = self.fetch_sp500_tickers()
            
            above_ma = 0
            total_checked = 0
            failed_tickers = []
            
            # Process in batches to show progress
            if show_progress:
                from tqdm import tqdm
                ticker_iterator = tqdm(tickers, desc="Checking stocks vs 50MA", ncols=100)
            else:
                ticker_iterator = tickers
            
            for ticker in ticker_iterator:
                try:
                    # Fetch daily data for this ticker
                    if self.use_alpaca and self.data_provider:
                        df = self.data_provider.fetch_daily_bars(
                            ticker, 
                            days_back=config.MARKET_REGIME_MA_PERIOD + 10
                        )
                    else:
                        continue  # Skip if no data provider
                    
                    if df is not None and len(df) >= config.MARKET_REGIME_MA_PERIOD:
                        # Calculate 50-day MA
                        df['ma50'] = df['close'].rolling(window=config.MARKET_REGIME_MA_PERIOD).mean()
                        
                        # Check if current price > 50MA
                        current_price = df['close'].iloc[-1]
                        ma50 = df['ma50'].iloc[-1]
                        
                        if not pd.isna(ma50):
                            total_checked += 1
                            if current_price > ma50:
                                above_ma += 1
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    logger.debug(f"Error checking {ticker}: {e}")
                    failed_tickers.append(ticker)
            
            # Calculate percentage
            if total_checked > 0:
                breadth_pct = (above_ma / total_checked) * 100
            else:
                breadth_pct = 50.0  # Default to neutral if calculation fails
            
            # Prepare detailed results
            breadth_details = {
                'breadth_pct': breadth_pct,
                'stocks_above': above_ma,
                'stocks_checked': total_checked,
                'total_tickers': len(tickers),
                'failed_tickers': failed_tickers,
                'timestamp': datetime.now().isoformat(),
                'calculation_time': time.time()  # Track how long it took
            }
            
            # Cache the results
            self.save_cache(breadth_details, 'market_breadth', cache_key)
            
            logger.info(f"Market breadth: {breadth_pct:.1f}% ({above_ma}/{total_checked} stocks above 50MA)")
            
            return breadth_pct, breadth_details
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            # Return neutral breadth on error
            return 50.0, {'breadth_pct': 50.0, 'error': str(e)}
    
    def check_market_regime(self, skip_breadth=False) -> bool:
        """
        Enhanced market regime check with three factors:
        1. SPY > 50-day MA
        2. Market breadth > 60% (stocks above 50MA)
        3. VIX < 25
        
        All three must pass for bullish confirmation
        
        Args:
            skip_breadth: Skip breadth calculation for faster check
        
        Returns:
            True if bullish (continue scan), False if not bullish (halt scan)
        """
        try:
            self.dashboard.console.print("\n" + "="*60)
            self.dashboard.console.print("[bold]MARKET REGIME ANALYSIS[/bold]")
            self.dashboard.console.print("="*60)
            
            # Try to get cached SPY data first
            cache_key = f"spy_daily_{datetime.now().strftime('%Y%m%d')}"
            spy_data = self.load_cache('market_regime', cache_key, 
                                     max_age_minutes=config.DAILY_CACHE_EXPIRY_HOURS * 60)
            
            if spy_data is None:
                # Fetch fresh SPY data
                if self.use_alpaca and self.data_provider:
                    spy_data = self.data_provider.fetch_daily_bars(
                        'SPY', 
                        days_back=config.MARKET_REGIME_MA_PERIOD + 10  # Extra days for MA calculation
                    )
                    if spy_data is not None:
                        self.save_cache(spy_data, 'market_regime', cache_key)
                else:
                    # Fallback to demo mode or skip
                    self.dashboard.console.print(
                        "[yellow]⚠ Unable to fetch SPY data. Proceeding without market regime check.[/yellow]"
                    )
                    return True
            
            if spy_data is None or len(spy_data) < config.MARKET_REGIME_MA_PERIOD:
                self.dashboard.console.print(
                    "[yellow]⚠ Insufficient SPY data for market regime check. Proceeding anyway.[/yellow]"
                )
                return True
            
            # Initialize regime factors tracking
            regime_factors = {
                'spy_trend': {'passed': False, 'details': ''},
                'market_breadth': {'passed': False, 'details': ''},
                'vix_level': {'passed': False, 'details': ''}
            }
            
            # Factor 1: SPY Trend
            spy_data['ma50'] = spy_data['close'].rolling(window=config.MARKET_REGIME_MA_PERIOD).mean()
            current_spy = spy_data['close'].iloc[-1]
            ma50 = spy_data['ma50'].iloc[-1]
            spy_bullish = current_spy > ma50
            
            if spy_bullish:
                pct_above = ((current_spy - ma50) / ma50) * 100
                regime_factors['spy_trend']['passed'] = True
                regime_factors['spy_trend']['details'] = f"${current_spy:.2f} > ${ma50:.2f} (+{pct_above:.1f}%)"
                spy_status = "[green]✅[/green]"
            else:
                pct_below = ((ma50 - current_spy) / ma50) * 100
                regime_factors['spy_trend']['passed'] = False
                regime_factors['spy_trend']['details'] = f"${current_spy:.2f} < ${ma50:.2f} (-{pct_below:.1f}%)"
                spy_status = "[red]❌[/red]"
            
            # Factor 2: VIX Level
            vix_level = 18.0  # Default
            if self.use_alpaca and self.data_provider:
                vix_level = self.data_provider.fetch_vix_data()
            
            vix_bullish = vix_level < config.VIX_THRESHOLD
            regime_factors['vix_level']['passed'] = vix_bullish
            
            if vix_bullish:
                regime_factors['vix_level']['details'] = f"{vix_level:.1f} (below {config.VIX_THRESHOLD})"
                vix_status = "[green]✅[/green]"
            else:
                regime_factors['vix_level']['details'] = f"{vix_level:.1f} (above {config.VIX_THRESHOLD})"
                vix_status = "[red]❌[/red]"
            
            # Factor 3: Market Breadth (optional based on skip_breadth)
            if not skip_breadth:
                breadth_pct, breadth_details = self.calculate_market_breadth(show_progress=True)
                breadth_bullish = breadth_pct > config.MARKET_BREADTH_THRESHOLD
                regime_factors['market_breadth']['passed'] = breadth_bullish
                
                if breadth_bullish:
                    regime_factors['market_breadth']['details'] = f"{breadth_pct:.1f}% stocks > 50MA"
                    breadth_status = "[green]✅[/green]"
                else:
                    regime_factors['market_breadth']['details'] = f"{breadth_pct:.1f}% stocks > 50MA"
                    breadth_status = "[red]❌[/red]"
            else:
                # Skip breadth check
                regime_factors['market_breadth']['details'] = "Skipped (--fast-regime)"
                breadth_status = "[yellow]⚠[/yellow]"
                breadth_bullish = True  # Don't block on skipped check
            
            # Display all factors
            self.dashboard.console.print(f"\n📊 SPY Trend:      {spy_status} {regime_factors['spy_trend']['details']}")
            self.dashboard.console.print(f"😨 VIX Level:      {vix_status} {regime_factors['vix_level']['details']}")
            if not skip_breadth:
                self.dashboard.console.print(f"📈 Market Breadth: {breadth_status} {regime_factors['market_breadth']['details']}")
            
            # Determine overall regime
            factors_passed = sum(1 for f in [spy_bullish, vix_bullish, breadth_bullish if not skip_breadth else True] if f)
            total_factors = 2 if skip_breadth else 3
            is_bullish = factors_passed == total_factors
            
            # Display overall verdict
            self.dashboard.console.print("\n" + "="*60)
            if is_bullish:
                self.dashboard.console.print(
                    f"[green]✅ MARKET REGIME: BULLISH ({factors_passed}/{total_factors} factors positive)[/green]\n"
                    f"[green]Market conditions favorable for bullish strategies.[/green]"
                )
                # Store regime data when bullish
                breadth_pct_val = breadth_pct if not skip_breadth else 60
                self.market_regime = {
                    'breadth_pct': breadth_pct_val,
                    'vix_level': vix_level,
                    'spy_above_ma': spy_bullish,
                    'is_bullish': True
                }
                return True
            else:
                self.dashboard.console.print(
                    f"[yellow]⚠️  MARKET REGIME: NOT BULLISH ({factors_passed}/{total_factors} factors positive)[/yellow]\n"
                    f"[yellow]Market conditions not ideal for bullish strategies.[/yellow]"
                )
                if config.DEFENSIVE_MODE_ENABLED:
                    self.dashboard.console.print("[yellow]🛡️ Consider defensive strategies (puts, hedges)[/yellow]")
                else:
                    self.dashboard.console.print("[yellow]💡 Tip: Reduce position sizes or consider defensive strategies[/yellow]")
                
                # Store regime data even if not bullish
                breadth_pct_val = breadth_pct if not skip_breadth else 50
                self.market_regime = {
                    'breadth_pct': breadth_pct_val,
                    'vix_level': vix_level,
                    'spy_above_ma': spy_bullish,
                    'is_bullish': False
                }
                return False
                
        except Exception as e:
            logger.warning(f"Error checking market regime: {e}")
            self.dashboard.console.print(
                "[yellow]⚠ Could not determine market regime. Proceeding with scan.[/yellow]"
            )
            # Set default regime data
            self.market_regime = {
                'breadth_pct': 50,
                'vix_level': 20,
                'spy_above_ma': True,
                'is_bullish': True
            }
            return True
    
    def add_options_recommendations(self, analyses: List[Dict], scan_type: str = 'sp500', scan_context: Dict = None):
        """
        Add options contract recommendations to analysis results
        
        Args:
            analyses: List of top stock analyses
            scan_type: 'watchlist' or 'sp500' - determines filtering logic
        """
        if not self.options_analyzer:
            return
        
        # Check if trading is allowed by risk manager
        can_trade, reason = self.risk_manager.can_place_trade()
        if not can_trade:
            logger.warning(f"Options recommendations blocked by risk manager: {reason}")
            # Still fetch options but they'll be marked as blocked
        
        # Determine which stocks to process based on scan type
        if scan_type == 'watchlist':
            # For watchlist: fetch options for ALL displayed stocks
            stocks_to_process = analyses
            logger.info(f"Watchlist mode: Fetching options for all {len(stocks_to_process)} stocks")
        else:
            # For S&P 500: only actionable signals
            actionable_signals = ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']
            stocks_to_process = [a for a in analyses if a['signal']['type'] in actionable_signals]
            
            if not stocks_to_process:
                logger.info("No actionable signals found for options recommendations")
                return
            
            logger.info(f"S&P 500 mode: Fetching options for {len(stocks_to_process)} stocks with actionable signals")
        
        # Process stocks (limit to prevent excessive API calls)
        max_stocks = 10 if scan_type == 'watchlist' else 5
        for analysis in stocks_to_process[:max_stocks]:
            try:
                ticker = analysis['ticker']
                signal_type = analysis['signal']['type']
                stock_price = analysis['current_price']
                
                logger.info(f"Fetching options for {ticker} ({signal_type})")
                
                # Get optimal contracts
                contracts = self.options_analyzer.get_optimal_contracts(
                    ticker, signal_type, stock_price, scan_context
                )
                
                if contracts:
                    # Add to analysis
                    analysis['options_contracts'] = [c.to_dict() for c in contracts]
                    
                    # Log recommendations
                    recommendation = self.options_analyzer.format_recommendation(
                        ticker, contracts, signal_type, stock_price
                    )
                    logger.info(recommendation)
                else:
                    analysis['options_contracts'] = []
                    logger.info(f"No liquid options found for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching options for {analysis['ticker']}: {e}")
                analysis['options_contracts'] = []
    
    def save_error_log(self):
        """Save error log"""
        if not self.errors:
            return
        
        error_file = os.path.join(config.LOG_DIR, 'error_log.txt')
        with open(error_file, 'a') as f:
            f.write(f"\n\n=== Scan at {datetime.now()} ===\n")
            for error in self.errors:
                f.write(f"{error['timestamp']}: {error['ticker']} - {error['error']}\n")
    
    def run_scan(self, skip_breadth=False, top_count=None, filter_signals=None, auto_export=False) -> List[Dict]:
        """Run a complete scan of stocks (S&P 500 or watchlist)"""
        scan_time = datetime.now()
        self.errors = []
        
        # Build scan context with market timing info
        scan_context = self.build_scan_context(self.scanner_mode)
        
        # Check market regime first (unless skipped)
        if not self.skip_regime:
            if not self.demo_mode:
                self.market_regime_bullish = self.check_market_regime(skip_breadth=skip_breadth)
            else:
                self.market_regime = {'breadth_pct': 50, 'vix_level': 20, 'spy_above_ma': True, 'is_bullish': True}
                self.market_regime_bullish = True
        else:
            # Skip regime check - assume neutral
            self.market_regime = {'breadth_pct': 50, 'vix_level': 20, 'spy_above_ma': True, 'is_bullish': True}
            self.market_regime_bullish = True
            logger.info("Skipping market regime check as requested")
        
        # Determine effective mode and update context
        self.effective_mode = self.determine_effective_mode()
        scan_context['scan_mode'] = self.effective_mode
        
        # Display mode information
        mode_display = {
            'bullish': '[green]📈 BULLISH MODE[/green] - Scanning for call/put-sell opportunities',
            'bearish': '[red]📉 BEARISH MODE[/red] - Scanning for put/call-sell opportunities',
            'mixed': '[yellow]🔄 MIXED MODE[/yellow] - Showing trend-confirmed opportunities'
        }
        self.dashboard.console.print(f"\n{mode_display.get(self.effective_mode, 'ADAPTIVE MODE')}\n")
        
        # Get tickers (watchlist or S&P 500)
        if self.watchlist_file:
            self.dashboard.display_success(f"Loading watchlist: {self.watchlist_file}...")
            tickers = self.fetch_watchlist_tickers(self.watchlist_file)
            scan_type = 'watchlist'
        else:
            self.dashboard.display_success("Fetching S&P 500 constituents...")
            tickers = self.fetch_sp500_tickers()
            scan_type = 'sp500'
        
        # Process stocks with progress bar
        analyses = []
        self.dashboard.console.print(f"\nProcessing {len(tickers)} stocks...")
        
        # Log if market is closed
        if not scan_context['is_market_open']:
            logger.info(f"Market is CLOSED - analyzing for {scan_context['reference_date'].strftime('%Y-%m-%d')}")
        
        with tqdm(total=len(tickers), desc="Scanning", ncols=100) as pbar:
            # Process in batches to manage memory and API limits
            batch_size = 10
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                
                # Process batch sequentially (due to API rate limits)
                for ticker in batch:
                    pbar.set_description(f"Processing {ticker}")
                    analysis = self.process_stock(ticker)
                    if analysis:
                        analyses.append(analysis)
                    pbar.update(1)
        
        # Rank stocks
        ranked_analyses = rank_stocks(analyses)
        
        # Save results
        self.save_results(ranked_analyses, scan_time, scan_type)
        self.save_error_log()
        
        # Filter results if specified
        display_analyses = ranked_analyses
        if filter_signals:
            display_analyses = [a for a in ranked_analyses if a['signal']['type'] in filter_signals]
        
        # Limit results if specified
        if top_count:
            display_analyses = display_analyses[:top_count]
        
        # Fetch options recommendations if requested
        if self.fetch_options and self.options_analyzer:
            logger.info("Fetching options contract recommendations for top stocks...")
            # Pass scan_type to determine filtering logic
            self.add_options_recommendations(display_analyses, scan_type, scan_context)
        
        # Auto-export if requested
        if auto_export:
            self.export_to_csv(display_analyses, scan_time)
        
        # Display results with market regime status and watchlist info
        self.dashboard.display_results(
            display_analyses, 
            scan_time, 
            self.errors,
            market_regime_bullish=getattr(self, 'market_regime_bullish', True),
            mode=self.effective_mode,
            scan_type=scan_type,
            watchlist_file=self.watchlist_file,
            scan_context=scan_context
        )
        
        # Display summary for watchlist including invalid tickers
        if self.watchlist_file:
            self.dashboard.console.print(f"\n✓ Scanned {len(analyses)} of {len(tickers)} tickers successfully")
            if self.invalid_tickers:
                self.dashboard.console.print(f"[yellow]⚠ Invalid tickers skipped: {', '.join(self.invalid_tickers)}[/yellow]")
            
            # Show tickers that had errors during processing
            failed_tickers = [e['ticker'] for e in self.errors if e['ticker'] not in self.invalid_tickers]
            if failed_tickers:
                self.dashboard.console.print(f"[yellow]⚠ Failed to process: {', '.join(set(failed_tickers))}[/yellow]")
        
        return ranked_analyses
    
    def run_continuous(self):
        """Run continuous scanning with auto-refresh"""
        logger.info("Starting continuous scanning mode...")
        
        while self.running:
            try:
                # Run scan
                analyses = self.run_scan()
                
                # Wait for next scan or user input
                next_scan_time = datetime.now() + timedelta(
                    minutes=config.REFRESH_INTERVAL_MINUTES
                )
                
                self.dashboard.console.print(
                    f"\n[dim]Next automatic scan at "
                    f"{next_scan_time.strftime('%I:%M %p')}. "
                    f"Press 'R' to refresh now or 'Q' to quit.[/dim]"
                )
                
                # Wait for interval
                time.sleep(config.REFRESH_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                time.sleep(60)  # Wait before retrying
    
    def test_connection(self):
        """Test data provider connection"""
        if self.use_alpaca and self.data_provider:
            logger.info("Testing Alpaca connection...")
            if self.data_provider.test_connection():
                logger.info("✓ Alpaca connection successful!")
                return True
            else:
                logger.error("✗ Alpaca connection failed!")
                return False
        return True
    
    def export_to_csv(self, analyses: List[Dict], scan_time: datetime):
        """Export results to CSV file"""
        if not analyses:
            return
        
        filename = f"options_signals_{scan_time.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        data = []
        for a in analyses:
            data.append({
                'Ticker': a['ticker'],
                'Price': a['current_price'],
                'Change%': a['price_change_pct'],
                'Score': a['scores']['composite_score'],
                'Strategy': a['signal']['text'],
                'RSI': a['indicators']['rsi'],
                'MACD_Bullish': a['indicators']['macd_bullish'],
                'ATR': a['indicators'].get('atr_value', 0)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")
    
    def run_once(self, skip_breadth=False, top_count=None, filter_signals=None, auto_export=False):
        """Run a single scan"""
        logger.info("Running single scan...")
        
        # Test connection first
        if not self.demo_mode and not self.test_connection():
            logger.error("Cannot proceed without data connection")
            return
            
        analyses = self.run_scan(skip_breadth=skip_breadth, top_count=top_count, 
                                filter_signals=filter_signals, auto_export=auto_export)
        
        if analyses:
            self.dashboard.console.print(
                f"\n[green]✓[/green] Scan complete. "
                f"Found {len(analyses)} stocks with valid data."
            )
            
            # Count signals
            strong_buys = sum(1 for a in analyses 
                            if a['signal']['type'] == 'STRONG_BUY')
            buys = sum(1 for a in analyses 
                      if a['signal']['type'] == 'BUY')
            
            self.dashboard.console.print(
                f"[green]Strong Buy signals: {strong_buys}[/green] | "
                f"[green]Buy signals: {buys}[/green]"
            )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='S&P 500 Options Trading Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python sp500_options_scanner.py --top 20
    Show top 20 opportunities
  
  python sp500_options_scanner.py --filter STRONG_BUY,BUY
    Show only STRONG_BUY and BUY signals
  
  python sp500_options_scanner.py --top 30 --export
    Show top 30 and export to CSV
  
  python sp500_options_scanner.py --demo --top 5
    Run in demo mode showing top 5
        ''')
    
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    parser.add_argument('--continuous', action='store_true', help='Run continuous scanning with auto-refresh')
    parser.add_argument('--finnhub', action='store_true', help='Use legacy Finnhub API instead of Alpaca')
    parser.add_argument('--quick', action='store_true', help='Quick mode using cached data')
    parser.add_argument('--fast-regime', action='store_true', help='Skip market breadth check for faster results')
    parser.add_argument('--regime-only', action='store_true', help='Check market regime only and exit')
    parser.add_argument('--warm-cache', action='store_true', help='Pre-calculate market breadth and cache it')
    
    # New arguments for display control
    parser.add_argument('--top', type=int, metavar='N', help='Show top N results (default: 20)')
    parser.add_argument('--filter', type=str, metavar='SIGNALS', 
                       help='Filter by signal types (comma-separated: STRONG_BUY,BUY,HOLD,AVOID)')
    parser.add_argument('--export', action='store_true', help='Auto-export results to CSV')
    
    # Mode selection arguments
    parser.add_argument('--mode', 
                       choices=['adaptive', 'bullish', 'bearish', 'mixed'],
                       default='adaptive',
                       help='Scanner mode (default: adaptive)')
    parser.add_argument('--bearish', 
                       action='store_true',
                       help='Shortcut for --mode bearish')
    
    # Watchlist arguments
    parser.add_argument('--watchlist', 
                       type=str,
                       metavar='FILE',
                       help='Use custom watchlist file instead of S&P 500')
    parser.add_argument('--no-regime',
                       action='store_true',
                       help='Skip market regime check for faster results')
    
    # Options recommendations
    parser.add_argument('--options',
                       action='store_true',
                       help='Include specific options contract recommendations')
    
    # Maximum Profit Scanner
    parser.add_argument('--max-profit',
                       action='store_true',
                       help='Run Maximum Profit scanner for high-gamma, explosive opportunities')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Process filter argument
    filter_signals = None
    if args.filter:
        filter_signals = [s.strip() for s in args.filter.split(',')]
        valid_signals = ['STRONG_BUY', 'BUY', 'HOLD', 'AVOID']
        filter_signals = [s for s in filter_signals if s in valid_signals]
        if filter_signals:
            logger.info(f"Filtering for signals: {', '.join(filter_signals)}")
    
    # Check if Maximum Profit mode is requested
    if args.max_profit:
        logger.warning("=" * 60)
        logger.warning("🚀 MAXIMUM PROFIT MODE - Finding explosive opportunities...")
        logger.warning("⚠️  HIGH RISK - These are speculative trades!")
        logger.warning("=" * 60)
        
        # Create minimal scanner for data provider
        # Force use_alpaca=True for max profit scanner to get data provider
        scanner = SP500OptionsScanner(
            demo_mode=False,  # Need data provider initialized
            use_alpaca=True,  # Force Alpaca for data provider
            quick_mode=False,
            scanner_mode='adaptive',
            skip_regime=True,  # Skip regime check for max profit
            fetch_options=False
        )
        
        # Initialize Maximum Profit Scanner
        max_scanner = MaxProfitScanner(
            data_provider=scanner.data_provider,
            risk_manager=scanner.risk_manager
        )
        
        # Run the scan
        opportunities = max_scanner.run_scan()
        
        # Display results with warnings
        scanner.dashboard.display_max_profit_results(opportunities)
        
        # Save summary
        if opportunities:
            logger.info(f"Found {len(opportunities)} high-gamma opportunities")
            for i, opp in enumerate(opportunities, 1):
                logger.info(f"{i}. {opp['symbol']} - Score: {opp['score']:.1f}/100")
        else:
            logger.warning("No opportunities found matching criteria")
        
        return
    
    # Determine scanner mode for regular scanner
    scanner_mode = args.mode
    if args.bearish:
        scanner_mode = 'bearish'
    
    # Create regular scanner
    scanner = SP500OptionsScanner(
        demo_mode=args.demo,
        use_alpaca=not args.finnhub,  # Use Alpaca by default
        quick_mode=args.quick,
        scanner_mode=scanner_mode,
        watchlist_file=args.watchlist,
        skip_regime=args.no_regime,
        fetch_options=args.options
    )
    
    # Handle special modes
    if args.warm_cache:
        logger.info("Warming cache with market breadth calculation...")
        scanner.calculate_market_breadth(show_progress=True)
        logger.info("Cache warmed successfully!")
        return
    
    if args.regime_only:
        logger.info("Checking market regime only...")
        is_bullish = scanner.check_market_regime(skip_breadth=args.fast_regime)
        if is_bullish:
            scanner.dashboard.console.print("\n[green]Market regime is BULLISH - conditions favorable for bullish options strategies![/green]")
        else:
            scanner.dashboard.console.print("\n[yellow]Market regime is NOT BULLISH - consider reducing position sizes or defensive strategies.[/yellow]")
            scanner.dashboard.console.print("[dim]Scanner will still run but with caution warnings when regime is not bullish.[/dim]")
        return
    
    # Determine top count (default to 20 if not specified)
    top_count = args.top if args.top else 20
    
    if args.continuous:
        scanner.run_continuous()
    else:
        scanner.run_once(skip_breadth=args.fast_regime, top_count=top_count, 
                        filter_signals=filter_signals, auto_export=args.export)


if __name__ == "__main__":
    main()