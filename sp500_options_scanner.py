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
    
    def __init__(self, demo_mode=False, use_alpaca=True, quick_mode=False):
        self.dashboard = OptionsScannnerDashboard()
        self.session = requests.Session()
        self.errors = []
        self.running = True
        self.demo_mode = demo_mode
        self.use_alpaca = use_alpaca
        self.quick_mode = quick_mode
        
        # Initialize data provider
        if self.use_alpaca and not self.demo_mode:
            self.data_provider = AlpacaDataProvider()
            logger.info("Using Alpaca Market Data API")
        else:
            self.data_provider = None
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
            if df is None or len(df) < config.MIN_REQUIRED_BARS:
                self.errors.append({
                    'ticker': ticker,
                    'error': 'Insufficient data',
                    'timestamp': datetime.now()
                })
                return None
            
            # Calculate indicators
            indicators = calculate_all_indicators(df)
            if indicators is None:
                self.errors.append({
                    'ticker': ticker,
                    'error': 'Failed to calculate indicators',
                    'timestamp': datetime.now()
                })
                return None
            
            # Analyze and generate signals
            analysis = analyze_stock(ticker, indicators)
            return analysis
            
        except Exception as e:
            self.errors.append({
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now()
            })
            logger.error(f"Error processing {ticker}: {e}")
            return None
    
    def save_results(self, analyses: List[Dict], scan_time: datetime):
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
        
        # Save to CSV
        filename = f"sp500_scan_{scan_time.strftime('%Y-%m-%d_%H%M')}.csv"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
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
                    f"[green]Proceeding with bullish opportunity scan...[/green]"
                )
                return True
            else:
                self.dashboard.console.print(
                    f"[red]❌ MARKET REGIME: NOT BULLISH ({factors_passed}/{total_factors} factors positive)[/red]\n"
                    f"[red]Halting bullish scan.[/red]"
                )
                if config.DEFENSIVE_MODE_ENABLED:
                    self.dashboard.console.print("[yellow]🛡️ Consider switching to defensive mode (future feature)[/yellow]")
                else:
                    self.dashboard.console.print("[yellow]💡 Tip: Consider defensive strategies or wait for better conditions[/yellow]")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking market regime: {e}")
            self.dashboard.console.print(
                "[yellow]⚠ Could not determine market regime. Proceeding with scan.[/yellow]"
            )
            return True
    
    def save_error_log(self):
        """Save error log"""
        if not self.errors:
            return
        
        error_file = os.path.join(config.LOG_DIR, 'error_log.txt')
        with open(error_file, 'a') as f:
            f.write(f"\n\n=== Scan at {datetime.now()} ===\n")
            for error in self.errors:
                f.write(f"{error['timestamp']}: {error['ticker']} - {error['error']}\n")
    
    def run_scan(self, skip_breadth=False) -> List[Dict]:
        """Run a complete scan of all S&P 500 stocks"""
        scan_time = datetime.now()
        self.errors = []
        
        # Check market regime first
        if not self.demo_mode:  # Skip market check in demo mode
            if not self.check_market_regime(skip_breadth=skip_breadth):
                # Market is not bullish, halt scan
                return []
        
        # Get S&P 500 tickers
        self.dashboard.display_success("Fetching S&P 500 constituents...")
        tickers = self.fetch_sp500_tickers()
        
        # Process stocks with progress bar
        analyses = []
        self.dashboard.console.print(f"\nProcessing {len(tickers)} stocks...")
        
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
        self.save_results(ranked_analyses, scan_time)
        self.save_error_log()
        
        # Display results
        self.dashboard.display_results(ranked_analyses, scan_time, self.errors)
        
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
    
    def run_once(self, skip_breadth=False):
        """Run a single scan"""
        logger.info("Running single scan...")
        
        # Test connection first
        if not self.demo_mode and not self.test_connection():
            logger.error("Cannot proceed without data connection")
            return
            
        analyses = self.run_scan(skip_breadth=skip_breadth)
        
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


def main():
    """Main entry point"""
    # Check for command line arguments
    demo_mode = '--demo' in sys.argv
    continuous_mode = '--continuous' in sys.argv
    use_finnhub = '--finnhub' in sys.argv  # Option to use legacy Finnhub
    quick_mode = '--quick' in sys.argv  # Quick mode using cached data
    fast_regime = '--fast-regime' in sys.argv  # Skip breadth check
    regime_only = '--regime-only' in sys.argv  # Check regime only
    warm_cache = '--warm-cache' in sys.argv  # Pre-calculate breadth
    
    # Create scanner
    scanner = SP500OptionsScanner(
        demo_mode=demo_mode,
        use_alpaca=not use_finnhub,  # Use Alpaca by default
        quick_mode=quick_mode
    )
    
    # Handle special modes
    if warm_cache:
        logger.info("Warming cache with market breadth calculation...")
        scanner.calculate_market_breadth(show_progress=True)
        logger.info("Cache warmed successfully!")
        return
    
    if regime_only:
        logger.info("Checking market regime only...")
        is_bullish = scanner.check_market_regime(skip_breadth=fast_regime)
        if is_bullish:
            scanner.dashboard.console.print("\n[green]Market regime is BULLISH - good for options trading![/green]")
        else:
            scanner.dashboard.console.print("\n[red]Market regime is NOT BULLISH - consider defensive strategies.[/red]")
        return
    
    if continuous_mode:
        scanner.run_continuous()
    else:
        scanner.run_once(skip_breadth=fast_regime)


if __name__ == "__main__":
    main()