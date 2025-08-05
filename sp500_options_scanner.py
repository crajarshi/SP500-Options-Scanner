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
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

import config
from indicators import calculate_all_indicators
from signals import analyze_stock, rank_stocks
from dashboard import OptionsScannnerDashboard
from demo_data_generator import generate_demo_intraday_data, get_demo_sp500_tickers

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
    
    def __init__(self, demo_mode=False):
        self.api_key = config.FINNHUB_API_KEY
        self.dashboard = OptionsScannnerDashboard()
        self.session = requests.Session()
        self.errors = []
        self.running = True
        self.demo_mode = demo_mode
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        if self.demo_mode:
            logger.info("Running in DEMO MODE - using simulated data")
        
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
        cached_data = self.load_cache(
            'intraday_data', 
            cache_identifier,
            max_age_minutes=config.CACHE_EXPIRY_MINUTES
        )
        if cached_data is not None:
            return cached_data
        
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
                'signal': analysis['signal']['text'],
                'rsi_value': analysis['indicators']['rsi'],
                'macd_bullish': analysis['indicators']['macd_bullish'],
                'scan_time': scan_time
            })
        
        df = pd.DataFrame(results_data)
        
        # Save to CSV
        filename = f"sp500_scan_{scan_time.strftime('%Y-%m-%d_%H%M')}.csv"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    
    def save_error_log(self):
        """Save error log"""
        if not self.errors:
            return
        
        error_file = os.path.join(config.LOG_DIR, 'error_log.txt')
        with open(error_file, 'a') as f:
            f.write(f"\n\n=== Scan at {datetime.now()} ===\n")
            for error in self.errors:
                f.write(f"{error['timestamp']}: {error['ticker']} - {error['error']}\n")
    
    def run_scan(self) -> List[Dict]:
        """Run a complete scan of all S&P 500 stocks"""
        scan_time = datetime.now()
        self.errors = []
        
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
    
    def run_once(self):
        """Run a single scan"""
        logger.info("Running single scan...")
        analyses = self.run_scan()
        
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
    
    # Create scanner
    scanner = SP500OptionsScanner(demo_mode=demo_mode)
    
    if continuous_mode:
        scanner.run_continuous()
    else:
        scanner.run_once()


if __name__ == "__main__":
    main()