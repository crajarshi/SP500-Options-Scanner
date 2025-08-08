"""
Alpaca Market Data API integration for fetching historical stock data
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
import time

import config

logger = logging.getLogger(__name__)


class AlpacaDataProvider:
    """
    Handles all interactions with Alpaca Market Data API
    """
    
    def __init__(self):
        self.api_key = config.ALPACA_API_KEY_ID
        self.secret_key = config.ALPACA_SECRET_KEY
        self.data_url = config.ALPACA_DATA_URL
        self.session = requests.Session()
        # Set authentication headers
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        })
    
    def get_sp500_tickers(self) -> List[str]:
        """
        Get S&P 500 tickers from Alpaca assets endpoint
        Note: Alpaca doesn't have a direct S&P 500 list, so we'll use Wikipedia
        """
        try:
            # Fetch from Wikipedia
            import pandas as pd
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean tickers (replace dots with dashes for compatibility)
            tickers = [t.replace('.', '-') for t in tickers]
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 list: {e}")
            # Return a default list
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
                   'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD']
    
    def fetch_bars(self, symbol: str, timeframe: str = '15Min', 
                   days_back: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch historical bars (OHLCV data) for a symbol
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            days_back: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        # Sanitize symbol for Alpaca
        clean_symbol = self.sanitize_symbol(symbol)
        
        try:
            # Calculate date range
            end = datetime.now()
            start = end - timedelta(days=days_back)
            
            # Format dates for API (RFC-3339 format)
            start_str = start.strftime('%Y-%m-%dT00:00:00Z')
            end_str = end.strftime('%Y-%m-%dT23:59:59Z')
            
            # Construct API endpoint with sanitized symbol
            endpoint = f"{self.data_url}/v2/stocks/{clean_symbol}/bars"
            
            # Parameters
            params = {
                'start': start_str,
                'end': end_str,
                'timeframe': timeframe,
                'limit': 10000,  # Max allowed
                'page_token': None,
                'asof': None,
                'feed': 'iex',  # Use IEX feed for paper trading
                'adjustment': 'raw'
            }
            
            # Make request
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got bars
            if 'bars' not in data or not data['bars']:
                logger.warning(f"No bars returned for {symbol}")
                return None
            
            # Convert to DataFrame
            bars = data['bars']
            df = pd.DataFrame(bars)
            
            # Rename columns to match our format
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for {symbol}, waiting...")
                time.sleep(30)  # Wait 30 seconds
                return None
            else:
                logger.error(f"HTTP error fetching data for {symbol}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def sanitize_symbol(self, symbol: str) -> str:
        """
        Sanitize symbol for Alpaca API
        Alpaca doesn't use dashes: BRK-B → BRKB, BF-B → BFB
        """
        return symbol.replace("-", "").replace(".", "")
    
    def fetch_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch latest quote for a symbol with error handling
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with latest quote data or None if error
        """
        # Sanitize symbol for Alpaca
        clean_symbol = self.sanitize_symbol(symbol)
        
        try:
            endpoint = f"{self.data_url}/v2/stocks/{clean_symbol}/quotes/latest"
            
            response = self.session.get(endpoint)
            
            # Handle specific HTTP errors
            if response.status_code == 400:
                logger.warning(f"Bad symbol: {symbol} (sanitized to {clean_symbol})")
                return None
            elif response.status_code == 429:
                logger.warning(f"Rate limit hit for {clean_symbol}, waiting...")
                time.sleep(2)  # Wait 2 seconds
                # Retry once
                response = self.session.get(endpoint)
                if response.status_code != 200:
                    return None
            elif response.status_code == 404:
                logger.debug(f"Symbol not found: {clean_symbol}")
                return None
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'quote' in data:
                quote_data = data['quote']
                # Add original symbol for tracking
                quote_data['original_symbol'] = symbol
                return quote_data
            else:
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for {clean_symbol}, skipping")
            else:
                logger.error(f"HTTP error fetching quote for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def fetch_daily_bars(self, symbol: str, days_back: int = 252) -> Optional[pd.DataFrame]:
        """
        Fetch daily bars (OHLCV data) for a symbol
        
        Args:
            symbol: Stock symbol
            days_back: Number of days of historical data to fetch (default: 252 for 1 year)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Calculate date range
            end = datetime.now()
            start = end - timedelta(days=days_back + 100)  # Add buffer for weekends/holidays
            
            # Format dates for API (RFC-3339 format)
            start_str = start.strftime('%Y-%m-%dT00:00:00Z')
            end_str = end.strftime('%Y-%m-%dT23:59:59Z')
            
            # Construct API endpoint with sanitized symbol
            endpoint = f"{self.data_url}/v2/stocks/{clean_symbol}/bars"
            
            # Parameters for daily bars
            params = {
                'start': start_str,
                'end': end_str,
                'timeframe': '1Day',  # Daily bars
                'limit': 10000,  # Max allowed
                'feed': 'iex',  # Use IEX feed
                'adjustment': 'split'  # Adjust for splits
            }
            
            # Make request
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got bars
            if 'bars' not in data or not data['bars']:
                logger.warning(f"No daily bars returned for {symbol}")
                return None
            
            # Convert to DataFrame
            bars = data['bars']
            df = pd.DataFrame(bars)
            
            # Rename columns to match our format
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Limit to requested days
            if len(df) > days_back:
                df = df.tail(days_back)
            
            logger.info(f"Fetched {len(df)} daily bars for {symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for {symbol}, waiting...")
                time.sleep(30)  # Wait 30 seconds
                return None
            else:
                logger.error(f"HTTP error fetching daily data for {symbol}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None
    
    def fetch_vix_data(self) -> Optional[float]:
        """
        Fetch current VIX level from Alpaca indices endpoint
        
        Returns:
            Latest VIX closing value or None if error
        """
        try:
            # VIX is available through the indices endpoint
            # Note: This requires Alpaca's data subscription that includes indices
            # For paper trading, we'll use a fallback approach
            
            # Try to fetch VIX as an index
            endpoint = f"{self.data_url}/v1beta3/quotes/latest"
            params = {
                'symbols': 'VIX',
                'feed': 'indicative'  # Use indicative feed for indices
            }
            
            response = self.session.get(endpoint, params=params)
            
            # If VIX not available, try alternative symbols
            if response.status_code == 404 or response.status_code == 403:
                # Try VIXY ETF as a proxy for VIX
                logger.info("VIX index not available, using VIXY ETF as proxy")
                endpoint = f"{self.data_url}/v2/stocks/VIXY/quotes/latest"
                response = self.session.get(endpoint)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'quote' in data:
                        # VIXY trades around 0.5x to 1x of VIX value
                        # Rough approximation: multiply by 1.5 for VIX estimate
                        vixy_price = data['quote'].get('ap', data['quote'].get('bp', 0))
                        estimated_vix = vixy_price * 1.5
                        logger.info(f"Estimated VIX from VIXY: {estimated_vix:.2f}")
                        return estimated_vix
            
            # Process standard VIX response
            if response.status_code == 200:
                data = response.json()
                if 'quotes' in data and 'VIX' in data['quotes']:
                    vix_quote = data['quotes']['VIX']
                    vix_value = vix_quote.get('ap', vix_quote.get('bp', 0))
                    logger.info(f"VIX level: {vix_value:.2f}")
                    return vix_value
            
            # Fallback to a default moderate value if unable to fetch
            logger.warning("Unable to fetch VIX, using default value of 18")
            return 18.0  # Historical average VIX
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            # Return a moderate default value rather than None
            return 18.0  # Historical average VIX
    
    def fetch_options_chain(self, symbol: str) -> Optional[Dict]:
        """
        Fetch options chain for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with options chain data organized by expiration and strike
            Format: {expiration_date: {strike: {call: {...}, put: {...}}}}
        """
        # Sanitize symbol for Alpaca
        clean_symbol = self.sanitize_symbol(symbol)
        
        try:
            # Use Alpaca options snapshots endpoint for complete chain with Greeks
            endpoint = f"{self.data_url}/v1beta1/options/snapshots/{clean_symbol}"
            
            # Add feed parameter for options data
            params = {
                'feed': 'indicative'  # Use indicative feed for options
            }
            
            response = self.session.get(endpoint, params=params)
            
            if response.status_code in [403, 404]:
                # Options data not available in subscription or symbol doesn't have options
                logger.info(f"Options data not available for {symbol} (status: {response.status_code}), using simulated data")
                return self._generate_simulated_options_chain(symbol)
            
            response.raise_for_status()
            data = response.json()
            
            # Parse Alpaca options snapshots response
            chain = {}
            
            # Check the actual response structure
            # The snapshots endpoint might return different formats
            if isinstance(data, dict):
                # Could be {'snapshots': [...]} or direct snapshots
                snapshots = data.get('snapshots', [])
                if not snapshots and 'contracts' in data:
                    # Alternative format with contracts
                    snapshots = data.get('contracts', [])
            elif isinstance(data, list):
                # Direct list of snapshots
                snapshots = data
            else:
                logger.warning(f"Unexpected response format for {symbol}: {type(data)}")
                snapshots = []
            
            for snapshot in snapshots:
                # Handle different snapshot formats
                if isinstance(snapshot, str):
                    # If it's just a symbol string, skip
                    continue
                elif not isinstance(snapshot, dict):
                    continue
                    
                # Extract contract details from symbol (format: AAPL240119C00150000)
                contract_symbol = snapshot.get('symbol', '')
                
                # Parse the OCC symbol format
                if len(contract_symbol) < 15:
                    continue
                    
                # Extract components
                underlying = symbol  # We already know this
                exp_date_str = contract_symbol[len(underlying):len(underlying)+6]  # YYMMDD
                contract_type = contract_symbol[len(underlying)+6].lower()  # 'c' or 'p'
                strike_str = contract_symbol[len(underlying)+7:]  # Strike * 1000
                
                # Convert date
                try:
                    exp_date = datetime.strptime(exp_date_str, '%y%m%d').strftime('%Y-%m-%d')
                    strike = float(strike_str) / 1000
                except:
                    continue
                
                if exp_date not in chain:
                    chain[exp_date] = {}
                if strike not in chain[exp_date]:
                    chain[exp_date][strike] = {}
                
                # Map contract type
                contract_type_full = 'call' if contract_type == 'c' else 'put'
                
                # Extract latest quote and Greeks
                latest_quote = snapshot.get('latestQuote', {})
                greeks = snapshot.get('greeks', {})
                
                # Add Greeks and market data
                chain[exp_date][strike][contract_type_full] = {
                    'symbol': contract_symbol,
                    'bid': latest_quote.get('bidPrice', 0),
                    'ask': latest_quote.get('askPrice', 0),
                    'last': snapshot.get('latestTrade', {}).get('price', 0),
                    'volume': snapshot.get('dailyBar', {}).get('volume', 0),
                    'open_interest': snapshot.get('openInterest', 0),
                    'delta': greeks.get('delta', 0),
                    'gamma': greeks.get('gamma', 0),
                    'theta': greeks.get('theta', 0),
                    'vega': greeks.get('vega', 0),
                    'implied_volatility': snapshot.get('impliedVolatility', 0)
                }
            
            if not chain:
                # No data from API, fall back to simulated
                logger.info(f"No options data from API for {symbol}, using simulated data")
                return self._generate_simulated_options_chain(symbol)
                
            logger.info(f"Fetched options chain for {symbol}: {len(chain)} expirations")
            return chain
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            # Fallback to simulated data
            return self._generate_simulated_options_chain(symbol)
    
    def _generate_simulated_options_chain(self, symbol: str) -> Dict:
        """
        Generate simulated options chain for testing/demo purposes
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Simulated options chain data
        """
        import numpy as np
        
        # Get current stock price
        quote = self.fetch_latest_quote(symbol)
        if quote:
            stock_price = (quote.get('ap', 0) + quote.get('bp', 0)) / 2
        else:
            stock_price = 100  # Default
        
        # Generate monthly expirations
        chain = {}
        base_date = datetime.now()
        
        for months_ahead in [1, 2, 3]:  # Next 3 monthly expirations
            # Find 3rd Friday of the month
            exp_date = self._get_monthly_expiration(base_date, months_ahead)
            exp_str = exp_date.strftime('%Y-%m-%d')
            chain[exp_str] = {}
            
            # Generate strikes around current price
            strikes = np.arange(
                stock_price * 0.85,  # 15% OTM
                stock_price * 1.15,  # 15% OTM
                stock_price * 0.025  # 2.5% increments
            )
            
            for strike in strikes:
                strike = round(strike, 2)
                days_to_exp = (exp_date - datetime.now()).days
                
                # Calculate theoretical values
                call_delta = self._black_scholes_delta(
                    stock_price, strike, days_to_exp/365, 0.05, 0.30, 'call'
                )
                put_delta = self._black_scholes_delta(
                    stock_price, strike, days_to_exp/365, 0.05, 0.30, 'put'
                )
                
                # Simulate bid-ask spreads
                call_mid = max(0.01, stock_price - strike) if strike < stock_price else 0.50
                put_mid = max(0.01, strike - stock_price) if strike > stock_price else 0.50
                
                spread_pct = 0.05  # 5% spread
                
                chain[exp_str][strike] = {
                    'call': {
                        'bid': round(call_mid * (1 - spread_pct/2), 2),
                        'ask': round(call_mid * (1 + spread_pct/2), 2),
                        'last': call_mid,
                        'volume': np.random.randint(0, 1000),
                        'open_interest': np.random.randint(100, 5000),
                        'delta': call_delta,
                        'implied_volatility': 0.30 + np.random.uniform(-0.05, 0.05)
                    },
                    'put': {
                        'bid': round(put_mid * (1 - spread_pct/2), 2),
                        'ask': round(put_mid * (1 + spread_pct/2), 2),
                        'last': put_mid,
                        'volume': np.random.randint(0, 1000),
                        'open_interest': np.random.randint(100, 5000),
                        'delta': put_delta,
                        'implied_volatility': 0.30 + np.random.uniform(-0.05, 0.05)
                    }
                }
        
        logger.info(f"Generated simulated options chain for {symbol}")
        return chain
    
    def calculate_beta(self, symbol: str, benchmark: str = 'SPY', period: int = 252) -> float:
        """
        Calculate beta coefficient vs benchmark
        
        Args:
            symbol: Stock symbol to calculate beta for
            benchmark: Benchmark symbol (default SPY)
            period: Number of trading days for calculation
            
        Returns:
            Beta coefficient
        """
        try:
            # Note: fetch_bars already handles symbol sanitization internally
            # Fetch historical data for both stock and benchmark
            stock_data = self.fetch_bars(symbol, '1Day', days_back=period)
            benchmark_data = self.fetch_bars(benchmark, '1Day', days_back=period)
            
            if stock_data is None or benchmark_data is None or len(stock_data) < 30:
                logger.warning(f"Insufficient data for beta calculation: {symbol}")
                return 1.0  # Default beta
            
            # Calculate daily returns
            stock_returns = stock_data['close'].pct_change().dropna()
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            
            # Align data by timestamp
            aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
            if len(aligned) < 30:  # Need minimum data points
                logger.warning(f"Insufficient aligned data for beta: {symbol}")
                return 1.0
            
            aligned.columns = ['stock', 'benchmark']
            
            # Calculate beta = covariance(stock, market) / variance(market)
            covariance = aligned['stock'].cov(aligned['benchmark'])
            variance = aligned['benchmark'].var()
            
            if variance > 0:
                beta = covariance / variance
                logger.debug(f"Beta for {symbol}: {beta:.2f}")
                return beta
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return 1.0  # Default beta on error
    
    def calculate_iv_rank(self, symbol: str, current_iv: float = None) -> float:
        """
        Calculate IV rank (percentile) over past year
        Phase 1: Simple approximation based on current IV levels
        Phase 2 TODO: Store and calculate from 252 days of historical IV data
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility (if known)
            
        Returns:
            IV rank as percentage (0-100)
        """
        try:
            # Phase 1: Simple approximation based on typical IV ranges
            # This is a rough estimate until we implement historical IV storage
            
            if current_iv is None:
                # Try to get from options chain
                chain = self.fetch_options_chain(symbol)
                if chain:
                    # Get ATM option IV as proxy
                    ivs = []
                    for exp_date, strikes in chain.items():
                        for strike, data in strikes.items():
                            if 'call' in data and 'implied_volatility' in data['call']:
                                ivs.append(data['call']['implied_volatility'])
                    
                    if ivs:
                        current_iv = np.median(ivs)
                    else:
                        logger.warning(f"No IV data available for {symbol}")
                        return 50.0  # Default middle rank
                else:
                    return 50.0
            
            # Rough approximation based on typical IV ranges
            # TODO: Replace with actual historical percentile calculation
            if current_iv < 0.15:
                return 5
            elif current_iv < 0.20:
                return 15
            elif current_iv < 0.25:
                return 25
            elif current_iv < 0.30:
                return 40
            elif current_iv < 0.35:
                return 50
            elif current_iv < 0.40:
                return 60
            elif current_iv < 0.50:
                return 70
            elif current_iv < 0.60:
                return 80
            elif current_iv < 0.80:
                return 90
            else:
                return 95
                
        except Exception as e:
            logger.error(f"Error calculating IV rank for {symbol}: {e}")
            return 50.0  # Default middle rank
    
    def get_historical_volatility(self, symbol: str, period: int = 30) -> float:
        """
        Calculate historical volatility (realized volatility)
        
        Args:
            symbol: Stock symbol
            period: Number of trading days for calculation
            
        Returns:
            Annualized historical volatility
        """
        try:
            # Fetch historical data
            data = self.fetch_bars(symbol, '1Day', days_back=period * 2)
            
            if data is None or len(data) < period:
                logger.warning(f"Insufficient data for HV calculation: {symbol}")
                return 0.30  # Default 30% volatility
            
            # Calculate daily returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate standard deviation of returns
            daily_vol = returns.std()
            
            # Annualize (assuming 252 trading days)
            annual_vol = daily_vol * np.sqrt(252)
            
            logger.debug(f"Historical volatility for {symbol}: {annual_vol:.2%}")
            return annual_vol
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {e}")
            return 0.30  # Default volatility
    
    def _get_monthly_expiration(self, base_date: datetime, months_ahead: int) -> datetime:
        """
        Get the monthly options expiration date (3rd Friday)
        
        Args:
            base_date: Starting date
            months_ahead: Number of months ahead
            
        Returns:
            Expiration date (3rd Friday of the month)
        """
        # Move to target month
        target_date = base_date + timedelta(days=30 * months_ahead)
        
        # Find first day of month
        first_day = target_date.replace(day=1)
        
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday
    
    def _black_scholes_delta(self, S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str) -> float:
        """
        Calculate Black-Scholes delta for option
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Delta value
        """
        from scipy.stats import norm
        
        if T <= 0:
            return 0
        
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def test_connection(self) -> bool:
        """
        Test if Alpaca API connection is working
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with account endpoint
            endpoint = f"{config.ALPACA_BASE_URL}/v2/account"
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            account_data = response.json()
            logger.info(f"Connected to Alpaca. Account status: {account_data.get('status', 'Unknown')}")
            
            # Test market data endpoint
            test_symbol = 'AAPL'
            df = self.fetch_bars(test_symbol, timeframe='15Min', days_back=1)
            
            if df is not None and not df.empty:
                logger.info(f"Market data test successful. Got {len(df)} bars for {test_symbol}")
                return True
            else:
                logger.error("Market data test failed")
                return False
                
        except Exception as e:
            logger.error(f"Alpaca connection test failed: {e}")
            return False
    
    def get_market_clock(self) -> Optional[Dict]:
        """
        Get current market status from Alpaca Clock API
        
        Returns:
            Dict with market clock information or None if error
            {
                'is_open': bool,
                'timestamp': datetime,
                'next_open': datetime,
                'next_close': datetime
            }
        """
        try:
            # Use Alpaca trading API v2 endpoint for clock
            endpoint = f"{config.ALPACA_BASE_URL}/v2/clock"
            
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse timestamps
            return {
                'is_open': data.get('is_open', False),
                'timestamp': datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')) if 'timestamp' in data else datetime.now(),
                'next_open': datetime.fromisoformat(data['next_open'].replace('Z', '+00:00')) if 'next_open' in data else None,
                'next_close': datetime.fromisoformat(data['next_close'].replace('Z', '+00:00')) if 'next_close' in data else None
            }
            
        except Exception as e:
            logger.error(f"Error fetching market clock: {e}")
            # Return default based on current time
            from pytz import timezone
            et = timezone('US/Eastern')
            now_et = datetime.now(et)
            hour = now_et.hour
            minute = now_et.minute
            
            # Simple market hours check (9:30 AM - 4:00 PM ET)
            is_open = (
                now_et.weekday() < 5 and  # Monday-Friday
                ((hour == 9 and minute >= 30) or (10 <= hour < 16))
            )
            
            return {
                'is_open': is_open,
                'timestamp': datetime.now(),
                'next_open': None,
                'next_close': None
            }
    
    def get_next_trading_day(self) -> datetime:
        """
        Get the next trading day from Alpaca Calendar API
        
        Returns:
            datetime object for next trading day at market open
        """
        try:
            # Get calendar for next 7 days using v2 API
            endpoint = f"{config.ALPACA_BASE_URL}/v2/calendar"
            
            start_date = datetime.now().strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            
            params = {
                'start': start_date,
                'end': end_date
            }
            
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            calendar_days = response.json()
            
            if calendar_days:
                # Get the first trading day from today
                today = datetime.now().date()
                
                for day in calendar_days:
                    trading_date = datetime.strptime(day['date'], '%Y-%m-%d').date()
                    
                    # If it's today and market is still open, use today
                    if trading_date == today:
                        clock = self.get_market_clock()
                        if clock and clock['is_open']:
                            # Return today at current time
                            return datetime.now()
                    
                    # If it's a future day, return that day at market open
                    if trading_date > today:
                        # Parse market open time
                        open_time = datetime.strptime(
                            f"{day['date']} {day['open']}", 
                            '%Y-%m-%d %H:%M'
                        )
                        return open_time
                
            # Fallback: return next weekday at 9:30 AM ET
            from pytz import timezone
            et = timezone('US/Eastern')
            next_day = datetime.now(et) + timedelta(days=1)
            
            # Skip to Monday if it's weekend
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            
            # Set to market open time
            return next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            
        except Exception as e:
            logger.error(f"Error fetching next trading day: {e}")
            # Fallback to next weekday
            next_day = datetime.now() + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)
            return next_day.replace(hour=9, minute=30, second=0, microsecond=0)