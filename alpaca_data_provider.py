"""
Alpaca Market Data API integration for fetching historical stock data
"""
import requests
import pandas as pd
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
        try:
            # Calculate date range
            end = datetime.now()
            start = end - timedelta(days=days_back)
            
            # Format dates for API (RFC-3339 format)
            start_str = start.strftime('%Y-%m-%dT00:00:00Z')
            end_str = end.strftime('%Y-%m-%dT23:59:59Z')
            
            # Construct API endpoint
            endpoint = f"{self.data_url}/v2/stocks/{symbol}/bars"
            
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
    
    def fetch_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch latest quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with latest quote data or None if error
        """
        try:
            endpoint = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
            
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            data = response.json()
            
            if 'quote' in data:
                return data['quote']
            else:
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
            
            # Construct API endpoint
            endpoint = f"{self.data_url}/v2/stocks/{symbol}/bars"
            
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