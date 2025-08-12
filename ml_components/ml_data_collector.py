"""
Machine Learning Data Collector
Collects and prepares historical data for deep learning model training
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import AlpacaDataProvider
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpaca_data_provider import AlpacaDataProvider

logger = logging.getLogger(__name__)


class MLDataCollector:
    """Collects and prepares historical data for ML model training"""
    
    def __init__(self, tickers: List[str], start_date: str = None, end_date: str = None,
                 label_threshold: float = 0.05, label_days: int = 10, 
                 cache_dir: str = "ml_data_cache"):
        """
        Initialize data collector
        
        Args:
            tickers: List of stock tickers to collect data for
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            label_threshold: Profit threshold for positive labels (5% = 0.05)
            label_days: Number of days to look ahead for labeling
            cache_dir: Directory to cache collected data
        """
        self.tickers = tickers
        self.start_date = start_date or '2015-01-01'
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.label_threshold = label_threshold
        self.label_days = label_days
        self.cache_dir = cache_dir
        
        # Initialize Alpaca data provider
        self.data_provider = AlpacaDataProvider()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Feature columns to calculate
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'volume_ratio', 'price_change_1d', 'price_change_5d', 'price_change_20d',
            'volatility_20d', 'volume_ma_ratio', 'high_low_ratio', 'close_open_ratio',
            'upper_shadow', 'lower_shadow', 'body_ratio', 'trend_strength',
            'support_distance', 'resistance_distance'
        ]
        
    def collect_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Collect historical data for a single stock using Alpaca API
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with historical data and calculated features
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_raw.pkl")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                    # Check if cache has enough recent data
                    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index.max()
                        if last_date >= pd.to_datetime(self.end_date) - timedelta(days=7):
                            logger.info(f"Loaded {ticker} from cache")
                            return df
            except:
                pass
        
        try:
            # Calculate total days needed
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
            total_days = (end_dt - start_dt).days
            
            # Fetch data in chunks (Alpaca has limits on historical data)
            all_data = []
            chunk_size = 365  # Fetch 1 year at a time
            max_days_per_request = 1000  # Alpaca's limit
            
            current_end = end_dt
            while current_end > start_dt:
                current_start = max(current_end - timedelta(days=min(chunk_size, max_days_per_request)), start_dt)
                
                # Calculate days to fetch for this chunk
                days_to_fetch = min((current_end - current_start).days + 1, max_days_per_request)
                
                logger.info(f"Fetching {ticker} data from {current_start.date()} to {current_end.date()}")
                
                # Fetch daily bars from Alpaca
                df_chunk = self.data_provider.fetch_daily_bars(
                    ticker, 
                    days_back=days_to_fetch
                )
                
                if df_chunk is not None and not df_chunk.empty:
                    # Filter to our date range
                    df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])
                    
                    # Make timestamps timezone-naive for comparison
                    if df_chunk['timestamp'].dt.tz is not None:
                        df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_localize(None)
                    
                    df_chunk = df_chunk[
                        (df_chunk['timestamp'] >= current_start) & 
                        (df_chunk['timestamp'] <= current_end)
                    ]
                    
                    if not df_chunk.empty:
                        all_data.append(df_chunk)
                    
                # Move to next chunk
                current_end = current_start - timedelta(days=1)
                
                # Add delay to respect rate limits
                time.sleep(0.5)
            
            if not all_data:
                logger.warning(f"No data found for {ticker}")
                return None
            
            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            logger.info(f"Collected {len(df)} days of data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=50).mean()
        
        # Price change features
        df['price_change_1d'] = df['Close'].pct_change(1)
        df['price_change_5d'] = df['Close'].pct_change(5)
        df['price_change_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        df['volatility_20d'] = df['returns'].rolling(window=20).std()
        df['volatility_50d'] = df['returns'].rolling(window=50).std()
        
        # Candlestick patterns
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Shadows
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        df['body_ratio'] = np.abs(df['Close'] - df['Open']) / df['Close']
        
        # Trend strength
        df['trend_strength'] = (df['Close'] - df['sma_50']) / df['sma_50']
        
        # Support and resistance
        df['support_distance'] = (df['Close'] - df['Low'].rolling(window=20).min()) / df['Close']
        df['resistance_distance'] = (df['High'].rolling(window=20).max() - df['Close']) / df['Close']
        
        # Market microstructure
        df['spread'] = df['High'] - df['Low']
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for supervised learning
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with labels added
        """
        # Calculate future returns
        df['future_return'] = df['Close'].shift(-self.label_days) / df['Close'] - 1
        
        # Create binary labels
        df['label'] = (df['future_return'] > self.label_threshold).astype(int)
        
        # Create multi-class labels for more granular predictions
        conditions = [
            df['future_return'] < -self.label_threshold,
            (df['future_return'] >= -self.label_threshold) & (df['future_return'] <= self.label_threshold),
            df['future_return'] > self.label_threshold
        ]
        choices = [0, 1, 2]  # 0: sell, 1: hold, 2: buy
        df['multi_label'] = np.select(conditions, choices, default=1)
        
        # Add confidence score (magnitude of move)
        df['confidence_score'] = np.abs(df['future_return'])
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/GRU models
        
        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        # Fill NaN values
        df[feature_cols] = df[feature_cols].ffill().fillna(0)
        
        X, y = [], []
        
        for i in range(sequence_length, len(df) - self.label_days):
            X.append(df[feature_cols].iloc[i-sequence_length:i].values)
            y.append(df['label'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def collect_all_data(self, max_workers: int = 10) -> pd.DataFrame:
        """
        Collect data for all tickers in parallel
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            Combined DataFrame with all stock data
        """
        all_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_ticker, ticker): ticker 
                      for ticker in self.tickers}
            
            for future in tqdm(as_completed(futures), total=len(self.tickers), 
                              desc="Collecting data"):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_data.append(result)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save combined dataset
            output_file = os.path.join(self.cache_dir, 'combined_ml_dataset.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(combined_df, f)
            
            logger.info(f"Saved combined dataset to {output_file}")
            return combined_df
        else:
            logger.warning("No data collected")
            return pd.DataFrame()
    
    def process_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Process a single ticker: collect data, calculate features, create labels
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Processed DataFrame or None
        """
        # Collect raw data
        df = self.collect_stock_data(ticker)
        if df is None:
            return None
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create labels
        df = self.create_labels(df)
        
        # Drop rows with NaN values in critical columns
        df = df.dropna(subset=['label'] + [col for col in self.feature_columns if col in df.columns])
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for feature importance analysis
        
        Args:
            df: Combined dataset
            
        Returns:
            DataFrame ready for feature importance analysis
        """
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        # Create a clean dataset for feature importance
        clean_df = df[feature_cols + ['label', 'ticker']].copy()
        clean_df = clean_df.dropna()
        
        return clean_df
    
    def create_train_test_split(self, df: pd.DataFrame, 
                              train_end_date: str = '2022-12-31',
                              val_end_date: str = '2023-12-31') -> Dict:
        """
        Create chronological train/validation/test split
        
        Args:
            df: Combined dataset
            train_end_date: End date for training data
            val_end_date: End date for validation data
            
        Returns:
            Dictionary with train, validation, and test datasets
        """
        df['date'] = pd.to_datetime(df.index)
        
        train_df = df[df['date'] <= train_end_date]
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)]
        test_df = df[df['date'] > val_end_date]
        
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }


def main():
    """Example usage of the data collector"""
    # Get S&P 500 tickers (simplified list for demo)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    
    # Initialize collector
    collector = MLDataCollector(
        tickers=tickers,
        start_date='2015-01-01',
        label_threshold=0.05,  # 5% profit target
        label_days=10  # 10-day holding period
    )
    
    # Collect all data
    print("Collecting historical data...")
    df = collector.collect_all_data()
    
    if not df.empty:
        print(f"\nCollected {len(df)} data points")
        print(f"Features shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")
        
        # Create train/test split
        splits = collector.create_train_test_split(df)
        print(f"\nTrain samples: {len(splits['train'])}")
        print(f"Validation samples: {len(splits['validation'])}")
        print(f"Test samples: {len(splits['test'])}")


if __name__ == "__main__":
    main()