"""
Demo data generator for testing when API access is limited
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_demo_intraday_data(ticker: str, days: int = 3) -> pd.DataFrame:
    """
    Generate realistic demo intraday data for testing
    
    Args:
        ticker: Stock symbol
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Base prices for some popular stocks
    base_prices = {
        'AAPL': 180.0,
        'MSFT': 380.0,
        'GOOGL': 140.0,
        'AMZN': 170.0,
        'META': 480.0,
        'NVDA': 700.0,
        'TSLA': 240.0,
        'BRK.B': 350.0,
        'JPM': 150.0,
        'JNJ': 155.0
    }
    
    # Get base price or use random
    base_price = base_prices.get(ticker, random.uniform(50, 500))
    
    # Generate timestamps (15-minute intervals during market hours)
    timestamps = []
    current_time = datetime.now() - timedelta(days=days)
    
    while current_time < datetime.now():
        # Only include market hours (9:30 AM - 4:00 PM ET)
        if current_time.weekday() < 5:  # Monday = 0, Friday = 4
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            time_slot = market_open
            while time_slot <= market_close:
                timestamps.append(time_slot)
                time_slot += timedelta(minutes=15)
        
        current_time += timedelta(days=1)
    
    # Generate price data with realistic patterns
    data = []
    current_price = base_price
    
    for ts in timestamps:
        # Add some intraday volatility
        volatility = 0.002  # 0.2% volatility per 15 minutes
        trend = -0.00001 if random.random() > 0.5 else 0.00001  # Slight trend
        
        # Generate OHLC
        open_price = current_price
        high_price = open_price * (1 + abs(np.random.normal(0, volatility)))
        low_price = open_price * (1 - abs(np.random.normal(0, volatility)))
        close_price = open_price * (1 + np.random.normal(trend, volatility))
        
        # Ensure high/low bounds
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (higher during open/close)
        hour = ts.hour
        if hour == 9 or hour == 15:
            volume_multiplier = 2.0
        else:
            volume_multiplier = 1.0
        
        volume = int(random.uniform(1000000, 5000000) * volume_multiplier)
        
        data.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        current_price = close_price
    
    return pd.DataFrame(data)


def get_demo_sp500_tickers() -> list:
    """Get a demo list of S&P 500 tickers for testing"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK.B', 'JPM', 'JNJ',
        'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'ADBE', 'NFLX',
        'CRM', 'XOM', 'PFE', 'KO', 'PEP', 'TMO', 'CSCO', 'ABT', 'CVX', 'NKE'
    ]