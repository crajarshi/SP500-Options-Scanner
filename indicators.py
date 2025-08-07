"""
Technical indicator calculations for intraday data
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

import config


def calculate_rsi(prices: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        prices: Series of closing prices
        period: RSI period (default: 14 bars)
    
    Returns:
        RSI values as pandas Series
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(prices: pd.Series, 
                            period: int = config.BOLLINGER_PERIOD,
                            std_dev: int = config.BOLLINGER_STD) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Series of closing prices
        period: Moving average period (default: 20 bars)
        std_dev: Number of standard deviations (default: 2)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    # Calculate moving average and standard deviation
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower


def calculate_macd(prices: pd.Series,
                  fast: int = config.MACD_FAST,
                  slow: int = config.MACD_SLOW,
                  signal: int = config.MACD_SIGNAL) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default: 12 bars)
        slow: Slow EMA period (default: 26 bars)
        signal: Signal line EMA period (default: 9 bars)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate exponential moving averages
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default: 14 bars)
    
    Returns:
        ATR values as pandas Series
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    # True Range is the maximum of the three
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR using RMA (Running Moving Average)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_obv(prices: pd.Series, volumes: pd.Series, 
                 sma_period: int = config.OBV_SMA_PERIOD) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate On-Balance Volume and its SMA
    
    Args:
        prices: Series of closing prices
        volumes: Series of volume data
        sma_period: Period for OBV's simple moving average (default: 20 bars)
    
    Returns:
        Tuple of (obv, obv_sma)
    """
    # Calculate price direction
    price_diff = prices.diff()
    
    # Create volume direction based on price movement
    volume_direction = volumes.copy()
    volume_direction[price_diff < 0] *= -1
    volume_direction[price_diff == 0] = 0
    
    # Calculate OBV as cumulative sum
    obv = volume_direction.cumsum()
    
    # Calculate OBV's SMA
    obv_sma = obv.rolling(window=sma_period).mean()
    
    return obv, obv_sma


def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate all technical indicators for a stock
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
    
    Returns:
        Dictionary containing all indicator values and scores
    """
    if len(df) < config.MIN_REQUIRED_BARS:
        return None
    
    # Extract price and volume data
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    volumes = df['volume']
    
    # Calculate indicators
    rsi = calculate_rsi(close_prices)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close_prices)
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    obv, obv_sma = calculate_obv(close_prices, volumes)
    
    # Calculate ATR and its SMA
    atr = calculate_atr(high_prices, low_prices, close_prices, period=config.ATR_PERIOD)
    atr_sma = atr.rolling(window=config.ATR_SMA_PERIOD).mean()
    
    # Calculate volume metrics
    avg_volume = volumes.rolling(window=20).mean()  # 20-period average volume
    current_volume = volumes.iloc[-1]
    relative_volume = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
    
    # Get latest values (most recent bar)
    latest_idx = -1
    current_price = close_prices.iloc[latest_idx]
    
    # Handle NaN values by checking if indicators are calculated
    if pd.isna(rsi.iloc[latest_idx]) or pd.isna(atr.iloc[latest_idx]) or pd.isna(atr_sma.iloc[latest_idx]):
        return None
    
    indicators = {
        'current_price': current_price,
        'rsi': {
            'value': rsi.iloc[latest_idx],
            'threshold_oversold': config.RSI_OVERSOLD,
            'threshold_overbought': config.RSI_OVERBOUGHT
        },
        'bollinger': {
            'upper': upper_bb.iloc[latest_idx],
            'middle': middle_bb.iloc[latest_idx],
            'lower': lower_bb.iloc[latest_idx],
            'position': (current_price - lower_bb.iloc[latest_idx]) / 
                       (upper_bb.iloc[latest_idx] - lower_bb.iloc[latest_idx])
        },
        'macd': {
            'macd_line': macd_line.iloc[latest_idx],
            'signal_line': signal_line.iloc[latest_idx],
            'histogram': histogram.iloc[latest_idx],
            'bullish': macd_line.iloc[latest_idx] > signal_line.iloc[latest_idx]
        },
        'obv': {
            'current': obv.iloc[latest_idx],
            'sma': obv_sma.iloc[latest_idx],
            'above_sma': obv.iloc[latest_idx] > obv_sma.iloc[latest_idx]
        },
        'atr': {
            'value': atr.iloc[latest_idx],
            'sma': atr_sma.iloc[latest_idx],
            'above_sma': atr.iloc[latest_idx] > atr_sma.iloc[latest_idx],
            'trend': 'Rising' if atr.iloc[latest_idx] > atr_sma.iloc[latest_idx] else 'Falling'
        },
        'volume': {
            'current': current_volume,
            'average': avg_volume.iloc[latest_idx],
            'relative': relative_volume,
            'above_average': relative_volume > 1.0
        }
    }
    
    # Add price change information
    indicators['price_change'] = {
        'open': df['open'].iloc[0],
        'current': current_price,
        'change_pct': ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100,
        'high': df['high'].max(),
        'low': df['low'].min()
    }
    
    return indicators