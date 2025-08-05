"""
Technical indicator calculations for intraday data
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import pandas_ta as ta

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
    return ta.rsi(prices, length=period)


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
    bb = ta.bbands(prices, length=period, std=std_dev)
    if bb is not None and not bb.empty:
        # pandas_ta returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        lower = bb.iloc[:, 0]  # Lower band
        middle = bb.iloc[:, 1]  # Middle band (SMA)
        upper = bb.iloc[:, 2]  # Upper band
        return upper, middle, lower
    else:
        # Fallback calculation if pandas_ta fails
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
    macd_result = ta.macd(prices, fast=fast, slow=slow, signal=signal)
    if macd_result is not None and not macd_result.empty:
        # pandas_ta returns columns like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        macd_line = macd_result.iloc[:, 0]
        histogram = macd_result.iloc[:, 1]
        signal_line = macd_result.iloc[:, 2]
        return macd_line, signal_line, histogram
    else:
        # Fallback calculation
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


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
    # Calculate OBV
    obv = ta.obv(prices, volumes)
    if obv is None:
        # Fallback calculation
        price_diff = prices.diff()
        volume_direction = volumes.copy()
        volume_direction[price_diff < 0] *= -1
        volume_direction[price_diff == 0] = 0
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
    volumes = df['volume']
    
    # Calculate indicators
    rsi = calculate_rsi(close_prices)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close_prices)
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    obv, obv_sma = calculate_obv(close_prices, volumes)
    
    # Get latest values (most recent bar)
    latest_idx = -1
    current_price = close_prices.iloc[latest_idx]
    
    # Handle NaN values by checking if indicators are calculated
    if pd.isna(rsi.iloc[latest_idx]):
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