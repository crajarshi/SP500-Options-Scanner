"""
Technical indicator calculations for intraday data
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

import config

logger = logging.getLogger(__name__)


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


def calculate_stock_trend(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate individual stock trend using strict dual confirmation
    Both EMA20 and SMA50 must align for trend confirmation
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with trend direction and strength
    """
    if len(df) < 50:
        return {'direction': 'neutral', 'strength': 0, 'ema20': None, 'sma50': None}
    
    close_prices = df['close']
    
    # Short-term: 20-period EMA
    ema20 = close_prices.ewm(span=20, adjust=False).mean()
    
    # Medium-term: 50-period SMA
    sma50 = close_prices.rolling(window=50).mean()
    
    current_price = close_prices.iloc[-1]
    ema20_value = ema20.iloc[-1]
    sma50_value = sma50.iloc[-1]
    
    # Check if we have valid moving averages
    if pd.isna(sma50_value):
        return {'direction': 'neutral', 'strength': 0, 'ema20': ema20_value, 'sma50': None}
    
    # Calculate distances from moving averages
    ema20_distance = ((current_price - ema20_value) / ema20_value) * 100
    sma50_distance = ((current_price - sma50_value) / sma50_value) * 100
    
    # Both must be true for bullish confirmation
    above_ema20 = current_price > ema20_value
    above_sma50 = current_price > sma50_value
    
    # Calculate trend strength (average of distances)
    avg_distance = (abs(ema20_distance) + abs(sma50_distance)) / 2
    
    # Strict trend confirmation - BOTH must align
    if above_ema20 and above_sma50 and ema20_distance > 0.5 and sma50_distance > 0:
        return {
            'direction': 'bullish',
            'strength': avg_distance,
            'ema20': ema20_value,
            'sma50': sma50_value,
            'ema20_distance': ema20_distance,
            'sma50_distance': sma50_distance
        }
    elif not above_ema20 and not above_sma50 and ema20_distance < -0.5 and sma50_distance < 0:
        return {
            'direction': 'bearish',
            'strength': avg_distance,
            'ema20': ema20_value,
            'sma50': sma50_value,
            'ema20_distance': ema20_distance,
            'sma50_distance': sma50_distance
        }
    else:
        return {
            'direction': 'neutral',
            'strength': 0,
            'ema20': ema20_value,
            'sma50': sma50_value,
            'ema20_distance': ema20_distance,
            'sma50_distance': sma50_distance
        }


def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate all technical indicators for a stock
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
    
    Returns:
        Dictionary containing all indicator values and scores
    """
    # Defensive check: Ensure we have enough bars for all indicators
    # Required periods: RSI(14), BB(20), MACD(26), ATR(14), ATR_SMA(30), Trend(50)
    required_bars = max(
        config.RSI_PERIOD,           # 14
        config.BOLLINGER_PERIOD,     # 20
        config.MACD_SLOW,            # 26
        config.ATR_SMA_PERIOD,       # 30
        50                           # For SMA50 in trend calculation
    )
    
    if len(df) < required_bars:
        logger.warning(f"Insufficient bars for indicators: have {len(df)}, need {required_bars}")
        logger.debug(f"Minimum requirements - RSI:{config.RSI_PERIOD}, BB:{config.BOLLINGER_PERIOD}, "
                    f"MACD:{config.MACD_SLOW}, ATR_SMA:{config.ATR_SMA_PERIOD}, Trend:50")
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
    
    # Calculate stock trend
    trend = calculate_stock_trend(df)
    
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
        },
        'trend': trend
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