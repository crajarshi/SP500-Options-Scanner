"""
Signal generation and scoring logic for options trading
"""
from typing import Dict, Optional, Tuple
import config


def calculate_rsi_score(rsi_value: float) -> float:
    """
    Calculate RSI score (0-100)
    
    Scoring logic:
    - RSI < 30: Score = 100 (oversold)
    - RSI 30-70: Score = 100 - ((RSI - 30) * 2.5)
    - RSI > 70: Score = 0 (overbought)
    """
    if rsi_value < config.RSI_OVERSOLD:
        return 100.0
    elif rsi_value > config.RSI_OVERBOUGHT:
        return 0.0
    else:
        # Linear scaling between 30 and 70
        return 100.0 - ((rsi_value - config.RSI_OVERSOLD) * 2.5)


def calculate_bollinger_score(position: float) -> float:
    """
    Calculate Bollinger Band score (0-100)
    
    Position = (Price - Lower Band) / (Upper Band - Lower Band)
    Score = 100 - (position * 100)
    Higher score when price is closer to lower band
    """
    # Clamp position between 0 and 1
    position = max(0, min(1, position))
    return 100.0 - (position * 100.0)


def calculate_macd_score(macd_bullish: bool) -> float:
    """
    Calculate MACD score (0-100)
    
    Simple binary scoring:
    - MACD > Signal: Score = 100 (bullish)
    - MACD <= Signal: Score = 0 (bearish)
    """
    return 100.0 if macd_bullish else 0.0


def calculate_obv_score(obv_above_sma: bool) -> float:
    """
    Calculate OBV score (0-100)
    
    Simple binary scoring:
    - OBV > SMA: Score = 100 (positive volume)
    - OBV <= SMA: Score = 0 (negative volume)
    """
    return 100.0 if obv_above_sma else 0.0


def calculate_atr_score(atr_above_sma: bool) -> float:
    """
    Calculate ATR score (0-100)
    
    Simple binary scoring:
    - ATR > SMA: Score = 100 (expanding volatility)
    - ATR <= SMA: Score = 0 (contracting volatility)
    """
    return 100.0 if atr_above_sma else 0.0


def calculate_composite_score(indicators: Dict) -> Dict[str, float]:
    """
    Calculate individual and composite scores from indicators
    
    Args:
        indicators: Dictionary of indicator values
    
    Returns:
        Dictionary with individual scores and weighted composite score
    """
    # Calculate individual scores
    rsi_score = calculate_rsi_score(indicators['rsi']['value'])
    bollinger_score = calculate_bollinger_score(indicators['bollinger']['position'])
    macd_score = calculate_macd_score(indicators['macd']['bullish'])
    obv_score = calculate_obv_score(indicators['obv']['above_sma'])
    atr_score = calculate_atr_score(indicators['atr']['above_sma'])
    
    # Calculate weighted composite score
    composite_score = (
        rsi_score * config.WEIGHT_RSI +
        macd_score * config.WEIGHT_MACD +
        bollinger_score * config.WEIGHT_BOLLINGER +
        obv_score * config.WEIGHT_OBV +
        atr_score * config.WEIGHT_ATR
    )
    
    return {
        'rsi_score': round(rsi_score, config.DECIMAL_PLACES),
        'macd_score': round(macd_score, config.DECIMAL_PLACES),
        'bollinger_score': round(bollinger_score, config.DECIMAL_PLACES),
        'obv_score': round(obv_score, config.DECIMAL_PLACES),
        'atr_score': round(atr_score, config.DECIMAL_PLACES),
        'composite_score': round(composite_score, config.DECIMAL_PLACES)
    }


def generate_signal(composite_score: float) -> Tuple[str, str, str]:
    """
    Generate trading signal based on composite score
    
    Args:
        composite_score: Weighted average score (0-100)
    
    Returns:
        Tuple of (signal_type, signal_text, signal_emoji)
    """
    if composite_score > config.SIGNAL_STRONG_BUY:
        return 'STRONG_BUY', 'STRONG BUY: Sell Put / Buy Call', '🟢'
    elif composite_score > config.SIGNAL_BUY:
        return 'BUY', 'BUY: Sell Put / Buy Call', '🟢'
    elif config.SIGNAL_HOLD_MIN <= composite_score <= config.SIGNAL_HOLD_MAX:
        return 'HOLD', 'HOLD: No Signal', '⚪'
    else:
        return 'AVOID', 'AVOID: No Bullish Edge', '🔴'


def analyze_stock(ticker: str, indicators: Dict) -> Optional[Dict]:
    """
    Complete analysis for a single stock
    
    Args:
        ticker: Stock symbol
        indicators: Dictionary of calculated indicators
    
    Returns:
        Complete analysis results or None if insufficient data
    """
    if not indicators:
        return None
    
    # Calculate scores
    scores = calculate_composite_score(indicators)
    
    # Generate signal
    signal_type, signal_text, signal_emoji = generate_signal(scores['composite_score'])
    
    # Build complete analysis
    analysis = {
        'ticker': ticker,
        'current_price': indicators['current_price'],
        'price_change_pct': indicators['price_change']['change_pct'],
        'scores': scores,
        'signal': {
            'type': signal_type,
            'text': signal_text,
            'emoji': signal_emoji
        },
        'indicators': {
            'rsi': indicators['rsi']['value'],
            'macd_bullish': indicators['macd']['bullish'],
            'bb_position': indicators['bollinger']['position'],
            'obv_above_sma': indicators['obv']['above_sma'],
            'atr_value': indicators['atr']['value'],
            'atr_trend': indicators['atr']['trend']
        }
    }
    
    return analysis


def rank_stocks(analyses: list) -> list:
    """
    Rank stocks by composite score (descending)
    
    Args:
        analyses: List of stock analysis dictionaries
    
    Returns:
        Sorted list of analyses
    """
    # Filter out None values
    valid_analyses = [a for a in analyses if a is not None]
    
    # Sort by composite score (highest first)
    return sorted(valid_analyses, 
                 key=lambda x: x['scores']['composite_score'], 
                 reverse=True)