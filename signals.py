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


def calculate_volume_score(relative_volume: float) -> float:
    """
    Calculate volume score (0-100)
    
    Scoring logic:
    - Relative volume > 2.0: Score = 100 (very high volume)
    - Relative volume 1.5-2.0: Score = 75
    - Relative volume 1.0-1.5: Score = 50
    - Relative volume < 1.0: Score = 0 (below average)
    """
    if relative_volume >= 2.0:
        return 100.0
    elif relative_volume >= 1.5:
        return 75.0
    elif relative_volume >= 1.0:
        return 50.0
    else:
        return 0.0


# ============== BEARISH SCORING FUNCTIONS ==============

def calculate_bearish_rsi_score(rsi_value: float) -> float:
    """
    Calculate bearish RSI score (0-100)
    
    Scoring logic (inverse of bullish):
    - RSI > 70: Score = 100 (overbought - ready to fall)
    - RSI 30-70: Linear scale favoring higher values
    - RSI < 30: Score = 0 (oversold - not bearish)
    """
    if rsi_value > config.RSI_OVERBOUGHT:
        return 100.0
    elif rsi_value < config.RSI_OVERSOLD:
        return 0.0
    else:
        # Linear scaling: higher RSI = higher bearish score
        return (rsi_value - config.RSI_OVERSOLD) * 2.5


def calculate_bearish_bollinger_score(position: float) -> float:
    """
    Calculate bearish Bollinger Band score (0-100)
    
    Position = (Price - Lower Band) / (Upper Band - Lower Band)
    Score = position * 100 (inverse of bullish)
    Higher score when price is closer to upper band
    """
    position = max(0, min(1, position))
    return position * 100.0


def calculate_bearish_macd_score(macd_bullish: bool) -> float:
    """
    Calculate bearish MACD score (0-100)
    
    Simple binary scoring (inverse of bullish):
    - MACD < Signal: Score = 100 (bearish crossover)
    - MACD >= Signal: Score = 0 (bullish)
    """
    return 0.0 if macd_bullish else 100.0


def calculate_bearish_obv_score(obv_above_sma: bool) -> float:
    """
    Calculate bearish OBV score (0-100)
    
    Simple binary scoring (inverse of bullish):
    - OBV < SMA: Score = 100 (distribution/selling pressure)
    - OBV >= SMA: Score = 0 (accumulation)
    """
    return 0.0 if obv_above_sma else 100.0


def calculate_bearish_volume_score(relative_volume: float, price_change_pct: float) -> float:
    """
    Calculate bearish volume score (0-100)
    
    High volume on down days indicates strong selling pressure
    """
    if price_change_pct < -0.5 and relative_volume >= 2.0:
        return 100.0  # Strong selling with very high volume
    elif price_change_pct < -0.5 and relative_volume >= 1.5:
        return 75.0   # Strong selling with high volume
    elif price_change_pct < 0 and relative_volume >= 1.0:
        return 50.0   # Selling with average+ volume
    else:
        return 0.0    # No bearish volume signal


def calculate_bearish_atr_score(atr_above_sma: bool) -> float:
    """
    Calculate bearish ATR score (0-100)
    
    Same as bullish - volatility expansion is good for options regardless of direction
    """
    return 100.0 if atr_above_sma else 0.0


def calculate_bullish_composite(indicators: Dict) -> Dict[str, float]:
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
    volume_score = calculate_volume_score(indicators.get('volume', {}).get('relative', 1.0))
    
    # Calculate weighted composite score
    composite_score = (
        rsi_score * config.WEIGHT_RSI +
        macd_score * config.WEIGHT_MACD +
        bollinger_score * config.WEIGHT_BOLLINGER +
        obv_score * config.WEIGHT_OBV +
        volume_score * config.WEIGHT_VOLUME +
        atr_score * config.WEIGHT_ATR
    )
    
    return {
        'rsi_score': round(rsi_score, config.DECIMAL_PLACES),
        'macd_score': round(macd_score, config.DECIMAL_PLACES),
        'bollinger_score': round(bollinger_score, config.DECIMAL_PLACES),
        'obv_score': round(obv_score, config.DECIMAL_PLACES),
        'volume_score': round(volume_score, config.DECIMAL_PLACES),
        'atr_score': round(atr_score, config.DECIMAL_PLACES),
        'composite_score': round(composite_score, config.DECIMAL_PLACES)
    }


def calculate_bearish_composite(indicators: Dict) -> Dict[str, float]:
    """
    Calculate bearish individual and composite scores
    
    Args:
        indicators: Dictionary of indicator values
    
    Returns:
        Dictionary with individual scores and weighted composite score
    """
    # Calculate individual bearish scores
    rsi_score = calculate_bearish_rsi_score(indicators['rsi']['value'])
    bollinger_score = calculate_bearish_bollinger_score(indicators['bollinger']['position'])
    macd_score = calculate_bearish_macd_score(indicators['macd']['bullish'])
    obv_score = calculate_bearish_obv_score(indicators['obv']['above_sma'])
    atr_score = calculate_bearish_atr_score(indicators['atr']['above_sma'])
    
    # Bearish volume score needs price change
    price_change_pct = indicators.get('price_change', {}).get('change_pct', 0)
    relative_volume = indicators.get('volume', {}).get('relative', 1.0)
    volume_score = calculate_bearish_volume_score(relative_volume, price_change_pct)
    
    # Calculate weighted composite score
    composite_score = (
        rsi_score * config.WEIGHT_RSI +
        macd_score * config.WEIGHT_MACD +
        bollinger_score * config.WEIGHT_BOLLINGER +
        obv_score * config.WEIGHT_OBV +
        volume_score * config.WEIGHT_VOLUME +
        atr_score * config.WEIGHT_ATR
    )
    
    return {
        'rsi_score': round(rsi_score, config.DECIMAL_PLACES),
        'macd_score': round(macd_score, config.DECIMAL_PLACES),
        'bollinger_score': round(bollinger_score, config.DECIMAL_PLACES),
        'obv_score': round(obv_score, config.DECIMAL_PLACES),
        'volume_score': round(volume_score, config.DECIMAL_PLACES),
        'atr_score': round(atr_score, config.DECIMAL_PLACES),
        'composite_score': round(composite_score, config.DECIMAL_PLACES)
    }


def calculate_composite_score(indicators: Dict, mode: str = 'adaptive', market_regime: Dict = None) -> Optional[Dict]:
    """
    Main adaptive scoring function that determines which scoring to use
    
    Args:
        indicators: Dictionary of indicator values
        mode: 'adaptive', 'bullish', 'bearish', or 'mixed'
        market_regime: Market regime data for adaptive mode
    
    Returns:
        Dictionary with scores or None if filtered out
    """
    # Determine effective mode
    effective_mode = mode
    if mode == 'adaptive' and market_regime:
        breadth = market_regime.get('breadth_pct', 50)
        vix = market_regime.get('vix_level', 20)
        
        if breadth < 40 or vix > 25:
            effective_mode = 'bearish'
        elif breadth > 60 and vix < 20:
            effective_mode = 'bullish'
        else:
            effective_mode = 'mixed'
    
    # Get stock trend
    stock_trend = indicators.get('trend', {})
    trend_direction = stock_trend.get('direction', 'neutral')
    
    # Calculate both scores
    bullish_scores = calculate_bullish_composite(indicators)
    bearish_scores = calculate_bearish_composite(indicators)
    
    # Apply mode logic
    if effective_mode == 'bullish':
        return {
            **bullish_scores,
            'score_type': 'bullish',
            'mode': effective_mode
        }
    
    elif effective_mode == 'bearish':
        return {
            **bearish_scores,
            'score_type': 'bearish',
            'mode': effective_mode
        }
    
    elif effective_mode == 'mixed':
        # In mixed mode, require trend confirmation
        if trend_direction == 'bullish':
            return {
                **bullish_scores,
                'score_type': 'bullish',
                'mode': effective_mode,
                'trend_confirmed': True
            }
        elif trend_direction == 'bearish':
            return {
                **bearish_scores,
                'score_type': 'bearish',
                'mode': effective_mode,
                'trend_confirmed': True
            }
        else:
            # Neutral trend in mixed market - filter out
            return None
    
    # Default to bullish if mode not recognized
    return {
        **bullish_scores,
        'score_type': 'bullish',
        'mode': 'default'
    }


def generate_signal(composite_score: float, score_type: str = 'bullish') -> Tuple[str, str, str]:
    """
    Generate trading signal based on composite score and type
    
    Args:
        composite_score: Weighted average score (0-100)
        score_type: 'bullish' or 'bearish' scoring type
    
    Returns:
        Tuple of (signal_type, signal_text, signal_emoji)
    """
    if score_type == 'bullish':
        # Bullish scoring interpretation
        if composite_score > config.SIGNAL_STRONG_BUY:
            return 'STRONG_BUY', 'SELL PUT / BUY CALL', '🟢'
        elif composite_score > config.SIGNAL_BUY:
            return 'BUY', 'BULL SPREAD / SELL PUT', '🟢'
        elif composite_score > config.SIGNAL_NEUTRAL_BULLISH:
            return 'WEAK_BUY', 'SELL OTM PUT', '🟡'
        else:
            return 'NEUTRAL', 'NO ACTION', '⚪'
    
    elif score_type == 'bearish':
        # Bearish scoring interpretation
        if composite_score > config.SIGNAL_STRONG_BUY:
            return 'STRONG_SELL', 'BUY PUT / SELL CALL', '🔴'
        elif composite_score > config.SIGNAL_BUY:
            return 'SELL', 'BEAR SPREAD / BUY PUT', '🔴'
        elif composite_score > config.SIGNAL_NEUTRAL_BULLISH:
            return 'WEAK_SELL', 'BUY OTM PUT', '🟠'
        else:
            return 'NEUTRAL', 'NO ACTION', '⚪'
    
    # Default fallback
    return 'HOLD', 'NO SIGNAL', '⚪'


def analyze_stock(ticker: str, indicators: Dict, mode: str = 'adaptive', market_regime: Dict = None) -> Optional[Dict]:
    """
    Complete analysis for a single stock with mode awareness
    
    Args:
        ticker: Stock symbol
        indicators: Dictionary of calculated indicators
        mode: Scanner mode ('adaptive', 'bullish', 'bearish', 'mixed')
        market_regime: Market regime data for adaptive mode
    
    Returns:
        Complete analysis results or None if insufficient data or filtered
    """
    if not indicators:
        return None
    
    # Calculate adaptive scores
    scores = calculate_composite_score(indicators, mode, market_regime)
    
    # Check if filtered out (e.g., neutral stock in mixed mode)
    if scores is None:
        return None
    
    # Apply stricter threshold in mixed mode
    if scores.get('mode') == 'mixed' and scores['composite_score'] < 70:
        return None
    
    # Generate signal based on score type
    score_type = scores.get('score_type', 'bullish')
    signal_type, signal_text, signal_emoji = generate_signal(scores['composite_score'], score_type)
    
    # Build complete analysis
    analysis = {
        'ticker': ticker,
        'current_price': indicators['current_price'],
        'price_change_pct': indicators['price_change']['change_pct'],
        'scores': scores,
        'score_type': score_type,
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
            'atr_trend': indicators['atr']['trend'],
            'volume_relative': indicators.get('volume', {}).get('relative', 1.0)
        },
        'trend': indicators.get('trend', {})
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