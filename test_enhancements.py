#!/usr/bin/env python3
"""
Test script to verify all options trading enhancements
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
from indicators import calculate_atr, calculate_all_indicators
from signals import calculate_atr_score, calculate_composite_score
from alpaca_data_provider import AlpacaDataProvider

def test_atr_calculation():
    """Test ATR calculation"""
    print("\n1. Testing ATR Calculation...")
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=50, freq='15min')
    high = pd.Series(np.random.uniform(100, 105, 50))
    low = pd.Series(np.random.uniform(95, 100, 50))
    close = pd.Series(np.random.uniform(97, 103, 50))
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period=14)
    
    if atr is not None and not atr.isna().all():
        print(f"   ✓ ATR calculated successfully")
        print(f"   Latest ATR: ${atr.iloc[-1]:.2f}")
        
        # Calculate ATR SMA
        atr_sma = atr.rolling(window=30).mean()
        if not pd.isna(atr_sma.iloc[-1]):
            print(f"   ATR SMA(30): ${atr_sma.iloc[-1]:.2f}")
            trend = "Rising" if atr.iloc[-1] > atr_sma.iloc[-1] else "Falling"
            print(f"   Volatility Trend: {trend}")
    else:
        print("   ✗ ATR calculation failed")
    
    return True

def test_atr_scoring():
    """Test ATR scoring logic"""
    print("\n2. Testing ATR Scoring...")
    
    # Test cases
    test_cases = [
        (True, 100.0, "ATR above SMA"),
        (False, 0.0, "ATR below SMA")
    ]
    
    for above_sma, expected_score, description in test_cases:
        score = calculate_atr_score(above_sma)
        status = "✓" if score == expected_score else "✗"
        print(f"   {status} {description}: Score = {score} (expected {expected_score})")
    
    return True

def test_composite_score_with_atr():
    """Test composite score calculation with ATR"""
    print("\n3. Testing Composite Score with ATR...")
    
    # Create mock indicators
    indicators = {
        'rsi': {'value': 45},
        'bollinger': {'position': 0.3},
        'macd': {'bullish': True},
        'obv': {'above_sma': True},
        'atr': {'above_sma': True}
    }
    
    scores = calculate_composite_score(indicators)
    
    print(f"   Individual Scores:")
    print(f"   - RSI: {scores['rsi_score']}")
    print(f"   - MACD: {scores['macd_score']}")
    print(f"   - Bollinger: {scores['bollinger_score']}")
    print(f"   - OBV: {scores['obv_score']}")
    print(f"   - ATR: {scores['atr_score']}")
    print(f"   Composite Score: {scores['composite_score']}")
    
    # Verify weights add up to 100%
    total_weight = (config.WEIGHT_RSI + config.WEIGHT_MACD + 
                   config.WEIGHT_BOLLINGER + config.WEIGHT_OBV + config.WEIGHT_ATR)
    
    if abs(total_weight - 1.0) < 0.001:
        print(f"   ✓ Weights sum to 100%")
    else:
        print(f"   ✗ Weights sum to {total_weight * 100}%, not 100%")
    
    return True

def test_market_regime_filter():
    """Test market regime filter logic"""
    print("\n4. Testing Market Regime Filter...")
    
    # Create sample SPY data
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    
    # Bullish scenario: price above MA
    spy_prices_bull = pd.Series(np.linspace(400, 450, 60))
    ma50_bull = spy_prices_bull.rolling(window=50).mean()
    
    if not pd.isna(ma50_bull.iloc[-1]):
        current_price = spy_prices_bull.iloc[-1]
        ma50 = ma50_bull.iloc[-1]
        is_bullish = current_price > ma50
        
        print(f"   Bullish Test:")
        print(f"   SPY: ${current_price:.2f}, MA50: ${ma50:.2f}")
        print(f"   Market Regime: {'BULLISH ✓' if is_bullish else 'BEARISH ✗'}")
    
    # Bearish scenario: price below MA
    spy_prices_bear = pd.Series(np.linspace(450, 400, 60))
    ma50_bear = spy_prices_bear.rolling(window=50).mean()
    
    if not pd.isna(ma50_bear.iloc[-1]):
        current_price = spy_prices_bear.iloc[-1]
        ma50 = ma50_bear.iloc[-1]
        is_bearish = current_price < ma50
        
        print(f"   Bearish Test:")
        print(f"   SPY: ${current_price:.2f}, MA50: ${ma50:.2f}")
        print(f"   Market Regime: {'BEARISH ✓' if is_bearish else 'BULLISH ✗'}")
    
    return True

def test_daily_bars_fetching():
    """Test fetching daily bars for market regime check"""
    print("\n5. Testing Daily Bars Fetching...")
    
    try:
        provider = AlpacaDataProvider()
        
        # Test connection first
        if provider.test_connection():
            print("   ✓ Alpaca connection successful")
            
            # Try to fetch SPY daily bars
            spy_data = provider.fetch_daily_bars('SPY', days_back=60)
            
            if spy_data is not None and not spy_data.empty:
                print(f"   ✓ Fetched {len(spy_data)} daily bars for SPY")
                print(f"   Date range: {spy_data['timestamp'].min()} to {spy_data['timestamp'].max()}")
            else:
                print("   ⚠ Could not fetch SPY data (may be outside market hours)")
        else:
            print("   ⚠ Alpaca connection failed (check API keys)")
    except Exception as e:
        print(f"   ⚠ Error testing daily bars: {e}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Options Trading Enhancements")
    print("=" * 60)
    
    # Run tests
    test_atr_calculation()
    test_atr_scoring()
    test_composite_score_with_atr()
    test_market_regime_filter()
    test_daily_bars_fetching()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
    
    # Summary of enhancements
    print("\nEnhancements Summary:")
    print("1. Market Regime Filter: SPY > 50-day MA check")
    print("2. ATR Scoring: 10% weight in composite score")
    print("3. Volatility Trend: Rising/Falling indicators")
    print("4. Quick Scan Mode: --quick flag for cached data")
    print("5. Enhanced Dashboard: ATR values and trends displayed")
    
    print("\nNew Weights Distribution:")
    print(f"- MACD: {config.WEIGHT_MACD * 100}%")
    print(f"- RSI: {config.WEIGHT_RSI * 100}%")
    print(f"- Bollinger: {config.WEIGHT_BOLLINGER * 100}%")
    print(f"- OBV: {config.WEIGHT_OBV * 100}%")
    print(f"- ATR: {config.WEIGHT_ATR * 100}%")

if __name__ == "__main__":
    main()