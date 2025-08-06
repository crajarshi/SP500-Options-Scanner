#!/usr/bin/env python3
"""
Test script for enhanced market regime detection
Tests the three-factor market regime filter
"""
import sys
import pandas as pd
from datetime import datetime
from alpaca_data_provider import AlpacaDataProvider
import config

def test_spy_trend():
    """Test SPY vs 50-day MA"""
    print("\n" + "="*60)
    print("Testing SPY Trend Analysis")
    print("="*60)
    
    try:
        provider = AlpacaDataProvider()
        
        # Fetch SPY data
        spy_data = provider.fetch_daily_bars('SPY', days_back=60)
        
        if spy_data is not None and not spy_data.empty:
            # Calculate 50-day MA
            spy_data['ma50'] = spy_data['close'].rolling(window=50).mean()
            
            current_price = spy_data['close'].iloc[-1]
            ma50 = spy_data['ma50'].iloc[-1]
            
            if pd.notna(ma50):
                is_bullish = current_price > ma50
                
                print(f"SPY Current Price: ${current_price:.2f}")
                print(f"50-day MA: ${ma50:.2f}")
                
                if is_bullish:
                    pct_above = ((current_price - ma50) / ma50) * 100
                    print(f"Status: ✅ BULLISH (+{pct_above:.1f}% above MA)")
                else:
                    pct_below = ((ma50 - current_price) / ma50) * 100
                    print(f"Status: ❌ BEARISH (-{pct_below:.1f}% below MA)")
                
                return is_bullish
            else:
                print("⚠️  Insufficient data for 50-day MA calculation")
        else:
            print("⚠️  Could not fetch SPY data")
    
    except Exception as e:
        print(f"❌ Error testing SPY trend: {e}")
    
    return None

def test_vix_level():
    """Test VIX level"""
    print("\n" + "="*60)
    print("Testing VIX Level")
    print("="*60)
    
    try:
        provider = AlpacaDataProvider()
        
        # Fetch VIX data
        vix_level = provider.fetch_vix_data()
        
        if vix_level:
            is_bullish = vix_level < config.VIX_THRESHOLD
            
            print(f"VIX Level: {vix_level:.2f}")
            print(f"Threshold: {config.VIX_THRESHOLD}")
            
            if is_bullish:
                print(f"Status: ✅ LOW VOLATILITY (below threshold)")
            else:
                print(f"Status: ❌ HIGH VOLATILITY (above threshold)")
            
            # Additional context
            if vix_level < config.VIX_WARNING_THRESHOLD:
                print(f"Note: VIX below {config.VIX_WARNING_THRESHOLD} - Very calm market")
            elif vix_level > config.EXTREME_VIX_THRESHOLD:
                print(f"Warning: VIX above {config.EXTREME_VIX_THRESHOLD} - Extreme volatility!")
            
            return is_bullish
        else:
            print("⚠️  Could not fetch VIX data")
    
    except Exception as e:
        print(f"❌ Error testing VIX: {e}")
    
    return None

def test_market_breadth_sample():
    """Test market breadth with a small sample"""
    print("\n" + "="*60)
    print("Testing Market Breadth (Sample)")
    print("="*60)
    
    try:
        provider = AlpacaDataProvider()
        
        # Test with a small sample of stocks
        sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
                         'NVDA', 'TSLA', 'JPM', 'JNJ', 'V']
        
        above_ma = 0
        total_checked = 0
        
        print(f"Checking sample of {len(sample_tickers)} stocks...")
        
        for ticker in sample_tickers:
            try:
                df = provider.fetch_daily_bars(ticker, days_back=60)
                
                if df is not None and len(df) >= 50:
                    df['ma50'] = df['close'].rolling(window=50).mean()
                    current = df['close'].iloc[-1]
                    ma50 = df['ma50'].iloc[-1]
                    
                    if pd.notna(ma50):
                        total_checked += 1
                        if current > ma50:
                            above_ma += 1
                            print(f"  {ticker}: ✅ Above 50MA")
                        else:
                            print(f"  {ticker}: ❌ Below 50MA")
            except:
                print(f"  {ticker}: ⚠️  Skipped")
        
        if total_checked > 0:
            breadth_pct = (above_ma / total_checked) * 100
            print(f"\nSample Breadth: {breadth_pct:.1f}% ({above_ma}/{total_checked} above 50MA)")
            
            is_bullish = breadth_pct > config.MARKET_BREADTH_THRESHOLD
            if is_bullish:
                print(f"Status: ✅ STRONG BREADTH (>{config.MARKET_BREADTH_THRESHOLD}%)")
            else:
                print(f"Status: ❌ WEAK BREADTH (<{config.MARKET_BREADTH_THRESHOLD}%)")
            
            return is_bullish
        else:
            print("⚠️  No stocks could be checked")
    
    except Exception as e:
        print(f"❌ Error testing breadth: {e}")
    
    return None

def test_regime_check():
    """Test the complete regime check"""
    print("\n" + "="*60)
    print("COMPLETE MARKET REGIME CHECK")
    print("="*60)
    
    # Run individual tests
    spy_bullish = test_spy_trend()
    vix_bullish = test_vix_level()
    breadth_bullish = test_market_breadth_sample()
    
    # Summary
    print("\n" + "="*60)
    print("REGIME SUMMARY")
    print("="*60)
    
    factors_passed = 0
    
    if spy_bullish is not None:
        print(f"1. SPY Trend:      {'✅ Bullish' if spy_bullish else '❌ Bearish'}")
        if spy_bullish:
            factors_passed += 1
    else:
        print(f"1. SPY Trend:      ⚠️  Unknown")
    
    if vix_bullish is not None:
        print(f"2. VIX Level:      {'✅ Low' if vix_bullish else '❌ High'}")
        if vix_bullish:
            factors_passed += 1
    else:
        print(f"2. VIX Level:      ⚠️  Unknown")
    
    if breadth_bullish is not None:
        print(f"3. Market Breadth: {'✅ Strong' if breadth_bullish else '❌ Weak'}")
        if breadth_bullish:
            factors_passed += 1
    else:
        print(f"3. Market Breadth: ⚠️  Unknown")
    
    print("\n" + "-"*60)
    
    if factors_passed == 3:
        print("✅ MARKET REGIME: BULLISH (3/3 factors)")
        print("Recommendation: Proceed with bullish options strategies")
    elif factors_passed == 2:
        print("⚠️  MARKET REGIME: MIXED (2/3 factors)")
        print("Recommendation: Exercise caution, reduce position sizes")
    else:
        print("❌ MARKET REGIME: NOT BULLISH ({}/3 factors)".format(factors_passed))
        print("Recommendation: Avoid bullish strategies, consider defensive positions")

def test_command_line_options():
    """Test command line regime check"""
    print("\n" + "="*60)
    print("Testing Command Line Options")
    print("="*60)
    
    print("\nAvailable commands:")
    print("  python sp500_options_scanner.py --regime-only")
    print("    → Check regime without running scan")
    print("\n  python sp500_options_scanner.py --fast-regime")
    print("    → Skip breadth calculation for faster check")
    print("\n  python sp500_options_scanner.py --warm-cache")
    print("    → Pre-calculate and cache market breadth")
    
    print("\nConfiguration thresholds (config.py):")
    print(f"  - SPY MA Period: {config.MARKET_REGIME_MA_PERIOD} days")
    print(f"  - Breadth Threshold: {config.MARKET_BREADTH_THRESHOLD}%")
    print(f"  - VIX Threshold: {config.VIX_THRESHOLD}")
    print(f"  - VIX Warning: {config.VIX_WARNING_THRESHOLD}")

def main():
    """Run all tests"""
    print("="*60)
    print("Enhanced Market Regime Detection Test")
    print("="*60)
    
    if '--quick' in sys.argv:
        # Quick test - regime only
        test_regime_check()
    elif '--spy' in sys.argv:
        # Test SPY only
        test_spy_trend()
    elif '--vix' in sys.argv:
        # Test VIX only
        test_vix_level()
    elif '--breadth' in sys.argv:
        # Test breadth only
        test_market_breadth_sample()
    elif '--help' in sys.argv:
        # Show command options
        test_command_line_options()
    else:
        # Full test
        test_regime_check()
        test_command_line_options()

if __name__ == "__main__":
    main()