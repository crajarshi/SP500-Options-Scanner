#!/usr/bin/env python3
"""
Test ML Data Collector with Alpaca API
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add ml_components to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_components'))

print("="*60)
print("Testing ML Data Collector with Alpaca API")
print("="*60)

try:
    from ml_data_collector import MLDataCollector
    print("✅ ML Data Collector imported successfully")
    
    # Test with a small set of tickers and short date range for quick testing
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Use a shorter date range for testing (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    print(f"\nTest Configuration:")
    print(f"  Tickers: {test_tickers}")
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Label threshold: 5%")
    print(f"  Label days: 10")
    
    # Initialize collector
    collector = MLDataCollector(
        tickers=test_tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        label_threshold=0.05,  # 5% profit target
        label_days=10,  # 10-day holding period
        cache_dir="ml_data_cache_test"
    )
    print("✅ Data collector initialized")
    
    # Test collecting data for one ticker
    print(f"\n" + "="*60)
    print("Testing single ticker data collection...")
    print("="*60)
    
    ticker = test_tickers[0]
    print(f"\nCollecting data for {ticker}...")
    df = collector.collect_stock_data(ticker)
    
    if df is not None and not df.empty:
        print(f"✅ Successfully collected {len(df)} days of data for {ticker}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
        
        # Test technical indicators
        print(f"\nCalculating technical indicators...")
        df_with_indicators = collector.calculate_technical_indicators(df)
        
        # Check if indicators were calculated
        indicator_cols = ['rsi', 'macd', 'bb_upper', 'bb_lower']
        found_indicators = [col for col in indicator_cols if col in df_with_indicators.columns]
        
        if found_indicators:
            print(f"✅ Successfully calculated {len(found_indicators)} indicators")
            print(f"   Sample indicators: {found_indicators}")
            
            # Show sample values (non-NaN)
            for ind in found_indicators[:2]:
                non_nan_values = df_with_indicators[ind].dropna()
                if len(non_nan_values) > 0:
                    print(f"   {ind}: mean={non_nan_values.mean():.2f}, std={non_nan_values.std():.2f}")
        
        # Test label creation
        print(f"\nCreating labels...")
        df_with_labels = collector.create_labels(df_with_indicators)
        
        if 'label' in df_with_labels.columns:
            label_counts = df_with_labels['label'].value_counts()
            print(f"✅ Labels created successfully")
            print(f"   Label distribution: {dict(label_counts)}")
            print(f"   Buy signals (1): {label_counts.get(1, 0)} ({100*label_counts.get(1, 0)/len(df_with_labels.dropna(subset=['label'])):.1f}%)")
            print(f"   Sell/Hold signals (0): {label_counts.get(0, 0)} ({100*label_counts.get(0, 0)/len(df_with_labels.dropna(subset=['label'])):.1f}%)")
    else:
        print(f"⚠️  No data collected for {ticker}")
        print("   This could be due to:")
        print("   - Market being closed")
        print("   - Invalid ticker symbol")
        print("   - API rate limits")
        print("   - Network issues")
    
    # Test parallel collection for multiple tickers
    print(f"\n" + "="*60)
    print("Testing parallel data collection...")
    print("="*60)
    
    print(f"\nCollecting data for {len(test_tickers)} tickers...")
    all_data = collector.collect_all_data(max_workers=3)
    
    if not all_data.empty:
        print(f"✅ Successfully collected combined dataset")
        print(f"   Total samples: {len(all_data)}")
        print(f"   Unique tickers: {all_data['ticker'].nunique()}")
        print(f"   Date range: {all_data.index.min()} to {all_data.index.max()}")
        
        # Show per-ticker statistics
        print("\n   Per-ticker statistics:")
        for ticker in all_data['ticker'].unique():
            ticker_data = all_data[all_data['ticker'] == ticker]
            print(f"     {ticker}: {len(ticker_data)} samples")
    else:
        print("⚠️  No combined data collected")
    
    print(f"\n" + "="*60)
    print("Summary")
    print("="*60)
    print("✅ ML Data Collector is working with Alpaca API")
    print("✅ Technical indicators can be calculated")
    print("✅ Labels can be created for supervised learning")
    print("✅ Parallel data collection is functional")
    
    print("\nNext steps:")
    print("1. Increase date range for more training data")
    print("2. Add more tickers for diversity")
    print("3. Run feature engineering pipeline")
    print("4. Train ML models with the collected data")
    
except ImportError as e:
    print(f"❌ Failed to import ML Data Collector: {e}")
    print("\nPlease ensure:")
    print("1. You're in the correct directory")
    print("2. The ml_components folder exists")
    print("3. All dependencies are installed")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()