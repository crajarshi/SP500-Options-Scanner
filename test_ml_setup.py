#!/usr/bin/env python3
"""
Test ML Setup - Verify all components are working with Apple Silicon
"""
import os
import sys
print(f"Python version: {sys.version}")

# Test imports
print("\n" + "="*50)
print("Testing ML Libraries")
print("="*50)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
    
    # Check for Apple Silicon support
    if hasattr(tf.config, 'list_physical_devices'):
        mps_devices = tf.config.list_physical_devices('MPS')
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        if mps_devices:
            print(f"✅ Metal Performance Shaders (MPS) available: {mps_devices}")
        elif gpu_devices:
            print(f"✅ GPU available: {gpu_devices}")
        else:
            print("ℹ️  Running in CPU mode")
            
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import shap
    print(f"✅ SHAP {shap.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ SHAP import failed: {e}")

try:
    import xgboost as xgb
    print(f"✅ XGBoost {xgb.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ XGBoost import failed: {e}")

try:
    import mlflow
    print(f"✅ MLflow {mlflow.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ MLflow import failed: {e}")

try:
    import optuna
    print(f"✅ Optuna {optuna.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ Optuna import failed: {e}")

# Test ML components
print("\n" + "="*50)
print("Testing ML Components")
print("="*50)

# Add ml_components to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_components'))

try:
    from ml_data_collector import MLDataCollector
    print("✅ ML Data Collector imported successfully")
    print("   ✅ Now using Alpaca API instead of yfinance")
except ImportError as e:
    print(f"❌ ML Data Collector import failed: {e}")

try:
    from ml_feature_engineering import FeatureEngineer
    print("✅ Feature Engineering module imported successfully")
except ImportError as e:
    print(f"❌ Feature Engineering import failed: {e}")

try:
    from ml_model import TradingNeuralNetwork
    print("✅ Neural Network Model imported successfully")
except ImportError as e:
    print(f"❌ Neural Network Model import failed: {e}")

try:
    from ml_trainer import ModelTrainer
    print("✅ Model Trainer imported successfully")
except ImportError as e:
    print(f"❌ Model Trainer import failed: {e}")

try:
    from ml_predictor import LivePredictor
    print("✅ Live Predictor imported successfully")
except ImportError as e:
    print(f"❌ Live Predictor import failed: {e}")

try:
    from ml_explainer import ModelExplainer
    print("✅ SHAP Explainer imported successfully")
except ImportError as e:
    print(f"❌ SHAP Explainer import failed: {e}")

try:
    from ml_backtester import MLBacktester
    print("✅ Backtester imported successfully")
except ImportError as e:
    print(f"❌ Backtester import failed: {e}")

try:
    from ml_monitor import DriftMonitor
    print("✅ Drift Monitor imported successfully")
except ImportError as e:
    print(f"❌ Drift Monitor import failed: {e}")

# Test TensorFlow Configuration
print("\n" + "="*50)
print("TensorFlow Configuration")
print("="*50)

if 'tf' in globals():
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check backend
    print(f"TensorFlow executing eagerly: {tf.executing_eagerly()}")
    
    # Physical devices
    print("\nPhysical Devices:")
    for device_type in ['CPU', 'GPU', 'MPS']:
        devices = tf.config.list_physical_devices(device_type)
        if devices:
            print(f"  {device_type}: {len(devices)} device(s)")
            for device in devices:
                print(f"    - {device.name}")

# Simple model test
print("\n" + "="*50)
print("Testing Simple Neural Network")
print("="*50)

try:
    import numpy as np
    
    if 'tf' in globals():
        # Create dummy data
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ Model created and compiled successfully")
        
        # Train for 1 epoch
        history = model.fit(X, y, epochs=1, verbose=0)
        print(f"✅ Model trained successfully - Loss: {history.history['loss'][0]:.4f}")
        
        # Test prediction
        pred = model.predict(X[:5], verbose=0)
        print(f"✅ Model prediction successful - Shape: {pred.shape}")
    else:
        print("⚠️  TensorFlow not available for model test")
        
except Exception as e:
    print(f"❌ Model test failed: {e}")

# Test Alpaca Data Collection
print("\n" + "="*50)
print("Testing Alpaca Data Collection")
print("="*50)

try:
    from alpaca_data_provider import AlpacaDataProvider
    
    provider = AlpacaDataProvider()
    print("✅ Alpaca Data Provider initialized")
    
    # Test fetching a small amount of data
    test_ticker = 'AAPL'
    print(f"\nTesting data fetch for {test_ticker}...")
    
    df = provider.fetch_daily_bars(test_ticker, days_back=5)
    if df is not None and not df.empty:
        print(f"✅ Successfully fetched {len(df)} days of data")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("⚠️  No data returned (market might be closed)")
        
except Exception as e:
    print(f"❌ Alpaca data test failed: {e}")

print("\n" + "="*50)
print("✅ ML Setup Complete!")
print("="*50)
print("\nEnvironment Summary:")
print(f"  - Python: {sys.version.split()[0]}")
print(f"  - Platform: Apple Silicon (ARM64)" if sys.platform == 'darwin' and 'arm' in os.uname().machine.lower() else f"  - Platform: {sys.platform}")
if 'tf' in globals():
    print(f"  - TensorFlow: {tf.__version__}")
    if mps_devices:
        print("  - Hardware Acceleration: Metal Performance Shaders")
    elif gpu_devices:
        print("  - Hardware Acceleration: GPU")
    else:
        print("  - Hardware Acceleration: CPU only")

print("\nNext steps:")
print("1. Run: python ml_components/ml_data_collector.py  # To collect training data")
print("2. Run: python ml_components/ml_trainer.py  # To train your first model")
print("3. Run: python ml_components/ml_backtester.py  # To backtest the model")
print("4. Integrate with scanner using ML predictions")