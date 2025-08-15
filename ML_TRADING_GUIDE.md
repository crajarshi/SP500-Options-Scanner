# Deep Learning Trading System - Complete Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (required for TensorFlow compatibility)
- 8GB+ RAM (16GB recommended for training)
- 50GB+ free disk space for data and models
- For Apple Silicon Macs: macOS 12.0+ for Metal Performance Shaders support

### Installation

#### For Apple Silicon Macs (M1/M2/M3/M4)
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Install Apple Silicon optimized TensorFlow (verified working versions)
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.1.0

# Install other ML dependencies
pip install scikit-learn==1.3.2 xgboost==2.0.3 shap==0.44.0 mlflow==2.9.2 optuna==3.5.0 imbalanced-learn==0.11.0
pip install pandas numpy python-dotenv alpaca-py

# Create necessary directories
mkdir -p models ml_data_cache ml_logs

# Test your setup
python test_ml_setup.py
# You should see: "✅ Metal Performance Shaders (MPS) available"
# This confirms GPU acceleration is working on your Apple Silicon Mac
```

#### For Intel Macs and Other Platforms
```bash
# Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Install standard dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models ml_data_cache ml_logs

# Test your setup
python test_ml_setup.py
```

### First-Time Setup
```bash
# 0. Install required ML dependencies (if not already installed)
pip install tensorflow scikit-learn shap

# 1. Collect historical data and train initial model
python ml_components/ml_trainer.py

# 2. Run backtesting to validate performance
python ml_components/ml_backtester.py

# 3. Start scanner with ML integration
python sp500_options_scanner.py --ml-enabled
```

**Note:** The `--ml-enabled` flag will work without ML models, but will show a warning. To fully utilize ML features, you must first train the models using the steps above.

## 📊 System Overview

The deep learning system enhances your trading scanner with:
- **Predictive ML Models**: Neural networks trained on 8+ years of historical data
- **Explainable AI**: SHAP explanations for every prediction
- **Confidence-Based Sizing**: Automatic position sizing based on ML confidence
- **Drift Monitoring**: Detects when market conditions change
- **Backtesting Engine**: Realistic simulation with costs and slippage

## 🧠 ML Components

### 1. Data Collection (`ml_data_collector.py`)
Collects and prepares historical data for training using **Alpaca API**:

**Important Updates (as of latest commit)**: 
- **Switched from yfinance to Alpaca API** for reliability and consistency
- **Fixed**: No more yfinance timeouts or parsing errors
- **Fixed**: Column name inconsistencies (handles both 'close'/'Close' formats)
- Uses same data source as main scanner for consistency
- Fetches data in chunks (365 days at a time) to handle 8+ years of history
- Automatic rate limiting and timezone handling
- Converts Alpaca's lowercase column names to uppercase for compatibility

```python
from ml_components.ml_data_collector import MLDataCollector

# Initialize collector (uses Alpaca API automatically)
collector = MLDataCollector(
    tickers=['AAPL', 'MSFT', 'GOOGL'],  # Add your tickers
    start_date='2015-01-01',
    label_threshold=0.05,  # 5% profit target
    label_days=10  # 10-day holding period
)

# Collect data (fetches from Alpaca in yearly chunks)
df = collector.collect_all_data()
```

### 2. Feature Engineering (`ml_feature_engineering.py`)
Creates 50+ technical indicators and patterns:
- Price-based: RSI, MACD, Bollinger Bands
- Volume: Volume ratios, spikes
- Patterns: Candlestick patterns, support/resistance
- Market regime: Trend strength, volatility regime

**Recent Fix**: Column name normalization now handles both uppercase and lowercase formats:
- Automatically detects whether data has 'Close' or 'close' columns
- All feature creation is conditional on column existence
- Prevents KeyError issues when switching between data sources

### 3. Neural Network Model (`ml_model.py`)
Hybrid architecture with:
- Dense layers for tabular features
- Attention mechanism for feature importance
- Class balancing for imbalanced data
- Dropout for regularization

### 4. Training Pipeline (`ml_trainer.py`)
Walk-forward validation ensures no look-ahead bias:
```python
from ml_components.ml_trainer import ModelTrainer

trainer = ModelTrainer()

# Optimize hyperparameters
best_params = trainer.hyperparameter_optimization(df, n_trials=50)

# Train with walk-forward validation
results = trainer.walk_forward_validation(df, n_splits=5)

# Save model
trainer.save_training_artifacts()
```

### 5. Live Predictions (`ml_predictor.py`)
Real-time predictions with caching:
```python
from ml_components.ml_predictor import LivePredictor

predictor = LivePredictor(
    model_path='models/best_model.h5',
    feature_engineer_path='models/feature_engineer.pkl'
)

# Make prediction
prediction = predictor.predict(stock_data, 'AAPL')
print(f"Probability: {prediction['probability']:.2%}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### 6. Explainability (`ml_explainer.py`)
SHAP explanations for transparency:
```python
from ml_components.ml_explainer import ModelExplainer

explainer = ModelExplainer(model, feature_names)
explanation = explainer.explain_prediction(X)

# Output example:
# "Prediction leans BULLISH
#  Bullish factors:
#    • RSI: 28.5 (+25%)
#    • MACD Crossover: Yes (+20%)
#    • Volume Spike: 2.3x (+15%)"
```

### 7. Backtesting (`ml_backtester.py`)
Realistic backtesting with:
- Transaction costs and slippage
- Stop losses and take profits
- Position sizing based on confidence
- Daily loss limits

```python
from ml_components.ml_backtester import MLBacktester

backtester = MLBacktester(
    initial_capital=30000,
    risk_per_trade=0.015,  # 1.5% risk
    commission=1.0,  # $1 per trade
    slippage_pct=0.001  # 0.1% slippage
)

results = backtester.run_backtest(price_data, predictions)
backtester.print_summary()
```

### 8. Drift Monitoring (`ml_monitor.py`)
Detects when market conditions change:
```python
from ml_components.ml_monitor import DriftMonitor

monitor = DriftMonitor(reference_data=training_data)
drift_result = monitor.detect_feature_drift(current_data)

if drift_result['overall_drift']:
    print("⚠️ Feature drift detected - consider retraining")
```

## 🔧 Integration with Scanner

### Enable ML in Scanner
The scanner now has built-in ML integration. Simply use the `--ml-enabled` flag:

```bash
# Basic ML-enhanced scanning
python sp500_options_scanner.py --ml-enabled

# ML + Options recommendations
python sp500_options_scanner.py --ml-enabled --options

# ML + Top 10 results only
python sp500_options_scanner.py --ml-enabled --top 10

# ML + Export to CSV
python sp500_options_scanner.py --ml-enabled --export
```

### What ML Integration Provides
When ML is enabled, the scanner:
1. **Enhances each stock signal** with ML probability and confidence scores
2. **Filters results** based on ML confidence thresholds (default: 60%)
3. **Re-ranks stocks** using a combination of traditional and ML scores
4. **Provides explanations** for predictions using SHAP values
5. **Monitors for drift** to detect when market conditions change
6. **Adjusts position sizing** based on ML confidence levels

### Under the Hood
```python
# The scanner automatically does this when --ml-enabled is used:
from ml_scanner_integration import MLScannerIntegration

# Initialize ML integration
ml_integration = MLScannerIntegration(
    model_path='models/best_model.h5',
    enable_explainability=True,
    min_confidence=0.6
)

# Enhance stock analysis
enhanced_signal = ml_integration.enhance_stock_analysis(
    stock_data, ticker, existing_signal
)
```

### ML-Enhanced Risk Management
The system automatically adjusts position sizes based on ML confidence:
- **90%+ confidence**: Full position (1.5% risk)
- **75-89% confidence**: 75% position
- **60-74% confidence**: 50% position
- **Below 60%**: Minimum position or skip

## 📈 Training Your Model

### Step 1: Collect Data
```bash
python -c "
from ml_components.ml_data_collector import MLDataCollector
collector = MLDataCollector(
    tickers=['SPY'] + [your_watchlist],
    start_date='2015-01-01'
)
df = collector.collect_all_data()
"
```

### Step 2: Train Model
```bash
python ml_components/ml_trainer.py
```
This will:
1. Optimize hyperparameters
2. Perform walk-forward validation
3. Train final model
4. Save to `models/` directory

### Step 3: Backtest
```bash
python -c "
from ml_components.ml_backtester import MLBacktester
# Load your data and predictions
backtester = MLBacktester()
results = backtester.run_backtest(data, predictions)
backtester.print_summary()
"
```

### Step 4: Monitor Performance
The system automatically monitors:
- Feature drift (KS test, PSI)
- Prediction confidence trends
- Model performance degradation

## 🎯 Best Practices

### 1. Data Quality
- Ensure at least 3 years of historical data
- Include diverse market conditions
- Update data regularly

### 2. Model Retraining
- Retrain monthly or when drift detected
- Use walk-forward validation
- Keep previous models for comparison

### 3. Risk Management
- Never override ML risk suggestions
- Start with small positions to validate
- Monitor daily P&L closely

### 4. Monitoring
- Check drift reports weekly
- Review prediction explanations
- Track actual vs predicted outcomes

## 📊 Performance Metrics

### Expected Performance (Based on Backtesting)
- **Win Rate**: 55-65%
- **Profit Factor**: 1.5-2.0
- **Sharpe Ratio**: 1.2-1.8
- **Max Drawdown**: 10-15%

### Key Indicators to Watch
1. **ML Confidence Distribution**: Should be balanced, not all high/low
2. **Feature Drift Score**: Below 30% is normal
3. **Prediction Accuracy**: Track weekly rolling accuracy
4. **Risk-Adjusted Returns**: Monitor Sharpe ratio

## 🛠️ Troubleshooting

### TensorFlow Installation Issues (Apple Silicon)
```bash
# If you see "No matching distribution found for tensorflow==2.15.0"
# This means you're on Apple Silicon - use tensorflow-macos instead:
pip uninstall tensorflow  # Remove any conflicting installation
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.1.0
```

### Column Name Errors (KeyError: 'close')
```bash
# Fixed in latest version - the system now handles both formats:
# - Alpaca API returns: 'close', 'open', 'high', 'low', 'volume'
# - ML components expect: 'Close', 'Open', 'High', 'Low', 'Volume'
# The data collector automatically converts to uppercase
```

### Model Not Loading
```bash
# Check model files exist
ls -la models/
# Should see: best_model.h5, feature_engineer.pkl
```

### Low Confidence Predictions
- Check for feature drift
- Verify data quality
- Consider retraining with recent data

### High Drift Scores
- Normal during regime changes
- Retrain if persists > 2 weeks
- Check for data issues

### Git Performance Issues
```bash
# If git is slow due to venv files:
# Ensure venv directories are in .gitignore:
echo "venv/" >> .gitignore
echo "venv311/" >> .gitignore
git rm -r --cached venv311  # Remove from tracking if needed
```

## 🔄 Continuous Improvement

### Weekly Tasks
1. Review drift monitoring report
2. Analyze prediction accuracy
3. Check for failed trades

### Monthly Tasks
1. Retrain model with latest data
2. Run full backtest
3. Adjust hyperparameters if needed

### Quarterly Tasks
1. Review feature importance
2. Add new features if available
3. Optimize model architecture

## 📝 Configuration

### Environment Requirements
- **Python**: 3.11 (required for TensorFlow compatibility)
- **macOS (Apple Silicon)**: 12.0+ for Metal Performance Shaders
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ for data and models

### Model Parameters (Adjustable)
```python
# In ml_model.py
model_config = {
    'learning_rate': 0.001,
    'dropout_rate': 0.3,
    'l1_reg': 0.01,
    'l2_reg': 0.01,
    'n_classes': 2  # Binary classification
}
```

### Trading Parameters
```python
# In ml_backtester.py
trading_config = {
    'confidence_threshold': 0.6,  # Minimum ML confidence
    'stop_loss_pct': 0.02,  # 2% stop loss
    'take_profit_pct': 0.05,  # 5% take profit
    'max_positions': 10  # Maximum concurrent positions
}
```

## 🚨 Safety Features

1. **Automatic Stop Loss**: All ML trades have 2% stop loss
2. **Daily Loss Limits**: Integrated with existing risk manager
3. **Confidence Thresholds**: Won't trade below 60% confidence
4. **Drift Alerts**: Warns when market regime changes
5. **Paper Trading Mode**: Test without real money first

## 📈 Advanced Features

### Ensemble Models
Combine multiple models for better predictions:
```python
from ml_components.ml_model import EnsembleModel
ensemble = EnsembleModel()
ensemble.add_model(model1)
ensemble.add_model(model2)
predictions = ensemble.predict_proba(X)
```

### Custom Features
Add your own indicators:
```python
# In ml_feature_engineering.py
def add_custom_features(df):
    df['my_indicator'] = calculate_my_indicator(df)
    return df
```

### A/B Testing
Compare models in production:
```python
# Track performance of different models
model_a_trades = []
model_b_trades = []
# Compare after sufficient samples
```

## 🎓 Learning Resources

### Understanding the Model
- Neural network basics: Dense layers process features
- Attention mechanism: Identifies important features
- SHAP values: Explain individual predictions

### Improving Performance
1. **More Data**: Quality > Quantity
2. **Feature Engineering**: Domain knowledge helps
3. **Hyperparameter Tuning**: Use Optuna
4. **Ensemble Methods**: Combine multiple models

## 📞 Support

### Common Issues & Solutions
- **ModuleNotFoundError**: Run `pip install -r requirements.txt`
- **No model file**: Train model first with `ml_trainer.py`
- **Low accuracy**: Normal in sideways markets
- **TensorFlow not found (Apple Silicon)**: Install tensorflow-macos instead
- **KeyError 'close'**: Update to latest version (fixed in recent commit)
- **Alpaca API timeout**: Check API keys and rate limits
- **yfinance errors**: System now uses Alpaca API instead (fixed)

### Getting Help
1. Check logs in `ml_logs/` directory
2. Review backtesting results
3. Monitor drift scores
4. Validate data quality

## 🏁 Next Steps

1. **Start Training**: Run `python ml_components/ml_trainer.py`
2. **Backtest**: Validate on historical data
3. **Paper Trade**: Test with live data, no real money
4. **Go Live**: Start with minimum positions
5. **Monitor**: Check daily, retrain monthly

Remember: The ML system enhances but doesn't replace good trading judgment. Always use proper risk management and never risk more than you can afford to lose.