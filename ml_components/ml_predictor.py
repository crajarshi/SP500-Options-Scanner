"""
Real-time ML Prediction Engine for Live Trading
Integrates trained models with live market data
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import threading
import time
from collections import deque

from ml_feature_engineering import FeatureEngineer
from ml_model import TradingNeuralNetwork, EnsembleModel

logger = logging.getLogger(__name__)


class LivePredictor:
    """Handles real-time predictions for live trading"""
    
    def __init__(self, 
                 model_path: str = 'models/best_model.h5',
                 feature_engineer_path: str = 'models/feature_engineer.pkl',
                 cache_predictions: bool = True,
                 cache_ttl: int = 300):  # 5 minutes
        """
        Initialize live predictor
        
        Args:
            model_path: Path to trained model
            feature_engineer_path: Path to feature engineer
            cache_predictions: Whether to cache predictions
            cache_ttl: Cache time-to-live in seconds
        """
        self.model = None
        self.feature_engineer = None
        self.cache_predictions = cache_predictions
        self.cache_ttl = cache_ttl
        self.prediction_cache = {}
        self.cache_timestamps = {}
        
        # Load model and feature engineer
        self.load_model(model_path, feature_engineer_path)
        
        # Performance tracking
        self.prediction_history = deque(maxlen=1000)
        self.prediction_times = deque(maxlen=100)
        
    def load_model(self, model_path: str, feature_engineer_path: str):
        """
        Load trained model and feature engineer
        
        Args:
            model_path: Path to model file
            feature_engineer_path: Path to feature engineer file
        """
        try:
            # Load model
            if os.path.exists(model_path):
                self.model = TradingNeuralNetwork.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.error(f"Model file not found: {model_path}")
                
            # Load feature engineer
            if os.path.exists(feature_engineer_path):
                self.feature_engineer = FeatureEngineer.load(feature_engineer_path)
                logger.info(f"Feature engineer loaded from {feature_engineer_path}")
            else:
                logger.error(f"Feature engineer file not found: {feature_engineer_path}")
                
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
    
    def prepare_live_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Prepare features from live market data
        
        Args:
            data: Raw market data DataFrame
            ticker: Stock ticker
            
        Returns:
            DataFrame with engineered features
        """
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns for {ticker}")
            return pd.DataFrame()
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Calculate basic technical indicators
        data = self.calculate_live_indicators(data)
        
        # Apply feature engineering
        if self.feature_engineer:
            data = self.feature_engineer.engineer_features(data)
        
        return data
    
    def calculate_live_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for live data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        # Price returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Volume features
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-10)
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(window=50).mean() + 1e-10)
        
        # Price changes
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility_20d'] = df['returns'].rolling(window=20).std()
        df['volatility_50d'] = df['returns'].rolling(window=50).std()
        
        # Candlestick features
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
        df['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['close'] + 1e-10)
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['close'] + 1e-10)
        df['body_ratio'] = np.abs(df['close'] - df['open']) / (df['close'] + 1e-10)
        
        # Support/Resistance
        df['support_distance'] = (df['close'] - df['low'].rolling(window=20).min()) / (df['close'] + 1e-10)
        df['resistance_distance'] = (df['high'].rolling(window=20).max() - df['close']) / (df['close'] + 1e-10)
        
        # Trend strength
        df['trend_strength'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        
        # Market microstructure
        df['spread'] = df['high'] - df['low']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df
    
    def predict(self, data: pd.DataFrame, ticker: str) -> Dict:
        """
        Make prediction for a single stock
        
        Args:
            data: Market data for the stock
            ticker: Stock ticker
            
        Returns:
            Prediction dictionary with probability and confidence
        """
        start_time = time.time()
        
        # Check cache
        if self.cache_predictions and ticker in self.prediction_cache:
            cache_age = time.time() - self.cache_timestamps.get(ticker, 0)
            if cache_age < self.cache_ttl:
                logger.debug(f"Using cached prediction for {ticker}")
                return self.prediction_cache[ticker]
        
        try:
            # Prepare features
            features_df = self.prepare_live_features(data, ticker)
            
            if features_df.empty or len(features_df) < 50:
                logger.warning(f"Insufficient data for {ticker}")
                return self._empty_prediction()
            
            # Get latest row with all features
            latest_features = features_df.iloc[-1:].copy()
            
            # Transform features
            X = self.feature_engineer.transform(latest_features)
            
            # Make prediction
            if self.model:
                proba = self.model.predict_proba(X)
                prediction = self.model.predict(X)[0]
                
                # Get attention weights if available
                attention_weights = None
                try:
                    attention_weights = self.model.get_attention_weights(X)[0]
                except:
                    pass
                
                result = {
                    'ticker': ticker,
                    'timestamp': datetime.now(),
                    'prediction': int(prediction),
                    'probability': float(proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]),
                    'confidence': float(max(proba[0])),
                    'signal': self._get_signal_strength(proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]),
                    'attention_weights': attention_weights,
                    'features': {
                        'rsi': float(latest_features['rsi'].iloc[0]) if 'rsi' in latest_features else None,
                        'macd': float(latest_features['macd'].iloc[0]) if 'macd' in latest_features else None,
                        'bb_position': float(latest_features['bb_position'].iloc[0]) if 'bb_position' in latest_features else None,
                        'volume_ratio': float(latest_features['volume_ratio'].iloc[0]) if 'volume_ratio' in latest_features else None,
                        'trend_strength': float(latest_features['trend_strength'].iloc[0]) if 'trend_strength' in latest_features else None
                    }
                }
                
                # Cache prediction
                if self.cache_predictions:
                    self.prediction_cache[ticker] = result
                    self.cache_timestamps[ticker] = time.time()
                
                # Track performance
                prediction_time = time.time() - start_time
                self.prediction_times.append(prediction_time)
                self.prediction_history.append(result)
                
                logger.debug(f"Prediction for {ticker} completed in {prediction_time:.3f}s")
                
                return result
            else:
                return self._empty_prediction()
                
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {e}")
            return self._empty_prediction()
    
    def predict_batch(self, stocks_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Make predictions for multiple stocks
        
        Args:
            stocks_data: Dictionary of ticker -> DataFrame
            
        Returns:
            Dictionary of ticker -> prediction
        """
        predictions = {}
        
        for ticker, data in stocks_data.items():
            predictions[ticker] = self.predict(data, ticker)
        
        return predictions
    
    def _get_signal_strength(self, probability: float) -> str:
        """
        Convert probability to signal strength
        
        Args:
            probability: Prediction probability
            
        Returns:
            Signal strength category
        """
        if probability >= 0.85:
            return 'STRONG_BUY'
        elif probability >= 0.70:
            return 'BUY'
        elif probability >= 0.55:
            return 'WEAK_BUY'
        elif probability >= 0.45:
            return 'NEUTRAL'
        elif probability >= 0.30:
            return 'WEAK_SELL'
        elif probability >= 0.15:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _empty_prediction(self) -> Dict:
        """Return empty prediction structure"""
        return {
            'ticker': '',
            'timestamp': datetime.now(),
            'prediction': 0,
            'probability': 0.5,
            'confidence': 0.5,
            'signal': 'NEUTRAL',
            'attention_weights': None,
            'features': {}
        }
    
    def get_top_predictions(self, predictions: Dict[str, Dict], 
                           top_n: int = 10,
                           min_probability: float = 0.6) -> List[Dict]:
        """
        Get top N predictions by probability
        
        Args:
            predictions: Dictionary of predictions
            top_n: Number of top predictions to return
            min_probability: Minimum probability threshold
            
        Returns:
            List of top predictions
        """
        # Filter by minimum probability
        filtered = [p for p in predictions.values() 
                   if p['probability'] >= min_probability]
        
        # Sort by probability
        sorted_predictions = sorted(filtered, 
                                  key=lambda x: x['probability'], 
                                  reverse=True)
        
        return sorted_predictions[:top_n]
    
    def get_performance_stats(self) -> Dict:
        """
        Get predictor performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.prediction_times:
            return {}
        
        return {
            'total_predictions': len(self.prediction_history),
            'avg_prediction_time': np.mean(self.prediction_times),
            'min_prediction_time': np.min(self.prediction_times),
            'max_prediction_time': np.max(self.prediction_times),
            'cache_size': len(self.prediction_cache),
            'model_loaded': self.model is not None,
            'feature_engineer_loaded': self.feature_engineer is not None
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Prediction cache cleared")
    
    def update_model(self, new_model_path: str):
        """
        Update model with a new version
        
        Args:
            new_model_path: Path to new model
        """
        try:
            self.model = TradingNeuralNetwork.load(new_model_path)
            self.clear_cache()
            logger.info(f"Model updated from {new_model_path}")
        except Exception as e:
            logger.error(f"Error updating model: {e}")


class PredictionAggregator:
    """Aggregates predictions from multiple models or timeframes"""
    
    def __init__(self, predictors: List[LivePredictor] = None):
        """
        Initialize aggregator
        
        Args:
            predictors: List of predictor instances
        """
        self.predictors = predictors or []
        
    def add_predictor(self, predictor: LivePredictor):
        """Add a predictor to the aggregator"""
        self.predictors.append(predictor)
        
    def aggregate_predictions(self, data: pd.DataFrame, ticker: str,
                            weights: List[float] = None) -> Dict:
        """
        Aggregate predictions from multiple predictors
        
        Args:
            data: Market data
            ticker: Stock ticker
            weights: Weights for each predictor
            
        Returns:
            Aggregated prediction
        """
        if not self.predictors:
            return {}
        
        if weights is None:
            weights = [1.0 / len(self.predictors)] * len(self.predictors)
        
        predictions = []
        probabilities = []
        
        for predictor, weight in zip(self.predictors, weights):
            pred = predictor.predict(data, ticker)
            predictions.append(pred)
            probabilities.append(pred['probability'] * weight)
        
        # Aggregate
        avg_probability = sum(probabilities)
        avg_prediction = 1 if avg_probability > 0.5 else 0
        
        return {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'prediction': avg_prediction,
            'probability': avg_probability,
            'confidence': max([p['confidence'] for p in predictions]),
            'signal': self._get_signal_strength(avg_probability),
            'n_models': len(self.predictors),
            'individual_predictions': predictions
        }
    
    def _get_signal_strength(self, probability: float) -> str:
        """Convert probability to signal strength"""
        if probability >= 0.85:
            return 'STRONG_BUY'
        elif probability >= 0.70:
            return 'BUY'
        elif probability >= 0.55:
            return 'WEAK_BUY'
        elif probability >= 0.45:
            return 'NEUTRAL'
        elif probability >= 0.30:
            return 'WEAK_SELL'
        elif probability >= 0.15:
            return 'SELL'
        else:
            return 'STRONG_SELL'