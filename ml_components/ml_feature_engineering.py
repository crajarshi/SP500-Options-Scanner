"""
Feature Engineering Pipeline for ML Trading Model
Handles feature creation, scaling, and preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import pickle
import os
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles all feature engineering and preprocessing for ML models"""
    
    def __init__(self, scaler_type: str = 'standard', 
                 use_pca: bool = False, pca_components: int = 50):
        """
        Initialize feature engineering pipeline
        
        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components to keep
        """
        self.scaler_type = scaler_type
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # Initialize scalers
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        # Feature groups for organized processing
        self.feature_groups = {
            'price': ['close', 'open', 'high', 'low', 'typical_price'],
            'volume': ['volume', 'volume_ratio', 'volume_ma_ratio'],
            'technical': ['rsi', 'macd', 'macd_signal', 'macd_histogram'],
            'bollinger': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'],
            'moving_averages': ['sma_20', 'sma_50', 'ema_12', 'ema_26'],
            'volatility': ['volatility_20d', 'volatility_50d', 'atr'],
            'patterns': ['upper_shadow', 'lower_shadow', 'body_ratio', 'high_low_ratio'],
            'momentum': ['price_change_1d', 'price_change_5d', 'price_change_20d'],
            'market_structure': ['support_distance', 'resistance_distance', 'trend_strength']
        }
        
        self.fitted = False
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key indicators
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with interaction features added
        """
        # Normalize column names to handle both uppercase and lowercase
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
        
        # RSI extremes
        if 'rsi' in df.columns:
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # Bollinger Band signals
        if 'bb_width' in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(20).mean()
            df['price_bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                                  (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
            df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & 
                                  (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Volume anomalies
        if volume_col in df.columns:
            df['volume_spike'] = (df[volume_col] > df[volume_col].rolling(20).mean() * 2).astype(int)
        
        # Trend combinations (only if columns exist)
        if 'trend_strength' in df.columns and 'price_change_5d' in df.columns:
            df['trend_momentum'] = df['trend_strength'] * df['price_change_5d']
        
        # Support/Resistance proximity (only if columns exist)
        if 'support_distance' in df.columns:
            df['near_support'] = (df['support_distance'] < 0.02).astype(int)
        if 'resistance_distance' in df.columns:
            df['near_resistance'] = (df['resistance_distance'] < 0.02).astype(int)
        
        # Volatility-adjusted returns (only if columns exist)
        if 'price_change_5d' in df.columns and 'volatility_20d' in df.columns:
            df['sharpe_5d'] = df['price_change_5d'] / (df['volatility_20d'] + 0.0001)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           features: List[str], 
                           lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged features for time series patterns
        
        Args:
            df: DataFrame with features
            features: List of features to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        for feature in features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                              features: List[str],
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling statistics features
        
        Args:
            df: DataFrame with features
            features: List of features to calculate rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        for feature in features:
            if feature in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{feature}_ma_{window}'] = df[feature].rolling(window).mean()
                    # Rolling std
                    df[f'{feature}_std_{window}'] = df[feature].rolling(window).std()
                    # Rolling min/max
                    df[f'{feature}_min_{window}'] = df[feature].rolling(window).min()
                    df[f'{feature}_max_{window}'] = df[feature].rolling(window).max()
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline
        
        Args:
            df: Raw DataFrame with basic features
            
        Returns:
            DataFrame with all engineered features
        """
        # Normalize column names to handle both uppercase and lowercase
        close_col = 'Close' if 'Close' in df.columns else 'close'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create lag features for key indicators
        lag_features = ['rsi', 'macd', 'volume_ratio', 'volatility_20d']
        # Filter to only existing columns
        lag_features = [f for f in lag_features if f in df.columns]
        if lag_features:
            df = self.create_lag_features(df, lag_features)
        
        # Create rolling features for price and volume
        rolling_features = []
        if close_col in df.columns:
            rolling_features.append(close_col)
        if volume_col in df.columns:
            rolling_features.append(volume_col)
        if 'volatility_20d' in df.columns:
            rolling_features.append('volatility_20d')
        
        if rolling_features:
            df = self.create_rolling_features(df, rolling_features, windows=[5, 10, 20])
        
        # Technical pattern recognition
        df = self.add_candlestick_patterns(df)
        
        # Market regime features
        df = self.add_market_regime_features(df)
        
        return df
    
    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick pattern recognition features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern features
        """
        # Normalize column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Only add patterns if we have the required columns
        if close_col in df.columns and open_col in df.columns:
            # Doji pattern
            df['doji'] = (np.abs(df[close_col] - df[open_col]) / df[close_col] < 0.001).astype(int)
            
            # Engulfing patterns
            df['bullish_engulfing'] = ((df[close_col] > df[open_col]) & 
                                       (df[close_col].shift(1) < df[open_col].shift(1)) &
                                       (df[open_col] < df[close_col].shift(1)) &
                                       (df[close_col] > df[open_col].shift(1))).astype(int)
            
            df['bearish_engulfing'] = ((df[close_col] < df[open_col]) & 
                                       (df[close_col].shift(1) > df[open_col].shift(1)) &
                                       (df[open_col] > df[close_col].shift(1)) &
                                       (df[close_col] < df[open_col].shift(1))).astype(int)
        
        # Hammer and shooting star patterns (only if shadow columns exist)
        if 'lower_shadow' in df.columns and 'upper_shadow' in df.columns and 'body_ratio' in df.columns:
            # Hammer pattern
            df['hammer'] = ((df['lower_shadow'] > df['body_ratio'] * 2) & 
                           (df['upper_shadow'] < df['body_ratio'] * 0.5)).astype(int)
            
            # Shooting star
            df['shooting_star'] = ((df['upper_shadow'] > df['body_ratio'] * 2) & 
                                   (df['lower_shadow'] < df['body_ratio'] * 0.5)).astype(int)
        
        return df
    
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime and context features
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with regime features
        """
        # Normalize column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        
        # Trend classification (only if moving averages exist)
        if 'sma_20' in df.columns and 'sma_50' in df.columns and close_col in df.columns:
            df['uptrend'] = ((df['sma_20'] > df['sma_50']) & 
                            (df[close_col] > df['sma_20'])).astype(int)
            df['downtrend'] = ((df['sma_20'] < df['sma_50']) & 
                              (df[close_col] < df['sma_20'])).astype(int)
            df['sideways'] = (~df['uptrend'] & ~df['downtrend']).astype(int)
        
        # Volatility regime (only if volatility column exists)
        if 'volatility_20d' in df.columns:
            vol_median = df['volatility_20d'].rolling(252).median()
            df['high_vol_regime'] = (df['volatility_20d'] > vol_median * 1.5).astype(int)
            df['low_vol_regime'] = (df['volatility_20d'] < vol_median * 0.5).astype(int)
        
        # Momentum regime (only if column exists)
        if 'price_change_20d' in df.columns:
            df['strong_momentum'] = (np.abs(df['price_change_20d']) > 0.1).astype(int)
        
        return df
    
    def fit(self, X: pd.DataFrame, feature_cols: List[str] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline
        
        Args:
            X: Training data
            feature_cols: List of feature columns to use
            
        Returns:
            Fitted FeatureEngineer object
        """
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if 'label' in feature_cols:
                feature_cols.remove('label')
        
        self.feature_cols = feature_cols
        
        # Fit scaler
        X_clean = X[feature_cols].fillna(0)
        self.scaler.fit(X_clean)
        
        # Fit PCA if enabled
        if self.use_pca:
            X_scaled = self.scaler.transform(X_clean)
            self.pca.fit(X_scaled)
        
        self.fitted = True
        logger.info(f"Feature engineering pipeline fitted with {len(feature_cols)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed feature array
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Select and clean features
        X_clean = X[self.feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_clean)
        
        # Apply PCA if enabled
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X: Training data
            feature_cols: List of feature columns
            
        Returns:
            Transformed feature array
        """
        self.fit(X, feature_cols)
        return self.transform(X)
    
    def save(self, filepath: str):
        """
        Save fitted pipeline to disk
        
        Args:
            filepath: Path to save pipeline
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_cols': self.feature_cols,
            'scaler_type': self.scaler_type,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Feature pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureEngineer':
        """
        Load fitted pipeline from disk
        
        Args:
            filepath: Path to load pipeline from
            
        Returns:
            Loaded FeatureEngineer object
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create new instance with saved parameters
        fe = cls(
            scaler_type=pipeline_data['scaler_type'],
            use_pca=pipeline_data['use_pca'],
            pca_components=pipeline_data['pca_components']
        )
        
        # Restore fitted components
        fe.scaler = pipeline_data['scaler']
        fe.pca = pipeline_data['pca']
        fe.feature_cols = pipeline_data['feature_cols']
        fe.fitted = True
        
        logger.info(f"Feature pipeline loaded from {filepath}")
        return fe
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names after transformation
        
        Returns:
            List of feature names
        """
        if not self.fitted:
            return []
        
        if self.use_pca:
            return [f'pca_{i}' for i in range(self.pca_components)]
        else:
            return self.feature_cols
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            logger.warning("Model doesn't have feature importance attributes")
            return pd.DataFrame()
        
        feature_names = self.get_feature_names()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


def create_feature_pipeline(config: Dict = None) -> FeatureEngineer:
    """
    Factory function to create configured feature pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FeatureEngineer instance
    """
    if config is None:
        config = {
            'scaler_type': 'standard',
            'use_pca': False,
            'pca_components': 50
        }
    
    return FeatureEngineer(**config)