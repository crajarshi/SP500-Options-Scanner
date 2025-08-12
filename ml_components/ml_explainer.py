"""
SHAP-based Model Explainability for Trading Predictions
Provides human-readable explanations for ML predictions
"""
import numpy as np
import pandas as pd
import shap
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Provides interpretable explanations for model predictions using SHAP"""
    
    def __init__(self, model, feature_names: List[str] = None, 
                 background_samples: int = 100):
        """
        Initialize model explainer
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            background_samples: Number of background samples for SHAP
        """
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples
        self.explainer = None
        self.background_data = None
        
    def initialize_explainer(self, X_train: np.ndarray):
        """
        Initialize SHAP explainer with background data
        
        Args:
            X_train: Training data for background samples
        """
        # Sample background data
        if len(X_train) > self.background_samples:
            indices = np.random.choice(len(X_train), self.background_samples, replace=False)
            self.background_data = X_train[indices]
        else:
            self.background_data = X_train
        
        # Create appropriate explainer based on model type
        try:
            # For neural networks, use DeepExplainer or GradientExplainer
            if hasattr(self.model, 'model'):  # TradingNeuralNetwork
                self.explainer = shap.GradientExplainer(self.model.model, self.background_data)
            else:
                # For tree-based models
                self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fallback to KernelExplainer (works with any model but slower)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                self.background_data
            )
        
        logger.info(f"SHAP explainer initialized with {len(self.background_data)} background samples")
    
    def explain_prediction(self, X: np.ndarray, top_n: int = 5) -> Dict:
        """
        Explain a single prediction
        
        Args:
            X: Feature vector for prediction (1D or 2D array)
            top_n: Number of top features to include in explanation
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            logger.error("Explainer not initialized. Call initialize_explainer first.")
            return {}
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if shap_values.ndim > 2:
                shap_values = shap_values[0]
            
            # Get feature contributions
            feature_contributions = self._get_feature_contributions(shap_values[0], X[0])
            
            # Get top contributors
            top_positive, top_negative = self._get_top_contributors(feature_contributions, top_n)
            
            # Generate human-readable explanation
            explanation_text = self._generate_explanation_text(
                feature_contributions, top_positive, top_negative
            )
            
            return {
                'shap_values': shap_values[0].tolist(),
                'feature_contributions': feature_contributions,
                'top_positive_factors': top_positive,
                'top_negative_factors': top_negative,
                'explanation': explanation_text,
                'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {}
    
    def _get_feature_contributions(self, shap_values: np.ndarray, 
                                  feature_values: np.ndarray) -> List[Dict]:
        """
        Get feature contributions with values and SHAP scores
        
        Args:
            shap_values: SHAP values for features
            feature_values: Actual feature values
            
        Returns:
            List of feature contribution dictionaries
        """
        contributions = []
        
        for i, (shap_val, feat_val) in enumerate(zip(shap_values, feature_values)):
            feature_name = self.feature_names[i] if self.feature_names and i < len(self.feature_names) else f'feature_{i}'
            
            contributions.append({
                'feature': feature_name,
                'value': float(feat_val),
                'shap_value': float(shap_val),
                'contribution_pct': 0  # Will be calculated later
            })
        
        # Calculate percentage contributions
        total_abs_shap = sum(abs(c['shap_value']) for c in contributions)
        if total_abs_shap > 0:
            for c in contributions:
                c['contribution_pct'] = (abs(c['shap_value']) / total_abs_shap) * 100
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return contributions
    
    def _get_top_contributors(self, contributions: List[Dict], 
                            top_n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Get top positive and negative contributors
        
        Args:
            contributions: List of feature contributions
            top_n: Number of top features to return
            
        Returns:
            Tuple of (top_positive, top_negative) contributors
        """
        positive = [c for c in contributions if c['shap_value'] > 0]
        negative = [c for c in contributions if c['shap_value'] < 0]
        
        top_positive = sorted(positive, key=lambda x: x['shap_value'], reverse=True)[:top_n]
        top_negative = sorted(negative, key=lambda x: x['shap_value'])[:top_n]
        
        return top_positive, top_negative
    
    def _generate_explanation_text(self, contributions: List[Dict],
                                  top_positive: List[Dict],
                                  top_negative: List[Dict]) -> str:
        """
        Generate human-readable explanation text
        
        Args:
            contributions: All feature contributions
            top_positive: Top positive contributors
            top_negative: Top negative contributors
            
        Returns:
            Explanation text
        """
        explanation_parts = []
        
        # Overall prediction strength
        total_positive = sum(c['shap_value'] for c in contributions if c['shap_value'] > 0)
        total_negative = sum(c['shap_value'] for c in contributions if c['shap_value'] < 0)
        
        if total_positive > abs(total_negative):
            explanation_parts.append("Prediction leans BULLISH")
        else:
            explanation_parts.append("Prediction leans BEARISH")
        
        # Top positive factors
        if top_positive:
            explanation_parts.append("\nBullish factors:")
            for factor in top_positive[:3]:
                explanation_parts.append(
                    f"  • {self._format_feature_name(factor['feature'])}: "
                    f"{factor['value']:.2f} (+{factor['contribution_pct']:.1f}%)"
                )
        
        # Top negative factors
        if top_negative:
            explanation_parts.append("\nBearish factors:")
            for factor in top_negative[:3]:
                explanation_parts.append(
                    f"  • {self._format_feature_name(factor['feature'])}: "
                    f"{factor['value']:.2f} (-{factor['contribution_pct']:.1f}%)"
                )
        
        return "\n".join(explanation_parts)
    
    def _format_feature_name(self, feature_name: str) -> str:
        """
        Format feature name for readability
        
        Args:
            feature_name: Raw feature name
            
        Returns:
            Formatted feature name
        """
        # Common feature name mappings
        name_map = {
            'rsi': 'RSI',
            'macd': 'MACD',
            'macd_signal': 'MACD Signal',
            'bb_position': 'Bollinger Band Position',
            'volume_ratio': 'Volume Ratio',
            'volatility_20d': '20-Day Volatility',
            'price_change_5d': '5-Day Price Change',
            'trend_strength': 'Trend Strength',
            'support_distance': 'Distance to Support',
            'resistance_distance': 'Distance to Resistance',
            'rsi_oversold': 'RSI Oversold Signal',
            'rsi_overbought': 'RSI Overbought Signal',
            'macd_bullish': 'MACD Bullish Cross',
            'volume_spike': 'Volume Spike'
        }
        
        return name_map.get(feature_name, feature_name.replace('_', ' ').title())
    
    def explain_batch(self, X: np.ndarray, top_n: int = 5) -> List[Dict]:
        """
        Explain multiple predictions
        
        Args:
            X: Feature matrix (2D array)
            top_n: Number of top features per explanation
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i in range(len(X)):
            explanation = self.explain_prediction(X[i:i+1], top_n)
            explanations.append(explanation)
        
        return explanations
    
    def plot_explanation(self, X: np.ndarray, save_path: str = None):
        """
        Create SHAP waterfall plot for prediction explanation
        
        Args:
            X: Feature vector to explain
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            logger.error("Explainer not initialized")
            return
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0] if isinstance(shap_values, list) else shap_values[0],
                    base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0.5,
                    feature_names=self.feature_names
                )
            )
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"Explanation plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating explanation plot: {e}")
    
    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP
        
        Args:
            X: Feature matrix for calculating importance
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.explainer is None:
            logger.error("Explainer not initialized")
            return pd.DataFrame()
        
        try:
            # Calculate SHAP values for all samples
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(mean_abs_shap))],
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            # Add percentage
            importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return pd.DataFrame()
    
    def save_explainer(self, filepath: str):
        """
        Save explainer to disk
        
        Args:
            filepath: Path to save explainer
        """
        explainer_data = {
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'background_data': self.background_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(explainer_data, f)
        
        logger.info(f"Explainer saved to {filepath}")
    
    @classmethod
    def load_explainer(cls, filepath: str, model) -> 'ModelExplainer':
        """
        Load explainer from disk
        
        Args:
            filepath: Path to load explainer from
            model: Model instance to attach to explainer
            
        Returns:
            Loaded ModelExplainer instance
        """
        with open(filepath, 'rb') as f:
            explainer_data = pickle.load(f)
        
        instance = cls(model, explainer_data['feature_names'])
        instance.explainer = explainer_data['explainer']
        instance.background_data = explainer_data['background_data']
        
        logger.info(f"Explainer loaded from {filepath}")
        return instance


def format_explanation_for_display(explanation: Dict) -> str:
    """
    Format SHAP explanation for display in scanner
    
    Args:
        explanation: Explanation dictionary from ModelExplainer
        
    Returns:
        Formatted string for display
    """
    if not explanation:
        return "No explanation available"
    
    lines = []
    
    # Add main explanation
    if 'explanation' in explanation:
        lines.append(explanation['explanation'])
    
    # Add confidence based on SHAP values
    if 'shap_values' in explanation:
        total_abs = sum(abs(v) for v in explanation['shap_values'])
        confidence = min(100, total_abs * 10)  # Scale to percentage
        lines.append(f"\nPrediction confidence: {confidence:.1f}%")
    
    return "\n".join(lines)