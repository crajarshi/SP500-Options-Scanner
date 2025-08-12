"""
ML Scanner Integration Module
Integrates deep learning predictions with the live options scanner
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

# Add ml_components to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_components'))

from ml_components.ml_predictor import LivePredictor
from ml_components.ml_explainer import ModelExplainer, format_explanation_for_display
from ml_components.ml_monitor import DriftMonitor

logger = logging.getLogger(__name__)


class MLScannerIntegration:
    """Integrates ML predictions with the live options scanner"""
    
    def __init__(self, 
                 model_path: str = 'models/best_model.h5',
                 feature_engineer_path: str = 'models/feature_engineer.pkl',
                 enable_explainability: bool = True,
                 enable_monitoring: bool = True,
                 min_confidence: float = 0.6):
        """
        Initialize ML integration
        
        Args:
            model_path: Path to trained model
            feature_engineer_path: Path to feature engineer
            enable_explainability: Whether to enable SHAP explanations
            enable_monitoring: Whether to enable drift monitoring
            min_confidence: Minimum confidence threshold for signals
        """
        self.min_confidence = min_confidence
        self.enable_explainability = enable_explainability
        self.enable_monitoring = enable_monitoring
        
        # Initialize predictor
        self.predictor = None
        self.explainer = None
        self.monitor = None
        
        # Load components
        self._initialize_components(model_path, feature_engineer_path)
        
        # Track predictions for monitoring
        self.prediction_history = []
        
    def _initialize_components(self, model_path: str, feature_engineer_path: str):
        """Initialize ML components"""
        try:
            # Initialize predictor
            self.predictor = LivePredictor(
                model_path=model_path,
                feature_engineer_path=feature_engineer_path,
                cache_predictions=True
            )
            logger.info("ML predictor initialized successfully")
            
            # Initialize explainer if enabled
            if self.enable_explainability and self.predictor.model:
                self.explainer = ModelExplainer(
                    model=self.predictor.model,
                    feature_names=self.predictor.feature_engineer.feature_cols if self.predictor.feature_engineer else None
                )
                logger.info("ML explainer initialized")
            
            # Initialize monitor if enabled
            if self.enable_monitoring:
                self.monitor = DriftMonitor(
                    feature_names=self.predictor.feature_engineer.feature_cols if self.predictor.feature_engineer else None
                )
                logger.info("Drift monitor initialized")
                
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
    
    def enhance_stock_analysis(self, stock_data: pd.DataFrame, 
                              ticker: str,
                              existing_signal: Dict) -> Dict:
        """
        Enhance existing stock analysis with ML predictions
        
        Args:
            stock_data: Historical data for the stock
            ticker: Stock ticker
            existing_signal: Existing signal from traditional analysis
            
        Returns:
            Enhanced signal with ML predictions
        """
        if not self.predictor:
            return existing_signal
        
        try:
            # Get ML prediction
            ml_prediction = self.predictor.predict(stock_data, ticker)
            
            if ml_prediction and ml_prediction['probability'] > 0:
                # Add ML components to signal
                existing_signal['ml_probability'] = ml_prediction['probability']
                existing_signal['ml_confidence'] = ml_prediction['confidence']
                existing_signal['ml_signal'] = ml_prediction['signal']
                
                # Adjust overall score based on ML confidence
                if 'score' in existing_signal:
                    # Weighted average: 60% traditional, 40% ML
                    ml_score = ml_prediction['probability'] * 100
                    existing_signal['combined_score'] = (
                        existing_signal['score'] * 0.6 + ml_score * 0.4
                    )
                else:
                    existing_signal['combined_score'] = ml_prediction['probability'] * 100
                
                # Add key ML features
                if 'features' in ml_prediction:
                    existing_signal['ml_features'] = ml_prediction['features']
                
                # Add explanation if available
                if self.enable_explainability and self.explainer:
                    explanation = self._get_explanation(stock_data, ticker)
                    if explanation:
                        existing_signal['ml_explanation'] = explanation
                
                # Track for monitoring
                if self.enable_monitoring:
                    self.prediction_history.append(ml_prediction)
                
                # Update signal strength based on ML
                existing_signal['signal_strength'] = self._calculate_signal_strength(
                    existing_signal.get('score', 50),
                    ml_prediction['probability'],
                    ml_prediction['confidence']
                )
                
            return existing_signal
            
        except Exception as e:
            logger.error(f"Error enhancing analysis for {ticker}: {e}")
            return existing_signal
    
    def _get_explanation(self, stock_data: pd.DataFrame, ticker: str) -> str:
        """Get SHAP explanation for prediction"""
        try:
            # Prepare features
            features_df = self.predictor.prepare_live_features(stock_data, ticker)
            if features_df.empty:
                return ""
            
            # Get latest features
            X = self.predictor.feature_engineer.transform(features_df.iloc[-1:])
            
            # Initialize explainer if needed
            if not self.explainer.explainer:
                # Use a sample of recent predictions as background
                self.explainer.initialize_explainer(X)
            
            # Get explanation
            explanation = self.explainer.explain_prediction(X)
            
            # Format for display
            return format_explanation_for_display(explanation)
            
        except Exception as e:
            logger.error(f"Error getting explanation: {e}")
            return ""
    
    def _calculate_signal_strength(self, traditional_score: float,
                                  ml_probability: float,
                                  ml_confidence: float) -> str:
        """
        Calculate combined signal strength
        
        Args:
            traditional_score: Traditional indicator score (0-100)
            ml_probability: ML probability (0-1)
            ml_confidence: ML confidence (0-1)
            
        Returns:
            Signal strength category
        """
        # Weighted combination
        combined = (traditional_score/100 * 0.5 + 
                   ml_probability * 0.3 + 
                   ml_confidence * 0.2)
        
        if combined >= 0.85:
            return "VERY STRONG"
        elif combined >= 0.70:
            return "STRONG"
        elif combined >= 0.55:
            return "MODERATE"
        elif combined >= 0.40:
            return "WEAK"
        else:
            return "VERY WEAK"
    
    def filter_by_ml_confidence(self, signals: List[Dict]) -> List[Dict]:
        """
        Filter signals by ML confidence threshold
        
        Args:
            signals: List of stock signals
            
        Returns:
            Filtered list of high-confidence signals
        """
        filtered = []
        
        for signal in signals:
            ml_conf = signal.get('ml_confidence', 0)
            ml_prob = signal.get('ml_probability', 0.5)
            
            # Include if ML confidence is high enough
            if ml_conf >= self.min_confidence:
                filtered.append(signal)
            # Also include strong traditional signals even if ML is uncertain
            elif signal.get('score', 0) > 80 and ml_prob > 0.4:
                filtered.append(signal)
        
        return filtered
    
    def rank_with_ml(self, signals: List[Dict]) -> List[Dict]:
        """
        Rank signals using ML predictions
        
        Args:
            signals: List of stock signals
            
        Returns:
            Ranked list of signals
        """
        # Calculate ranking score for each signal
        for signal in signals:
            # Components of ranking score
            traditional = signal.get('score', 50) / 100
            ml_prob = signal.get('ml_probability', 0.5)
            ml_conf = signal.get('ml_confidence', 0.5)
            volume_score = min(signal.get('volume_ratio', 1), 2) / 2
            
            # Weighted ranking
            signal['ml_rank_score'] = (
                traditional * 0.3 +
                ml_prob * 0.4 +
                ml_conf * 0.2 +
                volume_score * 0.1
            ) * 100
        
        # Sort by ML ranking score
        ranked = sorted(signals, key=lambda x: x.get('ml_rank_score', 0), reverse=True)
        
        return ranked
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        if not self.monitor:
            return {'status': 'disabled'}
        
        report = self.monitor.get_monitoring_report()
        
        # Add predictor stats
        if self.predictor:
            report['predictor_stats'] = self.predictor.get_performance_stats()
        
        return report
    
    def check_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Check for feature drift
        
        Args:
            current_data: Current market data
            
        Returns:
            Drift detection results
        """
        if not self.monitor:
            return {'drift_detected': False}
        
        # Prepare features
        features_df = pd.DataFrame()
        for ticker in current_data['ticker'].unique()[:10]:  # Sample 10 stocks
            ticker_data = current_data[current_data['ticker'] == ticker]
            prepared = self.predictor.prepare_live_features(ticker_data, ticker)
            if not prepared.empty:
                features_df = pd.concat([features_df, prepared.iloc[-1:]])
        
        if features_df.empty:
            return {'drift_detected': False}
        
        # Check drift
        drift_result = self.monitor.detect_feature_drift(features_df)
        
        return drift_result
    
    def update_risk_with_ml(self, risk_manager, contract_price: float,
                           ml_confidence: float) -> int:
        """
        Update position size based on ML confidence
        
        Args:
            risk_manager: RiskManager instance
            contract_price: Option contract price
            ml_confidence: ML model confidence
            
        Returns:
            Recommended number of contracts
        """
        return risk_manager.calculate_position_size(contract_price, ml_confidence)
    
    def format_ml_signal_display(self, signal: Dict) -> str:
        """
        Format ML-enhanced signal for display
        
        Args:
            signal: Signal dictionary with ML components
            
        Returns:
            Formatted string for display
        """
        lines = []
        
        # Basic info
        lines.append(f"📊 {signal.get('ticker', 'Unknown')}")
        lines.append(f"Traditional Score: {signal.get('score', 0):.1f}")
        
        # ML predictions
        if 'ml_probability' in signal:
            ml_prob = signal['ml_probability']
            ml_conf = signal['ml_confidence']
            
            # Probability with visual indicator
            if ml_prob > 0.7:
                prob_indicator = "🟢"
            elif ml_prob > 0.5:
                prob_indicator = "🟡"
            else:
                prob_indicator = "🔴"
            
            lines.append(f"{prob_indicator} ML Probability: {ml_prob:.1%}")
            lines.append(f"   ML Confidence: {ml_conf:.1%}")
            
            # Signal strength
            lines.append(f"   Signal: {signal.get('ml_signal', 'NEUTRAL')}")
            
            # Combined score
            if 'combined_score' in signal:
                lines.append(f"📈 Combined Score: {signal['combined_score']:.1f}")
        
        # ML explanation
        if 'ml_explanation' in signal:
            lines.append("\n📝 ML Explanation:")
            lines.append(signal['ml_explanation'])
        
        # Risk adjustment
        if 'ml_confidence' in signal:
            if signal['ml_confidence'] >= 0.9:
                lines.append("💪 High confidence - Full position size")
            elif signal['ml_confidence'] >= 0.75:
                lines.append("✅ Good confidence - 75% position size")
            elif signal['ml_confidence'] >= 0.6:
                lines.append("⚠️ Moderate confidence - 50% position size")
            else:
                lines.append("⚡ Low confidence - Minimum position")
        
        return "\n".join(lines)