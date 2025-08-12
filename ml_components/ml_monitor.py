"""
ML Model Drift Monitoring System
Detects feature drift and model performance degradation
"""
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import pickle
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitors feature drift and model performance degradation"""
    
    def __init__(self, 
                 reference_data: pd.DataFrame = None,
                 feature_names: List[str] = None,
                 drift_threshold: float = 0.05,  # p-value threshold for KS test
                 psi_threshold: float = 0.1,  # PSI threshold
                 performance_window: int = 100,  # Number of predictions to track
                 alert_threshold: int = 3):  # Consecutive drift detections before alert
        """
        Initialize drift monitor
        
        Args:
            reference_data: Reference dataset from training
            feature_names: List of feature names to monitor
            drift_threshold: Statistical threshold for drift detection
            psi_threshold: Population Stability Index threshold
            performance_window: Window size for performance tracking
            alert_threshold: Number of consecutive drifts before alert
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold
        self.performance_window = performance_window
        self.alert_threshold = alert_threshold
        
        # Calculate reference statistics
        self.reference_stats = {}
        if reference_data is not None:
            self.calculate_reference_stats()
        
        # Track current data
        self.current_data_buffer = deque(maxlen=performance_window)
        self.prediction_history = deque(maxlen=performance_window)
        
        # Drift tracking
        self.drift_scores = {}
        self.drift_history = []
        self.consecutive_drifts = 0
        self.alerts = []
        
        # Performance tracking
        self.performance_metrics = {
            'accuracy': deque(maxlen=performance_window),
            'confidence': deque(maxlen=performance_window),
            'prediction_distribution': deque(maxlen=performance_window)
        }
    
    def calculate_reference_stats(self):
        """Calculate statistics for reference data"""
        if self.reference_data is None:
            return
        
        for feature in self.feature_names:
            if feature in self.reference_data.columns:
                self.reference_stats[feature] = {
                    'mean': self.reference_data[feature].mean(),
                    'std': self.reference_data[feature].std(),
                    'median': self.reference_data[feature].median(),
                    'q25': self.reference_data[feature].quantile(0.25),
                    'q75': self.reference_data[feature].quantile(0.75),
                    'min': self.reference_data[feature].min(),
                    'max': self.reference_data[feature].max(),
                    'distribution': self.reference_data[feature].values
                }
        
        logger.info(f"Reference statistics calculated for {len(self.reference_stats)} features")
    
    def detect_drift_ks_test(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Args:
            current_data: Current batch of data
            
        Returns:
            Dictionary of feature -> p-value
        """
        ks_scores = {}
        
        for feature in self.feature_names:
            if feature not in current_data.columns or feature not in self.reference_stats:
                continue
            
            # Get distributions
            ref_dist = self.reference_stats[feature]['distribution']
            curr_dist = current_data[feature].values
            
            # Perform KS test
            try:
                statistic, p_value = stats.ks_2samp(ref_dist, curr_dist)
                ks_scores[feature] = p_value
            except Exception as e:
                logger.warning(f"KS test failed for {feature}: {e}")
                ks_scores[feature] = 1.0  # No drift
        
        return ks_scores
    
    def calculate_psi(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            current_data: Current batch of data
            
        Returns:
            Dictionary of feature -> PSI value
        """
        psi_scores = {}
        
        for feature in self.feature_names:
            if feature not in current_data.columns or feature not in self.reference_stats:
                continue
            
            try:
                # Create bins based on reference data
                ref_data = self.reference_stats[feature]['distribution']
                n_bins = 10
                
                # Calculate bin edges
                _, bin_edges = np.histogram(ref_data, bins=n_bins)
                
                # Calculate distributions
                ref_dist, _ = np.histogram(ref_data, bins=bin_edges)
                curr_dist, _ = np.histogram(current_data[feature].values, bins=bin_edges)
                
                # Normalize
                ref_dist = (ref_dist + 1) / (len(ref_data) + n_bins)
                curr_dist = (curr_dist + 1) / (len(current_data) + n_bins)
                
                # Calculate PSI
                psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
                psi_scores[feature] = psi
                
            except Exception as e:
                logger.warning(f"PSI calculation failed for {feature}: {e}")
                psi_scores[feature] = 0.0
        
        return psi_scores
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Comprehensive drift detection
        
        Args:
            current_data: Current batch of data
            
        Returns:
            Drift detection results
        """
        # KS test
        ks_scores = self.detect_drift_ks_test(current_data)
        
        # PSI
        psi_scores = self.calculate_psi(current_data)
        
        # Statistical drift
        stat_drift = self.detect_statistical_drift(current_data)
        
        # Combine results
        drift_detected = {}
        drift_details = {}
        
        for feature in self.feature_names:
            ks_drift = ks_scores.get(feature, 1.0) < self.drift_threshold
            psi_drift = psi_scores.get(feature, 0.0) > self.psi_threshold
            stat_drift_detected = stat_drift.get(feature, {}).get('drift', False)
            
            drift_detected[feature] = ks_drift or psi_drift or stat_drift_detected
            
            drift_details[feature] = {
                'ks_p_value': ks_scores.get(feature, 1.0),
                'psi': psi_scores.get(feature, 0.0),
                'statistical_drift': stat_drift.get(feature, {}),
                'drift_detected': drift_detected[feature]
            }
        
        # Overall drift
        overall_drift = sum(drift_detected.values()) / len(drift_detected) > 0.3
        
        result = {
            'timestamp': datetime.now(),
            'overall_drift': overall_drift,
            'features_drifted': sum(drift_detected.values()),
            'total_features': len(drift_detected),
            'drift_percentage': sum(drift_detected.values()) / len(drift_detected) * 100,
            'feature_details': drift_details
        }
        
        # Update tracking
        self.drift_history.append(result)
        self.drift_scores = drift_details
        
        # Check for consecutive drifts
        if overall_drift:
            self.consecutive_drifts += 1
            if self.consecutive_drifts >= self.alert_threshold:
                self.trigger_alert('feature_drift', result)
        else:
            self.consecutive_drifts = 0
        
        return result
    
    def detect_statistical_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift using statistical measures
        
        Args:
            current_data: Current batch of data
            
        Returns:
            Statistical drift results
        """
        stat_drift = {}
        
        for feature in self.feature_names:
            if feature not in current_data.columns or feature not in self.reference_stats:
                continue
            
            ref_stats = self.reference_stats[feature]
            curr_mean = current_data[feature].mean()
            curr_std = current_data[feature].std()
            
            # Z-score for mean shift
            if ref_stats['std'] > 0:
                z_score = abs(curr_mean - ref_stats['mean']) / ref_stats['std']
            else:
                z_score = 0
            
            # Variance ratio
            if ref_stats['std'] > 0:
                var_ratio = curr_std / ref_stats['std']
            else:
                var_ratio = 1
            
            # Detect drift
            mean_drift = z_score > 3  # 3 sigma rule
            var_drift = var_ratio < 0.5 or var_ratio > 2
            
            stat_drift[feature] = {
                'current_mean': curr_mean,
                'reference_mean': ref_stats['mean'],
                'z_score': z_score,
                'variance_ratio': var_ratio,
                'drift': mean_drift or var_drift
            }
        
        return stat_drift
    
    def monitor_prediction_performance(self, predictions: List[Dict]):
        """
        Monitor model prediction performance
        
        Args:
            predictions: List of prediction dictionaries
        """
        for pred in predictions:
            self.prediction_history.append(pred)
            
            # Track confidence
            if 'confidence' in pred:
                self.performance_metrics['confidence'].append(pred['confidence'])
            
            # Track prediction distribution
            if 'probability' in pred:
                self.performance_metrics['prediction_distribution'].append(pred['probability'])
        
        # Check for performance degradation
        self.check_performance_degradation()
    
    def check_performance_degradation(self):
        """Check for model performance degradation"""
        if len(self.performance_metrics['confidence']) < 50:
            return
        
        # Check confidence drop
        recent_confidence = list(self.performance_metrics['confidence'])[-20:]
        older_confidence = list(self.performance_metrics['confidence'])[-50:-20]
        
        avg_recent = np.mean(recent_confidence)
        avg_older = np.mean(older_confidence)
        
        if avg_recent < avg_older * 0.8:  # 20% drop in confidence
            self.trigger_alert('confidence_drop', {
                'recent_avg': avg_recent,
                'older_avg': avg_older,
                'drop_percentage': (1 - avg_recent/avg_older) * 100
            })
        
        # Check prediction distribution shift
        recent_dist = list(self.performance_metrics['prediction_distribution'])[-20:]
        if len(recent_dist) > 0:
            # Check if predictions are becoming too extreme or too neutral
            extreme_predictions = sum(1 for p in recent_dist if p > 0.9 or p < 0.1)
            neutral_predictions = sum(1 for p in recent_dist if 0.4 < p < 0.6)
            
            if extreme_predictions / len(recent_dist) > 0.5:
                self.trigger_alert('extreme_predictions', {
                    'percentage': extreme_predictions / len(recent_dist) * 100
                })
            
            if neutral_predictions / len(recent_dist) > 0.7:
                self.trigger_alert('neutral_predictions', {
                    'percentage': neutral_predictions / len(recent_dist) * 100
                })
    
    def trigger_alert(self, alert_type: str, details: Dict):
        """
        Trigger drift or performance alert
        
        Args:
            alert_type: Type of alert
            details: Alert details
        """
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'details': details,
            'severity': self._get_severity(alert_type, details)
        }
        
        self.alerts.append(alert)
        
        # Log alert
        if alert['severity'] == 'HIGH':
            logger.error(f"HIGH SEVERITY ALERT: {alert_type} - {details}")
        elif alert['severity'] == 'MEDIUM':
            logger.warning(f"MEDIUM SEVERITY ALERT: {alert_type} - {details}")
        else:
            logger.info(f"LOW SEVERITY ALERT: {alert_type} - {details}")
    
    def _get_severity(self, alert_type: str, details: Dict) -> str:
        """Determine alert severity"""
        if alert_type == 'feature_drift':
            drift_pct = details.get('drift_percentage', 0)
            if drift_pct > 50:
                return 'HIGH'
            elif drift_pct > 30:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        elif alert_type == 'confidence_drop':
            drop_pct = details.get('drop_percentage', 0)
            if drop_pct > 30:
                return 'HIGH'
            elif drop_pct > 20:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        elif alert_type in ['extreme_predictions', 'neutral_predictions']:
            return 'MEDIUM'
        
        return 'LOW'
    
    def get_monitoring_report(self) -> Dict:
        """
        Generate comprehensive monitoring report
        
        Returns:
            Monitoring report dictionary
        """
        report = {
            'timestamp': datetime.now(),
            'monitoring_window': self.performance_window,
            'total_predictions': len(self.prediction_history),
            'drift_status': {
                'consecutive_drifts': self.consecutive_drifts,
                'last_drift_check': self.drift_history[-1] if self.drift_history else None,
                'features_with_drift': [f for f, d in self.drift_scores.items() 
                                       if d.get('drift_detected', False)]
            },
            'performance_status': {
                'avg_confidence': np.mean(list(self.performance_metrics['confidence'])) 
                                if self.performance_metrics['confidence'] else 0,
                'confidence_trend': self._calculate_trend(self.performance_metrics['confidence']),
                'prediction_balance': self._calculate_prediction_balance()
            },
            'alerts': {
                'total_alerts': len(self.alerts),
                'recent_alerts': self.alerts[-5:] if self.alerts else [],
                'high_severity_alerts': [a for a in self.alerts if a['severity'] == 'HIGH']
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_trend(self, data: deque) -> str:
        """Calculate trend from data"""
        if len(data) < 10:
            return 'insufficient_data'
        
        recent = list(data)[-5:]
        older = list(data)[-10:-5]
        
        if np.mean(recent) > np.mean(older) * 1.1:
            return 'improving'
        elif np.mean(recent) < np.mean(older) * 0.9:
            return 'degrading'
        else:
            return 'stable'
    
    def _calculate_prediction_balance(self) -> Dict:
        """Calculate balance of predictions"""
        if not self.performance_metrics['prediction_distribution']:
            return {'balanced': True}
        
        dist = list(self.performance_metrics['prediction_distribution'])
        bullish = sum(1 for p in dist if p > 0.6) / len(dist)
        bearish = sum(1 for p in dist if p < 0.4) / len(dist)
        neutral = 1 - bullish - bearish
        
        return {
            'bullish_pct': bullish * 100,
            'bearish_pct': bearish * 100,
            'neutral_pct': neutral * 100,
            'balanced': 0.2 < bullish < 0.8 and 0.2 < bearish < 0.8
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if self.consecutive_drifts >= self.alert_threshold:
            recommendations.append("URGENT: Significant feature drift detected. Model retraining recommended.")
        
        if self.alerts:
            high_alerts = [a for a in self.alerts if a['severity'] == 'HIGH']
            if len(high_alerts) > 3:
                recommendations.append("Multiple high-severity alerts. Review model performance immediately.")
        
        if self.performance_metrics['confidence']:
            avg_conf = np.mean(list(self.performance_metrics['confidence']))
            if avg_conf < 0.6:
                recommendations.append("Low average confidence. Consider adjusting confidence thresholds.")
        
        balance = self._calculate_prediction_balance()
        if not balance.get('balanced', True):
            recommendations.append("Prediction distribution imbalanced. Check for market regime changes.")
        
        if not recommendations:
            recommendations.append("System operating normally. Continue monitoring.")
        
        return recommendations
    
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to file"""
        data = {
            'reference_stats': self.reference_stats,
            'drift_history': self.drift_history,
            'alerts': self.alerts,
            'performance_metrics': {
                'confidence': list(self.performance_metrics['confidence']),
                'prediction_distribution': list(self.performance_metrics['prediction_distribution'])
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Monitoring data saved to {filepath}")
    
    def load_monitoring_data(self, filepath: str):
        """Load monitoring data from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.reference_stats = data['reference_stats']
        self.drift_history = data['drift_history']
        self.alerts = data['alerts']
        self.performance_metrics['confidence'] = deque(
            data['performance_metrics']['confidence'], 
            maxlen=self.performance_window
        )
        self.performance_metrics['prediction_distribution'] = deque(
            data['performance_metrics']['prediction_distribution'],
            maxlen=self.performance_window
        )
        
        logger.info(f"Monitoring data loaded from {filepath}")