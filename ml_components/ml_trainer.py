"""
ML Model Training Pipeline with Walk-Forward Validation
Handles model training, validation, and hyperparameter optimization
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from tqdm import tqdm
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from ml_data_collector import MLDataCollector
from ml_feature_engineering import FeatureEngineer
from ml_model import TradingNeuralNetwork, EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles complete model training pipeline with walk-forward validation"""
    
    def __init__(self, 
                 data_collector: MLDataCollector = None,
                 feature_engineer: FeatureEngineer = None,
                 model_config: Dict = None):
        """
        Initialize model trainer
        
        Args:
            data_collector: Data collection instance
            feature_engineer: Feature engineering instance
            model_config: Model configuration dictionary
        """
        self.data_collector = data_collector
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        self.model_config = model_config or {
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l1_reg': 0.01,
            'l2_reg': 0.01
        }
        
        self.models = []
        self.best_model = None
        self.training_history = []
        
    def walk_forward_validation(self, 
                               df: pd.DataFrame,
                               n_splits: int = 5,
                               train_size: int = 252 * 3,  # 3 years
                               test_size: int = 252) -> List[Dict]:
        """
        Perform walk-forward validation
        
        Args:
            df: Complete dataset with features and labels
            n_splits: Number of walk-forward splits
            train_size: Training window size in days
            test_size: Testing window size in days
            
        Returns:
            List of validation results
        """
        results = []
        
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # Split data
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # Further split training data for validation
            val_size = len(train_data) // 5
            train_data, val_data = train_data[:-val_size], train_data[-val_size:]
            
            # Engineer features
            train_data = self.feature_engineer.engineer_features(train_data)
            val_data = self.feature_engineer.engineer_features(val_data)
            test_data = self.feature_engineer.engineer_features(test_data)
            
            # Prepare features and labels
            feature_cols = [col for col in train_data.columns 
                          if col not in ['label', 'multi_label', 'future_return', 'ticker', 'date']]
            
            # Fit feature engineer on training data
            self.feature_engineer.fit(train_data, feature_cols)
            
            X_train = self.feature_engineer.transform(train_data)
            y_train = train_data['label'].values
            
            X_val = self.feature_engineer.transform(val_data)
            y_val = val_data['label'].values
            
            X_test = self.feature_engineer.transform(test_data)
            y_test = test_data['label'].values
            
            # Train model
            model = TradingNeuralNetwork(
                input_shape=(X_train.shape[1],),
                **self.model_config
            )
            model.build_model()
            
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=50,
                batch_size=32,
                use_class_weights=True,
                early_stopping_patience=10
            )
            
            # Evaluate on test set
            metrics = model.evaluate(X_test, y_test)
            
            # Store results
            result = {
                'fold': fold,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'metrics': metrics,
                'model': model
            }
            
            results.append(result)
            self.models.append(model)
            
            logger.info(f"Fold {fold + 1} - Test Accuracy: {metrics.get('accuracy', 0):.4f}, "
                       f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        
        # Select best model based on average performance
        self.select_best_model(results)
        
        return results
    
    def select_best_model(self, results: List[Dict]):
        """
        Select the best model based on validation results
        
        Args:
            results: List of validation results
        """
        # Calculate average metrics for each model
        avg_scores = []
        for i, result in enumerate(results):
            metrics = result['metrics']
            # Use ROC-AUC as primary metric
            score = metrics.get('roc_auc', 0) * 0.5 + metrics.get('accuracy', 0) * 0.3 + metrics.get('f1', 0) * 0.2
            avg_scores.append(score)
        
        # Select best model
        best_idx = np.argmax(avg_scores)
        self.best_model = results[best_idx]['model']
        
        logger.info(f"Best model selected from fold {best_idx + 1} with score: {avg_scores[best_idx]:.4f}")
    
    def hyperparameter_optimization(self, 
                                  df: pd.DataFrame,
                                  n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            df: Training dataset
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
                'l1_reg': trial.suggest_loguniform('l1_reg', 1e-5, 1e-1),
                'l2_reg': trial.suggest_loguniform('l2_reg', 1e-5, 1e-1)
            }
            
            # Prepare data (simplified for optimization)
            df_sample = df.sample(min(10000, len(df)))
            df_sample = self.feature_engineer.engineer_features(df_sample)
            
            feature_cols = [col for col in df_sample.columns 
                          if col not in ['label', 'multi_label', 'future_return', 'ticker', 'date']]
            
            self.feature_engineer.fit(df_sample, feature_cols)
            X = self.feature_engineer.transform(df_sample)
            y = df_sample['label'].values
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model with suggested parameters
            model = TradingNeuralNetwork(
                input_shape=(X_train.shape[1],),
                **params
            )
            model.build_model()
            
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=20,  # Fewer epochs for optimization
                batch_size=32,
                use_class_weights=True,
                early_stopping_patience=5
            )
            
            # Return validation loss
            val_loss = min(history['val_loss'])
            return val_loss
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")
        
        return study.best_params
    
    def train_final_model(self, df: pd.DataFrame) -> TradingNeuralNetwork:
        """
        Train final model on all available data
        
        Args:
            df: Complete dataset
            
        Returns:
            Trained final model
        """
        logger.info("Training final model on complete dataset")
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df)
        
        # Prepare features
        feature_cols = [col for col in df.columns 
                       if col not in ['label', 'multi_label', 'future_return', 'ticker', 'date']]
        
        self.feature_engineer.fit(df, feature_cols)
        X = self.feature_engineer.transform(df)
        y = df['label'].values
        
        # Split for validation
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        model = TradingNeuralNetwork(
            input_shape=(X_train.shape[1],),
            **self.model_config
        )
        model.build_model()
        
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=32,
            use_class_weights=True,
            early_stopping_patience=15
        )
        
        # Evaluate
        metrics = model.evaluate(X_val, y_val)
        logger.info(f"Final model performance - Accuracy: {metrics.get('accuracy', 0):.4f}, "
                   f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        
        self.best_model = model
        return model
    
    def create_ensemble(self, n_models: int = 5) -> EnsembleModel:
        """
        Create an ensemble of models
        
        Args:
            n_models: Number of models in ensemble
            
        Returns:
            Ensemble model
        """
        if len(self.models) < n_models:
            logger.warning(f"Only {len(self.models)} models available for ensemble")
            n_models = len(self.models)
        
        # Select top n models based on performance
        ensemble = EnsembleModel()
        for model in self.models[:n_models]:
            ensemble.add_model(model)
        
        logger.info(f"Created ensemble with {n_models} models")
        return ensemble
    
    def save_training_artifacts(self, output_dir: str = 'models'):
        """
        Save all training artifacts
        
        Args:
            output_dir: Directory to save artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        if self.best_model:
            self.best_model.save(os.path.join(output_dir, 'best_model.h5'))
        
        # Save feature engineer
        self.feature_engineer.save(os.path.join(output_dir, 'feature_engineer.pkl'))
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.training_history, f)
        
        # Save model config
        with open(os.path.join(output_dir, 'model_config.pkl'), 'wb') as f:
            pickle.dump(self.model_config, f)
        
        logger.info(f"Training artifacts saved to {output_dir}")
    
    def load_training_artifacts(self, output_dir: str = 'models'):
        """
        Load training artifacts
        
        Args:
            output_dir: Directory containing artifacts
        """
        # Load best model
        model_path = os.path.join(output_dir, 'best_model.h5')
        if os.path.exists(model_path):
            self.best_model = TradingNeuralNetwork.load(model_path)
        
        # Load feature engineer
        fe_path = os.path.join(output_dir, 'feature_engineer.pkl')
        if os.path.exists(fe_path):
            self.feature_engineer = FeatureEngineer.load(fe_path)
        
        # Load training history
        history_path = os.path.join(output_dir, 'training_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
        
        # Load model config
        config_path = os.path.join(output_dir, 'model_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.model_config = pickle.load(f)
        
        logger.info(f"Training artifacts loaded from {output_dir}")


def main():
    """Example training pipeline"""
    # Initialize components
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    collector = MLDataCollector(
        tickers=tickers,
        start_date='2015-01-01',
        label_threshold=0.05,
        label_days=10
    )
    
    # Collect data
    print("Collecting historical data...")
    df = collector.collect_all_data()
    
    if df.empty:
        print("No data collected")
        return
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Optimize hyperparameters (optional)
    print("\nOptimizing hyperparameters...")
    best_params = trainer.hyperparameter_optimization(df, n_trials=10)
    trainer.model_config.update(best_params)
    
    # Perform walk-forward validation
    print("\nPerforming walk-forward validation...")
    results = trainer.walk_forward_validation(df, n_splits=3)
    
    # Print results
    print("\nValidation Results:")
    for result in results:
        print(f"Fold {result['fold'] + 1}:")
        print(f"  Test Period: {result['test_period'][0]} to {result['test_period'][1]}")
        print(f"  Accuracy: {result['metrics'].get('accuracy', 0):.4f}")
        print(f"  ROC-AUC: {result['metrics'].get('roc_auc', 0):.4f}")
    
    # Train final model
    print("\nTraining final model...")
    final_model = trainer.train_final_model(df)
    
    # Save artifacts
    trainer.save_training_artifacts()
    print("\nTraining complete! Artifacts saved to 'models' directory")


if __name__ == "__main__":
    main()