"""
Deep Learning Model Architecture for Trading Predictions
Implements hybrid neural network with attention mechanism
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l1_l2
from sklearn.utils.class_weight import compute_class_weight
import logging
from typing import Dict, Tuple, Optional, List
import pickle

logger = logging.getLogger(__name__)


class TradingNeuralNetwork:
    """Hybrid neural network for trading signal prediction"""
    
    def __init__(self, 
                 input_shape: Tuple[int],
                 n_classes: int = 2,
                 learning_rate: float = 0.001,
                 dropout_rate: float = 0.3,
                 l1_reg: float = 0.01,
                 l2_reg: float = 0.01):
        """
        Initialize the neural network model
        
        Args:
            input_shape: Shape of input features
            n_classes: Number of output classes (2 for binary, 3 for multi-class)
            learning_rate: Initial learning rate
            dropout_rate: Dropout rate for regularization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        self.model = None
        self.history = None
        self.class_weights = None
        
    def build_model(self) -> keras.Model:
        """
        Build the hybrid neural network architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Dense pathway for tabular features
        dense_path = layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )(inputs)
        dense_path = layers.BatchNormalization()(dense_path)
        dense_path = layers.Dropout(self.dropout_rate)(dense_path)
        
        dense_path = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )(dense_path)
        dense_path = layers.BatchNormalization()(dense_path)
        dense_path = layers.Dropout(self.dropout_rate)(dense_path)
        
        dense_path = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )(dense_path)
        dense_path = layers.BatchNormalization()(dense_path)
        dense_path = layers.Dropout(self.dropout_rate * 0.5)(dense_path)
        
        # Attention mechanism
        attention = layers.Dense(
            self.input_shape[0],
            activation='softmax',
            name='attention_weights'
        )(dense_path)
        
        # Apply attention to input features
        attended_features = layers.Multiply()([inputs, attention])
        
        # Combine paths
        combined = layers.Concatenate()([dense_path, attended_features])
        
        # Final dense layers
        final_dense = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )(combined)
        final_dense = layers.BatchNormalization()(final_dense)
        final_dense = layers.Dropout(self.dropout_rate * 0.5)(final_dense)
        
        # Output layer
        if self.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(final_dense)
        else:
            outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(final_dense)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        if self.n_classes == 2:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        logger.info(f"Model built with {model.count_params()} parameters")
        
        return model
    
    def build_lstm_model(self, sequence_length: int) -> keras.Model:
        """
        Build LSTM model for sequential data
        
        Args:
            sequence_length: Length of input sequences
            
        Returns:
            Compiled LSTM model
        """
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(sequence_length, self.input_shape[0])),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(32),
            layers.Dropout(self.dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.Dense(1 if self.n_classes == 2 else self.n_classes,
                        activation='sigmoid' if self.n_classes == 2 else 'softmax')
        ])
        
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        if self.n_classes == 2:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        logger.info(f"LSTM model built with {model.count_params()} parameters")
        
        return model
    
    def calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        classes = np.unique(y_train)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        self.class_weights = dict(zip(classes, weights))
        logger.info(f"Class weights calculated: {self.class_weights}")
        
        return self.class_weights
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             use_class_weights: bool = True,
             early_stopping_patience: int = 15,
             reduce_lr_patience: int = 10) -> Dict:
        """
        Train the model with advanced callbacks
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_class_weights: Whether to use class weights for imbalanced data
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Calculate class weights if needed
        if use_class_weights:
            self.calculate_class_weights(y_train)
        
        # Define callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            class_weight=self.class_weights if use_class_weights else None,
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history.history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X, verbose=0)
        
        # Convert to probabilities for binary classification
        if self.n_classes == 2 and predictions.shape[1] == 1:
            # For binary classification with sigmoid output
            proba = np.column_stack([1 - predictions, predictions])
        else:
            proba = predictions
        
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get class predictions
        
        Args:
            X: Features to predict
            threshold: Probability threshold for positive class
            
        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)
        
        if self.n_classes == 2:
            return (proba[:, 1] > threshold).astype(int)
        else:
            return np.argmax(proba, axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Get model metrics
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        # Add custom metrics
        predictions = self.predict(X_test)
        proba = self.predict_proba(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        metrics['classification_report'] = classification_report(y_test, predictions)
        metrics['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()
        
        # Calculate additional metrics for binary classification
        if self.n_classes == 2:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            
            metrics['roc_auc'] = roc_auc_score(y_test, proba[:, 1])
            metrics['precision'] = precision_score(y_test, predictions)
            metrics['recall'] = recall_score(y_test, predictions)
            metrics['f1'] = f1_score(y_test, predictions)
        
        logger.info(f"Model evaluation completed: Accuracy={metrics.get('accuracy', 0):.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model architecture and weights
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'class_weights': self.class_weights,
            'history': self.history.history if self.history else None
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TradingNeuralNetwork':
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded TradingNeuralNetwork instance
        """
        # Load model
        model = keras.models.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            input_shape=metadata['input_shape'],
            n_classes=metadata['n_classes'],
            learning_rate=metadata['learning_rate'],
            dropout_rate=metadata['dropout_rate']
        )
        
        instance.model = model
        instance.class_weights = metadata['class_weights']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights for interpretability
        
        Args:
            X: Input features
            
        Returns:
            Attention weights for each sample
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create a model that outputs attention weights
        attention_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('attention_weights').output
        )
        
        attention_weights = attention_model.predict(X, verbose=0)
        return attention_weights


class EnsembleModel:
    """Ensemble of multiple models for improved predictions"""
    
    def __init__(self, models: List[TradingNeuralNetwork] = None):
        """
        Initialize ensemble model
        
        Args:
            models: List of trained models
        """
        self.models = models or []
        
    def add_model(self, model: TradingNeuralNetwork):
        """Add a model to the ensemble"""
        self.models.append(model)
        
    def predict_proba(self, X: np.ndarray, weights: List[float] = None) -> np.ndarray:
        """
        Get ensemble probability predictions
        
        Args:
            X: Features to predict
            weights: Model weights for weighted average
            
        Returns:
            Averaged probability predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        predictions = []
        for model, weight in zip(self.models, weights):
            pred = model.predict_proba(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get ensemble class predictions
        
        Args:
            X: Features to predict
            threshold: Probability threshold
            
        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)
        
        if proba.shape[1] == 2:
            return (proba[:, 1] > threshold).astype(int)
        else:
            return np.argmax(proba, axis=1)