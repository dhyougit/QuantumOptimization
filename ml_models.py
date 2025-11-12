"""
Classical ML Baselines

Traditional machine learning models for comparison with quantum approaches.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import xgboost as xgb


class ClassicalDetector:
    """
    Classical machine learning detector for bad actors.
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Args:
            model_type: Type of classifier ('random_forest', 'xgboost', 'svm', 
                       'logistic', 'mlp', 'gradient_boosting')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._create_model(**kwargs)
        self.is_fitted = False
    
    def _create_model(self, **kwargs):
        """Create classical ML model."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                random_state=42
            )
        
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        
        elif self.model_type == 'svm':
            return SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=42
            )
        
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (64, 32)),
                activation=kwargs.get('activation', 'relu'),
                max_iter=kwargs.get('max_iter', 500),
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classical detector.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability scores.
        
        Args:
            X: Input features
            
        Returns:
            Probability scores for positive class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)[:, 1]
        else:
            # For models without predict_proba, use decision function
            decision = self.model.decision_function(X_scaled)
            proba = 1 / (1 + np.exp(-decision))
        
        return proba
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict]:
    """
    Compare multiple classical models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of results for each model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import time
    
    models = {
        'Logistic Regression': ClassicalDetector('logistic'),
        'Random Forest': ClassicalDetector('random_forest', n_estimators=100),
        'XGBoost': ClassicalDetector('xgboost', n_estimators=100),
        'SVM': ClassicalDetector('svm'),
        'MLP': ClassicalDetector('mlp', hidden_layer_sizes=(64, 32)),
        'Gradient Boosting': ClassicalDetector('gradient_boosting')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        # Compute metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'train_time': train_time,
            'inference_time_ms': inference_time
        }
        
        print(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
              f"F1: {results[name]['f1_score']:.4f}")
    
    return results
