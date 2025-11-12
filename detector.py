"""
Main Quantum Bad Actor Detector

High-level interface for detecting malicious users using quantum-inspired methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
import pickle
import os

from quantum_detector.quantum_classifier import (
    VariationalQuantumClassifier,
    HybridQuantumNN,
    QuantumKernelSVM,
    QuantumEnsemble
)
from quantum_detector.quantum_annealing import QuantumAnnealingDetector


class QuantumBadActorDetector:
    """
    Main interface for quantum-inspired bad actor detection.
    
    Provides a unified API for training and inference using various
    quantum-inspired algorithms.
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        method: str = 'vqc',
        n_layers: int = 3,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize quantum bad actor detector.
        
        Args:
            n_qubits: Number of qubits to use
            method: Detection method ('vqc', 'hybrid', 'kernel', 'qaoa', 'ensemble')
            n_layers: Number of quantum circuit layers
            device: Device to run on ('cpu', 'cuda')
            **kwargs: Additional method-specific parameters
        """
        self.n_qubits = n_qubits
        self.method = method
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.kwargs = kwargs
        
        self.model = None
        self.is_fitted = False
        self.feature_scaler = None
        self.n_features = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _create_model(self, n_features: int):
        """
        Create quantum model based on specified method.
        
        Args:
            n_features: Number of input features
        """
        if self.method == 'vqc':
            self.model = VariationalQuantumClassifier(
                n_features=n_features,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                use_classical_head=True
            ).to(self.device)
            
        elif self.method == 'hybrid':
            hidden_dims = self.kwargs.get('hidden_dims', [64, 32])
            self.model = HybridQuantumNN(
                n_features=n_features,
                hidden_dims=hidden_dims,
                n_qubits=self.n_qubits,
                n_quantum_layers=self.n_layers
            ).to(self.device)
            
        elif self.method == 'kernel':
            self.model = QuantumKernelSVM(
                n_qubits=self.n_qubits,
                gamma=self.kwargs.get('gamma', 1.0)
            )
            
        elif self.method == 'qaoa':
            self.model = QuantumAnnealingDetector(
                n_qubits=self.n_qubits
            )
            
        elif self.method == 'ensemble':
            n_estimators = self.kwargs.get('n_estimators', 5)
            self.model = QuantumEnsemble(
                n_features=n_features,
                n_estimators=n_estimators,
                n_qubits=self.n_qubits
            )
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess input data.
        
        Args:
            X: Input features
            fit: Whether to fit scaler
            
        Returns:
            Preprocessed features
        """
        if fit or self.feature_scaler is None:
            # Simple standardization
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0) + 1e-8
            self.feature_scaler = {'mean': mean, 'std': std}
        
        X_scaled = (X - self.feature_scaler['mean']) / self.feature_scaler['std']
        
        return X_scaled
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        """
        Train the quantum detector.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            verbose: Whether to print progress
        """
        self.n_features = X.shape[1]
        
        # Preprocess data
        X_train = self._preprocess_data(X, fit=True)
        
        if X_val is not None:
            X_val = self._preprocess_data(X_val, fit=False)
        
        # Create model
        if self.model is None:
            self._create_model(self.n_features)
        
        # Train based on method
        if self.method == 'kernel':
            # Quantum kernel SVM doesn't need iterative training
            self.model.fit(X_train, y)
            self.is_fitted = True
            
        elif self.method == 'qaoa':
            # QAOA-based detection
            self.model.fit(X_train, y)
            self.is_fitted = True
            
        else:
            # Neural network-based methods
            self._train_neural(
                X_train, y, X_val, y_val,
                epochs, batch_size, learning_rate, verbose
            )
    
    def _train_neural(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool
    ):
        """Train neural network-based quantum models."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Setup optimizer and loss
        if isinstance(self.model, QuantumEnsemble):
            # Optimize all ensemble members
            params = []
            for estimator in self.model.estimators:
                params.extend(list(estimator.parameters()))
            optimizer = optim.Adam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train() if hasattr(self.model, 'train') else None
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(self.model, QuantumEnsemble):
                    outputs = self.model.forward(batch_X)
                else:
                    outputs = self.model(batch_X)
                
                # Compute loss
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item() * batch_X.size(0)
                predictions = (outputs > 0.5).float()
                epoch_correct += (predictions == batch_y).sum().item()
                epoch_total += batch_X.size(0)
            
            # Compute epoch metrics
            avg_loss = epoch_loss / epoch_total
            avg_acc = epoch_correct / epoch_total
            
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_acc)
            
            # Validation
            if X_val is not None:
                val_loss, val_acc = self._validate(X_val, y_val, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        self.is_fitted = True
    
    def _validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate model on validation set."""
        
        self.model.eval() if hasattr(self.model, 'eval') else None
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, QuantumEnsemble):
                outputs = self.model.forward(X_val_tensor)
            else:
                outputs = self.model(X_val_tensor)
            
            loss = criterion(outputs, y_val_tensor)
            
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_val_tensor).float().mean()
        
        return loss.item(), accuracy.item()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict whether users are bad actors.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Preprocess
        X_processed = self._preprocess_data(X, fit=False)
        
        if self.method == 'kernel' or self.method == 'qaoa':
            # Non-neural methods
            predictions = self.model.predict(X_processed)
        else:
            # Neural methods
            self.model.eval() if hasattr(self.model, 'eval') else None
            
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            with torch.no_grad():
                if isinstance(self.model, QuantumEnsemble):
                    outputs = self.model.forward(X_tensor)
                else:
                    outputs = self.model(X_tensor)
                
                predictions = (outputs > 0.5).cpu().numpy().flatten()
        
        return predictions.astype(int)
    
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores for users.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Risk scores between 0 and 1 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Preprocess
        X_processed = self._preprocess_data(X, fit=False)
        
        if self.method == 'kernel':
            risk_scores = self.model.predict_proba(X_processed)
        elif self.method == 'qaoa':
            risk_scores = self.model.predict_risk(X_processed)
        else:
            # Neural methods
            self.model.eval() if hasattr(self.model, 'eval') else None
            
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            with torch.no_grad():
                if isinstance(self.model, QuantumEnsemble):
                    outputs = self.model.forward(X_tensor)
                else:
                    outputs = self.model(X_tensor)
                
                risk_scores = outputs.cpu().numpy().flatten()
        
        return risk_scores
    
    def save(self, filepath: str):
        """
        Save the detector to disk.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        save_dict = {
            'n_qubits': self.n_qubits,
            'method': self.method,
            'n_layers': self.n_layers,
            'n_features': self.n_features,
            'feature_scaler': self.feature_scaler,
            'kwargs': self.kwargs,
            'history': self.history
        }
        
        # Save model state
        if self.method in ['kernel', 'qaoa']:
            save_dict['model'] = self.model
        else:
            save_dict['model_state_dict'] = self.model.state_dict()
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu'):
        """
        Load a detector from disk.
        
        Args:
            filepath: Path to saved file
            device: Device to load to
            
        Returns:
            Loaded detector instance
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create detector instance
        detector = cls(
            n_qubits=save_dict['n_qubits'],
            method=save_dict['method'],
            n_layers=save_dict['n_layers'],
            device=device,
            **save_dict['kwargs']
        )
        
        detector.n_features = save_dict['n_features']
        detector.feature_scaler = save_dict['feature_scaler']
        detector.history = save_dict['history']
        
        # Load model
        if detector.method in ['kernel', 'qaoa']:
            detector.model = save_dict['model']
        else:
            detector._create_model(detector.n_features)
            detector.model.load_state_dict(save_dict['model_state_dict'])
        
        detector.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        
        return detector
