"""
Quantum-Inspired Classifier

Hybrid quantum-classical neural network for bad actor detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from quantum_detector.quantum_circuit import ParameterizedQuantumCircuit, QuantumFeatureMap, QuantumMeasurement


class VariationalQuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier (VQC) for binary classification.
    
    Combines quantum feature encoding with variational quantum circuits
    and classical post-processing for risk scoring.
    """
    
    def __init__(
        self,
        n_features: int,
        n_qubits: int = 8,
        n_layers: int = 3,
        encoding_type: str = 'amplitude',
        use_classical_head: bool = True
    ):
        """
        Args:
            n_features: Number of input features
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of variational layers
            encoding_type: Type of quantum encoding
            use_classical_head: Whether to use classical neural network head
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2 ** n_qubits
        
        # Quantum components
        self.feature_map = QuantumFeatureMap(n_qubits, encoding_type)
        self.quantum_circuit = ParameterizedQuantumCircuit(n_qubits, n_layers)
        self.measurement = QuantumMeasurement(n_qubits)
        
        # Feature preprocessing (reduce dimensionality if needed)
        if n_features > self.state_dim:
            self.feature_reducer = nn.Linear(n_features, self.state_dim)
        else:
            self.feature_reducer = None
        
        # Classical post-processing head
        if use_classical_head:
            # Map quantum measurements to risk score
            n_measurements = n_qubits  # Measure Z on each qubit
            self.classical_head = nn.Sequential(
                nn.Linear(n_measurements, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.classical_head = None
    
    def preprocess_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess features to appropriate dimension for quantum encoding.
        
        Args:
            x: Input features (batch_size, n_features)
            
        Returns:
            Processed features (batch_size, state_dim or less)
        """
        if self.feature_reducer is not None:
            x = self.feature_reducer(x)
        return x
    
    def quantum_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through quantum circuit.
        
        Args:
            x: Classical input features
            
        Returns:
            Quantum measurement results
        """
        batch_size = x.shape[0]
        measurements = []
        
        for i in range(batch_size):
            # Encode classical data to quantum state
            quantum_state = self.feature_map.encode(x[i])
            
            # Convert to torch tensor
            quantum_state_torch = torch.from_numpy(quantum_state).unsqueeze(0)
            
            # Apply variational quantum circuit
            output_state = self.quantum_circuit(quantum_state_torch)
            
            # Measure qubits
            output_state_np = output_state.detach().numpy()[0]
            
            # Measure Pauli-Z on each qubit
            qubit_measurements = []
            for q in range(self.n_qubits):
                z_exp = self.measurement.pauli_z_measurement(output_state_np, q)
                qubit_measurements.append(z_exp)
            
            measurements.append(qubit_measurements)
        
        return np.array(measurements)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: classical → quantum → classical.
        
        Args:
            x: Input features (batch_size, n_features)
            
        Returns:
            Risk scores (batch_size, 1)
        """
        # Preprocess features
        x_processed = self.preprocess_features(x)
        
        # Convert to numpy for quantum processing
        x_np = x_processed.detach().cpu().numpy()
        
        # Quantum forward pass
        quantum_features = self.quantum_forward(x_np)
        
        # Convert back to torch
        quantum_features_torch = torch.from_numpy(quantum_features).float().to(x.device)
        
        # Classical post-processing
        if self.classical_head is not None:
            output = self.classical_head(quantum_features_torch)
        else:
            # Simple aggregation if no classical head
            output = torch.sigmoid(quantum_features_torch.mean(dim=1, keepdim=True))
        
        return output


class HybridQuantumNN(nn.Module):
    """
    Hybrid quantum-classical neural network.
    
    Combines classical feature extraction with quantum processing.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list = [64, 32],
        n_qubits: int = 6,
        n_quantum_layers: int = 2
    ):
        """
        Args:
            n_features: Number of input features
            hidden_dims: Dimensions of classical hidden layers
            n_qubits: Number of qubits
            n_quantum_layers: Number of quantum circuit layers
        """
        super().__init__()
        
        # Classical feature extraction
        layers = []
        prev_dim = n_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.classical_encoder = nn.Sequential(*layers)
        
        # Quantum processing layer
        self.vqc = VariationalQuantumClassifier(
            n_features=prev_dim,
            n_qubits=n_qubits,
            n_layers=n_quantum_layers,
            use_classical_head=False
        )
        
        # Final classification head
        self.output_layer = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network.
        
        Args:
            x: Input features (batch_size, n_features)
            
        Returns:
            Risk scores (batch_size, 1)
        """
        # Classical feature extraction
        classical_features = self.classical_encoder(x)
        
        # Quantum processing
        quantum_features = self.vqc(classical_features)
        
        # Final classification
        output = self.output_layer(quantum_features)
        
        return output


class QuantumKernelSVM:
    """
    Support Vector Machine with quantum kernel.
    
    Uses quantum feature maps to compute kernel similarities.
    """
    
    def __init__(self, n_qubits: int = 8, gamma: float = 1.0):
        """
        Args:
            n_qubits: Number of qubits for quantum feature map
            gamma: Kernel parameter
        """
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.feature_map = QuantumFeatureMap(n_qubits, encoding_type='amplitude')
        
        # SVM parameters (learned during training)
        self.support_vectors = None
        self.support_labels = None
        self.alphas = None
        self.bias = 0.0
    
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum kernel between two samples.
        
        K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Kernel value
        """
        # Encode both samples
        state1 = self.feature_map.encode(x1)
        state2 = self.feature_map.encode(x2)
        
        # Compute inner product
        inner_product = np.abs(np.vdot(state1, state2)) ** 2
        
        return inner_product
    
    def compute_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute kernel matrix between samples.
        
        Args:
            X: First set of samples (n_samples_x, n_features)
            Y: Second set of samples (n_samples_y, n_features) or None
            
        Returns:
            Kernel matrix (n_samples_x, n_samples_y)
        """
        if Y is None:
            Y = X
        
        n_x = X.shape[0]
        n_y = Y.shape[0]
        
        kernel_matrix = np.zeros((n_x, n_y))
        
        for i in range(n_x):
            for j in range(n_y):
                kernel_matrix[i, j] = self.quantum_kernel(X[i], Y[j])
        
        return kernel_matrix
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train quantum kernel SVM.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        # Simplified SVM training using quantum kernel
        # In practice, use proper SVM optimization (e.g., SMO algorithm)
        
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.compute_kernel_matrix(X)
        
        # Simplified: use all samples as support vectors
        self.support_vectors = X
        self.support_labels = y
        self.alphas = np.ones(n_samples) / n_samples
        
        # Compute bias
        predictions = self._decision_function(X)
        self.bias = np.mean(y - predictions)
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        Args:
            X: Input samples
            
        Returns:
            Decision values
        """
        if self.support_vectors is None:
            raise ValueError("Model not trained yet")
        
        K = self.compute_kernel_matrix(X, self.support_vectors)
        decision = K @ (self.alphas * self.support_labels) + self.bias
        
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted labels
        """
        decision = self._decision_function(X)
        return (decision > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability scores.
        
        Args:
            X: Input samples
            
        Returns:
            Probability scores
        """
        decision = self._decision_function(X)
        # Convert decision values to probabilities using sigmoid
        proba = 1 / (1 + np.exp(-decision))
        return proba


class QuantumEnsemble:
    """
    Ensemble of quantum classifiers for improved robustness.
    """
    
    def __init__(
        self,
        n_features: int,
        n_estimators: int = 5,
        n_qubits: int = 6
    ):
        """
        Args:
            n_features: Number of input features
            n_estimators: Number of ensemble members
            n_qubits: Number of qubits per classifier
        """
        self.n_estimators = n_estimators
        
        # Create ensemble of VQCs with different initializations
        self.estimators = []
        for i in range(n_estimators):
            vqc = VariationalQuantumClassifier(
                n_features=n_features,
                n_qubits=n_qubits,
                n_layers=2,
                use_classical_head=True
            )
            self.estimators.append(vqc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features
            
        Returns:
            Averaged risk scores
        """
        predictions = []
        
        for estimator in self.estimators:
            pred = estimator(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    def train(self):
        """Set all estimators to training mode."""
        for estimator in self.estimators:
            estimator.train()
    
    def eval(self):
        """Set all estimators to evaluation mode."""
        for estimator in self.estimators:
            estimator.eval()
