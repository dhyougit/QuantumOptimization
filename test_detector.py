"""
Unit Tests for Quantum Bad Actor Detector

Basic tests to ensure functionality.
"""

import numpy as np
import pytest
from quantum_detector.detector import QuantumBadActorDetector
from data.behavior_generator import generate_sample_data


def test_data_generation():
    """Test synthetic data generation."""
    user_ids, features, labels = generate_sample_data(n_users=100, bad_actor_ratio=0.2)
    
    assert len(user_ids) == 100
    assert features.shape == (100, 18)
    assert len(labels) == 100
    assert np.sum(labels == 1) == 20  # 20% bad actors


def test_detector_initialization():
    """Test detector initialization."""
    detector = QuantumBadActorDetector(n_qubits=4, method='vqc')
    
    assert detector.n_qubits == 4
    assert detector.method == 'vqc'
    assert not detector.is_fitted


def test_detector_training():
    """Test detector training."""
    # Generate small dataset
    _, X, y = generate_sample_data(n_users=50, bad_actor_ratio=0.2)
    
    # Create and train detector
    detector = QuantumBadActorDetector(n_qubits=4, method='vqc', n_layers=1)
    detector.fit(X, y, epochs=2, batch_size=10, verbose=False)
    
    assert detector.is_fitted
    assert detector.n_features == X.shape[1]


def test_detector_prediction():
    """Test detector prediction."""
    # Generate dataset
    _, X, y = generate_sample_data(n_users=50, bad_actor_ratio=0.2)
    
    # Train detector
    detector = QuantumBadActorDetector(n_qubits=4, method='vqc', n_layers=1)
    detector.fit(X, y, epochs=2, batch_size=10, verbose=False)
    
    # Make predictions
    predictions = detector.predict(X[:10])
    risk_scores = detector.predict_risk(X[:10])
    
    assert len(predictions) == 10
    assert len(risk_scores) == 10
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= r <= 1 for r in risk_scores)


def test_detector_save_load():
    """Test detector save and load."""
    import tempfile
    import os
    
    # Generate dataset
    _, X, y = generate_sample_data(n_users=50, bad_actor_ratio=0.2)
    
    # Train detector
    detector = QuantumBadActorDetector(n_qubits=4, method='vqc', n_layers=1)
    detector.fit(X, y, epochs=2, batch_size=10, verbose=False)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        detector.save(temp_path)
        
        # Load
        loaded_detector = QuantumBadActorDetector.load(temp_path)
        
        assert loaded_detector.is_fitted
        assert loaded_detector.n_qubits == 4
        
        # Test predictions match
        pred1 = detector.predict(X[:5])
        pred2 = loaded_detector.predict(X[:5])
        assert np.array_equal(pred1, pred2)
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_classical_detector():
    """Test classical baseline detector."""
    from classical_baseline.ml_models import ClassicalDetector
    
    # Generate dataset
    _, X, y = generate_sample_data(n_users=100, bad_actor_ratio=0.2)
    
    # Train detector
    detector = ClassicalDetector('random_forest')
    detector.fit(X, y)
    
    # Predict
    predictions = detector.predict(X[:10])
    probas = detector.predict_proba(X[:10])
    
    assert len(predictions) == 10
    assert len(probas) == 10
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= p <= 1 for p in probas)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
