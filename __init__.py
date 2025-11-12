"""
Quantum Detector Package

Quantum-inspired bad actor detection using variational quantum algorithms.
"""

from .detector import QuantumBadActorDetector
from .quantum_classifier import VariationalQuantumClassifier, HybridQuantumNN
from .quantum_circuit import QuantumCircuit, ParameterizedQuantumCircuit, QuantumFeatureMap
from .quantum_annealing import QuantumAnnealingDetector

__version__ = '1.0.0'

__all__ = [
    'QuantumBadActorDetector',
    'VariationalQuantumClassifier',
    'HybridQuantumNN',
    'QuantumCircuit',
    'ParameterizedQuantumCircuit',
    'QuantumFeatureMap',
    'QuantumAnnealingDetector'
]
