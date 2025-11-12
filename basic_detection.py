"""
Basic Detection Example

Demonstrates simple usage of the quantum bad actor detector.
"""

import numpy as np
from quantum_detector.detector import QuantumBadActorDetector
from data.behavior_generator import generate_sample_data, get_feature_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print(" Quantum Bad Actor Detection - Basic Example")
    print("=" * 70)
    
    # Generate sample data
    print("\n1. Generating synthetic user behavior data...")
    user_ids, features, labels = generate_sample_data(
        n_users=1000,
        bad_actor_ratio=0.2,
        seed=42
    )
    
    print(f"   Generated {len(user_ids)} users")
    print(f"   Normal users: {np.sum(labels == 0)}")
    print(f"   Bad actors: {np.sum(labels == 1)}")
    print(f"   Features: {features.shape[1]}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create quantum detector
    print("\n3. Initializing Quantum Bad Actor Detector...")
    detector = QuantumBadActorDetector(
        n_qubits=8,
        method='vqc',  # Variational Quantum Classifier
        n_layers=2,
        device='cpu'
    )
    
    print("   Method: Variational Quantum Classifier (VQC)")
    print("   Qubits: 8")
    print("   Circuit layers: 2")
    
    # Train detector
    print("\n4. Training quantum detector...")
    print("   (This may take a few minutes...)")
    
    detector.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        verbose=True
    )
    
    # Make predictions
    print("\n5. Making predictions on test set...")
    predictions = detector.predict(X_test)
    risk_scores = detector.predict_risk(X_test)
    
    # Evaluate
    print("\n6. Evaluation Results:")
    print("\n" + "=" * 70)
    print(classification_report(y_test, predictions, 
                               target_names=['Normal', 'Bad Actor']))
    print("=" * 70)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(f"                 Predicted Normal  |  Predicted Bad Actor")
    print(f"Actual Normal    {cm[0,0]:8d}          |  {cm[0,1]:8d}")
    print(f"Actual Bad Actor {cm[1,0]:8d}          |  {cm[1,1]:8d}")
    
    # Show some examples
    print("\n7. Sample Predictions:")
    print("-" * 70)
    print("User ID | True Label | Prediction | Risk Score")
    print("-" * 70)
    
    for i in range(min(10, len(X_test))):
        true_label = "Bad Actor" if y_test[i] == 1 else "Normal"
        pred_label = "Bad Actor" if predictions[i] == 1 else "Normal"
        print(f"{i:7d} | {true_label:10s} | {pred_label:10s} | {risk_scores[i]:.4f}")
    
    # Plot training history
    print("\n8. Plotting training history...")
    plot_training_history(detector.history)
    
    # Plot risk score distribution
    print("\n9. Plotting risk score distribution...")
    plot_risk_distribution(risk_scores, y_test)
    
    # Save model
    print("\n10. Saving model...")
    detector.save('models/quantum_detector_basic.pkl')
    print("    Model saved to: models/quantum_detector_basic.pkl")
    
    print("\n" + "=" * 70)
    print(" Example completed successfully!")
    print("=" * 70)


def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Progress - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    print("    Training history saved to: results/training_history.png")
    plt.close()


def plot_risk_distribution(risk_scores, labels):
    """Plot distribution of risk scores for normal vs bad actors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal_scores = risk_scores[labels == 0]
    bad_scores = risk_scores[labels == 1]
    
    ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal Users', color='green', edgecolor='black')
    ax.hist(bad_scores, bins=30, alpha=0.6, label='Bad Actors', color='red', edgecolor='black')
    
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add threshold line at 0.5
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    plt.tight_layout()
    plt.savefig('results/risk_distribution.png', dpi=300, bbox_inches='tight')
    print("    Risk distribution saved to: results/risk_distribution.png")
    plt.close()


if __name__ == '__main__':
    # Create directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    main()
