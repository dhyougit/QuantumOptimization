"""
Quantum vs Classical Comparison

Comprehensive comparison between quantum-inspired and classical methods.
"""

import numpy as np
import time
from quantum_detector.detector import QuantumBadActorDetector
from classical_baseline.ml_models import ClassicalDetector, compare_models
from data.behavior_generator import generate_sample_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd


def benchmark_detector(detector, X_train, y_train, X_test, y_test, name):
    """Benchmark a single detector."""
    print(f"\nBenchmarking {name}...")
    
    # Training time
    start_time = time.time()
    if isinstance(detector, QuantumBadActorDetector):
        detector.fit(X_train, y_train, epochs=30, batch_size=32, verbose=False)
    else:
        detector.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Inference time
    start_time = time.time()
    predictions = detector.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    # Get probability scores
    if isinstance(detector, QuantumBadActorDetector):
        probas = detector.predict_risk(X_test)
    else:
        probas = detector.predict_proba(X_test)
    
    # Compute metrics
    metrics = {
        'name': name,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probas),
        'train_time_s': train_time,
        'inference_time_ms': inference_time
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Train Time: {metrics['train_time_s']:.2f}s")
    print(f"  Inference Time: {metrics['inference_time_ms']:.4f}ms/sample")
    
    return metrics


def main():
    print("=" * 80)
    print(" Quantum vs Classical: Comprehensive Comparison")
    print("=" * 80)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    n_users = 2000
    user_ids, features, labels = generate_sample_data(
        n_users=n_users,
        bad_actor_ratio=0.2,
        seed=42
    )
    
    print(f"   Total users: {n_users}")
    print(f"   Features: {features.shape[1]}")
    print(f"   Bad actors: {np.sum(labels == 1)} ({np.mean(labels)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize detectors
    print("\n2. Initializing detectors...")
    
    detectors = {
        # Quantum-inspired methods
        'Quantum VQC': QuantumBadActorDetector(
            n_qubits=8, method='vqc', n_layers=2, device='cpu'
        ),
        'Quantum Hybrid': QuantumBadActorDetector(
            n_qubits=6, method='hybrid', n_layers=2, device='cpu',
            hidden_dims=[64, 32]
        ),
        'Quantum Kernel SVM': QuantumBadActorDetector(
            n_qubits=8, method='kernel', device='cpu'
        ),
        
        # Classical methods
        'Random Forest': ClassicalDetector('random_forest', n_estimators=100),
        'XGBoost': ClassicalDetector('xgboost', n_estimators=100),
        'SVM (RBF)': ClassicalDetector('svm', kernel='rbf'),
        'Logistic Regression': ClassicalDetector('logistic'),
        'Neural Network': ClassicalDetector('mlp', hidden_layer_sizes=(64, 32)),
    }
    
    # Benchmark all detectors
    print("\n3. Running benchmarks...")
    print("   (This will take several minutes...)")
    
    results = []
    for name, detector in detectors.items():
        try:
            metrics = benchmark_detector(detector, X_train, y_train, X_test, y_test, name)
            results.append(metrics)
        except Exception as e:
            print(f"   Error with {name}: {e}")
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Display results
    print("\n4. Results Summary:")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    
    # Find best models
    print("\n5. Best Models:")
    print(f"   Best Accuracy: {df_results.loc[df_results['accuracy'].idxmax(), 'name']} "
          f"({df_results['accuracy'].max():.4f})")
    print(f"   Best F1 Score: {df_results.loc[df_results['f1_score'].idxmax(), 'name']} "
          f"({df_results['f1_score'].max():.4f})")
    print(f"   Fastest Training: {df_results.loc[df_results['train_time_s'].idxmin(), 'name']} "
          f"({df_results['train_time_s'].min():.2f}s)")
    print(f"   Fastest Inference: {df_results.loc[df_results['inference_time_ms'].idxmin(), 'name']} "
          f"({df_results['inference_time_ms'].min():.4f}ms)")
    
    # Calculate quantum advantage
    print("\n6. Quantum Advantage Analysis:")
    quantum_results = df_results[df_results['name'].str.contains('Quantum')]
    classical_results = df_results[~df_results['name'].str.contains('Quantum')]
    
    print(f"   Average Quantum F1: {quantum_results['f1_score'].mean():.4f}")
    print(f"   Average Classical F1: {classical_results['f1_score'].mean():.4f}")
    print(f"   Quantum Advantage: {(quantum_results['f1_score'].mean() / classical_results['f1_score'].mean() - 1) * 100:.2f}%")
    
    # Visualizations
    print("\n7. Generating visualizations...")
    plot_comparison(df_results)
    
    # Save results
    print("\n8. Saving results...")
    df_results.to_csv('results/comparison_results.csv', index=False)
    print("   Results saved to: results/comparison_results.csv")
    
    print("\n" + "=" * 80)
    print(" Comparison completed!")
    print("=" * 80)


def plot_comparison(df_results):
    """Create comprehensive comparison plots."""
    
    # Separate quantum and classical
    quantum_mask = df_results['name'].str.contains('Quantum')
    
    # 1. Performance metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    titles = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        sorted_df = df_results.sort_values(metric, ascending=True)
        colors = ['#FF6B6B' if 'Quantum' in name else '#4ECDC4' 
                 for name in sorted_df['name']]
        
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['name'])
        ax.set_xlabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_df[metric])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Quantum'),
        Patch(facecolor='#4ECDC4', label='Classical')
    ]
    axes[0, 0].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("   Performance comparison saved to: results/performance_comparison.png")
    plt.close()
    
    # 2. Speed comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training time
    sorted_df = df_results.sort_values('train_time_s', ascending=True)
    colors = ['#FF6B6B' if 'Quantum' in name else '#4ECDC4' 
             for name in sorted_df['name']]
    
    bars = ax1.barh(range(len(sorted_df)), sorted_df['train_time_s'], color=colors)
    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(sorted_df['name'])
    ax1.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Inference time
    sorted_df = df_results.sort_values('inference_time_ms', ascending=True)
    colors = ['#FF6B6B' if 'Quantum' in name else '#4ECDC4' 
             for name in sorted_df['name']]
    
    bars = ax2.barh(range(len(sorted_df)), sorted_df['inference_time_ms'], color=colors)
    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels(sorted_df['name'])
    ax2.set_xlabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/speed_comparison.png', dpi=300, bbox_inches='tight')
    print("   Speed comparison saved to: results/speed_comparison.png")
    plt.close()
    
    # 3. F1 vs Speed scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    quantum_df = df_results[df_results['name'].str.contains('Quantum')]
    classical_df = df_results[~df_results['name'].str.contains('Quantum')]
    
    ax.scatter(classical_df['inference_time_ms'], classical_df['f1_score'], 
              s=200, c='#4ECDC4', alpha=0.7, edgecolors='black', linewidth=2,
              label='Classical', marker='o')
    ax.scatter(quantum_df['inference_time_ms'], quantum_df['f1_score'], 
              s=200, c='#FF6B6B', alpha=0.7, edgecolors='black', linewidth=2,
              label='Quantum', marker='s')
    
    # Annotate points
    for _, row in df_results.iterrows():
        ax.annotate(row['name'], 
                   (row['inference_time_ms'], row['f1_score']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score vs Inference Speed', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/f1_vs_speed.png', dpi=300, bbox_inches='tight')
    print("   F1 vs Speed plot saved to: results/f1_vs_speed.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()
