"""
Visualization Utilities

Tools for visualizing attention patterns and knowledge retrieval.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional


def visualize_attention_heatmap(
    attention_weights,
    tokens=None,
    knowledge_items=None,
    title="Attention Heatmap",
    figsize=(10, 8),
    save_path=None
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weight matrix (seq_len, seq_len) or (seq_len, n_knowledge)
        tokens: Optional list of token strings
        knowledge_items: Optional list of knowledge item descriptions
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        cmap='YlOrRd',
        cbar=True,
        square=True,
        xticklabels=knowledge_items if knowledge_items else False,
        yticklabels=tokens if tokens else False,
        linewidths=0.5
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Keys' if knowledge_items is None else 'Knowledge Items', fontsize=12)
    plt.ylabel('Queries (Tokens)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_multi_head_attention(
    attention_weights,
    n_heads=8,
    tokens=None,
    title="Multi-Head Attention",
    figsize=(15, 10),
    save_path=None
):
    """
    Visualize attention weights from multiple heads.
    
    Args:
        attention_weights: Attention weights (n_heads, seq_len, seq_len)
        n_heads: Number of attention heads
        tokens: Optional list of token strings
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(np.sqrt(n_heads)))
    n_cols = int(np.ceil(n_heads / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        
        sns.heatmap(
            attention_weights[head_idx],
            cmap='YlOrRd',
            cbar=True,
            square=True,
            xticklabels=tokens if tokens and head_idx >= n_heads - n_cols else False,
            yticklabels=tokens if tokens and head_idx % n_cols == 0 else False,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_title(f'Head {head_idx + 1}', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_knowledge_retrieval(
    query_text,
    retrieved_items,
    scores,
    top_k=5,
    figsize=(12, 6),
    save_path=None
):
    """
    Visualize top-k retrieved knowledge items with scores.
    
    Args:
        query_text: Query text
        retrieved_items: List of retrieved knowledge text
        scores: Retrieval scores
        top_k: Number of items to visualize
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()
    
    # Take top-k
    top_k = min(top_k, len(retrieved_items))
    retrieved_items = retrieved_items[:top_k]
    scores = scores[:top_k]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    y_pos = np.arange(top_k)
    colors = plt.cm.viridis(scores / scores.max())
    
    bars = ax.barh(y_pos, scores, color=colors)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f' {score:.4f}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Item {i+1}' for i in range(top_k)])
    ax.invert_yaxis()
    ax.set_xlabel('Relevance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top-{top_k} Retrieved Knowledge\nQuery: "{query_text}"', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add knowledge text as annotations
    for i, text in enumerate(retrieved_items):
        # Truncate long text
        display_text = text[:80] + '...' if len(text) > 80 else text
        ax.text(
            -0.02,
            i,
            display_text,
            ha='right',
            va='center',
            fontsize=9,
            transform=ax.get_yaxis_transform()
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_training_curves(
    train_losses,
    val_losses,
    train_accuracies=None,
    val_accuracies=None,
    figsize=(14, 5),
    save_path=None
):
    """
    Visualize training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: Optional list of training accuracies
        val_accuracies: Optional list of validation accuracies
        figsize: Figure size
        save_path: Optional path to save figure
    """
    n_plots = 2 if train_accuracies is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax1 = axes[0]
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies if provided
    if train_accuracies is not None and n_plots > 1:
        ax2 = axes[1]
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_knowledge_attention_flow(
    self_attention,
    knowledge_attention,
    tokens,
    layer_idx,
    figsize=(16, 6),
    save_path=None
):
    """
    Visualize both self-attention and knowledge attention side by side.
    
    Args:
        self_attention: Self-attention weights (seq_len, seq_len)
        knowledge_attention: Knowledge attention weights (seq_len, n_knowledge)
        tokens: List of token strings
        layer_idx: Layer index for title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Self-attention
    if isinstance(self_attention, torch.Tensor):
        self_attention = self_attention.cpu().detach().numpy()
    
    sns.heatmap(
        self_attention,
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=tokens,
        yticklabels=tokens,
        ax=ax1
    )
    ax1.set_title(f'Layer {layer_idx}: Self-Attention', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Keys (Tokens)', fontsize=10)
    ax1.set_ylabel('Queries (Tokens)', fontsize=10)
    
    # Knowledge attention
    if isinstance(knowledge_attention, torch.Tensor):
        knowledge_attention = knowledge_attention.cpu().detach().numpy()
    
    sns.heatmap(
        knowledge_attention,
        cmap='Reds',
        cbar=True,
        xticklabels=False,
        yticklabels=tokens,
        ax=ax2
    )
    ax2.set_title(f'Layer {layer_idx}: Knowledge Attention', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Knowledge Items', fontsize=10)
    ax2.set_ylabel('Queries (Tokens)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_attention_animation(
    attention_weights_per_layer,
    tokens,
    save_path='attention_animation.gif',
    duration=1000
):
    """
    Create an animated visualization of attention across layers.
    
    Requires: imageio
    
    Args:
        attention_weights_per_layer: List of attention weights, one per layer
        tokens: List of token strings
        save_path: Path to save GIF
        duration: Duration per frame in milliseconds
    """
    try:
        import imageio
        from PIL import Image
        import io
    except ImportError:
        print("Please install imageio and pillow: pip install imageio pillow")
        return
    
    images = []
    
    for layer_idx, attn_weights in enumerate(attention_weights_per_layer):
        # Create frame
        plt.figure(figsize=(10, 8))
        
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.cpu().detach().numpy()
        
        sns.heatmap(
            attn_weights,
            cmap='YlOrRd',
            cbar=True,
            square=True,
            xticklabels=tokens,
            yticklabels=tokens
        )
        
        plt.title(f'Layer {layer_idx + 1} Attention', fontsize=14, fontweight='bold')
        plt.xlabel('Keys', fontsize=12)
        plt.ylabel('Queries', fontsize=12)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close()
        buf.close()
    
    # Save as GIF
    imageio.mimsave(save_path, images, duration=duration)
    print(f"Animation saved to {save_path}")
