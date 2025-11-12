"""
Knowledge Attention Mechanism for Transformer Models

This module implements a specialized attention mechanism that allows
transformers to attend to external knowledge bases during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KnowledgeAttention(nn.Module):
    """
    Knowledge Attention layer that attends to external knowledge base.
    
    This mechanism computes attention between input queries and knowledge base
    entries, allowing the model to dynamically retrieve and integrate relevant
    external knowledge.
    """
    
    def __init__(self, d_model, n_heads, knowledge_dim=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            knowledge_dim: Dimension of knowledge embeddings (default: d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.knowledge_dim = knowledge_dim or d_model
        
        # Query projection for input
        self.query_proj = nn.Linear(d_model, d_model)
        
        # Key and value projections for knowledge
        self.key_proj = nn.Linear(self.knowledge_dim, d_model)
        self.value_proj = nn.Linear(self.knowledge_dim, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, knowledge_embeddings, knowledge_mask=None, return_attention=False):
        """
        Forward pass of knowledge attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            knowledge_embeddings: Knowledge base embeddings (batch_size, n_knowledge, knowledge_dim)
            knowledge_mask: Optional mask for knowledge entries (batch_size, n_knowledge)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attended output (batch_size, seq_len, d_model)
            attention_weights: Optional attention weights if return_attention=True
        """
        batch_size, seq_len, _ = x.shape
        n_knowledge = knowledge_embeddings.shape[1]
        
        # Project queries, keys, values
        Q = self.query_proj(x)  # (batch, seq_len, d_model)
        K = self.key_proj(knowledge_embeddings)  # (batch, n_knowledge, d_model)
        V = self.value_proj(knowledge_embeddings)  # (batch, n_knowledge, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_knowledge, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_knowledge, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, n_heads, seq_len, n_knowledge)
        
        # Apply knowledge mask if provided
        if knowledge_mask is not None:
            # Expand mask for heads and seq_len
            mask = knowledge_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_knowledge)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # (batch, n_heads, seq_len, head_dim)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        if return_attention:
            # Average attention across heads for visualization
            avg_attention = attention_weights.mean(dim=1)
            return output, avg_attention
        
        return output


class GatedKnowledgeFusion(nn.Module):
    """
    Gated fusion mechanism to combine input representations with knowledge.
    
    This module learns to dynamically balance between the original input
    and the knowledge-augmented representations.
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x, knowledge_context):
        """
        Args:
            x: Original input (batch_size, seq_len, d_model)
            knowledge_context: Knowledge context (batch_size, seq_len, d_model)
            
        Returns:
            Fused representation (batch_size, seq_len, d_model)
        """
        # Concatenate input and knowledge
        combined = torch.cat([x, knowledge_context], dim=-1)
        
        # Compute gate values
        gate_values = torch.sigmoid(self.gate(combined))
        
        # Gated fusion
        output = gate_values * knowledge_context + (1 - gate_values) * x
        
        return output


class KnowledgeEnhancedSelfAttention(nn.Module):
    """
    Self-attention layer enhanced with knowledge attention.
    
    Combines standard self-attention with knowledge attention to allow
    the model to attend to both input context and external knowledge.
    """
    
    def __init__(self, d_model, n_heads, n_knowledge_heads=None, dropout=0.1):
        super().__init__()
        
        n_knowledge_heads = n_knowledge_heads or (n_heads // 2)
        
        # Standard self-attention
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Knowledge attention
        self.knowledge_attention = KnowledgeAttention(
            d_model, n_knowledge_heads, dropout=dropout
        )
        
        # Fusion mechanism
        self.fusion = GatedKnowledgeFusion(d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, knowledge_embeddings=None, mask=None, 
                knowledge_mask=None, use_knowledge=True, return_attention=False):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            knowledge_embeddings: Optional knowledge embeddings
            mask: Self-attention mask
            knowledge_mask: Knowledge attention mask
            use_knowledge: Whether to use knowledge attention
            return_attention: Whether to return attention weights
            
        Returns:
            output: Enhanced representation
            attention_weights: Optional attention weights dict
        """
        attention_dict = {}
        
        # Self-attention
        attn_output, self_attn_weights = self.self_attention(
            x, x, x, attn_mask=mask, need_weights=return_attention
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        if return_attention:
            attention_dict['self_attention'] = self_attn_weights
        
        # Knowledge attention (if enabled and knowledge is provided)
        if use_knowledge and knowledge_embeddings is not None:
            if return_attention:
                k_context, k_attn_weights = self.knowledge_attention(
                    x, knowledge_embeddings, knowledge_mask, return_attention=True
                )
                attention_dict['knowledge_attention'] = k_attn_weights
            else:
                k_context = self.knowledge_attention(
                    x, knowledge_embeddings, knowledge_mask
                )
            
            # Fuse with knowledge
            x = self.fusion(x, k_context)
            x = self.norm2(x)
        
        if return_attention:
            return x, attention_dict
        return x


class SparseKnowledgeAttention(nn.Module):
    """
    Sparse knowledge attention that only attends to top-k relevant knowledge items.
    
    More efficient for large knowledge bases by using sparse attention.
    """
    
    def __init__(self, d_model, n_heads, top_k=10, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, knowledge_embeddings, return_attention=False):
        """
        Sparse attention to top-k knowledge items.
        """
        batch_size, seq_len, _ = x.shape
        n_knowledge = knowledge_embeddings.shape[1]
        
        Q = self.query_proj(x)
        K = self.key_proj(knowledge_embeddings)
        V = self.value_proj(knowledge_embeddings)
        
        # Reshape for attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_knowledge, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_knowledge, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Select top-k scores
        top_k = min(self.top_k, n_knowledge)
        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)
        
        # Create sparse attention weights
        sparse_weights = F.softmax(topk_scores, dim=-1)
        sparse_weights = self.dropout(sparse_weights)
        
        # Gather top-k values
        batch_idx = torch.arange(batch_size).view(-1, 1, 1, 1).expand_as(topk_indices)
        head_idx = torch.arange(self.n_heads).view(1, -1, 1, 1).expand_as(topk_indices)
        
        # Select values using gathered indices
        selected_V = V[batch_idx, head_idx, topk_indices, :]
        
        # Apply attention
        context = torch.matmul(sparse_weights.unsqueeze(-2), selected_V).squeeze(-2)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        if return_attention:
            return output, (topk_indices, sparse_weights)
        return output
