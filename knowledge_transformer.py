"""
Knowledge-Enhanced Transformer Model

Complete transformer model with integrated knowledge attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.knowledge_attention import KnowledgeEnhancedSelfAttention, KnowledgeAttention
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class KnowledgeEnhancedTransformerBlock(nn.Module):
    """
    Single transformer block enhanced with knowledge attention.
    """
    
    def __init__(self, d_model, n_heads, d_ff, n_knowledge_heads=None, dropout=0.1):
        super().__init__()
        
        # Knowledge-enhanced self-attention
        self.attention = KnowledgeEnhancedSelfAttention(
            d_model, n_heads, n_knowledge_heads, dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, knowledge_embeddings=None, mask=None, 
                knowledge_mask=None, use_knowledge=True):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            knowledge_embeddings: External knowledge embeddings
            mask: Self-attention mask
            knowledge_mask: Knowledge attention mask
            use_knowledge: Whether to use knowledge attention
        """
        # Attention with knowledge
        attn_out = self.attention(
            x, knowledge_embeddings, mask, knowledge_mask, use_knowledge
        )
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(attn_out)
        output = self.norm(attn_out + self.dropout(ff_out))
        
        return output


class KnowledgeEnhancedTransformer(nn.Module):
    """
    Complete transformer model with knowledge attention integration.
    
    This model can operate in two modes:
    1. Standard mode: Functions as a regular transformer
    2. Knowledge-enhanced mode: Uses external knowledge base for improved accuracy
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        n_knowledge_heads=None,
        max_len=5000,
        dropout=0.1,
        knowledge_base=None
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            n_knowledge_heads: Number of heads for knowledge attention
            max_len: Maximum sequence length
            dropout: Dropout probability
            knowledge_base: Optional external knowledge base
        """
        super().__init__()
        
        self.d_model = d_model
        self.knowledge_base = knowledge_base
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            KnowledgeEnhancedTransformerBlock(
                d_model, n_heads, d_ff, n_knowledge_heads, dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        input_ids,
        knowledge_embeddings=None,
        mask=None,
        knowledge_mask=None,
        use_knowledge=True,
        return_embeddings=False
    ):
        """
        Forward pass of the knowledge-enhanced transformer.
        
        Args:
            input_ids: Input token ids (batch_size, seq_len)
            knowledge_embeddings: Pre-computed knowledge embeddings
            mask: Attention mask
            knowledge_mask: Knowledge mask
            use_knowledge: Whether to use knowledge attention
            return_embeddings: Whether to return final embeddings
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            embeddings: Optional final embeddings if return_embeddings=True
        """
        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Retrieve knowledge if using knowledge base and no embeddings provided
        if use_knowledge and knowledge_embeddings is None and self.knowledge_base is not None:
            knowledge_embeddings = self.knowledge_base.retrieve_batch(input_ids)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, knowledge_embeddings, mask, knowledge_mask, use_knowledge)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        if return_embeddings:
            return logits, x
        return logits
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        use_knowledge=True
    ):
        """
        Generate text using the model.
        
        Args:
            input_ids: Initial input tokens (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            use_knowledge: Whether to use knowledge during generation
            
        Returns:
            generated_ids: Generated token ids
        """
        self.eval()
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for current sequence
                logits = self.forward(generated, use_knowledge=use_knowledge)
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply nucleus (top-p) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Break if EOS token (assuming token_id 0 is EOS)
                if (next_token == 0).all():
                    break
        
        return generated
    
    def get_attention_weights(self, input_ids, knowledge_embeddings=None, use_knowledge=True):
        """
        Get attention weights for visualization.
        
        Returns attention weights from all layers.
        """
        self.eval()
        
        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Retrieve knowledge if needed
        if use_knowledge and knowledge_embeddings is None and self.knowledge_base is not None:
            knowledge_embeddings = self.knowledge_base.retrieve_batch(input_ids)
        
        attention_weights_list = []
        
        with torch.no_grad():
            for layer in self.layers:
                x, attn_weights = layer.attention(
                    x, knowledge_embeddings, None, None, use_knowledge, return_attention=True
                )
                attention_weights_list.append(attn_weights)
                
                # Apply feed-forward
                ff_out = layer.feed_forward(x)
                x = layer.norm(x + layer.dropout(ff_out))
        
        return attention_weights_list


class KnowledgeEncoder(nn.Module):
    """
    Encoder for knowledge base entries.
    
    Converts raw knowledge into embeddings compatible with the transformer.
    """
    
    def __init__(self, input_dim, d_model, n_layers=2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(d_model))
            current_dim = d_model
        
        layers.append(nn.Linear(current_dim, d_model))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, knowledge_features):
        """
        Encode knowledge features into embeddings.
        
        Args:
            knowledge_features: Raw knowledge features (batch_size, n_knowledge, input_dim)
            
        Returns:
            knowledge_embeddings: Encoded embeddings (batch_size, n_knowledge, d_model)
        """
        return self.encoder(knowledge_features)
