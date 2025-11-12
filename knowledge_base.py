"""
Knowledge Base Management System

Handles storage, retrieval, and encoding of external knowledge for the transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import List, Dict, Optional, Union


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving external knowledge.
    
    Supports multiple retrieval methods:
    - Dense retrieval using embeddings
    - Sparse retrieval using keyword matching
    - Hybrid retrieval combining both approaches
    """
    
    def __init__(self, knowledge_dim=512, retrieval_method='dense'):
        """
        Args:
            knowledge_dim: Dimension of knowledge embeddings
            retrieval_method: 'dense', 'sparse', or 'hybrid'
        """
        self.knowledge_dim = knowledge_dim
        self.retrieval_method = retrieval_method
        
        self.knowledge_entries = []
        self.knowledge_embeddings = None
        self.knowledge_metadata = []
        
    def add_entry(self, text, embedding=None, metadata=None):
        """
        Add a knowledge entry to the database.
        
        Args:
            text: Text content of the knowledge
            embedding: Pre-computed embedding (optional)
            metadata: Additional metadata (dict)
        """
        entry = {
            'text': text,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        self.knowledge_entries.append(entry)
        
    def build_index(self, encoder_model=None):
        """
        Build the retrieval index for the knowledge base.
        
        Args:
            encoder_model: Model to encode knowledge entries
        """
        if encoder_model is not None:
            embeddings = []
            for entry in self.knowledge_entries:
                if entry['embedding'] is None:
                    # Encode the text
                    with torch.no_grad():
                        emb = encoder_model.encode(entry['text'])
                        entry['embedding'] = emb
                embeddings.append(entry['embedding'])
            
            self.knowledge_embeddings = torch.stack(embeddings)
        else:
            # Use pre-computed embeddings
            embeddings = [entry['embedding'] for entry in self.knowledge_entries 
                         if entry['embedding'] is not None]
            if embeddings:
                self.knowledge_embeddings = torch.stack(embeddings)
                
    def retrieve(self, query_embedding, top_k=10, threshold=0.0):
        """
        Retrieve top-k relevant knowledge entries.
        
        Args:
            query_embedding: Query embedding (d_model,)
            top_k: Number of entries to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            retrieved_embeddings: Embeddings of retrieved knowledge
            retrieved_indices: Indices of retrieved knowledge
            scores: Relevance scores
        """
        if self.knowledge_embeddings is None:
            raise ValueError("Knowledge base index not built. Call build_index() first.")
        
        # Normalize embeddings
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        knowledge_norm = F.normalize(self.knowledge_embeddings, p=2, dim=1)
        
        # Compute similarity scores
        scores = torch.matmul(query_norm, knowledge_norm.t()).squeeze(0)
        
        # Get top-k
        top_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        # Apply threshold
        mask = top_scores >= threshold
        top_scores = top_scores[mask]
        top_indices = top_indices[mask]
        
        retrieved_embeddings = self.knowledge_embeddings[top_indices]
        
        return retrieved_embeddings, top_indices, top_scores
    
    def retrieve_batch(self, query_embeddings, top_k=10):
        """
        Retrieve knowledge for a batch of queries.
        
        Args:
            query_embeddings: Query embeddings (batch_size, seq_len, d_model)
            top_k: Number of entries to retrieve per query
            
        Returns:
            retrieved_embeddings: (batch_size, top_k, d_model)
        """
        if self.knowledge_embeddings is None:
            # Return dummy embeddings if no knowledge available
            batch_size = query_embeddings.shape[0]
            return torch.zeros(batch_size, top_k, self.knowledge_dim, 
                             device=query_embeddings.device)
        
        batch_size, seq_len, d_model = query_embeddings.shape
        
        # Average query embeddings across sequence
        avg_query = query_embeddings.mean(dim=1)  # (batch_size, d_model)
        
        # Normalize
        query_norm = F.normalize(avg_query, p=2, dim=1)
        knowledge_norm = F.normalize(self.knowledge_embeddings, p=2, dim=1)
        
        # Compute similarities
        scores = torch.matmul(query_norm, knowledge_norm.t())
        # (batch_size, n_knowledge)
        
        # Get top-k for each query
        top_k = min(top_k, self.knowledge_embeddings.shape[0])
        _, top_indices = torch.topk(scores, k=top_k, dim=1)
        
        # Gather embeddings
        retrieved = self.knowledge_embeddings[top_indices]
        # (batch_size, top_k, d_model)
        
        return retrieved
    
    def get_entry_text(self, indices):
        """
        Get text content for given indices.
        
        Args:
            indices: Tensor or list of indices
            
        Returns:
            List of text entries
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        return [self.knowledge_entries[i]['text'] for i in indices]
    
    def save(self, filepath):
        """Save knowledge base to file."""
        data = {
            'knowledge_dim': self.knowledge_dim,
            'retrieval_method': self.retrieval_method,
            'entries': [
                {
                    'text': entry['text'],
                    'embedding': entry['embedding'].cpu().numpy().tolist() 
                                if entry['embedding'] is not None else None,
                    'metadata': entry['metadata']
                }
                for entry in self.knowledge_entries
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_file(cls, filepath):
        """Load knowledge base from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        kb = cls(
            knowledge_dim=data['knowledge_dim'],
            retrieval_method=data['retrieval_method']
        )
        
        for entry_data in data['entries']:
            embedding = torch.tensor(entry_data['embedding']) \
                       if entry_data['embedding'] is not None else None
            kb.add_entry(
                text=entry_data['text'],
                embedding=embedding,
                metadata=entry_data['metadata']
            )
        
        if kb.knowledge_entries and kb.knowledge_entries[0]['embedding'] is not None:
            kb.build_index()
        
        return kb
    
    @classmethod
    def from_dict_list(cls, entries, knowledge_dim=512):
        """
        Create knowledge base from a list of dictionaries.
        
        Args:
            entries: List of dicts with 'text' and optionally 'embedding' and 'metadata'
            knowledge_dim: Dimension of knowledge embeddings
        """
        kb = cls(knowledge_dim=knowledge_dim)
        
        for entry in entries:
            kb.add_entry(
                text=entry['text'],
                embedding=entry.get('embedding'),
                metadata=entry.get('metadata', {})
            )
        
        return kb


class KnowledgeRetriever(nn.Module):
    """
    Neural retriever for knowledge base queries.
    
    Learns to retrieve relevant knowledge given input context.
    """
    
    def __init__(self, d_model, knowledge_dim, n_heads=4):
        super().__init__()
        
        self.d_model = d_model
        self.knowledge_dim = knowledge_dim
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, knowledge_dim)
        )
        
        # Scoring network
        self.scorer = nn.Sequential(
            nn.Linear(knowledge_dim * 2, knowledge_dim),
            nn.ReLU(),
            nn.Linear(knowledge_dim, 1)
        )
        
    def forward(self, context_embedding, knowledge_embeddings, top_k=10):
        """
        Retrieve relevant knowledge using learned retrieval.
        
        Args:
            context_embedding: Context representation (batch_size, d_model)
            knowledge_embeddings: All knowledge embeddings (n_knowledge, knowledge_dim)
            top_k: Number of items to retrieve
            
        Returns:
            retrieved_embeddings: Top-k knowledge embeddings
            scores: Relevance scores
        """
        batch_size = context_embedding.shape[0]
        n_knowledge = knowledge_embeddings.shape[0]
        
        # Encode query
        query = self.query_encoder(context_embedding)  # (batch, knowledge_dim)
        
        # Expand for scoring
        query_expanded = query.unsqueeze(1).expand(-1, n_knowledge, -1)
        knowledge_expanded = knowledge_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate and score
        combined = torch.cat([query_expanded, knowledge_expanded], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (batch, n_knowledge)
        
        # Get top-k
        top_k = min(top_k, n_knowledge)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        
        # Gather top-k embeddings
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, top_k)
        retrieved = knowledge_embeddings[top_indices]
        
        return retrieved, top_scores


class SimpleKnowledgeEncoder(nn.Module):
    """
    Simple encoder for converting text to knowledge embeddings.
    
    In practice, you would use a pre-trained language model like BERT.
    This is a simplified version for demonstration.
    """
    
    def __init__(self, vocab_size, d_model, knowledge_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=2
        )
        self.projection = nn.Linear(d_model, knowledge_dim)
        
    def encode(self, input_ids):
        """
        Encode input tokens into knowledge embedding.
        
        Args:
            input_ids: Token IDs (seq_len,) or (batch, seq_len)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Embed and encode
        x = self.embedding(input_ids)
        x = self.encoder(x)
        
        # Pool and project
        pooled = x.mean(dim=1)
        knowledge_emb = self.projection(pooled)
        
        return knowledge_emb.squeeze(0) if input_ids.shape[0] == 1 else knowledge_emb
