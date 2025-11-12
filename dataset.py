"""
Dataset for Knowledge-Augmented Training

Handles data loading with knowledge annotations.
"""

import torch
from torch.utils.data import Dataset
import json
import random


class KnowledgeAugmentedDataset(Dataset):
    """
    Dataset that pairs training examples with relevant knowledge.
    """
    
    def __init__(self, data_path, knowledge_base, split='train', 
                 val_split=0.1, max_length=512, seed=42):
        """
        Args:
            data_path: Path to training data (JSONL format)
            knowledge_base: KnowledgeBase instance
            split: 'train' or 'val'
            val_split: Fraction of data to use for validation
            max_length: Maximum sequence length
            seed: Random seed for split
        """
        self.knowledge_base = knowledge_base
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            all_data = [json.loads(line) for line in f]
        
        # Split data
        random.seed(seed)
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * (1 - val_split))
        
        if split == 'train':
            self.data = all_data[:split_idx]
        else:
            self.data = all_data[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get input and target sequences
        input_ids = torch.tensor(item['input_ids'][:self.max_length])
        target_ids = torch.tensor(item['target_ids'][:self.max_length])
        
        # Retrieve relevant knowledge
        knowledge_query = item.get('knowledge_query', input_ids)
        if isinstance(knowledge_query, list):
            knowledge_query = torch.tensor(knowledge_query)
        
        # Get knowledge embeddings
        # For simplicity, we'll use a dummy embedding here
        # In practice, you'd encode the knowledge_query properly
        if self.knowledge_base is not None and len(self.knowledge_base.knowledge_entries) > 0:
            # Create a dummy query embedding (mean of input embeddings)
            dummy_query = torch.randn(512)  # d_model=512
            knowledge_embeddings, _, _ = self.knowledge_base.retrieve(
                dummy_query, top_k=10
            )
        else:
            knowledge_embeddings = None
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'knowledge_embeddings': knowledge_embeddings
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and knowledge.
    """
    # Find max length in batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pad sequences
    input_ids = []
    target_ids = []
    knowledge_embeddings_list = []
    
    for item in batch:
        # Pad input
        input_pad_len = max_len - item['input_ids'].size(0)
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(input_pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Pad target
        target_pad_len = max_len - item['target_ids'].size(0)
        padded_target = torch.cat([
            item['target_ids'],
            torch.zeros(target_pad_len, dtype=torch.long)
        ])
        target_ids.append(padded_target)
        
        # Knowledge embeddings
        if item['knowledge_embeddings'] is not None:
            knowledge_embeddings_list.append(item['knowledge_embeddings'])
    
    # Stack
    input_ids = torch.stack(input_ids)
    target_ids = torch.stack(target_ids)
    
    # Handle knowledge embeddings
    if knowledge_embeddings_list:
        # Find max number of knowledge items
        max_k = max(k.size(0) for k in knowledge_embeddings_list)
        k_dim = knowledge_embeddings_list[0].size(1)
        
        # Pad knowledge embeddings
        padded_knowledge = []
        for k_emb in knowledge_embeddings_list:
            pad_len = max_k - k_emb.size(0)
            if pad_len > 0:
                padding = torch.zeros(pad_len, k_dim)
                k_emb = torch.cat([k_emb, padding], dim=0)
            padded_knowledge.append(k_emb)
        
        knowledge_embeddings = torch.stack(padded_knowledge)
    else:
        knowledge_embeddings = None
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'knowledge_embeddings': knowledge_embeddings
    }
