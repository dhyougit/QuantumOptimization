"""
Evaluation Metrics

Utility functions for evaluating knowledge-enhanced models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from collections import defaultdict


def compute_perplexity(logits, targets, ignore_index=-100):
    """
    Compute perplexity from logits and targets.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        targets: Target token ids (batch_size, seq_len)
        ignore_index: Index to ignore (padding)
        
    Returns:
        Perplexity score
    """
    # Reshape
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    
    # Compute cross-entropy
    ce_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Compute perplexity
    perplexity = torch.exp(ce_loss)
    
    return perplexity.item()


def compute_accuracy(logits, targets, ignore_index=-100):
    """
    Compute token-level accuracy.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        targets: Target token ids (batch_size, seq_len)
        ignore_index: Index to ignore (padding)
        
    Returns:
        Accuracy score
    """
    predictions = logits.argmax(dim=-1)
    
    # Create mask for valid positions
    mask = targets != ignore_index
    
    # Compute accuracy
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_top_k_accuracy(logits, targets, k=5, ignore_index=-100):
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        targets: Target token ids (batch_size, seq_len)
        k: Top-k parameter
        ignore_index: Index to ignore
        
    Returns:
        Top-k accuracy score
    """
    # Get top-k predictions
    _, top_k_pred = torch.topk(logits, k=k, dim=-1)
    
    # Expand targets for comparison
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_pred)
    
    # Check if target is in top-k
    correct = (top_k_pred == targets_expanded).any(dim=-1)
    
    # Apply mask
    mask = targets != ignore_index
    correct = correct & mask
    
    # Compute accuracy
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_knowledge_retrieval_metrics(
    retrieved_knowledge,
    relevant_knowledge,
    k_values=[1, 3, 5, 10]
):
    """
    Compute retrieval metrics (Precision, Recall, F1).
    
    Args:
        retrieved_knowledge: List of retrieved knowledge IDs
        relevant_knowledge: List of relevant knowledge IDs
        k_values: List of k values for metrics@k
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    for k in k_values:
        retrieved_at_k = set(retrieved_knowledge[:k])
        relevant_set = set(relevant_knowledge)
        
        # Precision@k
        if len(retrieved_at_k) > 0:
            precision = len(retrieved_at_k & relevant_set) / len(retrieved_at_k)
        else:
            precision = 0.0
        
        # Recall@k
        if len(relevant_set) > 0:
            recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
        else:
            recall = 0.0
        
        # F1@k
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'f1@{k}'] = f1
    
    return metrics


def compute_knowledge_attribution_score(
    model_output,
    knowledge_embeddings,
    attention_weights
):
    """
    Compute how much the model attributes its output to external knowledge.
    
    Args:
        model_output: Model output representation
        knowledge_embeddings: Retrieved knowledge embeddings
        attention_weights: Attention weights over knowledge
        
    Returns:
        Attribution score
    """
    # Compute weighted knowledge representation
    weighted_knowledge = torch.matmul(
        attention_weights.unsqueeze(1),
        knowledge_embeddings
    ).squeeze(1)
    
    # Compute similarity
    similarity = F.cosine_similarity(
        model_output,
        weighted_knowledge,
        dim=-1
    )
    
    return similarity.mean().item()


def compute_factual_correctness(
    predictions: List[str],
    ground_truth: List[str],
    knowledge_base
):
    """
    Compute factual correctness by checking against knowledge base.
    
    Args:
        predictions: List of predicted texts
        ground_truth: List of ground truth texts
        knowledge_base: KnowledgeBase instance
        
    Returns:
        Factual correctness score
    """
    # This is a simplified version
    # In practice, you'd use more sophisticated fact-checking
    
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truth):
        # Simple string matching
        if pred.strip().lower() == truth.strip().lower():
            correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_diversity_metrics(sequences: List[List[int]]):
    """
    Compute diversity metrics for generated sequences.
    
    Args:
        sequences: List of generated sequences (token IDs)
        
    Returns:
        Dictionary of diversity metrics
    """
    # Compute n-gram diversity
    def get_ngrams(seq, n):
        return set(tuple(seq[i:i+n]) for i in range(len(seq) - n + 1))
    
    all_unigrams = set()
    all_bigrams = set()
    all_trigrams = set()
    total_tokens = 0
    
    for seq in sequences:
        all_unigrams.update(get_ngrams(seq, 1))
        all_bigrams.update(get_ngrams(seq, 2))
        all_trigrams.update(get_ngrams(seq, 3))
        total_tokens += len(seq)
    
    return {
        'unique_unigrams': len(all_unigrams),
        'unique_bigrams': len(all_bigrams),
        'unique_trigrams': len(all_trigrams),
        'unigram_diversity': len(all_unigrams) / max(total_tokens, 1),
        'bigram_diversity': len(all_bigrams) / max(total_tokens - 1, 1),
        'trigram_diversity': len(all_trigrams) / max(total_tokens - 2, 1)
    }


def compute_bleu_score(predictions: List[List[int]], references: List[List[int]], n=4):
    """
    Compute BLEU score for generated sequences.
    
    Simplified implementation for demonstration.
    """
    from collections import Counter
    
    def get_ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
    
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        scores_n = []
        
        for n_val in range(1, n + 1):
            pred_ngrams = Counter(get_ngrams(pred, n_val))
            ref_ngrams = Counter(get_ngrams(ref, n_val))
            
            # Compute matches
            matches = sum((pred_ngrams & ref_ngrams).values())
            total = max(sum(pred_ngrams.values()), 1)
            
            precision_n = matches / total
            scores_n.append(precision_n)
        
        # Geometric mean
        if all(s > 0 for s in scores_n):
            bleu = np.exp(np.mean(np.log(scores_n)))
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref) / max(len(pred), 1)))
        bleu = bp * bleu
        
        bleu_scores.append(bleu)
    
    return np.mean(bleu_scores)


class MetricsTracker:
    """
    Track and aggregate metrics during training/evaluation.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metric_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metric_dict.items():
            self.metrics[key].append(value)
    
    def get_average(self, metric_name: str):
        """Get average value for a metric."""
        if metric_name in self.metrics:
            return np.mean(self.metrics[metric_name])
        return None
    
    def get_all_averages(self):
        """Get averages for all metrics."""
        return {
            key: np.mean(values)
            for key, values in self.metrics.items()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
    
    def __repr__(self):
        avg_metrics = self.get_all_averages()
        return f"MetricsTracker({avg_metrics})"


def evaluate_model(model, dataloader, device, knowledge_base=None):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to run on
        knowledge_base: Optional knowledge base
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            knowledge_embeddings = batch.get('knowledge_embeddings')
            
            if knowledge_embeddings is not None:
                knowledge_embeddings = knowledge_embeddings.to(device)
            
            # Forward pass
            logits = model(
                input_ids,
                knowledge_embeddings=knowledge_embeddings,
                use_knowledge=knowledge_embeddings is not None
            )
            
            # Compute metrics
            metrics = {
                'perplexity': compute_perplexity(logits, target_ids),
                'accuracy': compute_accuracy(logits, target_ids),
                'top5_accuracy': compute_top_k_accuracy(logits, target_ids, k=5)
            }
            
            tracker.update(metrics)
    
    return tracker.get_all_averages()
