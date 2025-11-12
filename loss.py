"""
Loss Functions for Knowledge-Augmented Training

Custom loss functions that incorporate knowledge consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeAugmentedLoss(nn.Module):
    """
    Combined loss function for knowledge-enhanced transformer.
    
    Combines:
    1. Standard language modeling loss (cross-entropy)
    2. Knowledge consistency loss (optional)
    """
    
    def __init__(self, knowledge_weight=0.5, use_knowledge=True, 
                 ignore_index=-100, label_smoothing=0.1):
        """
        Args:
            knowledge_weight: Weight for knowledge loss component
            use_knowledge: Whether to use knowledge consistency loss
            ignore_index: Index to ignore in loss computation (for padding)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.knowledge_weight = knowledge_weight
        self.use_knowledge = use_knowledge
        self.ignore_index = ignore_index
        
        # Language modeling loss
        self.lm_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(self, logits, targets, knowledge_embeddings=None, model=None):
        """
        Compute combined loss.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token ids (batch_size, seq_len)
            knowledge_embeddings: Knowledge embeddings used
            model: Model instance (for accessing internal representations)
            
        Returns:
            Dictionary with loss components
        """
        # Language modeling loss
        batch_size, seq_len, vocab_size = logits.shape
        lm_loss = self.lm_criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        
        loss_dict = {
            'lm_loss': lm_loss,
            'total_loss': lm_loss
        }
        
        # Knowledge consistency loss
        if self.use_knowledge and knowledge_embeddings is not None:
            knowledge_loss = self.compute_knowledge_loss(
                logits, targets, knowledge_embeddings
            )
            
            loss_dict['knowledge_loss'] = knowledge_loss
            loss_dict['total_loss'] = lm_loss + self.knowledge_weight * knowledge_loss
        
        return loss_dict
    
    def compute_knowledge_loss(self, logits, targets, knowledge_embeddings):
        """
        Compute knowledge consistency loss.
        
        Encourages the model to be consistent with the retrieved knowledge.
        This is a simplified version - in practice, you might want more
        sophisticated consistency measures.
        """
        # Simple consistency loss: encourage model to attend to knowledge
        # This is a placeholder - implement based on your specific needs
        
        # For example: maximize similarity between predictions and knowledge
        # Here we just use a small regularization term
        knowledge_norm = torch.norm(knowledge_embeddings, p=2, dim=-1).mean()
        
        # Regularize to prevent degenerate solutions
        knowledge_loss = -torch.log(knowledge_norm + 1e-8)
        
        return knowledge_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better knowledge representations.
    
    Encourages the model to distinguish between relevant and irrelevant knowledge.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query_embeddings, positive_knowledge, negative_knowledge):
        """
        Args:
            query_embeddings: Query representations (batch_size, d_model)
            positive_knowledge: Relevant knowledge (batch_size, d_model)
            negative_knowledge: Irrelevant knowledge (batch_size, n_neg, d_model)
        """
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        pos_norm = F.normalize(positive_knowledge, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(query_norm * pos_norm, dim=1) / self.temperature
        
        # Negative similarities
        if negative_knowledge is not None:
            neg_norm = F.normalize(negative_knowledge, p=2, dim=2)
            neg_sim = torch.matmul(
                query_norm.unsqueeze(1),
                neg_norm.transpose(1, 2)
            ).squeeze(1) / self.temperature
            
            # Combine positive and negative
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            logits = pos_sim.unsqueeze(1)
        
        # Labels: positive is always first
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for transferring knowledge from a teacher model.
    """
    
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            targets: Ground truth labels
        """
        # Distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Student loss
        student_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return {
            'distill_loss': distill_loss,
            'student_loss': student_loss,
            'total_loss': total_loss
        }


class FactualConsistencyLoss(nn.Module):
    """
    Loss that encourages factual consistency with knowledge base.
    
    Useful for fact-checking and knowledge-grounded generation tasks.
    """
    
    def __init__(self, consistency_weight=1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(self, predictions, knowledge_facts, fact_labels):
        """
        Args:
            predictions: Model predictions (batch_size, seq_len, vocab_size)
            knowledge_facts: Embeddings of factual statements (batch_size, n_facts, d_model)
            fact_labels: Binary labels indicating if facts are consistent (batch_size, n_facts)
        """
        # Compute similarity between predictions and facts
        # This is a simplified version
        
        # Average prediction embeddings
        pred_mean = predictions.mean(dim=1)  # (batch_size, vocab_size)
        
        # Convert to common space (simplified)
        # In practice, you'd project both to the same embedding space
        
        # Binary cross-entropy for consistency
        # Placeholder implementation
        loss = torch.tensor(0.0, device=predictions.device)
        
        return loss
