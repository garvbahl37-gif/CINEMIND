"""
Loss Functions for Two-Tower Training
======================================
Implements production-grade contrastive losses:
- InfoNCE (in-batch negatives)
- Sampled Softmax
- Hard Negative Mining Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with in-batch negatives.
    
    For a batch of (user, item) pairs, treats all other items
    in the batch as negatives. This is efficient and effective
    for large batch training.
    
    Loss = -log(exp(sim(u, i+)) / sum(exp(sim(u, i))))
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            user_embeddings: (batch_size, dim) normalized user vectors
            item_embeddings: (batch_size, dim) normalized item vectors
            labels: (batch_size,) indices of positive items (default: diagonal)
            
        Returns:
            Scalar loss value
        """
        batch_size = user_embeddings.shape[0]
        
        # Compute similarity matrix
        # (batch_size, batch_size) - each user against all items
        logits = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature
        
        # Default labels: positive pair is on diagonal
        if labels is None:
            labels = torch.arange(batch_size, device=logits.device)
        
        # Cross-entropy loss (softmax over items)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax loss for efficient training with large item catalogs.
    
    Instead of computing softmax over all items, samples a subset
    of negative items per batch. More memory efficient than full softmax.
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        num_negatives: int = 100
    ):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        
    def forward(
        self,
        user_embeddings: torch.Tensor,
        positive_item_embeddings: torch.Tensor,
        negative_item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sampled softmax loss.
        
        Args:
            user_embeddings: (batch_size, dim)
            positive_item_embeddings: (batch_size, dim)
            negative_item_embeddings: (batch_size, num_neg, dim) or (num_neg, dim)
            
        Returns:
            Scalar loss value
        """
        batch_size = user_embeddings.shape[0]
        
        # Positive similarities
        pos_sim = (user_embeddings * positive_item_embeddings).sum(dim=-1) / self.temperature
        pos_sim = pos_sim.unsqueeze(1)  # (batch, 1)
        
        # Negative similarities
        if negative_item_embeddings.dim() == 2:
            # Shared negatives across batch
            neg_sim = torch.matmul(
                user_embeddings, negative_item_embeddings.T
            ) / self.temperature  # (batch, num_neg)
        else:
            # Per-user negatives
            neg_sim = torch.bmm(
                user_embeddings.unsqueeze(1),
                negative_item_embeddings.transpose(1, 2)
            ).squeeze(1) / self.temperature  # (batch, num_neg)
        
        # Concatenate positive and negative logits
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (batch, 1 + num_neg)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class HardNegativeLoss(nn.Module):
    """
    Hard Negative Mining Loss.
    
    Combines standard InfoNCE with additional hard negatives
    mined from a pre-built index or provided externally.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 0.5,
        margin: float = 0.0
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin
        self.info_nce = InfoNCELoss(temperature)
        
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        hard_negative_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss with hard negatives.
        
        Args:
            user_embeddings: (batch_size, dim)
            item_embeddings: (batch_size, dim) positive items
            hard_negative_embeddings: (batch_size, num_hard_neg, dim) hard negatives
            
        Returns:
            Scalar loss value
        """
        # Standard InfoNCE loss
        base_loss = self.info_nce(user_embeddings, item_embeddings)
        
        if hard_negative_embeddings is None:
            return base_loss
        
        # Hard negative loss
        batch_size = user_embeddings.shape[0]
        
        # Positive similarities
        pos_sim = (user_embeddings * item_embeddings).sum(dim=-1)  # (batch,)
        
        # Hard negative similarities
        hard_neg_sim = torch.bmm(
            user_embeddings.unsqueeze(1),
            hard_negative_embeddings.transpose(1, 2)
        ).squeeze(1)  # (batch, num_hard_neg)
        
        # Margin loss: want pos_sim > hard_neg_sim + margin
        margin_loss = F.relu(hard_neg_sim - pos_sim.unsqueeze(1) + self.margin)
        hard_loss = margin_loss.mean()
        
        # Combined loss
        total_loss = base_loss + self.hard_negative_weight * hard_loss
        
        return total_loss


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss.
    
    Pairwise ranking loss that maximizes the difference between
    positive and negative item scores.
    
    Loss = -log(sigmoid(pos_score - neg_score))
    """
    
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        user_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            user_embeddings: (batch_size, dim)
            positive_embeddings: (batch_size, dim)
            negative_embeddings: (batch_size, dim)
            
        Returns:
            Scalar loss value
        """
        pos_scores = (user_embeddings * positive_embeddings).sum(dim=-1)
        neg_scores = (user_embeddings * negative_embeddings).sum(dim=-1)
        
        loss = -F.logsigmoid(pos_scores - neg_scores - self.margin).mean()
        
        return loss


def get_loss_function(
    loss_type: str = 'infonce',
    temperature: float = 0.07,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: 'infonce', 'sampled_softmax', 'hard_negative', 'bpr'
        temperature: Temperature for softmax-based losses
        **kwargs: Additional arguments for specific losses
        
    Returns:
        Loss function module
    """
    loss_map = {
        'infonce': InfoNCELoss,
        'sampled_softmax': SampledSoftmaxLoss,
        'hard_negative': HardNegativeLoss,
        'bpr': BPRLoss,
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")
    
    loss_class = loss_map[loss_type]
    
    if loss_type in ['infonce', 'sampled_softmax', 'hard_negative']:
        return loss_class(temperature=temperature, **kwargs)
    else:
        return loss_class(**kwargs)
