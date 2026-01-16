"""
Production-Grade Two-Tower Model
=================================
Deep user and item towers for candidate retrieval.

Architecture:
- User Tower: ID embedding + behavior features + attention
- Item Tower: ID embedding + genre embedding + text embedding
- L2 normalized output for cosine similarity
- Support for mixed precision and gradient checkpointing
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNormMLP(nn.Module):
    """MLP block with layer normalization and residual connection."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.layer_norm(x + residual)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for sequence modeling."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        out = self.dropout(out)
        
        return self.layer_norm(out + residual)


class UserTower(nn.Module):
    """
    Deep User Tower for two-tower retrieval.
    
    Processes:
    - User ID embedding
    - User behavior features (rating stats, activity patterns)
    - Optional: sequence of recent items (with attention)
    """
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        num_behavior_features: int = 0,
        dropout: float = 0.1,
        use_attention: bool = False,
        num_attention_heads: int = 4
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_behavior_features = num_behavior_features
        self.use_attention = use_attention
        
        # User ID embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        
        # Behavior feature projection
        if num_behavior_features > 0:
            self.behavior_proj = nn.Linear(num_behavior_features, embedding_dim)
        
        # Attention for sequence modeling (optional)
        if use_attention:
            self.attention = MultiHeadSelfAttention(
                embedding_dim, num_attention_heads, dropout
            )
        
        # Deep MLP layers
        input_dim = embedding_dim * (2 if num_behavior_features > 0 else 1)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(LayerNormMLP(prev_dim, hidden_dim * 2, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(prev_dim, output_dim)
        
    def forward(
        self,
        user_ids: torch.Tensor,
        behavior_features: Optional[torch.Tensor] = None,
        item_sequence: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: (batch_size,) user indices
            behavior_features: (batch_size, num_features) user behavior features
            item_sequence: (batch_size, seq_len, dim) recent item embeddings (optional)
            sequence_mask: (batch_size, seq_len) mask for sequence (optional)
            
        Returns:
            (batch_size, output_dim) L2-normalized user embeddings
        """
        # User ID embedding
        user_emb = self.user_embedding(user_ids)  # (B, D)
        
        # Combine with behavior features if available
        if behavior_features is not None and self.num_behavior_features > 0:
            behavior_emb = self.behavior_proj(behavior_features)
            combined = torch.cat([user_emb, behavior_emb], dim=-1)
        else:
            combined = user_emb
        
        # Apply attention over item sequence if provided
        if self.use_attention and item_sequence is not None:
            # Add user embedding as first token
            seq_with_user = torch.cat([
                user_emb.unsqueeze(1), 
                item_sequence
            ], dim=1)
            attended = self.attention(seq_with_user, sequence_mask)
            # Take first token (user) output
            user_attended = attended[:, 0, :]
            combined = torch.cat([combined, user_attended], dim=-1)
        
        # Deep MLP
        hidden = self.mlp(combined)
        
        # Project to output dimension
        output = self.output_proj(hidden)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class ItemTower(nn.Module):
    """
    Deep Item Tower for two-tower retrieval.
    
    Processes:
    - Item ID embedding
    - Genre multi-hot embedding
    - Text embedding (from title/tags)
    - Item statistics features
    """
    
    def __init__(
        self,
        num_items: int,
        num_genres: int = 20,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        text_embedding_dim: int = 0,
        num_item_features: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.text_embedding_dim = text_embedding_dim
        self.num_item_features = num_item_features
        
        # Item ID embedding
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Genre embedding (multi-hot -> dense)
        self.num_genres = num_genres
        if num_genres > 0:
            self.genre_embedding = nn.Linear(num_genres, embedding_dim // 2)
        
        # Text embedding projection
        if text_embedding_dim > 0:
            self.text_proj = nn.Linear(text_embedding_dim, embedding_dim)
        
        # Item feature projection
        if num_item_features > 0:
            self.item_feat_proj = nn.Linear(num_item_features, embedding_dim // 2)
        
        # Calculate input dimension for MLP
        input_dim = embedding_dim  # ID embedding
        if num_genres > 0:
            input_dim += embedding_dim // 2  # Genre embedding
        if text_embedding_dim > 0:
            input_dim += embedding_dim  # Text embedding
        if num_item_features > 0:
            input_dim += embedding_dim // 2  # Item features
        
        # Deep MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(LayerNormMLP(prev_dim, hidden_dim * 2, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final projection
        self.output_proj = nn.Linear(prev_dim, output_dim)
        
    def forward(
        self,
        item_ids: torch.Tensor,
        genre_vectors: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            item_ids: (batch_size,) item indices
            genre_vectors: (batch_size, num_genres) multi-hot genre vectors
            text_embeddings: (batch_size, text_dim) pre-computed text embeddings
            item_features: (batch_size, num_features) item statistics
            
        Returns:
            (batch_size, output_dim) L2-normalized item embeddings
        """
        # Item ID embedding
        item_emb = self.item_embedding(item_ids)  # (B, D)
        
        embeddings = [item_emb]
        
        # Genre embedding
        if genre_vectors is not None and self.num_genres > 0:
            genre_emb = self.genre_embedding(genre_vectors.float())
            embeddings.append(genre_emb)
        
        # Text embedding
        if text_embeddings is not None and self.text_embedding_dim > 0:
            text_emb = self.text_proj(text_embeddings)
            embeddings.append(text_emb)
        
        # Item features
        if item_features is not None and self.num_item_features > 0:
            feat_emb = self.item_feat_proj(item_features)
            embeddings.append(feat_emb)
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Deep MLP
        hidden = self.mlp(combined)
        
        # Project to output dimension
        output = self.output_proj(hidden)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class TwoTowerModel(nn.Module):
    """
    Complete Two-Tower model for candidate retrieval.
    
    Features:
    - Separate user and item towers
    - Dot-product similarity with temperature scaling
    - Support for in-batch negatives
    - Mixed precision training support
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int = 20,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        text_embedding_dim: int = 0,
        num_user_features: int = 0,
        num_item_features: int = 0,
        dropout: float = 0.1,
        temperature: float = 0.07,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.temperature = temperature
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # User tower
        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_behavior_features=num_user_features,
            dropout=dropout
        )
        
        # Item tower
        self.item_tower = ItemTower(
            num_items=num_items,
            num_genres=num_genres,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            text_embedding_dim=text_embedding_dim,
            num_item_features=num_item_features,
            dropout=dropout
        )
        
        # Store config for inference
        self.config = {
            'num_users': num_users,
            'num_items': num_items,
            'num_genres': num_genres,
            'embedding_dim': embedding_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'text_embedding_dim': text_embedding_dim,
            'num_user_features': num_user_features,
            'num_item_features': num_item_features,
        }
        
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_genre_vectors: Optional[torch.Tensor] = None,
        item_text_embeddings: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing user and item embeddings.
        
        Returns:
            user_embeddings: (batch_size, output_dim)
            item_embeddings: (batch_size, output_dim)
            logits: (batch_size,) dot product similarities
        """
        # User tower
        if self.use_gradient_checkpointing and self.training:
            user_emb = checkpoint(
                self.user_tower,
                user_ids,
                user_features,
                use_reentrant=False
            )
        else:
            user_emb = self.user_tower(user_ids, user_features)
        
        # Item tower  
        if self.use_gradient_checkpointing and self.training:
            item_emb = checkpoint(
                self.item_tower,
                item_ids,
                item_genre_vectors,
                item_text_embeddings,
                item_features,
                use_reentrant=False
            )
        else:
            item_emb = self.item_tower(
                item_ids, 
                item_genre_vectors,
                item_text_embeddings,
                item_features
            )
        
        # Dot product similarity (already L2 normalized = cosine similarity)
        logits = (user_emb * item_emb).sum(dim=-1) / self.temperature
        
        return user_emb, item_emb, logits
    
    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get user embeddings for batch of users."""
        return self.user_tower(user_ids, user_features)
    
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_genre_vectors: Optional[torch.Tensor] = None,
        item_text_embeddings: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get item embeddings for batch of items."""
        return self.item_tower(
            item_ids,
            item_genre_vectors,
            item_text_embeddings,
            item_features
        )
    
    def compute_in_batch_logits(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits for all user-item pairs in batch (in-batch negatives).
        
        Args:
            user_emb: (batch_size, dim)
            item_emb: (batch_size, dim)
            
        Returns:
            logits: (batch_size, batch_size) similarity matrix
        """
        # All-pairs dot product
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature
        return logits


def create_model(
    num_users: int,
    num_items: int,
    num_genres: int = 20,
    config: Dict = None
) -> TwoTowerModel:
    """Factory function to create model with config."""
    default_config = {
        'embedding_dim': 128,
        'hidden_dims': [256, 128],
        'output_dim': 64,
        'dropout': 0.1,
        'temperature': 0.07,
    }
    
    if config:
        default_config.update(config)
    
    return TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        **default_config
    )
