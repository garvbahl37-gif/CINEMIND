"""
Production-Grade Training Pipeline
===================================
Implements FAANG-level training with:
- Mixed precision (AMP)
- Gradient accumulation
- Learning rate scheduling (warmup + cosine decay)
- Early stopping based on validation Recall@K
- Checkpoint management
- TensorBoard logging
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model import TwoTowerModel, create_model
from loss import InfoNCELoss, get_loss_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        self.base_path = Path(base_path)
        
        # Paths
        self.processed_path = self.base_path / "data" / "processed"
        self.features_path = self.base_path / "data" / "features"
        self.embeddings_path = self.base_path / "embeddings"
        self.checkpoint_path = self.base_path / "checkpoints"
        self.logs_path = self.base_path / "logs"
        
        # Model config
        self.embedding_dim = 128
        self.hidden_dims = [256, 128]
        self.output_dim = 64
        self.dropout = 0.1
        self.temperature = 0.07
        
        # Training config
        self.batch_size = 2048
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.warmup_epochs = 2
        self.grad_accumulation_steps = 1
        self.max_grad_norm = 1.0
        
        # Early stopping
        self.patience = 5
        self.min_delta = 0.001
        
        # Hardware
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = torch.cuda.is_available()
        self.num_workers = 4
        
        # Create directories
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)


class InteractionDataset(Dataset):
    """Dataset for user-item interactions."""
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        user_features: Optional[pd.DataFrame] = None,
        item_features: Optional[pd.DataFrame] = None,
        genre_columns: Optional[List[str]] = None,
        mode: str = 'train'
    ):
        self.user_ids = torch.LongTensor(interactions['user_idx'].values)
        self.item_ids = torch.LongTensor(interactions['item_idx'].values)
        
        if 'label' in interactions.columns:
            self.labels = torch.FloatTensor(interactions['label'].values)
        else:
            self.labels = torch.ones(len(interactions))
        
        # Optional features
        self.user_features = None
        self.item_features = None
        self.item_genres = None
        
        if user_features is not None:
            # Get feature columns (exclude user_idx)
            feat_cols = [c for c in user_features.columns if c != 'user_idx']
            self.user_feat_lookup = user_features.set_index('user_idx')[feat_cols].values.astype(np.float32)
            self.user_features = True
            
        if item_features is not None and genre_columns:
            # Genre features
            item_features = item_features.set_index('item_idx')
            valid_genre_cols = [c for c in genre_columns if c in item_features.columns]
            if valid_genre_cols:
                self.item_genre_lookup = item_features[valid_genre_cols].values.astype(np.float32)
                self.item_genres = True
        
        self.mode = mode
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        label = self.labels[idx]
        
        sample = {
            'user_id': user_id,
            'item_id': item_id,
            'label': label
        }
        
        if self.user_features is not None:
            sample['user_features'] = torch.FloatTensor(self.user_feat_lookup[user_id])
            
        if self.item_genres is not None:
            sample['item_genres'] = torch.FloatTensor(self.item_genre_lookup[item_id])
        
        return sample


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class CosineWarmupScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


class Trainer:
    """Production training pipeline."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.config.use_amp else None
        self.loss_fn = None
        self.early_stopping = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_metrics = []
        self.best_model_state = None
        self.best_epoch = 0
        
    def setup(self):
        """Load data and initialize model."""
        logger.info("Setting up trainer...")
        
        # Load metadata
        with open(self.config.processed_path / "metadata.json") as f:
            metadata = json.load(f)
        
        self.num_users = metadata['num_users']
        self.num_items = metadata['num_items']
        
        # Load data
        logger.info("Loading training data...")
        self.train_data = pd.read_parquet(
            self.config.processed_path / "train_interactions.parquet"
        )
        self.val_data = pd.read_parquet(
            self.config.processed_path / "val.parquet"
        )
        
        # Load features if available
        self.user_features = None
        self.item_features = None
        self.genre_columns = []
        
        user_feat_path = self.config.features_path / "user_features.parquet"
        item_feat_path = self.config.features_path / "item_features.parquet"
        
        if user_feat_path.exists():
            self.user_features = pd.read_parquet(user_feat_path)
            logger.info(f"Loaded user features: {self.user_features.shape}")
            
        if item_feat_path.exists():
            self.item_features = pd.read_parquet(item_feat_path)
            self.genre_columns = [c for c in self.item_features.columns if c.startswith('genre_')]
            logger.info(f"Loaded item features: {self.item_features.shape}")
        
        # Create datasets
        self.train_dataset = InteractionDataset(
            self.train_data,
            self.user_features,
            self.item_features,
            self.genre_columns,
            mode='train'
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create model
        logger.info("Initializing model...")
        self.model = TwoTowerModel(
            num_users=self.num_users,
            num_items=self.num_items,
            num_genres=len(self.genre_columns),
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            dropout=self.config.dropout,
            temperature=self.config.temperature
        ).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.num_epochs
        )
        
        # Loss function
        self.loss_fn = InfoNCELoss(temperature=self.config.temperature)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode='max'
        )
        
        logger.info("Trainer setup complete!")
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            
            item_genres = None
            if 'item_genres' in batch:
                item_genres = batch['item_genres'].to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.config.use_amp):
                user_emb, item_emb, _ = self.model(
                    user_ids, item_ids,
                    item_genre_vectors=item_genres
                )
                
                # InfoNCE loss with in-batch negatives
                loss = self.loss_fn(user_emb, item_emb)
                loss = loss / self.config.grad_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.grad_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, k: int = 100) -> Dict[str, float]:
        """Evaluate on validation set using Recall@K."""
        self.model.eval()
        
        # Get all item embeddings
        all_items = torch.arange(self.num_items, device=self.device)
        item_embeddings = []
        
        for i in range(0, self.num_items, 1024):
            batch_items = all_items[i:i+1024]
            item_emb = self.model.get_item_embeddings(batch_items)
            item_embeddings.append(item_emb.cpu())
            
        item_embeddings = torch.cat(item_embeddings, dim=0)  # (num_items, dim)
        
        # Build ground truth
        val_user_items = self.val_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        # Sample users for evaluation
        eval_users = list(val_user_items.keys())[:1000]  # Sample for speed
        
        hits = 0
        total = 0
        
        for user_idx in tqdm(eval_users, desc="Evaluating"):
            user_id = torch.LongTensor([user_idx]).to(self.device)
            user_emb = self.model.get_user_embeddings(user_id).cpu()
            
            # Compute similarities
            scores = torch.matmul(user_emb, item_embeddings.T).squeeze()
            
            # Get top-K
            _, top_k = torch.topk(scores, k)
            top_k = set(top_k.numpy())
            
            # Check hits
            ground_truth = val_user_items.get(user_idx, set())
            hits += len(top_k & ground_truth)
            total += len(ground_truth)
        
        recall = hits / max(total, 1)
        
        return {'recall@100': recall}
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.model.config
        }
        
        path = self.config.checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        # Save best model
        if metrics.get('recall@100', 0) >= self.early_stopping.best_score:
            best_path = self.config.checkpoint_path / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self):
        """Run full training loop."""
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Users: {self.num_users:,} | Items: {self.num_items:,}")
        logger.info(f"Train samples: {len(self.train_data):,}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            # Update learning rate
            lr = self.scheduler.step(epoch)
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs} | LR: {lr:.6f}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Evaluate
            metrics = self.evaluate()
            self.val_metrics.append(metrics)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Recall@100: {metrics['recall@100']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics)
            
            # Early stopping
            if self.early_stopping(metrics['recall@100']):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best Recall@100: {self.early_stopping.best_score:.4f}")
        logger.info("=" * 60)
    
    def export_embeddings(self):
        """Export user and item embeddings for FAISS indexing."""
        logger.info("Exporting embeddings...")
        
        # Load best model
        best_path = self.config.checkpoint_path / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Export item embeddings
        all_items = torch.arange(self.num_items, device=self.device)
        item_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, self.num_items, 1024), desc="Item embeddings"):
                batch_items = all_items[i:i+1024]
                item_emb = self.model.get_item_embeddings(batch_items)
                item_embeddings.append(item_emb.cpu().numpy())
        
        item_embeddings = np.vstack(item_embeddings).astype(np.float32)
        np.save(self.config.embeddings_path / "item_embeddings.npy", item_embeddings)
        logger.info(f"Saved item embeddings: {item_embeddings.shape}")
        
        # Export user embeddings
        all_users = torch.arange(self.num_users, device=self.device)
        user_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, self.num_users, 1024), desc="User embeddings"):
                batch_users = all_users[i:i+1024]
                user_emb = self.model.get_user_embeddings(batch_users)
                user_embeddings.append(user_emb.cpu().numpy())
        
        user_embeddings = np.vstack(user_embeddings).astype(np.float32)
        np.save(self.config.embeddings_path / "user_embeddings.npy", user_embeddings)
        logger.info(f"Saved user embeddings: {user_embeddings.shape}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-Tower Training")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    parser.add_argument('--export', action='store_true', help='Export embeddings after training')
    args = parser.parse_args()
    
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()
    
    if args.export:
        trainer.export_embeddings()


if __name__ == "__main__":
    main()
