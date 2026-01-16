"""
Inference Pipeline
===================
Generate embeddings from trained model for FAISS indexing.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from model import TwoTowerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings from trained Two-Tower model."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        self.base_path = Path(base_path)
        
        self.checkpoint_path = self.base_path / "checkpoints"
        self.embeddings_path = self.base_path / "embeddings"
        self.processed_path = self.base_path / "data" / "processed"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, checkpoint_name: str = "best_model.pt"):
        """Load trained model from checkpoint."""
        checkpoint_file = self.checkpoint_path / checkpoint_name
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Load metadata
        with open(self.processed_path / "metadata.json") as f:
            metadata = json.load(f)
        
        # Recreate model
        config = checkpoint.get('config', {})
        self.model = TwoTowerModel(
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            num_genres=config.get('num_genres', 20),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dims=config.get('hidden_dims', [256, 128]),
            output_dim=config.get('output_dim', 64),
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.num_users = metadata['num_users']
        self.num_items = metadata['num_items']
        
        logger.info(f"Model loaded: {self.num_users} users, {self.num_items} items")
        
    def generate_item_embeddings(self, batch_size: int = 1024) -> np.ndarray:
        """Generate embeddings for all items."""
        logger.info(f"Generating embeddings for {self.num_items} items...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, self.num_items, batch_size), desc="Item embeddings"):
                batch_ids = torch.arange(
                    i, min(i + batch_size, self.num_items),
                    device=self.device
                )
                batch_emb = self.model.get_item_embeddings(batch_ids)
                embeddings.append(batch_emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings).astype(np.float32)
        logger.info(f"Item embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def generate_user_embeddings(self, batch_size: int = 1024) -> np.ndarray:
        """Generate embeddings for all users."""
        logger.info(f"Generating embeddings for {self.num_users} users...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, self.num_users, batch_size), desc="User embeddings"):
                batch_ids = torch.arange(
                    i, min(i + batch_size, self.num_users),
                    device=self.device
                )
                batch_emb = self.model.get_user_embeddings(batch_ids)
                embeddings.append(batch_emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings).astype(np.float32)
        logger.info(f"User embeddings shape: {embeddings.shape}")
        
        return embeddings
        
    def generate_and_save(self):
        """Generate and save all embeddings."""
        logger.info("="*60)
        logger.info("Generating Embeddings")
        logger.info("="*60)
        
        # Generate
        item_embeddings = self.generate_item_embeddings()
        user_embeddings = self.generate_user_embeddings()
        
        # Save
        np.save(self.embeddings_path / "item_embeddings.npy", item_embeddings)
        np.save(self.embeddings_path / "user_embeddings.npy", user_embeddings)
        
        logger.info(f"Saved embeddings to: {self.embeddings_path}")
        
        return item_embeddings, user_embeddings


def main():
    """Generate embeddings from trained model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    
    generator = EmbeddingGenerator()
    generator.load_model(args.checkpoint)
    generator.generate_and_save()


if __name__ == "__main__":
    main()
