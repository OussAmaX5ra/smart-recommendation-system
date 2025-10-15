"""
Training pipeline for the hybrid recommendation system.
Trains CF, Temporal, and Hybrid models sequentially.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import sys

# Add models directory to path
sys.path.append('models')

import config
from models.collaborative_filtering import MatrixFactorization, train_cf_model
from models.temporal_model import LSTMRecommender, TransformerRecommender, train_temporal_model
from models.hybrid_model import HybridRecommender, train_hybrid_model


# ==================== DATASET CLASSES ====================

class CFDataset(Dataset):
    """Dataset for Collaborative Filtering."""

    def __init__(self, interactions, n_items, negative_samples=4):
        """
        Args:
            interactions: Sparse matrix of user-item interactions
            n_items: Total number of items
            negative_samples: Number of negative samples per positive
        """
        self.interactions = interactions
        self.n_items = n_items
        self.negative_samples = negative_samples

        # Get positive interactions
        self.user_indices, self.item_indices = interactions.nonzero()
        self.n_samples = len(self.user_indices)

    def __len__(self):
        return self.n_samples * (1 + self.negative_samples)

    def __getitem__(self, idx):
        # Positive or negative sample
        sample_idx = idx // (1 + self.negative_samples)
        is_positive = (idx % (1 + self.negative_samples)) == 0

        user_id = self.user_indices[sample_idx]

        if is_positive:
            item_id = self.item_indices[sample_idx]
            label = 1
        else:
            # Sample negative item
            item_id = np.random.randint(0, self.n_items)
            # Ensure it's truly negative
            while self.interactions[user_id, item_id] > 0:
                item_id = np.random.randint(0, self.n_items)
            label = 0

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


class TemporalDataset(Dataset):
    """Dataset for Temporal models."""

    def __init__(self, sequences, max_length=20):
        """
        Args:
            sequences: Dictionary of user sequences {user_idx: [item_ids]}
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.samples = []

        # Create training samples: predict next item from history
        for user_idx, seq in sequences.items():
            if len(seq) < 2:  # Need at least 2 items
                continue

            # Create multiple samples from the sequence
            for i in range(1, len(seq)):
                history = seq[:i]
                target = seq[i]
                self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]

        # Pad or truncate history
        if len(history) > self.max_length:
            history = history[-self.max_length:]

        sequence = history + [0] * (self.max_length - len(history))
        length = len(history)
        mask = [False] * length + [True] * (self.max_length - length)

        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }


class HybridDataset(Dataset):
    """Dataset for Hybrid model."""

    def __init__(self, interactions, sequences, demographic_features,
                 n_items, max_seq_length=20, negative_samples=4):
        """
        Args:
            interactions: Sparse matrix of interactions
            sequences: Dictionary of user sequences
            demographic_features: Dictionary {user_idx: features}
            n_items: Total number of items
            max_seq_length: Maximum sequence length
            negative_samples: Number of negative samples per positive
        """
        self.interactions = interactions
        self.sequences = sequences
        self.demographic_features = demographic_features
        self.n_items = n_items
        self.max_seq_length = max_seq_length
        self.negative_samples = negative_samples

        # Get positive interactions
        self.user_indices, self.item_indices = interactions.nonzero()
        self.n_samples = len(self.user_indices)

    def __len__(self):
        return self.n_samples * (1 + self.negative_samples)

    def __getitem__(self, idx):
        sample_idx = idx // (1 + self.negative_samples)
        is_positive = (idx % (1 + self.negative_samples)) == 0

        user_id = self.user_indices[sample_idx]

        if is_positive:
            item_id = self.item_indices[sample_idx]
            label = 1
        else:
            item_id = np.random.randint(0, self.n_items)
            while self.interactions[user_id, item_id] > 0:
                item_id = np.random.randint(0, self.n_items)
            label = 0

        # Get user sequence
        seq = self.sequences.get(user_id, [])
        if len(seq) > self.max_seq_length:
            seq = seq[-self.max_seq_length:]

        sequence = seq + [0] * (self.max_seq_length - len(seq))
        length = len(seq)
        mask = [False] * length + [True] * (self.max_seq_length - length)

        # Get demographic features
        demo = self.demographic_features.get(user_id, np.zeros(10))

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'demographic': torch.tensor(demo, dtype=torch.float),
            'length': torch.tensor(length, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float)
        }


# ==================== TRAINING FUNCTIONS ====================

def load_features():
    """Load all engineered features."""
    print("Loading features...")

    features = {}
    files = [
        'mappings', 'user_item_matrix', 'sequences',
        'demographic_features', 'product_features'
    ]

    for name in files:
        path = f"{config.PROCESSED_DATA_DIR}/{name}.pkl"
        with open(path, 'rb') as f:
            features[name] = pickle.load(f)

    print("✓ Features loaded!")
    return features


def train_cf(features, device):
    """Train Collaborative Filtering model."""
    print("\n" + "=" * 60)
    print("TRAINING COLLABORATIVE FILTERING MODEL")
    print("=" * 60)

    # Create dataset
    train_dataset = CFDataset(
        features['user_item_matrix'],
        features['mappings']['n_products'],
        negative_samples=4
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CF_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # Create model
    model = MatrixFactorization(
        n_users=features['mappings']['n_users'],
        n_items=features['mappings']['n_products'],
        embedding_dim=config.CF_CONFIG['embedding_dim']
    )

    # Train
    model, history = train_cf_model(
        model,
        train_loader,
        epochs=config.CF_CONFIG['epochs'],
        lr=config.CF_CONFIG['learning_rate'],
        weight_decay=config.CF_CONFIG['weight_decay'],
        device=device
    )

    # Save model
    torch.save(model.state_dict(), config.get_model_path('cf_model'))
    print(f"✓ Model saved to {config.get_model_path('cf_model')}")

    return model


def train_temporal(features, device):
    """Train Temporal model."""
    print("\n" + "=" * 60)
    print("TRAINING TEMPORAL MODEL")
    print("=" * 60)

    # Create dataset
    train_dataset = TemporalDataset(
        features['sequences']['sequences'],
        max_length=config.MAX_SEQUENCE_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TEMPORAL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # Create model (choose between LSTM and Transformer)
    model_type = config.TEMPORAL_CONFIG['model_type']
    n_items = features['mappings']['n_products']

    if model_type in ['lstm', 'gru', 'rnn']:
        model = LSTMRecommender(
            n_items=n_items,
            embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
            hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
            num_layers=config.TEMPORAL_CONFIG['num_layers'],
            dropout=config.TEMPORAL_CONFIG['dropout']
        )
    else:  # transformer
        model = TransformerRecommender(
            n_items=n_items,
            embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
            num_heads=config.TEMPORAL_CONFIG['num_heads'],
            num_layers=config.TEMPORAL_CONFIG['num_layers'],
            hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
            dropout=config.TEMPORAL_CONFIG['dropout'],
            max_seq_len=config.MAX_SEQUENCE_LENGTH
        )

    # Train
    model, history = train_temporal_model(
        model,
        train_loader,
        epochs=config.TEMPORAL_CONFIG['epochs'],
        lr=config.TEMPORAL_CONFIG['learning_rate'],
        device=device
    )

    # Save model
    torch.save(model.state_dict(), config.get_model_path('temporal_model'))
    print(f"✓ Model saved to {config.get_model_path('temporal_model')}")

    return model


def train_hybrid(cf_model, temporal_model, features, device):
    """Train Hybrid model."""
    print("\n" + "=" * 60)
    print("TRAINING HYBRID MODEL")
    print("=" * 60)

    # Prepare demographic features as dictionary
    demo_dict = {}
    demo_matrix = features['demographic_features']['matrix']
    for user_idx in range(len(demo_matrix)):
        demo_dict[user_idx] = demo_matrix[user_idx]

    # Create dataset
    train_dataset = HybridDataset(
        features['user_item_matrix'],
        features['sequences']['sequences'],
        demo_dict,
        features['mappings']['n_products'],
        max_seq_length=config.MAX_SEQUENCE_LENGTH,
        negative_samples=4
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.HYBRID_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # Create hybrid model
    model = HybridRecommender(
        cf_model=cf_model,
        temporal_model=temporal_model,
        n_users=features['mappings']['n_users'],
        n_items=features['mappings']['n_products'],
        demographic_dim=demo_matrix.shape[1],
        cf_weight=config.HYBRID_CONFIG['cf_weight'],
        temporal_weight=config.HYBRID_CONFIG['temporal_weight'],
        demographic_weight=config.HYBRID_CONFIG['demographic_weight'],
        fusion_method=config.HYBRID_CONFIG['fusion_method']
    )

    # Train
    model, history = train_hybrid_model(
        model,
        train_loader,
        epochs=config.HYBRID_CONFIG['epochs'],
        lr=config.HYBRID_CONFIG['learning_rate'],
        device=device
    )

    # Save model
    torch.save(model.state_dict(), config.get_model_path('hybrid_model'))
    print(f"✓ Model saved to {config.get_model_path('hybrid_model')}")

    return model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 60)

    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load features
    features = load_features()

    # Train models
    cf_model = train_cf(features, device)
    temporal_model = train_temporal(features, device)
    hybrid_model = train_hybrid(cf_model, temporal_model, features, device)

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Models saved in: {config.MODEL_DIR}")


if __name__ == "__main__":
    main()