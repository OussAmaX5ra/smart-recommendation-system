"""
Collaborative Filtering Model using Matrix Factorization.
Implements a neural collaborative filtering approach with user and item embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MatrixFactorization(nn.Module):
    """
    Matrix Factorization model for collaborative filtering.
    Uses embeddings for users and items with bias terms.
    """

    def __init__(self, n_users, n_items, embedding_dim=64, dropout=0.2):
        """
        Args:
            n_users: Number of users
            n_items: Number of items (products)
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate for regularization
        """
        super(MatrixFactorization, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with normal distribution."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        """
        Forward pass.

        Args:
            user_ids: Tensor of user indices [batch_size]
            item_ids: Tensor of item indices [batch_size]

        Returns:
            predictions: Predicted ratings [batch_size]
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embedding_dim]

        # Apply dropout
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        # Get biases
        user_b = self.user_bias(user_ids).squeeze()  # [batch_size]
        item_b = self.item_bias(item_ids).squeeze()  # [batch_size]

        # Compute prediction: dot product + biases
        dot_product = (user_emb * item_emb).sum(dim=1)  # [batch_size]
        prediction = dot_product + user_b + item_b + self.global_bias

        return torch.sigmoid(prediction)

    def get_user_embedding(self, user_ids):
        """Get user embeddings for given user IDs."""
        return self.user_embedding(user_ids)

    def get_item_embedding(self, item_ids):
        """Get item embeddings for given item IDs."""
        return self.item_embedding(item_ids)

    def predict_all(self, user_id):
        """
        Predict scores for all items for a given user.

        Args:
            user_id: Single user ID (int or tensor)

        Returns:
            scores: Scores for all items [n_items]
        """
        if isinstance(user_id, int):
            user_id = torch.tensor([user_id])

        # Get user embedding and bias
        user_emb = self.user_embedding(user_id)  # [1, embedding_dim]
        user_b = self.user_bias(user_id).squeeze()  # [1]

        # Get all item embeddings and biases
        all_items = torch.arange(self.n_items).to(user_emb.device)
        item_emb = self.item_embedding(all_items)  # [n_items, embedding_dim]
        item_b = self.item_bias(all_items).squeeze()  # [n_items]

        # Compute scores
        scores = torch.matmul(item_emb, user_emb.T).squeeze()  # [n_items]
        scores = scores + user_b + item_b + self.global_bias

        return torch.sigmoid(scores)


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    Combines matrix factorization with multi-layer perceptron.
    """

    def __init__(self, n_users, n_items, embedding_dim=64,
                 hidden_dims=[128, 64, 32], dropout=0.2):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            hidden_dims: List of hidden layer dimensions for MLP
            dropout: Dropout rate
        """
        super(NeuralCollaborativeFiltering, self).__init__()

        self.n_users = n_users
        self.n_items = n_items

        # Embeddings for GMF (Generalized Matrix Factorization)
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        # Embeddings for MLP
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()

        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final prediction layer
        self.output_layer = nn.Linear(embedding_dim + hidden_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        """Forward pass combining GMF and MLP."""
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf  # Element-wise product

        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)

        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)

        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)

        # Final prediction
        prediction = self.output_layer(concat).squeeze()
        return torch.sigmoid(prediction)

    def predict_all(self, user_id):
        """Predict scores for all items for a given user."""
        if isinstance(user_id, int):
            user_id = torch.tensor([user_id])

        # Get all items
        all_items = torch.arange(self.n_items).to(user_id.device)

        # Repeat user_id for all items
        user_ids = user_id.repeat(self.n_items)

        # Get predictions
        with torch.no_grad():
            scores = self.forward(user_ids, all_items)

        return scores


def train_cf_model(model, train_loader, val_loader=None,
                   epochs=50, lr=0.001, weight_decay=1e-5,
                   device='cuda', early_stopping_patience=5):
    """
    Train collaborative filtering model.

    Args:
        model: CF model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        early_stopping_patience: Patience for early stopping

    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.BCELoss()

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            labels = batch['label'].float().to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    user_ids = batch['user_id'].to(device)
                    item_ids = batch['item_id'].to(device)
                    labels = batch['label'].float().to(device)

                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

    return model, history