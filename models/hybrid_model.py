"""
Hybrid Recommendation Model.
Combines Collaborative Filtering, Temporal Model, and User Demographics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridRecommender(nn.Module):
    """
    Hybrid recommendation system combining multiple signals:
    1. Collaborative Filtering (user-item embeddings)
    2. Temporal patterns (sequential behavior)
    3. User demographics (age, gender, behavior stats)
    """

    def __init__(self, cf_model, temporal_model,
                 n_users, n_items, demographic_dim,
                 cf_weight=0.4, temporal_weight=0.4, demographic_weight=0.2,
                 fusion_method='weighted_sum', mlp_hidden_dims=[256, 128],
                 dropout=0.2):
        """
        Args:
            cf_model: Collaborative filtering model
            temporal_model: Temporal model (LSTM/Transformer)
            n_users: Number of users
            n_items: Number of items
            demographic_dim: Dimension of demographic features
            cf_weight: Weight for CF component
            temporal_weight: Weight for temporal component
            demographic_weight: Weight for demographic component
            fusion_method: 'weighted_sum' or 'mlp'
            mlp_hidden_dims: Hidden dimensions for MLP fusion
            dropout: Dropout rate
        """
        super(HybridRecommender, self).__init__()

        self.cf_model = cf_model
        self.temporal_model = temporal_model
        self.fusion_method = fusion_method

        # Weights for weighted sum fusion
        self.cf_weight = cf_weight
        self.temporal_weight = temporal_weight
        self.demographic_weight = demographic_weight

        # Demographic feature processor
        self.demographic_fc = nn.Sequential(
            nn.Linear(demographic_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # User-specific demographic projections
        self.user_demo_embedding = nn.Linear(64, n_items)

        # MLP fusion layers (if using MLP fusion)
        if fusion_method == 'mlp':
            # Input: CF score + temporal score + demographic features
            mlp_input_dim = 3  # Three score components
            self.fusion_mlp = nn.ModuleList()

            prev_dim = mlp_input_dim
            for hidden_dim in mlp_hidden_dims:
                self.fusion_mlp.append(nn.Linear(prev_dim, hidden_dim))
                self.fusion_mlp.append(nn.ReLU())
                self.fusion_mlp.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            self.fusion_output = nn.Linear(prev_dim, 1)

    def forward(self, user_ids, item_ids, sequences, demographic_features,
                sequence_mask=None, sequence_lengths=None):
        """
        Forward pass combining all components.

        Args:
            user_ids: User indices [batch_size]
            item_ids: Item indices [batch_size]
            sequences: Item sequences [batch_size, seq_len]
            demographic_features: User demographic features [batch_size, demo_dim]
            sequence_mask: Mask for sequences (for Transformer)
            sequence_lengths: Lengths for sequences (for LSTM)

        Returns:
            final_scores: Combined recommendation scores [batch_size]
        """
        # 1. Collaborative Filtering component
        cf_scores = self.cf_model(user_ids, item_ids)  # [batch_size]

        # 2. Temporal component
        if hasattr(self.temporal_model, 'lstm'):  # LSTM
            temporal_logits = self.temporal_model(sequences, sequence_lengths)
        else:  # Transformer
            temporal_logits = self.temporal_model(sequences, sequence_mask)

        # Get temporal scores for specific items
        temporal_scores = torch.gather(
            torch.softmax(temporal_logits, dim=-1),
            1,
            item_ids.unsqueeze(1)
        ).squeeze(1)  # [batch_size]

        # 3. Demographic component
        demo_features = self.demographic_fc(demographic_features)  # [batch_size, 64]
        demo_logits = self.user_demo_embedding(demo_features)  # [batch_size, n_items]
        demo_scores = torch.gather(
            torch.softmax(demo_logits, dim=-1),
            1,
            item_ids.unsqueeze(1)
        ).squeeze(1)  # [batch_size]

        # Fusion
        if self.fusion_method == 'weighted_sum':
            # Weighted sum of components
            final_scores = (
                    self.cf_weight * cf_scores +
                    self.temporal_weight * temporal_scores +
                    self.demographic_weight * demo_scores
            )

        elif self.fusion_method == 'mlp':
            # MLP fusion
            fusion_input = torch.stack([cf_scores, temporal_scores, demo_scores], dim=1)

            fusion_output = fusion_input
            for layer in self.fusion_mlp:
                fusion_output = layer(fusion_output)

            final_scores = self.fusion_output(fusion_output).squeeze(1)
            final_scores = torch.sigmoid(final_scores)

        return final_scores

    def predict_for_user(self, user_id, sequence, demographic_feature,
                         device='cuda', top_k=10, exclude_items=None):
        """
        Generate top-k recommendations for a single user.

        Args:
            user_id: User ID (int)
            sequence: User's item sequence [seq_len]
            demographic_feature: User's demographic features [demo_dim]
            device: Device
            top_k: Number of recommendations
            exclude_items: Set of item IDs to exclude (already purchased)

        Returns:
            top_items: Top-k recommended item indices
            top_scores: Scores for recommended items
        """
        self.eval()
        with torch.no_grad():
            # Prepare inputs
            user_tensor = torch.tensor([user_id]).to(device)
            sequence_tensor = torch.tensor([sequence]).to(device)
            demo_tensor = torch.tensor([demographic_feature]).float().to(device)

            # Get scores for all items
            n_items = self.cf_model.n_items

            # 1. CF scores for all items
            cf_scores_all = self.cf_model.predict_all(user_tensor)  # [n_items]

            # 2. Temporal scores for all items
            if hasattr(self.temporal_model, 'lstm'):
                lengths = torch.tensor([len([x for x in sequence if x != 0])]).to(device)
                temporal_logits = self.temporal_model(sequence_tensor, lengths)
            else:
                mask = (sequence_tensor == 0)
                temporal_logits = self.temporal_model(sequence_tensor, mask)

            temporal_scores_all = torch.softmax(temporal_logits, dim=-1).squeeze(0)  # [n_items]

            # 3. Demographic scores for all items
            demo_features = self.demographic_fc(demo_tensor)
            demo_logits = self.user_demo_embedding(demo_features)
            demo_scores_all = torch.softmax(demo_logits, dim=-1).squeeze(0)  # [n_items]

            # Combine scores
            if self.fusion_method == 'weighted_sum':
                final_scores = (
                        self.cf_weight * cf_scores_all +
                        self.temporal_weight * temporal_scores_all +
                        self.demographic_weight * demo_scores_all
                )
            else:  # MLP fusion
                # For simplicity with MLP, we'll use weighted sum for batch prediction
                final_scores = (
                        self.cf_weight * cf_scores_all +
                        self.temporal_weight * temporal_scores_all +
                        self.demographic_weight * demo_scores_all
                )

            # Exclude already purchased items
            if exclude_items is not None:
                for item_idx in exclude_items:
                    final_scores[item_idx] = -float('inf')

            # Get top-k
            top_scores, top_items = torch.topk(final_scores, top_k)

        return top_items.cpu().numpy(), top_scores.cpu().numpy()


def train_hybrid_model(model, train_loader, val_loader=None,
                       epochs=20, lr=0.0001, device='cuda',
                       early_stopping_patience=5):
    """
    Train hybrid model.

    Args:
        model: Hybrid model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        early_stopping_patience: Patience for early stopping

    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)

    # Freeze pretrained components initially
    for param in model.cf_model.parameters():
        param.requires_grad = False
    for param in model.temporal_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.BCELoss()

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Unfreeze all parameters after a few epochs
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr / 10)

        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            sequences = batch['sequence'].to(device)
            demographic = batch['demographic'].float().to(device)
            labels = batch['label'].float().to(device)

            # Get mask or lengths
            mask = batch.get('mask', None)
            lengths = batch.get('length', None)
            if mask is not None:
                mask = mask.to(device)
            if lengths is not None:
                lengths = lengths.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids, sequences, demographic,
                                sequence_mask=mask, sequence_lengths=lengths)
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
                    sequences = batch['sequence'].to(device)
                    demographic = batch['demographic'].float().to(device)
                    labels = batch['label'].float().to(device)

                    mask = batch.get('mask', None)
                    lengths = batch.get('length', None)
                    if mask is not None:
                        mask = mask.to(device)
                    if lengths is not None:
                        lengths = lengths.to(device)

                    predictions = model(user_ids, item_ids, sequences, demographic,
                                        sequence_mask=mask, sequence_lengths=lengths)
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