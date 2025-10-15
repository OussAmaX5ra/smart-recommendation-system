"""
Temporal models for sequential recommendation.
Implements RNN (LSTM/GRU) and Transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMRecommender(nn.Module):
    """
    LSTM-based sequential recommender.
    Predicts next items based on user's purchase history.
    """

    def __init__(self, n_items, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.2):
        """
        Args:
            n_items: Number of items
            embedding_dim: Dimension of item embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMRecommender, self).__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)

        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_items)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, sequences, lengths=None):
        """
        Forward pass.

        Args:
            sequences: Item sequences [batch_size, seq_len]
            lengths: Actual sequence lengths [batch_size]

        Returns:
            logits: Item scores [batch_size, n_items]
        """
        # Embed items
        embedded = self.item_embedding(sequences)  # [batch_size, seq_len, embedding_dim]

        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )

        # LSTM forward
        output, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get last output for each sequence
        if lengths is not None:
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
            last_output = output.gather(1, idx).squeeze(1)
        else:
            last_output = output[:, -1, :]  # [batch_size, hidden_dim]

        # Dropout and fully connected
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)  # [batch_size, n_items]

        return logits

    def predict(self, sequences, lengths=None, top_k=10):
        """Predict top-k items for given sequences."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(sequences, lengths)
            scores = torch.softmax(logits, dim=-1)
            top_scores, top_indices = torch.topk(scores, top_k, dim=-1)
        return top_indices, top_scores


class TransformerRecommender(nn.Module):
    """
    Transformer-based sequential recommender.
    Uses self-attention to model item sequences.
    """

    def __init__(self, n_items, embedding_dim=128, num_heads=4,
                 num_layers=2, hidden_dim=256, dropout=0.2, max_seq_len=50):
        """
        Args:
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Feedforward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(TransformerRecommender, self).__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, n_items)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sequences, mask=None):
        """
        Forward pass.

        Args:
            sequences: Item sequences [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len]

        Returns:
            logits: Item scores [batch_size, n_items]
        """
        # Embed items
        embedded = self.item_embedding(sequences)  # [batch_size, seq_len, embedding_dim]

        # Add positional encoding
        embedded = self.positional_encoding(embedded)

        # Create attention mask (padding mask)
        if mask is None:
            mask = (sequences == 0)  # Padding positions

        # Transformer forward
        output = self.transformer(
            embedded,
            src_key_padding_mask=mask
        )  # [batch_size, seq_len, embedding_dim]

        # Use last non-padding token
        # Get last valid position for each sequence
        lengths = (~mask).sum(dim=1)  # [batch_size]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
        last_output = output.gather(1, idx).squeeze(1)  # [batch_size, embedding_dim]

        # Dropout and fully connected
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)  # [batch_size, n_items]

        return logits

    def predict(self, sequences, mask=None, top_k=10):
        """Predict top-k items for given sequences."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(sequences, mask)
            scores = torch.softmax(logits, dim=-1)
            top_scores, top_indices = torch.topk(scores, top_k, dim=-1)
        return top_indices, top_scores


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


def train_temporal_model(model, train_loader, val_loader=None,
                         epochs=30, lr=0.0005, device='cuda',
                         early_stopping_patience=5):
    """
    Train temporal model.

    Args:
        model: Temporal model instance
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            targets = batch['target'].to(device)

            # Handle mask or lengths depending on model type
            if isinstance(model, LSTMRecommender):
                lengths = batch.get('length', None)
                if lengths is not None:
                    lengths = lengths.to(device)
                logits = model(sequences, lengths)
            else:  # Transformer
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(device)
                logits = model(sequences, mask)

            optimizer.zero_grad()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch['sequence'].to(device)
                    targets = batch['target'].to(device)

                    if isinstance(model, LSTMRecommender):
                        lengths = batch.get('length', None)
                        if lengths is not None:
                            lengths = lengths.to(device)
                        logits = model(sequences, lengths)
                    else:
                        mask = batch.get('mask', None)
                        if mask is not None:
                            mask = mask.to(device)
                        logits = model(sequences, mask)

                    loss = criterion(logits, targets)
                    val_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    return model, history