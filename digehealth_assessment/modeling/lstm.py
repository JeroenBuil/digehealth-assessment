"""LSTM model for bowel sound classification."""

import torch
import torch.nn as nn


class BowelSoundLSTM(nn.Module):
    """LSTM model for classifying bowel sounds from spectrogram features."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = 128,  # Default spectrogram height
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super(BowelSoundLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.num_directions,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, height, width) for spectrograms

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Handle different input shapes
        if x.dim() == 4:
            # (batch_size, channels, height, width) -> (batch_size, height, width)
            if x.size(1) == 1:
                x = x.squeeze(1)
            else:
                # If multiple channels, take mean across channels
                x = x.mean(dim=1)

        # (batch_size, height, width) -> (batch_size, width, height)
        # This treats each time step (width) as a sequence and height as features
        x = x.transpose(1, 2)

        # LSTM expects (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape

        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling over sequence dimension
        pooled = attn_out.mean(dim=1)

        # Classification
        output = self.classifier(pooled)

        return output


class BowelSoundLSTMWithFeatures(nn.Module):
    """LSTM model that can handle additional features alongside spectrograms."""

    def __init__(
        self,
        num_classes: int,
        spectrogram_height: int = 128,
        feature_dim: int = 0,  # Additional features dimension
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super(BowelSoundLSTMWithFeatures, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM for spectrogram features
        self.spectrogram_lstm = nn.LSTM(
            input_size=spectrogram_height,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # LSTM for additional features if provided
        if feature_dim > 0:
            self.feature_lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_size // 2,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            lstm_output_size = (
                hidden_size * self.num_directions
                + (hidden_size // 2) * self.num_directions
            )
        else:
            lstm_output_size = hidden_size * self.num_directions

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size, num_heads=8, dropout=dropout, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, spectrogram, features=None):
        """
        Forward pass through the LSTM model.

        Args:
            spectrogram: Spectrogram tensor of shape (batch_size, channels, height, width)
                        or (batch_size, height, width)
            features: Additional features tensor of shape (batch_size, seq_len, feature_dim) or None

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Process spectrogram
        if spectrogram.dim() == 4:
            if spectrogram.size(1) == 1:
                spectrogram = spectrogram.squeeze(1)
            else:
                spectrogram = spectrogram.mean(dim=1)

        # (batch_size, height, width) -> (batch_size, width, height)
        spectrogram = spectrogram.transpose(1, 2)

        # LSTM for spectrogram
        lstm_out, _ = self.spectrogram_lstm(spectrogram)

        # Process additional features if provided
        if features is not None and self.feature_dim > 0:
            feature_lstm_out, _ = self.feature_lstm(features)
            # Concatenate both LSTM outputs
            combined_out = torch.cat([lstm_out, feature_lstm_out], dim=-1)
        else:
            combined_out = lstm_out

        # Apply self-attention
        attn_out, _ = self.attention(combined_out, combined_out, combined_out)

        # Global average pooling
        pooled = attn_out.mean(dim=1)

        # Classification
        output = self.classifier(pooled)

        return output
