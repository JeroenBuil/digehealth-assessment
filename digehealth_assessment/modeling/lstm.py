import torch
import torch.nn as nn


class BowelSoundLSTM(nn.Module):
    """LSTM model for classifying bowel sounds from spectrogram features."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = 40,  # number of mel bands
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_heads: int = 8,
    ):
        super(BowelSoundLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,  # number of mel bands
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.num_directions,
            num_heads=attention_heads,
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
        Forward pass for spectrograms.

        Args:
            x: Input tensor, shape (B, 1, H, W) or (B, H, W)
               Here H=mel bands (40), W=time steps (25)

        Returns:
            logits: (B, num_classes)
        """
        # Handle channel dimension
        if x.dim() == 4:
            if x.size(1) == 1:
                x = x.squeeze(1)  # [B, H, W]
            else:
                x = x.mean(dim=1)  # combine channels if >1

        # Transpose to (B, seq_len=W, features=H)
        x = x.transpose(1, 2)  # [B, W, H]

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # [B, W, hidden_size*num_directions]

        # Apply attention over the sequence dimension
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, W, hidden*dir]

        # Global average pooling over time steps
        pooled = attn_out.mean(dim=1)  # [B, hidden*dir]

        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]

        return logits
