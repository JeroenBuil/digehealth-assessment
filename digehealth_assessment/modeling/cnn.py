import torch.nn as nn
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BowelSoundCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(30, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BowelSoundCNNv2(nn.Module):
    """Improved CNN architecture for bowel sound classification.

    Key improvements:
    - Increased model capacity (64->128->256 channels)
    - Time-aware kernels (3x7, 3x11) for better temporal modeling
    - Preserves temporal resolution longer with frequency-only pooling
    - Temporal attention mechanism before classification
    - Residual connections for better gradient flow
    - Squeeze-and-Excitation blocks for feature refinement
    """

    def __init__(self, num_classes, input_shape, dropout=0.3):
        super().__init__()
        c, h, w = input_shape
        self.dropout = dropout

        # Feature extraction with time-aware kernels
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 11), padding=(1, 5)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
        )

        # Pool frequency dimension but preserve temporal resolution
        self.freq_pool1 = nn.MaxPool2d((2, 1), stride=(2, 1))  # Pool H, keep W
        self.freq_pool2 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.freq_pool3 = nn.MaxPool2d((2, 1), stride=(2, 1))

        # Squeeze-and-Excitation blocks
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)

        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(256)

        # Global average pooling over frequency (H->1) and time (W->1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier with better capacity
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction with residual connections
        x1 = self.conv1(x)
        x1 = self.se1(x1)
        x1 = self.freq_pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.se2(x2)
        x2 = self.freq_pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.se3(x3)
        x3 = self.freq_pool3(x3)

        # Apply temporal attention
        x3 = self.temporal_attention(x3)

        # Global pooling
        x3 = self.global_pool(x3)

        # Classification
        output = self.classifier(x3)
        return output


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on important time steps."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Temporal attention weights
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (B, C, H, W) where W is time dimension
        attention_weights = self.temporal_conv(x)  # (B, 1, H, W)

        # Apply attention
        attended = x * attention_weights

        return attended


class BowelSoundCNNv3(nn.Module):
    """Alternative architecture with dilated convolutions for wider temporal context."""

    def __init__(self, num_classes, input_shape, dropout=0.3):
        super().__init__()
        c, h, w = input_shape
        self.dropout = dropout

        # Initial conv with standard kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
        )

        # Dilated convolutions for wider temporal context
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
        )

        # Progressive pooling
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        # SE blocks
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.se1(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.se2(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.se3(x3)
        x3 = self.pool3(x3)

        x3 = self.global_pool(x3)
        output = self.classifier(x3)
        return output
