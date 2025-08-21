"""Simple test script to verify LSTM model implementation."""

import torch
import numpy as np
from modeling.lstm import BowelSoundLSTM, BowelSoundLSTMWithFeatures


def test_lstm_model():
    """Test basic LSTM model functionality."""
    print("Testing LSTM model...")

    # Test basic LSTM
    model = BowelSoundLSTM(
        num_classes=5,
        input_size=128,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )

    # Create dummy input
    batch_size = 4
    channels = 1
    height = 128
    width = 50

    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 5)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (
        batch_size,
        5,
    ), f"Expected shape ({batch_size}, 5), got {output.shape}"
    print("✓ Basic LSTM test passed!")

    return model


def test_lstm_with_features():
    """Test LSTM model with additional features."""
    print("\nTesting LSTM with features...")

    model = BowelSoundLSTMWithFeatures(
        num_classes=5,
        spectrogram_height=128,
        feature_dim=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )

    # Create dummy inputs
    batch_size = 4
    channels = 1
    height = 128
    width = 50
    feature_dim = 20

    spectrogram = torch.randn(batch_size, channels, height, width)
    features = torch.randn(batch_size, width, feature_dim)

    # Forward pass
    output = model(spectrogram, features)

    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 5)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (
        batch_size,
        5,
    ), f"Expected shape ({batch_size}, 5), got {output.shape}"
    print("✓ LSTM with features test passed!")

    return model


if __name__ == "__main__":
    try:
        test_lstm_model()
        test_lstm_with_features()
        print("\n🎉 All LSTM tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
