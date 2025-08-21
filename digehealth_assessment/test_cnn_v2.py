"""Test script for the improved CNN architectures."""

import torch
import numpy as np
from modeling.cnn import BowelSoundCNNv2, BowelSoundCNNv3


def test_cnn_v2():
    """Test CNN v2 architecture."""
    print("Testing CNN v2...")

    # Create model
    model = BowelSoundCNNv2(
        num_classes=5,
        input_shape=(1, 128, 50),  # (channels, height, width)
        dropout=0.3,
    )

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 50)

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
    print("✓ CNN v2 test passed!")

    return model


def test_cnn_v3():
    """Test CNN v3 architecture."""
    print("\nTesting CNN v3...")

    # Create model
    model = BowelSoundCNNv3(
        num_classes=5,
        input_shape=(1, 128, 50),  # (channels, height, width)
        dropout=0.3,
    )

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 50)

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
    print("✓ CNN v3 test passed!")

    return model


def compare_architectures():
    """Compare the three CNN architectures."""
    print("\n" + "=" * 50)
    print("ARCHITECTURE COMPARISON")
    print("=" * 50)

    input_shape = (1, 128, 50)
    num_classes = 5

    # Original CNN
    from modeling.cnn import BowelSoundCNN

    original = BowelSoundCNN(num_classes, input_shape)
    original_params = sum(p.numel() for p in original.parameters())

    # CNN v2
    v2 = BowelSoundCNNv2(num_classes, input_shape, dropout=0.3)
    v2_params = sum(p.numel() for p in v2.parameters())

    # CNN v3
    v3 = BowelSoundCNNv3(num_classes, input_shape, dropout=0.3)
    v3_params = sum(p.numel() for p in v3.parameters())

    print(f"Original CNN: {original_params:,} parameters")
    print(f"CNN v2:       {v2_params:,} parameters")
    print(f"CNN v3:       {v3_params:,} parameters")

    print(f"\nParameter increase:")
    print(f"  v2 vs original: {v2_params/original_params:.1f}x")
    print(f"  v3 vs original: {v3_params/original_params:.1f}x")

    # Test forward pass for all
    x = torch.randn(2, 1, 128, 50)

    with torch.no_grad():
        out_orig = original(x)
        out_v2 = v2(x)
        out_v3 = v3(x)

    print(f"\nForward pass shapes:")
    print(f"  Original: {out_orig.shape}")
    print(f"  CNN v2:   {out_v2.shape}")
    print(f"  CNN v3:   {out_v3.shape}")


if __name__ == "__main__":
    try:
        test_cnn_v2()
        test_cnn_v3()
        compare_architectures()
        print("\n🎉 All CNN tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
