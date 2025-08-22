"""Script to visualize log mel spectrograms for each class.

This script plots:
1. Random examples of log mel spectrograms for each class
2. Average log mel spectrograms for each class
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from collections import defaultdict
import random

from modeling.preprocessing import (
    load_and_normalize_wav,
    extract_overlapping_segments,
    extract_mel_spectrogram,
)
from modeling.data_loading import load_blocked_split_features
from utils.config import EXTERNAL_DATA_DIR, FIGURES_DIR
from training_config import (
    DEFAULT_ALLOWED_LABELS,
    DEFAULT_TEST_FRACTION,
    DEFAULT_ENSURE_LABEL_COVERAGE,
)


def plot_random_class_examples(
    file_pairs,
    allowed_labels,
    window_size_sec=0.3,
    window_overlap=0.5,
    n_mels=13,
    n_example_per_class_per_file=20,
):
    """Plot random log mel spectrogram examples for each class."""

    # Dictionary to store spectrograms by class
    class_spectrograms = defaultdict(list)

    print("Loading and processing audio files...")

    # Process each file pair
    for wav_path, annotation_path in file_pairs:
        if not wav_path.exists() or not annotation_path.exists():
            print(f"Warning: Skipping {wav_path} or {annotation_path} - file not found")
            continue

        print(f"Processing {wav_path.name}...")

        # Extract segments and labels
        segments, labels = extract_overlapping_segments(
            wav_path,
            annotation_path,
            window_size_sec=window_size_sec,
        )

        # Load sample rate for spectrogram generation
        _, sample_rate = load_and_normalize_wav(wav_path)

        # Generate spectrograms for each segment
        # only store a limited number per class to avoid memory issues
        n_labels = np.unique(labels)
        label_counts = {label: 0 for label in n_labels}

        for segment, label in zip(segments, labels):
            if label in allowed_labels:
                if len(segment) > 0:  # Skip empty segments
                    if label_counts[label] < n_example_per_class_per_file:
                        label_counts[label] += 1
                        mel_spec = extract_mel_spectrogram(
                            segment, sample_rate, n_mels=n_mels, hop_length=None
                        )
                        class_spectrograms[label].append(mel_spec)

    # Plot random examples for each class
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"Random Log Mel Spectrogram Examples by Class (Window: {window_size_sec}s)",
        fontsize=16,
    )

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Define class colors for consistency
    class_colors = {
        "b": "red",  # bowel movement
        "mb": "darkred",  # major bowel movement
        "h": "orange",  # heart sound
        "n": "blue",  # normal
        "silence": "gray",  # silence
    }

    for idx, (class_name, spectrograms) in enumerate(class_spectrograms.items()):
        if idx >= 5:  # Only plot first 5 classes
            break

        if len(spectrograms) == 0:
            print(f"Warning: No spectrograms found for class '{class_name}'")
            continue

        # Choose a random spectrogram
        random_idx = random.randint(0, len(spectrograms) - 1)
        random_spec = spectrograms[random_idx]

        # Plot the spectrogram
        ax = axes[idx]
        im = ax.imshow(
            random_spec,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=-3,
            vmax=3,  # Consistent color scale
        )

        ax.set_title(
            f"Class: {class_name}\n(Example {random_idx + 1}/{len(spectrograms)})",
            color=class_colors.get(class_name, "black"),
        )
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        print(f"Class '{class_name}': {len(spectrograms)} spectrograms")

    # Hide the last subplot if we have fewer than 6 classes
    if len(class_spectrograms) < 6:
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.show()

    return class_spectrograms


def plot_class_averages(class_spectrograms):
    """Plot average log mel spectrograms for each class."""

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Average Log Mel Spectrograms by Class", fontsize=16)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Define class colors for consistency
    class_colors = {
        "b": "red",  # bowel movement
        "mb": "darkred",  # major bowel movement
        "h": "orange",  # heart sound
        "n": "blue",  # normal
        "silence": "gray",  # silence
    }

    for idx, (class_name, spectrograms) in enumerate(class_spectrograms.items()):
        if idx >= 5:  # Only plot first 5 classes
            break

        if len(spectrograms) == 0:
            continue

        # Compute average spectrogram
        avg_spec = np.mean(spectrograms, axis=0)

        # Plot the average spectrogram
        ax = axes[idx]
        im = ax.imshow(
            avg_spec,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=-3,
            vmax=3,  # Consistent color scale
        )

        ax.set_title(
            f"Class: {class_name}\n(Average of {len(spectrograms)} samples)",
            color=class_colors.get(class_name, "black"),
        )
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        print(f"Class '{class_name}' average: shape {avg_spec.shape}")

    # Hide the last subplot if we have fewer than 6 classes
    if len(class_spectrograms) < 6:
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.show()


# Example usage functions (can be imported and used elsewhere)
def get_class_spectrograms(file_pairs, window_size_sec=0.3, n_mels=13):
    """Get spectrograms organized by class."""
    class_spectrograms = defaultdict(list)

    for wav_path, annotation_path in file_pairs:
        if not wav_path.exists() or not annotation_path.exists():
            continue

        segments, labels = extract_overlapping_segments(
            wav_path, annotation_path, window_size_sec=window_size_sec
        )

        _, sample_rate = load_and_normalize_wav(wav_path)

        for segment, label in zip(segments, labels):
            if len(segment) > 0:
                mel_spec = extract_mel_spectrogram(segment, sample_rate, n_mels=n_mels)
                class_spectrograms[label].append(mel_spec)

    return dict(class_spectrograms)


def plot_single_class_comparison(class_spectrograms, class_name, n_examples=5):
    """Plot multiple examples from a single class for comparison.
    Args:
        class_spectrograms: Dictionary of spectrograms organized by class
        class_name: Name of the class to plot
        n_examples: Number of examples to plot
    Returns:
        None
    """
    if class_name not in class_spectrograms or len(class_spectrograms[class_name]) == 0:
        print(f"No spectrograms found for class '{class_name}'")
        return

    spectrograms = class_spectrograms[class_name]
    n_examples = min(n_examples, len(spectrograms))

    # Randomly select examples
    selected_indices = random.sample(range(len(spectrograms)), n_examples)

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]

    fig.suptitle(f"Multiple Examples from Class: {class_name}", fontsize=16)

    for i, idx in enumerate(selected_indices):
        spec = spectrograms[idx]
        im = axes[i].imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        axes[i].set_title(f"Example {i+1}")
        axes[i].set_xlabel("Time Frames")
        if i == 0:
            axes[i].set_ylabel("Mel Frequency Bins")

        plt.colorbar(im, ax=axes[i], shrink=0.8)

    plt.tight_layout()
    plt.show()


def main_visualization(file_pairs: list, allowed_labels: list, n_mels: int = 13):
    """Main function to run the visualization."""

    # Set random seed for reproducible plots
    random.seed(42)
    np.random.seed(42)

    print("Starting spectrogram visualization...")

    # Plot random examples
    class_spectrograms = plot_random_class_examples(
        file_pairs, allowed_labels, window_size_sec=0.3, n_mels=n_mels
    )

    # Plot class averages
    plot_class_averages(class_spectrograms)

    print("Visualization complete!")

    return class_spectrograms


if __name__ == "__main__":
    file_pairs = [
        (
            EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.wav",
            EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.txt",
        ),
        # (
        #     EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav",
        #     EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.txt",
        # ),
    ]
    main_visualization(
        file_pairs=file_pairs, allowed_labels=DEFAULT_ALLOWED_LABELS, n_mels=52
    )
