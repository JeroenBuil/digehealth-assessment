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

from preprocessing import (
    load_and_normalize_wav,
    extract_overlapping_segments,
    extract_mel_spectrogram,
)
from config import EXTERNAL_DATA_DIR


def plot_random_class_examples(file_pairs, window_size_sec=0.3, n_mels=128):
    """Plot random log mel spectrogram examples for each class."""

    # Dictionary to store spectrograms by class
    class_spectrograms = defaultdict(list)
    class_segments = defaultdict(list)

    print("Loading and processing audio files...")

    # Process each file pair
    for wav_path, annotation_path in file_pairs:
        if not wav_path.exists() or not annotation_path.exists():
            print(f"Warning: Skipping {wav_path} or {annotation_path} - file not found")
            continue

        print(f"Processing {wav_path.name}...")

        # Extract segments and labels
        segments, labels = extract_overlapping_segments(
            wav_path, annotation_path, window_size_sec=window_size_sec
        )

        # Load sample rate for spectrogram generation
        _, sample_rate = load_and_normalize_wav(wav_path)

        # Generate spectrograms for each segment
        for segment, label in zip(segments, labels):
            if len(segment) > 0:  # Skip empty segments
                mel_spec = extract_mel_spectrogram(segment, sample_rate, n_mels=n_mels)
                class_spectrograms[label].append(mel_spec)
                class_segments[label].append(segment)

    # Plot random examples for each class
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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

    return class_spectrograms, class_segments


def plot_class_averages(class_spectrograms):
    """Plot average log mel spectrograms for each class."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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


def plot_class_statistics(class_spectrograms):
    """Plot statistical information about each class."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Class Statistics and Distributions", fontsize=16)

    # Flatten axes
    axes = axes.flatten()

    # 1. Sample count per class
    ax1 = axes[0]
    class_names = list(class_spectrograms.keys())
    sample_counts = [len(class_spectrograms[name]) for name in class_names]

    bars = ax1.bar(
        class_names, sample_counts, color=["red", "darkred", "orange", "blue", "gray"]
    )
    ax1.set_title("Number of Samples per Class")
    ax1.set_ylabel("Sample Count")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{count}",
            ha="center",
            va="bottom",
        )

    # 2. Spectrogram energy distribution
    ax2 = axes[1]
    for class_name in class_names:
        if len(class_spectrograms[class_name]) > 0:
            # Calculate energy (mean of squared values) for each spectrogram
            energies = [np.mean(spec**2) for spec in class_spectrograms[class_name]]
            ax2.hist(energies, alpha=0.7, label=class_name, bins=20)

    ax2.set_title("Energy Distribution by Class")
    ax2.set_xlabel("Mean Squared Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    # 3. Frequency band analysis
    ax3 = axes[2]
    for class_name in class_names:
        if len(class_spectrograms[class_name]) > 0:
            # Calculate mean frequency response across time
            avg_spec = np.mean(class_spectrograms[class_name], axis=0)
            freq_response = np.mean(avg_spec, axis=1)  # Average across time
            mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=22050)
            ax3.plot(mel_freqs, freq_response, label=class_name, linewidth=2)

    ax3.set_title("Average Frequency Response by Class")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.legend()
    ax3.set_xscale("log")

    # 4. Temporal dynamics
    ax4 = axes[3]
    for class_name in class_names:
        if len(class_spectrograms[class_name]) > 0:
            # Calculate temporal envelope
            avg_spec = np.mean(class_spectrograms[class_name], axis=0)
            temporal_envelope = np.mean(avg_spec, axis=0)  # Average across frequency
            time_frames = np.arange(len(temporal_envelope))
            ax4.plot(time_frames, temporal_envelope, label=class_name, linewidth=2)

    ax4.set_title("Temporal Envelope by Class")
    ax4.set_xlabel("Time Frame")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def main_visualization():
    """Main function to run the visualization."""

    # Define file pairs (same as in training config)
    file_pairs = [
        (
            EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.wav",
            EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.txt",
        ),
        (
            EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav",
            EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.txt",
        ),
    ]

    # Set random seed for reproducible plots
    random.seed(42)
    np.random.seed(42)

    print("Starting spectrogram visualization...")

    # Plot random examples
    class_spectrograms, class_segments = plot_random_class_examples(
        file_pairs, window_size_sec=0.3, n_mels=128
    )

    # Plot class averages
    plot_class_averages(class_spectrograms)

    # Plot class statistics
    plot_class_statistics(class_spectrograms)

    print("Visualization complete!")

    return class_spectrograms, class_segments


# Example usage functions (can be imported and used elsewhere)
def get_class_spectrograms(file_pairs, window_size_sec=0.3, n_mels=128):
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
    """Plot multiple examples from a single class for comparison."""
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
