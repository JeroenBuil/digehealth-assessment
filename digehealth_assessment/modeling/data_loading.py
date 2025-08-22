"""Data loading and preprocessing utilities."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .preprocessing import (
    load_and_normalize_wav,
    extract_overlapping_segments,
    extract_mfcc_features,
    extract_mel_spectrogram,
)


def load_blocked_split_features(
    file_pairs: list,
    allowed_labels: list,
    feature_type: str,
    window_size_sec: float = 1,
    window_overlap: float = 0.75,
    test_fraction: float = 0.2,
    ensure_label_coverage: bool = True,
) -> Tuple[List, List, List, List]:
    """Load features and perform a time-ordered blocked split per file.

    The last `test_fraction` of segments from each file become that file's test block,
    keeping test segments consecutive in time. Optionally adjusts the split points to
    include at least one instance of each present class in the overall test set.
    """
    assert 0.0 < test_fraction < 1.0

    per_file_data = []
    all_present_labels = set()

    for wav_path, ann_path in file_pairs:
        wav_y, sample_rate = load_and_normalize_wav(wav_path)
        segments, labels = extract_overlapping_segments(
            wav_path=wav_path,
            annotation_path=ann_path,
            window_size_sec=window_size_sec,
            window_overlap=window_overlap,
        )
        # Normalize labels
        labels = [label if label in allowed_labels else "n" for label in labels]
        all_present_labels.update(labels)

        # Extract features in time order
        if feature_type == "mfcc":
            X_file = [
                extract_mfcc_features(seg, sample_rate=sample_rate) for seg in segments
            ]
        elif feature_type == "spectrogram":
            # Keep per-file list of variable-width spectrograms
            X_file = [
                extract_mel_spectrogram(seg, sample_rate=sample_rate, max_len=None)
                for seg in segments
            ]
        else:
            raise ValueError(
                f"Invalid feature_type: {feature_type}. Must be 'mfcc' or 'spectrogram'."
            )

        per_file_data.append(
            {
                "X": X_file,
                "y": labels,
            }
        )

    # Initial per-file split indices (keep last block for test)
    split_indices = []
    for item in per_file_data:
        n = len(item["y"])
        split_idx = max(0, min(n, int(round(n * (1.0 - test_fraction)))))
        # Ensure at least one test sample if possible
        if n > 0 and split_idx == n:
            split_idx = n - 1
        split_indices.append(split_idx)

    def build_sets():
        X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
        for item, split_idx in zip(per_file_data, split_indices):
            X_file, y_file = item["X"], item["y"]
            X_tr_list.append(X_file[:split_idx])
            y_tr_list.extend(y_file[:split_idx])
            X_te_list.append(X_file[split_idx:])
            y_te_list.extend(y_file[split_idx:])
        # Flatten lists without forcing into a numpy array to preserve variable widths
        X_train = [item for sub in X_tr_list for item in sub]
        X_test = [item for sub in X_te_list for item in sub]
        return X_train, y_tr_list, X_test, y_te_list

    # Optionally adjust split points to ensure label coverage in overall test set
    if ensure_label_coverage and len(per_file_data) > 0:
        _, _, _, y_test_tmp = build_sets()
        test_present = set(y_test_tmp)
        # Only try to cover labels that are present somewhere in the data
        missing = [lbl for lbl in all_present_labels if lbl not in test_present]
        if missing:
            for missing_lbl in missing:
                # Find a file containing the missing label
                for idx, item in enumerate(per_file_data):
                    y_file = item["y"]
                    if missing_lbl in y_file:
                        # Find last occurrence index of the label in that file
                        last_idx = max(
                            i for i, v in enumerate(y_file) if v == missing_lbl
                        )
                        # Ensure split index is at or before last occurrence to include it in test
                        if last_idx < split_indices[idx]:
                            split_indices[idx] = last_idx
                        break

    X_train, y_train, X_test, y_test = build_sets()
    return X_train, y_train, X_test, y_test


def load_data_and_extract_features(
    file_pairs: list,
    allowed_labels: set,
    feature_type: str,
    window_size_sec: float = 1,
    window_overlap: float = 0.75,
    verbose=True,
) -> Tuple[List, List]:
    """Load and extract features from audio files."""
    if feature_type not in ["mfcc", "spectrogram"]:
        raise ValueError(
            f"Invalid feature_type: {feature_type}. Must be 'mfcc' or 'spectrogram'."
        )

    if verbose:
        print("Loading and preparing data...")

    all_segments = []
    all_labels = []
    all_sample_rates = []
    for wav_path, ann_path in file_pairs:
        wav_y, sample_rate = load_and_normalize_wav(wav_path)
        segments, labels = extract_overlapping_segments(
            wav_path=wav_path,
            annotation_path=ann_path,
            window_size_sec=window_size_sec,
            window_overlap=window_overlap,
        )
        all_segments.extend(segments)
        all_labels.extend(labels)
        all_sample_rates.extend([sample_rate] * len(segments))

    # Only keep allowed labels, convert others to 'n'
    all_labels = [label if label in allowed_labels else "n" for label in all_labels]

    if feature_type == "mfcc":
        X = [
            extract_mfcc_features(seg, sample_rate=sr)
            for seg, sr in zip(all_segments, all_sample_rates)
        ]
    elif feature_type == "spectrogram":
        # Keep variable-width spectrograms as a list; avoid global padding
        X = [
            extract_mel_spectrogram(seg, sample_rate=sr, max_len=None)
            for seg, sr in zip(all_segments, all_sample_rates)
        ]
    else:
        raise ValueError(
            f"Invalid feature_type: {feature_type}. Must be 'mfcc' or 'spectrogram'."
        )

    y = all_labels
    return X, y
