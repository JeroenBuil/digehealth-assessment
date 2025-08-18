import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from config import EXTERNAL_DATA_DIR


def load_annotations(txt_path):
    df = pd.read_csv(txt_path, sep="\t", header=None, names=["start", "end", "label"])
    return df


def load_and_normalize_wav(wav_path):
    """Load a wav file, convert to mono, and normalize its amplitude to [-1, 1]."""
    y, sample_rate = librosa.load(wav_path, sr=None, mono=True)
    max_val = np.max(np.abs(y))
    y = y / max_val if max_val != 0 else y
    return y, sample_rate


def extract_segments(wav_path, annotation_path):
    y, sample_rate = load_and_normalize_wav(wav_path)
    annotations = load_annotations(annotation_path)
    segments = []
    for _, row in annotations.iterrows():
        start_sample = int(row["start"] * sample_rate)
        end_sample = int(row["end"] * sample_rate)
        segment = y[start_sample:end_sample]
        segments.append((segment, row["label"]))
    return segments


def extract_overlapping_segments(
    wav_path, annotation_path, window_size_sec=0.5, window_overlap=0.5
):
    """Extracts overlapping segments from a WAV file based on annotations.
    Args:
        wav_path (str or Path): Path to the WAV file.
        annotation_path (str or Path): Path to the annotation file.
        window_size_sec (float): Size of each segment in seconds.
        window_overlap (float): Overlap between segments as a fraction of window size. Value between 0 and 1.
    Returns:
        segments (list): List of audio segments.
        labels (list): Corresponding labels for each segment.
    """
    y, sample_rate = load_and_normalize_wav(wav_path)
    annotations = load_annotations(annotation_path)

    n_samples = len(y)
    win_length = int(window_size_sec * sample_rate)
    hop_length = int(window_overlap * window_size_sec * sample_rate)
    segments = []
    labels = []

    # Iterate over the audio signal with the specified hop length
    for start in range(0, n_samples - win_length + 1, hop_length):
        seg_start_time = start / sample_rate
        seg_end_time = (start + win_length) / sample_rate
        segment = y[start : start + win_length]

        # Find annotations that fully contain the segment
        contains = annotations[
            (annotations["start"] <= seg_start_time)
            & (annotations["end"] >= seg_end_time)
        ]
        if not contains.empty:
            # If the segment is fully inside an annotation, use that label
            label = contains.iloc[0]["label"]
        else:
            # Otherwise, check for overlapping annotations
            overlaps = annotations[
                (annotations["end"] > seg_start_time)
                & (annotations["start"] < seg_end_time)
            ].copy()
            if not overlaps.empty:
                # Find annotation with max overlap
                overlaps["overlap"] = overlaps.apply(
                    lambda row: min(row["end"], seg_end_time)
                    - max(row["start"], seg_start_time),
                    axis=1,
                )
                label = overlaps.loc[overlaps["overlap"].idxmax(), "label"]
            else:
                label = "s"  # add silence label if no annotations are present
        # Append the segment and its label
        segments.append(segment)
        labels.append(label)
    return segments, labels


def extract_features(segment, sample_rate):
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(segment)
    features = np.hstack(
        [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(zcr), np.std(zcr)]
    )
    return features


# Example usage
wav_path = EXTERNAL_DATA_DIR / "Tech Test/23M74M.wav"
annotation_path = EXTERNAL_DATA_DIR / "Tech Test/23M74M.txt"
wav_y, sample_rate = load_and_normalize_wav(wav_path)

# Extract overlapping segments and features
segments, labels = extract_overlapping_segments(wav_path, annotation_path)
X = np.array([extract_features(seg, sample_rate=sample_rate) for seg in segments])
y = np.array(labels)
