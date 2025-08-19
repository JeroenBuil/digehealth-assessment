import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from config import EXTERNAL_DATA_DIR

FALLBACK_LABEL = "silence"  # In case no annotations are present for a window, we assign a fallback label


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

        # Find annotations that overlap with the current segment
        overlaps = annotations[
            (annotations["end"] > seg_start_time)
            & (annotations["start"] < seg_end_time)
        ].copy()
        if overlaps.empty:
            label = FALLBACK_LABEL  # add silence label if no annotations are present
        else:
            if len(overlaps) == 1:
                # If only one annotation overlaps, use its label
                label = overlaps.iloc[0]["label"]
            else:
                # Assign label of annotation whose center is closest to segment center
                seg_center = (seg_start_time + seg_end_time) / 2
                centers = (overlaps["start"].values + overlaps["end"].values) / 2
                center_dists = np.abs(centers - seg_center)
                label = overlaps.iloc[center_dists.argmin()]["label"]
        # Append the segment and its label
        segments.append(segment)
        labels.append(label)
    return segments, labels


def extract_mfcc_features(segment, sample_rate):
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(segment)
    features = np.hstack(
        [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(zcr), np.std(zcr)]
    )
    return features


def extract_mel_spectrogram(
    segment, sample_rate, n_mels=128, n_fft=2048, hop_length=512, max_len=128
):
    """
    Extracts a fixed-size Mel-spectrogram for CNN input.
    - Pads or truncates to max_len frames.
    - Returns (n_mels, max_len) array.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate along time axis
    if log_mel_spec.shape[1] < max_len:
        pad_width = max_len - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        log_mel_spec = log_mel_spec[:, :max_len]

    return log_mel_spec


if __name__ == "__main__":
    """Example usage of the segment extraction function.
    This script extracts overlapping segments from a WAV file based on annotations
    and prints the number of segments and their labels.
    """
    # Example usage
    wav_path = EXTERNAL_DATA_DIR / "Tech Test/AS_1.wav"
    annotation_path = EXTERNAL_DATA_DIR / "Tech Test/AS_1.txt"
    segments, labels = extract_overlapping_segments(wav_path, annotation_path)
    print(f"Extracted {len(segments)} segments with labels: {labels[:10]}")
