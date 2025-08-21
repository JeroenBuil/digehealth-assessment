import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                # Prefer b/mb/h over n over silence; tie-break by center proximity
                def _priority(lbl: str) -> int:
                    if lbl in ("b", "mb", "h"):
                        return 3
                    if lbl == "n":
                        return 2
                    if lbl == "silence":
                        return 1
                    return 0

                seg_center = (seg_start_time + seg_end_time) / 2
                centers = (overlaps["start"].values + overlaps["end"].values) / 2
                # Compute per-row priority then choose best; if tie, closest center wins
                prios = overlaps["label"].apply(_priority).values
                best_prio = prios.max()
                candidate_idx = np.where(prios == best_prio)[0]
                if len(candidate_idx) == 1:
                    chosen = candidate_idx[0]
                else:
                    candidate_centers = centers[candidate_idx]
                    chosen_rel = np.abs(candidate_centers - seg_center).argmin()
                    chosen = candidate_idx[chosen_rel]
                label = overlaps.iloc[chosen]["label"]
        # Append the segment and its label
        segments.append(segment)
        labels.append(label)
    return segments, labels


def extract_overlapping_windows(wav_path, window_size_sec=0.5, window_overlap=0.75):
    """Extract overlapping fixed-size windows and their [start,end] times from audio.

    This does not require annotations and is suitable for inference.
    Returns:
        segments (list[np.ndarray]), times (list[tuple[float,float]]), sample_rate (int), hop_sec (float)
    """
    y, sample_rate = load_and_normalize_wav(wav_path)
    n_samples = len(y)
    win_length = int(window_size_sec * sample_rate)
    hop_length = int(window_overlap * window_size_sec * sample_rate)
    segments = []
    times = []
    for start in range(0, max(0, n_samples - win_length + 1), hop_length):
        segment = y[start : start + win_length]
        segments.append(segment)
        times.append((start / sample_rate, (start + win_length) / sample_rate))
    hop_sec = hop_length / sample_rate
    return segments, times, sample_rate, hop_sec


def assign_window_labels_from_annotations(times, ann_df, known_classes):
    """Assign a label to each window [start,end] using center-closest strategy.

    Falls back to 'silence' if no overlap or unknown label.
    """
    labels = []
    for s, e in times:
        overlaps = ann_df[(ann_df["end"] > s) & (ann_df["start"] < e)].copy()
        if overlaps.empty:
            label = "silence"
        else:
            if len(overlaps) == 1:
                label = overlaps.iloc[0]["label"]
            else:
                # Prefer b/mb/h over n over silence; tie-break by center proximity
                def _priority(lbl: str) -> int:
                    if lbl in ("b", "mb", "h"):
                        return 3
                    if lbl == "n":
                        return 2
                    if lbl == "silence":
                        return 1
                    return 0

                seg_center = (s + e) / 2
                centers = (overlaps["start"].values + overlaps["end"].values) / 2
                prios = overlaps["label"].apply(_priority).values
                best_prio = prios.max()
                candidate_idx = np.where(prios == best_prio)[0]
                if len(candidate_idx) == 1:
                    chosen = candidate_idx[0]
                else:
                    candidate_centers = centers[candidate_idx]
                    chosen_rel = np.abs(candidate_centers - seg_center).argmin()
                    chosen = candidate_idx[chosen_rel]
                label = overlaps.iloc[chosen]["label"]
        label = label if label in known_classes else "silence"
        labels.append(label)
    return labels


def extract_mfcc_features(segment, sample_rate):
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(segment)
    features = np.hstack(
        [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(zcr), np.std(zcr)]
    )
    return features


def extract_mel_spectrogram(
    segment, sample_rate, n_mels=128, n_fft=None, hop_length=None, max_len=None
):
    """Extracts a log-mel spectrogram from an audio segment.
    Args:
        segment (np.ndarray): Audio segment.
        sample_rate (int): Sample rate of the audio.
        n_mels (int): Number of mel bands.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        max_len (int): Maximum length of the output spectrogram.
    Returns:
        log_mel_spec (np.ndarray): Log-mel spectrogram of shape (n_mels, max_len).
    """
    # Choose a safe n_fft/hop_length for short segments
    seg_len = int(len(segment))
    if n_fft is None or (isinstance(n_fft, int) and n_fft > seg_len):
        if seg_len < 2:
            # Too short to compute FFT; return a zero spectrogram
            log_mel_spec = np.zeros((n_mels, max_len), dtype=np.float32)
            return log_mel_spec
        # Use largest power of two <= segment length (typical for FFT)
        n_fft_eff = int(2 ** np.floor(np.log2(seg_len)))
    else:
        n_fft_eff = int(n_fft)

    hop_length_eff = max(1, n_fft_eff // 4) if hop_length is None else int(hop_length)

    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft_eff,
        hop_length=hop_length_eff,
        power=2.0,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Optionally pad or truncate to a fixed width if max_len is provided
    if max_len is not None:
        if log_mel_spec.shape[1] < max_len:
            pad_width = max_len - log_mel_spec.shape[1]
            log_mel_spec = np.pad(
                log_mel_spec, ((0, 0), (0, pad_width)), mode="constant"
            )
        else:
            log_mel_spec = log_mel_spec[:, :max_len]

    # Normalize (zero mean, unit variance)
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (
        np.std(log_mel_spec) + 1e-6
    )

    return log_mel_spec


if __name__ == "__main__":
    """Example usage of the segment extraction function.
    This script extracts overlapping segments from a WAV file based on annotations
    and prints the number of segments and their labels.
    """
    # Example usage
    wav_path = EXTERNAL_DATA_DIR / "Tech Test/AS_1.wav"
    annotation_path = EXTERNAL_DATA_DIR / "Tech Test/AS_1.txt"

    # Load wave and read in sample rate
    wav_y, sample_rate = load_and_normalize_wav(wav_path)

    segments, labels = extract_overlapping_segments(wav_path, annotation_path)
    n_labels_to_print = 10
    print(
        f"Extracted {len(segments)} segments , the first {n_labels_to_print}: {labels[:n_labels_to_print]}"
    )

    # Example of extracting features from the first segment
    mel_spec = extract_mel_spectrogram(segments[0], sample_rate=sample_rate)
    print(f"Shape of extracted features: {mel_spec.shape}")

    # Visualise mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect="auto", origin="lower", cmap="viridis")
    plt.show()
