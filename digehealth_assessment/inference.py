import librosa
import numpy as np
from pathlib import Path
from import_librosa import extract_features

def predict_segments(audio_path, clf, sr=22050, window_size=1.0, hop_size=0.5):
    y, _ = librosa.load(audio_path, sr=sr)
    n_samples = len(y)
    win_length = int(window_size * sr)
    hop_length = int(hop_size * sr)
    segments = []
    times = []
    for start in range(0, n_samples - win_length + 1, hop_length):
        end = start + win_length
        segment = y[start:end]
        features = extract_features(segment, sr=sr).reshape(1, -1)
        pred = clf.predict(features)[0]
        segments.append(pred)
        times.append((start / sr, end / sr))
    # Merge consecutive segments with the same label
    results = []
    if segments:
        cur_label = segments[0]
        cur_start = times[0][0]
        cur_end = times[0][1]
        for i in range(1, len(segments)):
            if segments[i] == cur_label:
                cur_end = times[i][1]
            else:
                results.append((cur_start, cur_end, cur_label))
                cur_label = segments[i]
                cur_start = times[i][0]
                cur_end = times[i][1]
        results.append((cur_start, cur_end, cur_label))
    return results

# Example usage:
# from train_model import clf
# results = predict_segments('data/external/TechTest/file2.wav', clf)
# for start, end, label in results:
#     print(f"{start:.2f}\t{end:.2f}\t{label}")