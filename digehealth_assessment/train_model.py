import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import (
    extract_overlapping_segments,
    extract_features,
    load_and_normalize_wav,
)
from config import EXTERNAL_DATA_DIR
from pathlib import Path
from imblearn.over_sampling import SMOTE

# List of (wav, annotation) file pairs
file_pairs = [
    (
        EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav",
        EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.txt",
    ),
    (
        EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.wav",
        EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.txt",
    ),
]

all_segments = []
all_labels = []
all_sample_rates = []

print("Loading and segmenting audio files...")
for wav_path, ann_path in file_pairs:
    print(f"  Processing {wav_path.name} ...")
    wav_y, sample_rate = load_and_normalize_wav(wav_path)
    segments, labels = extract_overlapping_segments(
        wav_path=wav_path, annotation_path=ann_path
    )
    all_segments.extend(segments)
    all_labels.extend(labels)
    all_sample_rates.extend([sample_rate] * len(segments))

# Only keep 'b', 'mb', 'h' and 'n' labels, convert all remaining labels to 'n'
allowed_labels = {"b", "mb", "h", "n"}
all_labels = [label if label in allowed_labels else "n" for label in all_labels]

print("Extracting features from all segments...")
X = np.array(
    [
        extract_features(seg, sample_rate=sr)
        for seg, sr in zip(all_segments, all_sample_rates)
    ]
)
y = np.array(all_labels)

# Count and print all samples in each class before train/test split
unique, counts = np.unique(y, return_counts=True)
print("Class distribution BEFORE train/test split:")
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Count and print all samples in each class after train/test split
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("\nClass distribution in TRAIN set:")
for label, count in zip(unique_train, counts_train):
    print(f"  {label}: {count}")
print("\nClass distribution in TEST set:")
for label, count in zip(unique_test, counts_test):
    print(f"  {label}: {count}")

print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Balancing training data with SMOTE...")
# Find the minimum class count in y_train for SMOTE k_neighbors
(unique_train, counts_train) = np.unique(y_train, return_counts=True)
min_class_count = np.min(counts_train)
k_neighbors = max(1, min_class_count - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Training classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_bal, y_train_bal)

print("Evaluating on test set...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
