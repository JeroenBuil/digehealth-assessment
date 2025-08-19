import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

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

# Only keep 'b', 'mb', 'h', 'n' and silence labels, convert all remaining labels to 'n'
allowed_labels = {"b", "mb", "h", "n", "silence"}
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
print("\nClass distribution in TRAIN set:")
for label, count in zip(unique_train, counts_train):
    print(f"  {label}: {count}")


def balance_training_data(X_train, y_train):
    """
    Balance the training data using undersampling for majority classes and SMOTE for minority classes.
    Returns balanced X_train and y_train.
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    import numpy as np

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    target_size = int(np.median(counts_train))
    class_counts = dict(zip(unique_train, counts_train))
    under_classes = {
        label: target_size
        for label, count in class_counts.items()
        if count > target_size
    }
    over_classes = {
        label: target_size
        for label, count in class_counts.items()
        if count < target_size
    }

    # Step 1: Undersample majority classes
    under = (
        RandomUnderSampler(sampling_strategy=under_classes, random_state=42)
        if under_classes
        else None
    )

    # Step 2: Oversample minority classes
    smote = (
        SMOTE(sampling_strategy=over_classes, random_state=42) if over_classes else None
    )

    # Build pipeline dynamically
    steps = []
    if under:
        steps.append(("under", under))
    if smote:
        steps.append(("smote", smote))

    if steps:
        pipeline = Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    return X_res, y_res


print("Balancing training data...")
X_train, y_train = balance_training_data(X_train, y_train)

print("\nClass distribution in TRAIN set after balancing:")
unique_train_bal, counts_train_bal = np.unique(y_train, return_counts=True)
for label, count in zip(unique_train_bal, counts_train_bal):
    print(f"  {label}: {count}")

print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Evaluating on test set...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()
