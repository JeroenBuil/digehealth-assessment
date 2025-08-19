import numpy as np

from sklearn.preprocessing import StandardScaler
from preprocessing import (
    load_and_normalize_wav,
    extract_overlapping_segments,
    extract_mfcc_features,
    extract_spectrogram_features,
)


def load_data_and_extract_features(
    file_pairs: list,
    allowed_labels: set,
    feature_type: str,
    verbose=True,
):
    if feature_type not in ["mfcc", "spectrogram"]:
        ValueError(
            f"Invalid feature_type: {feature_type}. Must be 'mfcc' or 'spectrogram'."
        )

    if verbose == True:
        print("Loading and preparing data...")

    all_segments = []
    all_labels = []
    all_sample_rates = []
    for wav_path, ann_path in file_pairs:
        wav_y, sample_rate = load_and_normalize_wav(wav_path)
        segments, labels = extract_overlapping_segments(
            wav_path=wav_path, annotation_path=ann_path
        )
        all_segments.extend(segments)
        all_labels.extend(labels)
        all_sample_rates.extend([sample_rate] * len(segments))

    # Only keep allowed labels, convert others to 'n'
    all_labels = [label if label in allowed_labels else "n" for label in all_labels]

    if feature_type == "mfcc":
        X = np.array(
            [
                extract_mfcc_features(seg, sample_rate=sr)
                for seg, sr in zip(all_segments, all_sample_rates)
            ]
        )
    elif feature_type == "spectrogram":
        X = np.array(
            [
                extract_spectrogram_features(seg)
                for seg, sr in zip(all_segments, all_sample_rates)
            ]
        )
    else:
        raise ValueError(
            f"Invalid feature_type: {feature_type}. Must be 'mfcc' or 'spectrogram'."
        )

    y = np.array(all_labels)

    return X, y


def print_class_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    print(title)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")


def balance_training_data(X_train, y_train):
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

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

    under = (
        RandomUnderSampler(sampling_strategy=under_classes, random_state=42)
        if under_classes
        else None
    )
    smote = (
        SMOTE(sampling_strategy=over_classes, random_state=42) if over_classes else None
    )

    steps = []
    if under:
        steps.append(("under", under))
    if smote:
        steps.append(("smote", smote))

    if steps:
        from imblearn.pipeline import Pipeline

        pipeline = Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    return X_res, y_res


def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(clf, X_test, y_test):
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    import matplotlib.pyplot as plt

    print("Evaluating model...")

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
