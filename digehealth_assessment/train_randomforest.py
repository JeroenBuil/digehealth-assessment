import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

from config import EXTERNAL_DATA_DIR, MODELS_DIR, FIGURES_DIR
from pathlib import Path
from ml_pipeline import (
    load_blocked_split_features,
    print_class_distribution,
    balance_training_data,
)
from evaluation import plot_confusion_and_roc, plot_feature_importances
from preprocessing import extract_mfcc_features

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
allowed_labels = ["b", "mb", "h", "n", "silence"]

window_size_sec = 0.1  # Size of each segment in seconds
window_overlap = 0.5  # Overlap between segments as a fraction of window size

retrain_model = True  # Set to False to evaluate an existing model

print("Loading + splitting data + Extracting features... (this may take a while)")
# Load and prepare data (blocked split per file to keep test time-consecutive)
X_train, y_train, X_test, y_test = load_blocked_split_features(
    file_pairs=file_pairs,
    allowed_labels=allowed_labels,
    feature_type="mfcc",
    window_size_sec=window_size_sec,
    window_overlap=window_overlap,
    test_fraction=0.2,
    ensure_label_coverage=True,
)

print_class_distribution(y_train, "\nClass distribution in TRAIN set:")
print_class_distribution(y_test, "\nClass distribution in Test set:")

print("Balancing training data...")
X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
print_class_distribution(
    y_train_bal, "\nClass distribution in TRAIN set after balancing:"
)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_bal)
y_test_enc = le.transform(y_test)

# Convert MFCC features to numpy array for sklearn
X_train_array = np.array(X_train_bal)
X_test_array = np.array(X_test)

# Standardize features inline (fit on train, transform both)
print("Standardizing features...")
scaler = StandardScaler()
X_train_array = scaler.fit_transform(X_train_array)
X_test_array = scaler.transform(X_test_array)

# Train RandomForest classifier
print("Training RandomForest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_array, y_train_enc)

# Evaluate the model
print("Evaluating on test set...")
y_pred = clf.predict(X_test_array)
y_probs = clf.predict_proba(X_test_array)

test_acc = np.mean(y_pred == y_test_enc)
print(f"Test accuracy: {test_acc:.3f}")

print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Save model and create evaluation plots
model_path = (
    MODELS_DIR / f"bowel_sound_rf_win{window_size_sec}_overlap{window_overlap}.pth"
)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Save the trained model with metadata for inference
checkpoint = {
    "model": clf,
    "scaler": scaler,
    "classes": le.classes_.tolist(),
    "window_size_sec": window_size_sec,
    "window_overlap": window_overlap,
    "feature_type": "mfcc",
}
import pickle

pickle.dump(checkpoint, open(model_path, "wb"))
print(f"Saved model to {model_path}")

# Create evaluation plots
fig_name_stem = model_path.stem
save_path = FIGURES_DIR / f"{fig_name_stem}_cm_roc.png"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_confusion_and_roc(
    y_test_enc,
    y_probs,
    classes=le.classes_,
    save_path=save_path,
    show=True,
)

# Feature importance plot
fi_path = FIGURES_DIR / f"{fig_name_stem}_feature_importance.png"
plot_feature_importances(
    importances=clf.feature_importances_,
    save_path=fi_path,
    top_k=10,
    show=True,
)
print(f"Saved feature importance plot to {fi_path}")
