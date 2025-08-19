import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import EXTERNAL_DATA_DIR
from pathlib import Path
from ml_pipeline import (
    load_data_and_extract_features,
    print_class_distribution,
    balance_training_data,
    standardize_features,
    evaluate_model,
)

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
allowed_labels = {"b", "mb", "h", "n", "silence"}

# Load and prepare data
X, y = load_data_and_extract_features(file_pairs, allowed_labels, feature_type="mfcc")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print_class_distribution(y, "Class distribution BEFORE train/test split:")

print_class_distribution(y_train, "\nClass distribution in TRAIN set:")

print("Balancing training data...")
X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
print_class_distribution(
    y_train_bal, "\nClass distribution in TRAIN set after balancing:"
)

# Standardize features
print("Standardizing features...")
X_train_bal, X_test, scaler = standardize_features(X_train_bal, X_test)

# Train RandomForest classifier
print("Training RandomForest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_bal, y_train_bal)

# Evaluate the model
evaluate_model(clf, X_test, y_test)
