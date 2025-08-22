"""Random Forest training script for bowel sound classification."""

import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

from utils.config import MODELS_DIR, FIGURES_DIR
from modeling.data_loading import load_blocked_split_features
from modeling.data_utils import print_class_distribution
from modeling.model_training import (
    train_random_forest_model,
    save_model_checkpoint,
    create_evaluation_plots,
    create_feature_importance_plot,
)
from training_config import (
    DEFAULT_FILE_PAIRS,
    DEFAULT_ALLOWED_LABELS,
    DEFAULT_WINDOW_SIZE_SEC,
    DEFAULT_WINDOW_OVERLAP,
    DEFAULT_TEST_FRACTION,
    DEFAULT_ENSURE_LABEL_COVERAGE,
    DEFAULT_RETRAIN_MODEL,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_RANDOM_STATE,
)
from modeling.evaluation import build_mfcc_feature_names
from utils.model_utils import set_random_seeds


def main():
    """Main training function."""
    # Set random seeds for reproducible results
    set_random_seeds(seed=42)

    # Configuration
    file_pairs = DEFAULT_FILE_PAIRS
    allowed_labels = DEFAULT_ALLOWED_LABELS
    window_size_sec = DEFAULT_WINDOW_SIZE_SEC
    window_overlap = DEFAULT_WINDOW_OVERLAP
    retrain_model = DEFAULT_RETRAIN_MODEL

    print("Loading + splitting data + Extracting features... (this may take a while)")
    # Load and prepare data (blocked split per file to keep test time-consecutive)
    X_train, y_train, X_test, y_test = load_blocked_split_features(
        file_pairs=file_pairs,
        allowed_labels=allowed_labels,
        feature_type="mfcc",
        window_size_sec=window_size_sec,
        window_overlap=window_overlap,
        test_fraction=DEFAULT_TEST_FRACTION,
        ensure_label_coverage=DEFAULT_ENSURE_LABEL_COVERAGE,
    )

    print_class_distribution(y_train, "\nClass distribution in TRAIN set:")
    print_class_distribution(y_test, "\nClass distribution in Test set:")

    if retrain_model:
        # Train RandomForest classifier
        clf, scaler, le, X_test_scaled, y_test_enc = train_random_forest_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_estimators=DEFAULT_RF_N_ESTIMATORS,
            random_state=DEFAULT_RF_RANDOM_STATE,
        )

        # Save model and create evaluation plots
        model_path = (
            MODELS_DIR
            / f"bowel_sound_rf_win{window_size_sec}_overlap{window_overlap}.pth"
        )

        # Save the trained model with metadata for inference
        metadata = {
            "scaler": scaler,
            "classes": le.classes_.tolist(),
            "window_size_sec": window_size_sec,
            "window_overlap": window_overlap,
            "feature_type": "mfcc",
        }
        save_model_checkpoint(clf, model_path, metadata, model_type="random_forest")

    else:
        # Load existing model
        model_path = (
            MODELS_DIR
            / f"bowel_sound_rf_win{window_size_sec}_overlap{window_overlap}.pth"
        )
        import pickle

        checkpoint = pickle.load(open(model_path, "rb"))
        clf = checkpoint["model"]
        scaler = checkpoint["scaler"]
        le = checkpoint["classes"]

        # Transform test data
        X_test_array = np.array(X_test)
        X_test_scaled = scaler.transform(X_test_array)
        y_test_enc = le.transform(y_test)

    # Evaluate the model
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test_scaled)
    y_probs = clf.predict_proba(X_test_scaled)

    test_acc = np.mean(y_pred == y_test_enc)
    print(f"Test accuracy: {test_acc:.3f}")

    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # Create evaluation plots
    fig_name_stem = model_path.stem
    create_evaluation_plots(
        y_true=y_test_enc,
        y_scores=y_probs,
        classes=le.classes_,
        model_name=fig_name_stem,
        save_dir=FIGURES_DIR,
        show=True,
    )

    # Feature importance plot
    feature_names = build_mfcc_feature_names(n_mfcc=13, n_spectral_contrast_bands=7)
    create_feature_importance_plot(
        importances=clf.feature_importances_,
        model_name=fig_name_stem,
        save_dir=FIGURES_DIR,
        top_k=10,
        show=True,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    main()
