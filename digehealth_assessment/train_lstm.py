"""LSTM training script for bowel sound classification."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from config import MODELS_DIR, FIGURES_DIR
from pathlib import Path
from data_loading import load_blocked_split_features
from data_utils import print_class_distribution, make_weighted_sampler
from model_training import (
    train_lstm_model,
    evaluate_lstm_model,
    save_model_checkpoint,
    create_evaluation_plots,
)
from training_config import (
    DEFAULT_FILE_PAIRS,
    DEFAULT_ALLOWED_LABELS,
    DEFAULT_WINDOW_SIZE_SEC,
    DEFAULT_WINDOW_OVERLAP,
    DEFAULT_TEST_FRACTION,
    DEFAULT_ENSURE_LABEL_COVERAGE,
    DEFAULT_RETRAIN_MODEL,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_LSTM_LEARNING_RATE,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_HIDDEN_SIZE,
    DEFAULT_LSTM_NUM_LAYERS,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_BIDIRECTIONAL,
)
from modeling.lstm import BowelSoundLSTM
from modeling.datasets import SpectrogramDataset, pad_collate_spectrograms


def main():
    """Main training function."""
    # Configuration
    file_pairs = DEFAULT_FILE_PAIRS
    allowed_labels = DEFAULT_ALLOWED_LABELS
    window_size_sec = DEFAULT_WINDOW_SIZE_SEC
    window_overlap = DEFAULT_WINDOW_OVERLAP
    retrain_model = DEFAULT_RETRAIN_MODEL

    print("Loading and preparing data...")
    # Load and prepare data (blocked split per file to keep test time-consecutive)
    X_train, y_train, X_test, y_test = load_blocked_split_features(
        file_pairs=file_pairs,
        allowed_labels=allowed_labels,
        feature_type="spectrogram",
        window_size_sec=window_size_sec,
        window_overlap=window_overlap,
        test_fraction=DEFAULT_TEST_FRACTION,
        ensure_label_coverage=DEFAULT_ENSURE_LABEL_COVERAGE,
    )

    print_class_distribution(y_train, "\nClass distribution in TRAIN set:")
    print_class_distribution(y_test, "\nClass distribution in Test set:")

    print("Using WeightedRandomSampler for balancing...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    sampler, _ = make_weighted_sampler(y_train_enc)
    y_test_enc = le.transform(y_test)

    # Datasets (dataset converts to NCHW automatically; supports variable width)
    train_dataset = SpectrogramDataset(X_train, y_train_enc)
    test_dataset = SpectrogramDataset(X_test, y_test_enc)

    # DataLoaders with dynamic padding per batch for variable widths
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFAULT_LSTM_BATCH_SIZE,
        sampler=sampler,
        collate_fn=pad_collate_spectrograms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=DEFAULT_LSTM_BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_spectrograms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (
        MODELS_DIR
        / f"bowel_sound_lstm_win{window_size_sec}_overlap{window_overlap}.pth"
    )

    # Only train the model if retraining is enabled
    if retrain_model:
        # Define the LSTM model
        # Derive input shape from a sample (C,H,W); width can vary
        sample_shape = train_dataset[0][0].shape

        # For LSTM, we need the height (frequency bins) as input size
        # The width (time steps) will be handled as sequence length
        input_size = sample_shape[1] if len(sample_shape) >= 2 else 128

        model = BowelSoundLSTM(
            num_classes=len(le.classes_),
            input_size=input_size,
            hidden_size=DEFAULT_LSTM_HIDDEN_SIZE,
            num_layers=DEFAULT_LSTM_NUM_LAYERS,
            dropout=DEFAULT_LSTM_DROPOUT,
            bidirectional=DEFAULT_LSTM_BIDIRECTIONAL,
        ).to(device)

        print(f"LSTM Model Architecture:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {DEFAULT_LSTM_HIDDEN_SIZE}")
        print(f"  Num layers: {DEFAULT_LSTM_NUM_LAYERS}")
        print(f"  Dropout: {DEFAULT_LSTM_DROPOUT}")
        print(f"  Bidirectional: {DEFAULT_LSTM_BIDIRECTIONAL}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train the model
        train_losses, train_accuracies = train_lstm_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=DEFAULT_LSTM_EPOCHS,
            learning_rate=DEFAULT_LSTM_LEARNING_RATE,
            device=device,
        )

        # Save the trained model with metadata for inference
        metadata = {
            "state_dict": model.state_dict(),
            "classes": le.classes_.tolist(),
            "window_size_sec": window_size_sec,
            "window_overlap": window_overlap,
            "feature_type": "spectrogram",
            "input_shape": sample_shape,
            "model_config": {
                "hidden_size": DEFAULT_LSTM_HIDDEN_SIZE,
                "num_layers": DEFAULT_LSTM_NUM_LAYERS,
                "dropout": DEFAULT_LSTM_DROPOUT,
                "bidirectional": DEFAULT_LSTM_BIDIRECTIONAL,
                "input_size": input_size,
            },
        }
        save_model_checkpoint(model, model_path, metadata, model_type="lstm")

    else:
        # Load existing model
        sample_shape = train_dataset[0][0].shape
        input_size = sample_shape[1] if len(sample_shape) >= 2 else 128

        model = BowelSoundLSTM(
            num_classes=len(le.classes_),
            input_size=input_size,
            hidden_size=DEFAULT_LSTM_HIDDEN_SIZE,
            num_layers=DEFAULT_LSTM_NUM_LAYERS,
            dropout=DEFAULT_LSTM_DROPOUT,
            bidirectional=DEFAULT_LSTM_BIDIRECTIONAL,
        ).to(device)

        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
            model.load_state_dict(loaded_obj["state_dict"])
        else:
            # Backward compatibility with older checkpoints containing only state_dict
            model.load_state_dict(loaded_obj)

    print("Evaluating on test set...")
    all_preds, all_labels, all_probs = evaluate_lstm_model(model, test_loader, device)

    test_acc = np.mean(all_preds == all_labels)
    print(f"Test accuracy: {test_acc:.3f}")

    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Create evaluation plots
    fig_name_stem = model_path.stem
    create_evaluation_plots(
        y_true=all_labels,
        y_scores=all_probs,
        classes=le.classes_,
        model_name=fig_name_stem,
        save_dir=FIGURES_DIR,
        show=True,
    )


if __name__ == "__main__":
    main()
