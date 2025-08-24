"""Inference script for trained bowel sound classification models. (only works with RandomForest and CNN (not v2/v3) for now)"""

from pathlib import Path
from typing import List, Tuple
import pickle
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import typer
from loguru import logger
import matplotlib.pyplot as plt

from utils.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, EXTERNAL_DATA_DIR
from modeling.evaluation import plot_confusion_and_roc
from modeling.cnn import BowelSoundCNN
from modeling.datasets import SpectrogramDataset, collate_fixed_spectrograms
from modeling.preprocessing import (
    load_and_normalize_wav,
    extract_fixed_size_mel_spectrogram,
    load_annotations,
    assign_window_labels_from_annotations,
    extract_overlapping_windows,
)
from utils.model_utils import parse_window_from_model_name, align_classes_to_logits
from utils.io import write_events_txt
from utils.events import build_events_from_label_changes, _plot_events_section

app = typer.Typer()


def load_model(model_path: Path, device: torch.device):
    """Load a trained model (CNN or Random Forest)."""
    if model_path.suffix == ".pth":
        if model_path.name.startswith("bowel_sound_cnn"):
            logger.info(f"Loading CNN checkpoint: {model_path}")
            model_type = "cnn"
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            input_shape = checkpoint["input_shape"]
            scaler = None
            le = checkpoint["le"]
            classes = checkpoint["classes"]
            window_size_sec = checkpoint["window_size_sec"]
            window_overlap = checkpoint["window_overlap"]
            feature_type = checkpoint["feature_type"]

            # Load existing model
            model = BowelSoundCNN(num_classes=len(le.classes_)).to(device)

            model.load_state_dict(checkpoint["state_dict"])

        elif model_path.name.startswith("bowel_sound_rf"):
            logger.info(f"Loading CNN checkpoint: {model_path}")
            model_type = "random_forest"

            # Load existing model
            checkpoint = pickle.load(open(model_path, "rb"))
            model = checkpoint["model"]
            scaler = checkpoint["scaler"]
            le = checkpoint["le"]
            classes = checkpoint["classes"]
            window_size_sec = checkpoint["window_size_sec"]
            window_overlap = checkpoint["window_overlap"]
            feature_type = checkpoint["feature_type"]

        else:
            raise ValueError(
                f"Other model types (e.g. LSTM) not supported yet for Inference: {model_path.name}"
            )

    return model, classes, window_size_sec, window_overlap, model_type, scaler


def run_cnn_inference(
    model: BowelSoundCNN,
    segments: List[np.ndarray],
    device: torch.device,
    sample_rate: int,
    window_size_sec,
    batch_size: int = 64,
) -> Tuple[List[int], List[np.ndarray]]:
    """Run inference using a CNN model."""
    specs: List[np.ndarray] = [
        extract_fixed_size_mel_spectrogram(
            seg,
            n_mels=40,
            target_frames=25,
            sample_rate=sample_rate,
            window_size_sec=window_size_sec,
        )
        for seg in segments
    ]

    if not specs:
        logger.warning("No segments were generated from the audio. Exiting.")
        return [], []

    dataset = SpectrogramDataset(specs, np.zeros(len(specs), dtype=np.int64))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fixed_spectrograms,
    )

    logger.info("Running CNN model inference...")
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_probs


def run_random_forest_inference(
    model,
    scaler,
    segments: List[np.ndarray],
    sample_rate: int,
) -> Tuple[List[int], List[np.ndarray]]:
    """Run inference using a Random Forest model."""
    from modeling.preprocessing import extract_mfcc_features

    logger.info("Running Random Forest model inference...")

    # Extract MFCC features
    specs = [extract_mfcc_features(seg, sample_rate=sample_rate) for seg in segments]

    if not specs:
        logger.warning("No segments were generated from the audio. Exiting.")
        return [], []

    # Convert to numpy array and scale
    X = np.array(specs)
    X_scaled = scaler.transform(X)

    # Get predictions and probabilities
    all_preds = model.predict(X_scaled).tolist()
    all_probs = model.predict_proba(X_scaled)

    return all_preds, all_probs


@app.command()
def predict(
    audio_path: Path = typer.Argument(..., help="Path to WAV file"),
    model_path: Path = typer.Option(
        MODELS_DIR / "bowel_sound_cnn_win1_overlap0.75.pth",
        help="Path to trained model checkpoint",
    ),
    annotation_path: Path = typer.Option(
        None, help="Path to annotation TXT; defaults to audio with .txt"
    ),
    output_txt: Path = typer.Option(
        PROCESSED_DATA_DIR / "predictions.txt", help="Where to write TXT predictions"
    ),
):
    """Run inference using a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, classes, window_size_sec, window_overlap, model_type, scaler = load_model(
        model_path, device
    )

    # Segment audio using the same parameters
    logger.info("Segmenting audio...")
    segments, times, sample_rate, hop_sec = extract_overlapping_windows(
        audio_path, window_size_sec=window_size_sec, window_overlap=window_overlap
    )

    # Run inference based on model type
    if model_type == "cnn":
        all_preds, all_probs = run_cnn_inference(
            model=model,
            segments=segments,
            device=device,
            sample_rate=sample_rate,
            window_size_sec=window_size_sec,
        )
    elif model_type == "random_forest":
        all_preds, all_probs = run_random_forest_inference(
            model=model,
            scaler=scaler,
            segments=segments,
            sample_rate=sample_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not all_preds:
        logger.warning("No predictions generated. Exiting.")
        return

    # Align class names to model's output dimension dynamically
    classes = align_classes_to_logits(classes, all_probs)

    # Build events by label-change runs (simple, non-overlapping)
    events = build_events_from_label_changes(times, all_preds, classes)

    # Write outputs (TXT)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    write_events_txt(output_txt, events)

    # Evaluate with confusion matrix + ROC if annotations are available
    ann_path = annotation_path or audio_path.with_suffix(".txt")
    if ann_path.exists():
        logger.info(f"Loading annotations: {ann_path}")
        ann_df = load_annotations(ann_path)

        # Build y_true for each window using same assignment strategy as training
        y_true_labels = assign_window_labels_from_annotations(times, ann_df, classes)

        label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
        y_true = np.array(
            [label_to_idx.get(lbl, label_to_idx.get("n", 0)) for lbl in y_true_labels]
        )
        y_scores = np.vstack(all_probs)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"inference_{model_path.stem}_cm_roc.png"
        plot_confusion_and_roc(y_true, y_scores, classes, save_path=fig_path, show=True)
        logger.info(f"Saved evaluation figure: {fig_path}")

        # 15s events comparison plot (GT vs Pred)
        compare_path = FIGURES_DIR / f"inference_{model_path.stem}_events_compare.png"
        _plot_events_section(
            ann_df=ann_df,
            pred_events=events,
            classes=classes,
            start=0.0,
            duration=15.0,
            save_path=compare_path,
            show=True,
        )
        logger.info(f"Saved events comparison plot: {compare_path}")

    logger.success(f"Wrote predictions TXT: {output_txt}")


if __name__ == "__main__":
    """This function is used to test the model on a single audio file."""
    # Select default audio and model paths
    default_audio = EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav"
    default_ann = EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.txt"

    # Available models
    model_files = list(MODELS_DIR.glob("*_cnn_*.pth"))
    model_files.extend(MODELS_DIR.glob("*_rf_*.pth"))
    print(f"Available models:")
    for mf in model_files:
        print(f" - {mf.name}")

    # Selected model
    # default_model = MODELS_DIR / "bowel_sound_rf_win0.6_overlap0.5.pth"
    default_model = MODELS_DIR / "bowel_sound_cnn_win0.3_overlap0.5.pth"
    print(f"\nSelected model: {default_model}")

    predict(
        audio_path=default_audio,
        model_path=default_model,
        annotation_path=default_ann,
        output_txt=PROCESSED_DATA_DIR
        / f"{default_audio.stem}_{default_model.stem}_predictions.txt",
    )
