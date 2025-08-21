from pathlib import Path
from typing import List, Tuple

import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import typer
from loguru import logger
import matplotlib.pyplot as plt

from config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, EXTERNAL_DATA_DIR
from evaluation import plot_confusion_and_roc
from modeling.cnn import BowelSoundCNN
from modeling.datasets import SpectrogramDataset, pad_collate_spectrograms
from preprocessing import (
    load_and_normalize_wav,
    extract_mel_spectrogram,
    load_annotations,
    assign_window_labels_from_annotations,
    extract_overlapping_windows,
)
from utils.model_utils import parse_window_from_model_name, align_classes_to_logits
from utils.io import write_events_txt
from utils.events import build_events_from_label_changes, _plot_events_section

app = typer.Typer()



@app.command()
def predict(
    audio_path: Path = typer.Argument(..., help="Path to WAV file"),
    model_path: Path = typer.Option(
        MODELS_DIR / "bowel_sound_cnn_win1_overlap0.75.pth",
        help="Path to trained CNN checkpoint",
    ),
    annotation_path: Path = typer.Option(
        None, help="Path to annotation TXT; defaults to audio with .txt"
    ),
    output_txt: Path = typer.Option(
        PROCESSED_DATA_DIR / "predictions.txt", help="Where to write TXT predictions"
    ),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        classes = checkpoint.get("classes")
        # Parse window params from filename (overrides checkpoint values)
        window_size_sec, window_overlap = parse_window_from_model_name(model_path.stem)
        input_shape = tuple(checkpoint.get("input_shape"))
        model = BowelSoundCNN(num_classes=len(classes), input_shape=input_shape).to(
            device
        )
        model.load_state_dict(checkpoint["state_dict"])  # type: ignore[index]
    else:
        # Backward compatibility with state_dict-only checkpoints
        logger.warning("Old checkpoint format detected. Using default params.")
        classes = ["b", "mb", "h", "n", "silence"]
        # Parse from filename if possible; else default
        try:
            window_size_sec, window_overlap = parse_window_from_model_name(
                model_path.stem
            )
        except ValueError:
            raise ValueError(
                "Model filename must contain 'win<sec>_overlap<frac>', e.g., win0.5_overlap0.75"
            )
        y_for_shape, sr_for_shape = load_and_normalize_wav(audio_path)
        win_len = int(window_size_sec * sr_for_shape)
        dummy_seg = y_for_shape[:win_len]
        dummy_spec = extract_mel_spectrogram(dummy_seg, sample_rate=sr_for_shape)
        input_shape = (1, dummy_spec.shape[0], dummy_spec.shape[1])
        model = BowelSoundCNN(num_classes=len(classes), input_shape=input_shape).to(
            device
        )
        model.load_state_dict(checkpoint)

    model.eval()

    # Segment audio using the same parameters
    logger.info("Segmenting audio...")
    segments, times, sample_rate, hop_sec = extract_overlapping_windows(
        audio_path, window_size_sec=window_size_sec, window_overlap=window_overlap
    )

    specs: List[np.ndarray] = [
        extract_mel_spectrogram(seg, sample_rate=sample_rate) for seg in segments
    ]

    if not specs:
        logger.warning("No segments were generated from the audio. Exiting.")
        return

    dataset = SpectrogramDataset(specs, np.zeros(len(specs), dtype=np.int64))
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=pad_collate_spectrograms
    )

    logger.info("Running model inference...")
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

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
        fig_path = FIGURES_DIR / f"{model_path.stem}_cm_roc.png"
        plot_confusion_and_roc(y_true, y_scores, classes, save_path=fig_path, show=True)
        logger.info(f"Saved evaluation figure: {fig_path}")

        # 15s events comparison plot (GT vs Pred)
        compare_path = FIGURES_DIR / f"{model_path.stem}_events_compare.png"
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
    """This function is used to test the model on a single audio file.
    It is also used to generate the predictions.txt file.
    """
    # Hardcoded defaults for simple script execution
    default_audio = EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav"
    default_ann = EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.txt"
    default_model = MODELS_DIR / "bowel_sound_cnn_win0.2_overlap0.5.pth"

    predict(
        audio_path=default_audio,
        model_path=default_model,
        annotation_path=default_ann,
        output_txt=PROCESSED_DATA_DIR
        / f"{default_audio.stem}_{default_model.stem}_predictions.txt",
    )
