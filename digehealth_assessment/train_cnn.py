import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from config import EXTERNAL_DATA_DIR, MODELS_DIR, FIGURES_DIR
from pathlib import Path
from ml_pipeline import (
    load_blocked_split_features,
    print_class_distribution,
    balance_training_data,
)
from sampling import make_weighted_sampler
from evaluation import plot_confusion_and_roc
from modeling.cnn import BowelSoundCNN
from modeling.datasets import SpectrogramDataset, pad_collate_spectrograms

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

window_size_sec = 0.15  # Size of each segment in seconds
window_overlap = 0.5  # Overlap between segments as a fraction of window size

retrain_model = True  # Set to False to evaluate an existing model

# Load and prepare data (blocked split per file to keep test time-consecutive)
X_train, y_train, X_test, y_test = load_blocked_split_features(
    file_pairs=file_pairs,
    allowed_labels=allowed_labels,
    feature_type="spectrogram",
    window_size_sec=window_size_sec,
    window_overlap=window_overlap,
    test_fraction=0.2,
    ensure_label_coverage=True,
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
    train_dataset, batch_size=32, sampler=sampler, collate_fn=pad_collate_spectrograms
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_spectrograms
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = (
    MODELS_DIR / f"bowel_sound_cnn_win{window_size_sec}_overlap{window_overlap}.pth"
)
# Only train the model if retraining is enabled
if retrain_model:

    # Define the CNN model
    # Derive input shape from a sample (C,H,W); width can vary
    sample_shape = train_dataset[0][0].shape
    model = BowelSoundCNN(num_classes=len(le.classes_), input_shape=sample_shape).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training CNN model...")
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        train_acc = correct / total
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Acc: {train_acc:.4f}"
        )

    # Save the trained model with metadata for inference
    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": le.classes_.tolist(),
        "window_size_sec": window_size_sec,
        "window_overlap": window_overlap,
        "feature_type": "spectrogram",
        "input_shape": sample_shape,
    }
    torch.save(checkpoint, model_path)

else:
    sample_shape = train_dataset[0][0].shape
    model = BowelSoundCNN(num_classes=len(le.classes_), input_shape=sample_shape).to(
        device
    )
    loaded_obj = torch.load(model_path, map_location=device)
    if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
        model.load_state_dict(loaded_obj["state_dict"])
    else:
        # Backward compatibility with older checkpoints containing only state_dict
        model.load_state_dict(loaded_obj)

print("Evaluating on test set...")
model.eval()
all_preds = []
all_labels = []
all_probs = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test accuracy: {test_acc:.3f}")

print(classification_report(all_labels, all_preds, target_names=le.classes_))

fig_name_stem = model_path.stem
save_path = FIGURES_DIR / f"{fig_name_stem}_cm_roc.png"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_confusion_and_roc(
    np.array(all_labels),
    np.array(all_probs),
    classes=le.classes_,
    save_path=save_path,
    show=True,
)


# TODO: save plots to file
# TODO: update requirements.txt
# TODO: Update README.md
# TODO: Clean-up and refactor code
