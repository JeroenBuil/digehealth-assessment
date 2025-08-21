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

from config import EXTERNAL_DATA_DIR, MODELS_DIR
from pathlib import Path
from ml_pipeline import (
    load_data_and_extract_features,
    print_class_distribution,
    balance_training_data,
    evaluate_model,
    make_weighted_sampler,
    plot_roc_curve,
)
from modeling.cnn import BowelSoundCNN
from datasets import SpectrogramDataset


file_pairs = [
    # (
    #     EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav",
    #     EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.txt",
    # ),
    (
        EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.wav",
        EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.txt",
    ),
]
allowed_labels = ["b", "mb", "h", "n"]

window_size_sec = 0.3  # Size of each segment in seconds
window_overlap = 0.75  # Overlap between segments as a fraction of window size

# Load and prepare data
X, y = load_data_and_extract_features(
    file_pairs,
    allowed_labels,
    feature_type="spectrogram",
    window_size_sec=window_size_sec,
    window_overlap=window_overlap,
)

print("Using WeightedRandomSampler for balancing...")
le = LabelEncoder()
y = le.fit_transform(y)

# Dataset handles NCHW conversion internally
new_dataset = SpectrogramDataset(X, y)

# DataLoaders
new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = (
    MODELS_DIR / f"bowel_sound_cnn_win{window_size_sec}_overlap{window_overlap}.pth"
)
model = BowelSoundCNN(num_classes=len(le.classes_), input_shape=X.shape[1:]).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))

print("Evaluating on test set...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in new_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test accuracy: {test_acc:.3f}")

print(classification_report(all_labels, all_preds, target_names=le.classes_))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()

# Plot ROC curve
plot_roc_curve(all_labels, all_preds, classes=le.classes_)


def merge_events(preds):
    merged = []
    cur_start, cur_end, cur_class = preds[0]
    for start, end, cls in preds[1:]:
        if cls == cur_class:
            cur_end = end  # extend
        else:
            merged.append((cur_start, cur_end, cur_class))
            cur_start, cur_end, cur_class = start, end, cls
    merged.append((cur_start, cur_end, cur_class))
    return merged


# # Plot predictions vs true labels
# plt.figure(figsize=(10, 4))
# plt.plot(all_labels, label="True Labels", marker="o", linestyle="None")
# plt.plot(all_preds, label="Predicted Labels", marker="x", linestyle="None")
# # # Add red bar when the labels don't match
# # for i in range(len(all_labels)):
# #     if all_labels[i] != all_preds[i]:
# #         plt.axvline(x=i, color="red", linestyle="--", alpha=0.5)
# plt.yticks(np.arange(len(le.classes_)), le.classes_)
# plt.tight_layout()
# plt.xlabel("Segment Index")
# plt.ylabel("Label")
# plt.title("True vs Predicted Labels for New Data")
# plt.legend()
# plt.show()
