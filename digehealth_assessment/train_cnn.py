import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from config import EXTERNAL_DATA_DIR
from pathlib import Path
from ml_pipeline import (
    load_data_and_extract_features,
    print_class_distribution,
    balance_training_data,
    evaluate_model,
    BowelSoundCNN,
    make_weighted_sampler,
)


# Dataset class for PyTorch
class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
allowed_labels = ["b", "mb", "h", "n", "silence"]

# Load and prepare data
X, y = load_data_and_extract_features(
    file_pairs, allowed_labels, feature_type="spectrogram"
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print_class_distribution(y, "Class distribution BEFORE train/test split:")

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print_class_distribution(y_train, "\nClass distribution in TRAIN set:")


print("Using WeightedRandomSampler for balancing...")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
sampler, _ = make_weighted_sampler(y_train_enc)
y_test_enc = le.transform(y_test)

# Ensure correct shape
if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

# Datasets
train_dataset = SpectrogramDataset(X_train, y_train_enc)
test_dataset = SpectrogramDataset(X_test, y_test_enc)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BowelSoundCNN(num_classes=len(le.classes_), input_shape=X_train.shape[1:]).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training CNN...")
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

print("Evaluating on test set...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
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
