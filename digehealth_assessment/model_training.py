"""Common model training utilities and functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

from config import MODELS_DIR, FIGURES_DIR
from evaluation import plot_confusion_and_roc, plot_feature_importances
from data_utils import print_class_distribution, balance_training_data


def train_cnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    device: torch.device = None,
) -> Tuple[List[float], List[float]]:
    """Train a CNN model and return training history."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []

    print("Training CNN model...")
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

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )

    return train_losses, train_accuracies


def evaluate_cnn_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a trained CNN model on test data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device = None,
) -> Tuple[List[float], List[float]]:
    """Train an LSTM model and return training history."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    train_losses = []
    train_accuracies = []

    print("Training LSTM model...")
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

            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Update learning rate based on loss
        scheduler.step(epoch_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    return train_losses, train_accuracies


def evaluate_lstm_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a trained LSTM model on test data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_random_forest_model(
    X_train: List,
    y_train: List,
    X_test: List,
    y_test: List,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[
    RandomForestClassifier, StandardScaler, LabelEncoder, np.ndarray, np.ndarray
]:
    """Train a Random Forest model with preprocessing."""
    # Balance training data
    print("Balancing training data...")
    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
    print_class_distribution(y_train_bal, "Class distribution after balancing:")

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_bal)
    y_test_enc = le.transform(y_test)

    # Convert to numpy arrays
    X_train_array = np.array(X_train_bal)
    X_test_array = np.array(X_test)

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_array)
    X_test_scaled = scaler.transform(X_test_array)

    # Train model
    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train_scaled, y_train_enc)

    return clf, scaler, le, X_test_scaled, y_test_enc


def save_model_checkpoint(
    model: Any,
    model_path: Path,
    metadata: Dict[str, Any],
    model_type: str = "cnn",
) -> None:
    """Save a trained model with metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type in ["cnn", "lstm"]:
        torch.save(metadata, model_path)
    elif model_type == "random_forest":
        import pickle

        checkpoint = {"model": model, **metadata}
        pickle.dump(checkpoint, open(model_path, "wb"))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Saved model to {model_path}")


def create_evaluation_plots(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes: List[str],
    model_name: str,
    save_dir: Path = None,
    show: bool = True,
) -> None:
    """Create and save evaluation plots."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    save_dir.mkdir(parents=True, exist_ok=True)

    # Main evaluation plot
    save_path = save_dir / f"{model_name}_cm_roc.png"
    plot_confusion_and_roc(y_true, y_scores, classes, save_path=save_path, show=show)
    print(f"Saved evaluation plot: {save_path}")


def create_feature_importance_plot(
    importances: np.ndarray,
    model_name: str,
    save_dir: Path = None,
    top_k: int = 25,
    show: bool = True,
    feature_names: List[str] = None,
) -> None:
    """Create and save feature importance plot."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{model_name}_feature_importance.png"
    plot_feature_importances(
        importances=importances,
        save_path=save_path,
        top_k=top_k,
        show=show,
        feature_names=feature_names,
    )
    print(f"Saved feature importance plot: {save_path}")
