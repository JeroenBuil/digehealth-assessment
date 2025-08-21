import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def evaluate_model(clf, X_test, y_test):
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, class_name in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i], label=f"ROC curve of {class_name} (area = {roc_auc[i]:.2f})"
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_and_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes,
    save_path=None,
    show: bool = True,
):
    """Plot a single 2x2 figure with:
    - Top-left: Confusion Matrix (counts)
    - Top-right: ROC curves (per-class + micro-average)
    - Bottom-left: Confusion Matrix (normalized per true class)
    - Bottom-right: Precision–Recall curves (per-class + micro-average)

    Args:
        y_true: Array of integer class labels (shape: [N]).
        y_scores: Predicted probabilities/logits per class (shape: [N, C]).
        classes: Iterable of class names ordered by index.
        save_path: Optional path to save the figure.
        show: Whether to show the figure interactively.
    Returns:
        (fig, (ax_cm_counts, ax_roc, ax_cm_norm, ax_pr))
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_cm_counts = axes[0, 0]
    ax_roc = axes[0, 1]
    ax_cm_norm = axes[1, 0]
    ax_pr = axes[1, 1]

    # Predictions and label index set
    y_pred = np.argmax(y_scores, axis=1)
    all_labels = np.arange(len(classes))

    # Confusion matrix (counts)
    cm_counts = confusion_matrix(y_true, y_pred, labels=all_labels)
    disp_counts = ConfusionMatrixDisplay(
        confusion_matrix=cm_counts, display_labels=classes
    )
    disp_counts.plot(ax=ax_cm_counts, colorbar=False)
    ax_cm_counts.set_title("Confusion Matrix (Counts)")
    ax_cm_counts.set_xlabel("Predicted label")
    ax_cm_counts.set_ylabel("True label")

    # Confusion matrix (normalized per true class)
    cm_norm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize="true")
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
    disp_norm.plot(ax=ax_cm_norm, colorbar=False, values_format=".2f")
    ax_cm_norm.set_title("Confusion Matrix (Normalized)")
    ax_cm_norm.set_xlabel("Predicted label")
    ax_cm_norm.set_ylabel("True label")

    # Prepare one-hot for micro/macro metrics
    num_classes = len(classes)
    y_true_onehot = np.eye(num_classes)[y_true]

    # ROC per class + micro-average
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{class_name} (AUC={auc_val:.2f})")
    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax_roc.plot(
        fpr_micro,
        tpr_micro,
        label=f"micro-average (AUC={auc_micro:.2f})",
        color="black",
        linestyle=":",
        linewidth=2,
    )
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(loc="lower right")

    # Precision–Recall curves per class + micro-average
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
        ap = average_precision_score(y_true == i, y_scores[:, i])
        ax_pr.plot(recall, precision, label=f"{class_name} (AP={ap:.2f})")
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_onehot.ravel(), y_scores.ravel()
    )
    ap_micro = average_precision_score(y_true_onehot, y_scores, average="micro")
    ax_pr.plot(
        recall_micro,
        precision_micro,
        label=f"micro-average (AP={ap_micro:.2f})",
        color="black",
        linestyle=":",
        linewidth=2,
    )
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall Curves")
    ax_pr.legend(loc="lower left")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_cm_counts, ax_roc, ax_cm_norm, ax_pr)



def build_mfcc_feature_names(
    num_mfcc: int = 13, num_contrast_bands: int = 7
) -> list[str]:
    names: list[str] = []
    for prefix in ("mfcc", "dmfcc", "ddmfcc"):
        names.extend([f"{prefix}_mean_{i+1}" for i in range(num_mfcc)])
        names.extend([f"{prefix}_std_{i+1}" for i in range(num_mfcc)])
    for prefix in ["zcr"]:
        names.extend([f"{prefix}_mean", f"{prefix}_std"])
    return names


def plot_feature_importances(
    importances: np.ndarray,
    save_path: Path | None = None,
    top_k: int = 25,
    show: bool = True,
    feature_names: list[str] | None = None,
    num_mfcc: int = 13,
    num_contrast_bands: int = 7,
) -> None:
    if feature_names is None:
        feature_names = build_mfcc_feature_names(
            num_mfcc=num_mfcc, num_contrast_bands=num_contrast_bands
        )
    if len(feature_names) != importances.shape[0]:
        feature_names = [f"f{i}" for i in range(importances.shape[0])]

    order = np.argsort(importances)[::-1]
    topk = min(top_k, len(order))
    sel = order[:topk]

    plt.figure(figsize=(10, max(6, topk * 0.35)))
    plt.barh(range(topk), importances[sel][::-1], color="tab:blue")
    plt.yticks(range(topk), [feature_names[i] for i in sel][::-1])
    plt.xlabel("Importance")
    plt.title("RandomForest Feature Importances (Top)")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
