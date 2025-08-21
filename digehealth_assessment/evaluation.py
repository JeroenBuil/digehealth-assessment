import matplotlib.pyplot as plt
import numpy as np
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


def plot_timeline_with_correctness(
    times,
    predicted_indices,
    classes,
    merged_events_df,
    y_true_labels,
    ann_df,
    max_seconds: float | None = None,
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    # Choose which label counts as silence on y=0
    silence_label = (
        "silence" if "silence" in classes else ("n" if "n" in classes else classes[0])
    )
    other_labels = [c for c in classes if c != silence_label]
    classes_order = [silence_label] + other_labels

    y_pos = {c: i for i, c in enumerate(classes_order)}
    class_to_color = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(classes_order)}

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_title("Ground truth lines and predicted markers with mismatch overlay")

    # Optionally trim to a 0..max_seconds window
    if max_seconds is not None:
        # Windows overlapping [0, max_seconds]
        keep_idx = [i for i, (s, e) in enumerate(times) if (e > 0 and s < max_seconds)]
        times = [times[i] for i in keep_idx]
        predicted_indices = [predicted_indices[i] for i in keep_idx]
        y_true_labels = [y_true_labels[i] for i in keep_idx]

        # Trim merged events to window
        if len(merged_events_df) > 0:
            merged_events_df = merged_events_df[
                (merged_events_df["end"] > 0)
                & (merged_events_df["start"] < max_seconds)
            ].copy()
            merged_events_df.loc[:, "start"] = merged_events_df["start"].clip(lower=0)
            merged_events_df.loc[:, "end"] = merged_events_df["end"].clip(
                upper=max_seconds
            )

        # Trim annotations to window
        if ann_df is not None and len(ann_df) > 0:
            ann_df = ann_df[
                (ann_df["end"] > 0) & (ann_df["start"] < max_seconds)
            ].copy()
            ann_df.loc[:, "start"] = ann_df["start"].clip(lower=0)
            ann_df.loc[:, "end"] = ann_df["end"].clip(upper=max_seconds)

    # Correctness overlay per window (only red when mismatch)
    idx_to_label = {i: lbl for i, lbl in enumerate(classes)}
    pred_labels_per_win = [idx_to_label[i] for i in predicted_indices]
    for (s, e), pred_lbl, true_lbl in zip(times, pred_labels_per_win, y_true_labels):
        if pred_lbl != true_lbl:
            ax.axvspan(s, e, facecolor=(1, 0, 0, 0.18), edgecolor="none")

    # Plot ground truth annotation events as lines at class y-position
    for _, row in ann_df.iterrows():
        lbl = row["label"] if row["label"] in classes else silence_label
        start = float(row["start"]) if "start" in row else float(row.start)
        end = float(row["end"]) if "end" in row else float(row.end)
        y = y_pos.get(lbl, y_pos[silence_label])
        ax.plot([start, end], [y, y], color=class_to_color.get(lbl, "k"), linewidth=3)

    # Plot merged predicted events as markers at event midpoints (darker shade)
    for _, ev in merged_events_df.iterrows():
        lbl = ev["predicted"]
        y = y_pos.get(lbl, y_pos[silence_label])
        x = (float(ev["start"]) + float(ev["end"])) / 2.0
        base = class_to_color.get(lbl, (0, 0, 0, 1))
        dark = (
            (base[0] * 0.6, base[1] * 0.6, base[2] * 0.6, 1.0)
            if len(base) == 4
            else base
        )
        ax.scatter([x], [y], color=dark, s=40, marker="o", zorder=3)

    ax.set_yticks([y_pos[c] for c in classes_order])
    ax.set_yticklabels(classes_order)
    ax.set_xlabel("Time (s)")
    if max_seconds is not None:
        ax.set_xlim(0, max_seconds)
    handles = [
        plt.Line2D([0], [0], color=class_to_color[c], lw=3) for c in classes_order
    ]
    ax.legend(handles, classes_order, loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
