from typing import List, Tuple


def build_events_from_label_changes(
    times: List[Tuple[float, float]], preds: List[int], classes: List[str]
) -> List[Tuple[float, float, str]]:
    """Build events by grouping consecutive identical labels.

    Uses the start time of each window as the event boundary.
    Event start = start of first window in the run; event end = end of last window in the run.
    """
    if not times or not preds:
        return []
    assert len(times) == len(preds)

    events: List[Tuple[float, float, str]] = []
    current_label = preds[0]
    current_start = times[0][0]

    for i in range(1, len(preds)):
        if preds[i] != current_label:
            # Close previous run at the end of the previous window
            prev_end = times[i - 1][1]
            events.append((current_start, prev_end, classes[current_label]))
            # Start new run
            current_label = preds[i]
            current_start = times[i][0]

    # Close final run
    final_end = times[-1][1]
    events.append((current_start, final_end, classes[current_label]))
    return events


def _plot_events_section(
    ann_df: pd.DataFrame,
    pred_events: List[Tuple[float, float, str]],
    classes: List[str],
    start: float = 0.0,
    duration: float = 15.0,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot a comparison of annotated (GT) vs predicted events over a time section."""
    end = start + duration

    # Prepare and clip annotation events
    ann_clip = ann_df[(ann_df["end"] > start) & (ann_df["start"] < end)].copy()
    if len(ann_clip) > 0:
        ann_clip.loc[:, "start"] = ann_clip["start"].clip(lower=start)
        ann_clip.loc[:, "end"] = ann_clip["end"].clip(upper=end)

    # Prepare and clip predicted events
    pred_clip: List[Tuple[float, float, str]] = []
    for s, e, lbl in pred_events:
        if e > start and s < end:
            pred_clip.append((max(s, start), min(e, end), lbl))

    # Colors by class
    label_order = classes
    base_colors = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(label_order)}
    pred_colors = dict(base_colors)
    if "silence" in pred_colors:
        pred_colors["silence"] = (
            0.9,
            0.9,
            0.9,
            1.0,
        )  # very light grey for predicted silence

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 3.6), sharex=True, constrained_layout=False
    )

    # Top: Annotations (filled bands)
    axes[0].set_title("Annotated events (GT)")
    for _, row in ann_clip.iterrows():
        lbl = row["label"] if row["label"] in base_colors else label_order[0]
        axes[0].axvspan(
            float(row["start"]),
            float(row["end"]),
            ymin=0.15,
            ymax=0.85,
            facecolor=base_colors.get(lbl, "k"),
            edgecolor=base_colors.get(lbl, "k"),
            alpha=0.7,
        )
    axes[0].set_yticks([])
    axes[0].set_xlim(start, end)
    axes[0].grid(True, axis="x", alpha=0.2)

    # Bottom: Predictions (filled bands)
    axes[1].set_title("Predicted events")
    for s, e, lbl in pred_clip:
        color = pred_colors.get(lbl, "k")
        axes[1].axvspan(
            float(s),
            float(e),
            ymin=0.15,
            ymax=0.85,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(start, end)
    axes[1].grid(True, axis="x", alpha=0.2)

    # Legend (compact) based on GT colors
    handles = [plt.Line2D([0], [0], color=base_colors[c], lw=10) for c in label_order]
    axes[0].legend(handles, label_order, loc="upper right", frameon=False, fontsize=9)

    # Reduce whitespace around the figure
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, right=0.98, left=0.06, bottom=0.16, hspace=0.35)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
