from typing import List, Tuple, Optional
import numpy as np


def merge_events(preds: List[Tuple[float, float, str]]):
    if not preds:
        return []
    merged: List[Tuple[float, float, str]] = []
    cur_start, cur_end, cur_class = preds[0]
    for start, end, cls in preds[1:]:
        if cls == cur_class:
            cur_end = max(cur_end, end)
        else:
            # Ensure no overlap: next start should not be < current end
            if start < cur_end:
                start = cur_end
            merged.append((cur_start, cur_end, cur_class))
            cur_start, cur_end, cur_class = start, end, cls
    merged.append((cur_start, cur_end, cur_class))
    return merged


def enforce_non_overlapping(events: List[Tuple[float, float, str]]):
    """Force non-overlapping, contiguous events by clamping each start to prev end.

    Also enforces end >= start.
    """
    if not events:
        return []
    adjusted: List[Tuple[float, float, str]] = []
    prev_end = events[0][0]
    for start, end, label in events:
        start = max(start, prev_end)
        end = max(end, start)
        adjusted.append((start, end, label))
        prev_end = end
    return adjusted


def _label_priority(label: str) -> int:
    if label in ("b", "mb", "h"):
        return 3
    if label == "n":
        return 2
    if label == "silence":
        return 1
    return 0


def build_non_overlapping_events_from_windows(
    times: List[Tuple[float, float]],
    preds: List[int],
    classes: List[str],
    hop_sec: float,
    probs: Optional[np.ndarray] = None,
) -> List[Tuple[float, float, str]]:
    """Convert overlapping window predictions into non-overlapping events using priority.

    - Builds a time grid in steps of hop_sec
    - For each slice [t, t+hop_sec], chooses a label among overlapping windows:
      priority b/mb/h > n > silence; tie-breaker uses per-class probability if provided,
      otherwise nearest window center to slice center.
    - Merges adjacent slices of same label into continuous events.
    """
    if not times:
        return []
    t_start = times[0][0]
    t_end = max(e for _, e in times)
    # Safety for hop_sec
    if hop_sec <= 0:
        # Fallback to min distance between starts
        hop_candidates = np.diff([s for s, _ in times])
        hop_sec = (
            float(np.median(hop_candidates))
            if len(hop_candidates)
            else (times[0][1] - times[0][0])
        )

    idx_to_label = {i: lbl for i, lbl in enumerate(classes)}
    events: List[Tuple[float, float, str]] = []

    t = t_start
    while t < t_end - 1e-9:
        seg_start = t
        seg_end = min(t + hop_sec, t_end)
        seg_center = 0.5 * (seg_start + seg_end)
        # Find overlapping windows
        overlapping = [
            i for i, (s, e) in enumerate(times) if e > seg_start and s < seg_end
        ]
        if not overlapping:
            label = (
                "silence"
                if "silence" in classes
                else ("n" if "n" in classes else classes[0])
            )
        else:
            # Build candidate labels with priority
            candidates = {}
            for i in overlapping:
                lbl = idx_to_label.get(preds[i], classes[0])
                pr = _label_priority(lbl)
                if lbl not in candidates or pr > candidates[lbl]["priority"]:
                    # Store best priority and a score for tie-break
                    score = None
                    if probs is not None and 0 <= preds[i] < probs.shape[1]:
                        score = float(probs[i, preds[i]])
                    # distance tie-break as negative distance (so larger is better when None score)
                    dist = -abs(((times[i][0] + times[i][1]) * 0.5) - seg_center)
                    candidates[lbl] = {"priority": pr, "score": score, "dist": dist}
                else:
                    # Update tie-breakers
                    if probs is not None and 0 <= preds[i] < probs.shape[1]:
                        candidates[lbl]["score"] = max(
                            (
                                candidates[lbl]["score"]
                                if candidates[lbl]["score"] is not None
                                else -np.inf
                            ),
                            float(probs[i, preds[i]]),
                        )
                    dist = -abs(((times[i][0] + times[i][1]) * 0.5) - seg_center)
                    candidates[lbl]["dist"] = max(candidates[lbl]["dist"], dist)

            # Choose best label by priority, then by probability score, then by proximity
            best = None
            for lbl, meta in candidates.items():
                key = (
                    meta["priority"],
                    meta["score"] if meta["score"] is not None else -np.inf,
                    meta["dist"],
                )
                if best is None or key > best[0]:
                    best = (key, lbl)
            label = best[1] if best else classes[0]

        if not events or events[-1][2] != label:
            events.append((seg_start, seg_end, label))
        else:
            # extend
            events[-1] = (events[-1][0], seg_end, label)
        t = seg_end

    return events


def _merge_adjacent_events(
    events: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    if not events:
        return []
    merged: List[Tuple[float, float, str]] = []
    cur_start, cur_end, cur_label = events[0]
    for start, end, label in events[1:]:
        if label == cur_label:
            # Extend current event if the label stays the same
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = start, end, label
    merged.append((cur_start, cur_end, cur_label))
    return merged
