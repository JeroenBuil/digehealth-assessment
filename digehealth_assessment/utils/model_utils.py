import re
from typing import List, Tuple
import numpy as np


def parse_window_from_model_name(model_stem: str) -> Tuple[float, float]:
    """Parse window size (sec) and overlap (fraction) from model filename stem.

    Expected pattern: win<sec>_overlap<frac>, e.g., win0.5_overlap0.75
    """
    m = re.search(r"win([0-9.]+)_overlap([0-9.]+)", model_stem)
    if not m:
        raise ValueError(
            "Model filename must contain 'win<sec>_overlap<frac>', e.g., win0.5_overlap0.75"
        )
    return float(m.group(1)), float(m.group(2))


def align_classes_to_logits(
    classes: List[str] | None, probs_list: List[np.ndarray]
) -> List[str]:
    """Ensure classes length matches model output dimension.

    - Trims or extends provided classes to match logits dim
    - Prefers dropping 'silence' if it causes a single-class mismatch
    - If classes is None, synthesizes a default set
    """
    num_outputs = (
        int(np.vstack(probs_list).shape[1])
        if len(probs_list)
        else (len(classes) if classes else 0)
    )
    if classes is None or len(classes) == 0:
        base = ["b", "mb", "h", "n", "silence"]
        return base[:num_outputs] if num_outputs > 0 else base

    if len(classes) == num_outputs:
        return list(classes)

    if len(classes) > num_outputs:
        if "silence" in classes and len(classes) - 1 == num_outputs:
            return [c for c in classes if c != "silence"]
        return list(classes)[:num_outputs]

    # len(classes) < num_outputs: extend
    extension = [f"class_{i}" for i in range(len(classes), num_outputs)]
    return list(classes) + extension
