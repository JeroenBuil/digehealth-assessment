from pathlib import Path
from typing import Iterable, Tuple


def write_events_txt(output_path: Path, events: Iterable[Tuple[float, float, str]]):
    """Write events to a TXT file.

    Args:
        output_path: Path to the output file.
        events: Iterable of tuples containing (start, end, label).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("start\tend\tlabel\n")
        for start, end, label in events:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")
