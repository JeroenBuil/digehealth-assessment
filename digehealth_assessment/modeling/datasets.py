import numpy as np
import torch
from torch.utils.data import Dataset


def _to_nchw(array: np.ndarray) -> np.ndarray:
    """Normalize spectrogram array to shape (N, C, H, W).

    Accepts arrays of shape:
    - (N, H, W) -> adds C=1
    - (N, H, W, C) -> transposes to (N, C, H, W)
    - (N, C, H, W) -> returned as-is
    """
    if array.ndim != 3 and array.ndim != 4:
        raise ValueError("Expected X with 3 or 4 dims: (N,H,W) or (N,H,W,C)/(N,C,H,W)")
    if array.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        array = array[..., np.newaxis]
        array = np.transpose(array, (0, 3, 1, 2))
        return array
    # ndim == 4
    # If channels last (N, H, W, C), move to (N, C, H, W)
    if array.shape[1] not in (1, 2, 3, 4, 5) and array.shape[-1] in (1, 2, 3, 4, 5):
        array = np.transpose(array, (0, 3, 1, 2))
    return array


class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        """Dataset that supports variable-width spectrograms.

        X can be:
        - list of (H, W) numpy arrays (preferred for variable widths)
        - numpy array shaped (N, H, W) or (N, C, H, W) for fixed width
        """
        self.variable_width = isinstance(X, list)
        if self.variable_width:
            # Store list of tensors with channel dim added lazily
            self.X_list = [
                (
                    torch.tensor(x[np.newaxis, ...], dtype=torch.float32)
                    if isinstance(x, np.ndarray)
                    else x
                )
                for x in X
            ]
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            X = _to_nchw(np.asarray(X))
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_list) if self.variable_width else len(self.X)

    def __getitem__(self, idx):
        if self.variable_width:
            return self.X_list[idx], self.y[idx]
        return self.X[idx], self.y[idx]


def pad_collate_spectrograms(batch):
    """Collate function that pads variable-width spectrograms to the max width in batch.

    batch: list of (tensor[C,H,W], label)
    Returns: (batch_tensor[B,C,H,W], labels[B])
    """
    xs, ys = zip(*batch)
    # Determine max height and width in this batch
    c0, h0, w0 = xs[0].shape
    max_h = max(x.shape[1] for x in xs)
    max_w = max(x.shape[2] for x in xs)
    padded = []
    for x in xs:
        c, h, w = x.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        padded.append(x)
    batch_x = torch.stack(padded, dim=0)
    batch_y = torch.tensor(ys, dtype=torch.long)
    return batch_x, batch_y


def collate_fixed_spectrograms(batch):
    """
    Collate function for fixed-size spectrograms in list form.

    Args:
        batch: list of tuples (spectrogram [H, W], label)

    Returns:
        batch_x: torch.Tensor of shape [B, 1, H, W]
        batch_y: torch.Tensor of shape [B]
    """
    xs, ys = zip(*batch)

    # Ensure all spectrograms are tensors, float32, remove extra singleton dims
    xs = [
        (
            x.detach().clone().squeeze().float()
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=torch.float32)
        )
        for x in xs
    ]

    # Add channel dimension: [1, H, W]
    xs = [x.unsqueeze(0) for x in xs]

    # Stack into batch: [B, 1, H, W]
    batch_x = torch.stack(xs, dim=0)

    # Labels as tensor
    batch_y = torch.tensor(ys, dtype=torch.long)

    return batch_x, batch_y
