import numpy as np
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(y_train_enc):
    class_counts = np.bincount(y_train_enc)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train_enc]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, y_train_enc
