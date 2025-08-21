"""Data utility functions for balancing, sampling, and analysis."""

import numpy as np
from typing import List, Tuple, Any


def print_class_distribution(y: List[Any], title: str = "") -> None:
    """Print the distribution of classes in a dataset."""
    unique, counts = np.unique(y, return_counts=True)
    if title:
        print(title)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")


def balance_training_data(X_train: List, y_train: List) -> Tuple[List, List]:
    """Balance training data using undersampling and SMOTE."""
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    target_size = int(np.median(counts_train))
    class_counts = dict(zip(unique_train, counts_train))

    under_classes = {
        label: target_size
        for label, count in class_counts.items()
        if count > target_size
    }
    over_classes = {
        label: target_size
        for label, count in class_counts.items()
        if count < target_size
    }

    under = (
        RandomUnderSampler(sampling_strategy=under_classes, random_state=42)
        if under_classes
        else None
    )
    smote = (
        SMOTE(sampling_strategy=over_classes, random_state=42) if over_classes else None
    )

    steps = []
    if under:
        steps.append(("under", under))
    if smote:
        steps.append(("smote", smote))

    if steps:
        pipeline = Pipeline(steps)
        X_res, y_res = pipeline.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    return X_res, y_res


def make_weighted_sampler(y_train: np.ndarray) -> Tuple[Any, np.ndarray]:
    """Create a weighted sampler for imbalanced datasets."""
    from torch.utils.data import WeightedRandomSampler

    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = 1.0 / counts
    sample_weights = class_weights[y_train]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(y_train), replacement=True
    )

    return sampler, sample_weights
