"""Legacy ML pipeline utilities - most functions moved to dedicated modules.

This file is kept for backward compatibility but most functionality has been
moved to more focused modules:
- data_loading.py: Data loading and splitting functions
- data_utils.py: Data utility functions (balancing, sampling, analysis)
- model_training.py: Model training utilities
"""

# Import the moved functions for backward compatibility
from data_loading import load_blocked_split_features, load_data_and_extract_features
from data_utils import (
    print_class_distribution,
    balance_training_data,
    make_weighted_sampler,
)

# Re-export for backward compatibility
__all__ = [
    "load_blocked_split_features",
    "load_data_and_extract_features",
    "print_class_distribution",
    "balance_training_data",
    "make_weighted_sampler",
]
