"""DigeHealth Assessment - Bowel Sound Classification Package.

This package provides tools for training and evaluating machine learning models
for bowel sound classification from audio data.
"""

# Core modules
from . import config
from . import preprocessing
from . import evaluation

# Data handling
from . import data_loading
from . import data_utils

# Model training
from . import model_training
from . import training_config

# Model definitions
from . import modeling

# Utilities
from . import utils

# Training scripts
from . import train_cnn
from . import train_randomforest

# Inference
from . import inference

__version__ = "0.0.1"
__author__ = "Jeroen Buil"

__all__ = [
    "config",
    "preprocessing",
    "evaluation",
    "data_loading",
    "data_utils",
    "model_training",
    "training_config",
    "modeling",
    "utils",
    "train_cnn",
    "train_randomforest",
    "inference",
]
