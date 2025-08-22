"""DigeHealth Assessment - Bowel Sound Classification Package.

This package provides tools for training and evaluating machine learning models
for bowel sound classification from audio data.
"""

# Core modules
from .utils import config
from .modeling import preprocessing
from .modeling import evaluation

# Data handling
from .modeling import data_loading
from .modeling import data_utils

# Model training
from .modeling import model_training
from . import training_config

# Model definitions
from . import modeling

# Utilities
from . import utils

# Training scripts
from . import train_cnn_old
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
    "train_cnn_old",
    "train_randomforest",
    "inference",
]
