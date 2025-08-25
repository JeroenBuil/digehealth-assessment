"""Common training configuration parameters."""

from pathlib import Path
from utils.config import EXTERNAL_DATA_DIR

# Common file pairs for training
DEFAULT_FILE_PAIRS = [
    (
        EXTERNAL_DATA_DIR / "Tech Test" / "AS_1.wav",
        EXTERNAL_DATA_DIR / "Tech Test" / "AS_1_cleaned.txt",
    ),
    (
        EXTERNAL_DATA_DIR / "Tech Test" / "23M74M.wav",
        EXTERNAL_DATA_DIR / "Tech Test" / "23M74M_cleaned.txt",
    ),
]

# Common labels
DEFAULT_ALLOWED_LABELS = ["b", "mb", "h", "n", "silence"]

# Common window parameters
DEFAULT_WINDOW_SIZE_SEC = 0.3
DEFAULT_WINDOW_OVERLAP = 0.5

# Training parameters
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_ENSURE_LABEL_COVERAGE = True
DEFAULT_RETRAIN_MODEL = False

# CNN specific parameters
DEFAULT_CNN_EPOCHS = 20
DEFAULT_CNN_LEARNING_RATE = 0.001
DEFAULT_CNN_BATCH_SIZE = 32

# LSTM specific parameters
DEFAULT_LSTM_EPOCHS = 30
DEFAULT_LSTM_LEARNING_RATE = 0.001
DEFAULT_LSTM_BATCH_SIZE = 16  # Smaller batch size for LSTM memory efficiency
DEFAULT_LSTM_HIDDEN_SIZE = 128
DEFAULT_LSTM_NUM_LAYERS = 2
DEFAULT_LSTM_DROPOUT = 0.3
DEFAULT_LSTM_BIDIRECTIONAL = True

# Random Forest specific parameters
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_RANDOM_STATE = 42

# Data balancing
DEFAULT_USE_WEIGHTED_SAMPLER = True
