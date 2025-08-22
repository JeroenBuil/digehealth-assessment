# DigeHealth Assessment - Bowel Sound Classification

My's submission for the coding assessment for the ML Engineer at DigeHealth for classifying bowel sounds from audio recordings. This project implements multiple ML approaches including CNN, LSTM, and Random Forest models to identify and classify different types of bowel sounds.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Overview
Goal: develop a proof-of-concept ML model for identifying bowel sounds in audio data and differentiating between 3 main classes:
**Classes**:
- `b` - Single burst bowel movement
- `mb` - Multiple burst bowel movement  
- `h` - Harmonic sound
- `n` - Noise
- `silence` -  Default label when there is no annotation

The model should identify the start time, end time and type of each bowel sound. There are some other labels for voice and noise, these can be ignored or used.

The assignment includes, explorative data analysis, and several machine learning approaches to tackle this challenge.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd digehealth-assessment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## 💾 Data Requirements

**IMPORTANT**: This project requires the 'Tech Test' dataset from DigeHealth, which is **NOT INCLUDED** in this repository.

### Data Setup
1. **Obtain the dataset** from DigeHealth
2. **Place files in** `data/external/Tech Test/`:
   - `23M74M.wav` - Audio recording 1
   - `23M74M.txt` - Annotations for recording 1
   - `AS_1.wav` - Audio recording 2  
   - `AS_1.txt` - Annotations for recording 2

**Note**: Without this data, all scripts will fail with "file not found" errors.

## 📊 Usage Examples

### **Training Models**

```bash
# Train CNN with default settings
python -m digehealth_assessment.train_cnn

# Train LSTM with custom parameters
python -m digehealth_assessment.train_lstm

# Train Random Forest
python -m digehealth_assessment.train_randomforest

# Train improved CNN architectures
python -m digehealth_assessment.train_cnn_v2
```

### **Testing Models**

```bash
# Test LSTM implementation
python -m digehealth_assessment.test_lstm

# Test CNN architectures
python -m digehealth_assessment.test_cnn_v2
```

### **Data Visualization**

```bash
# Plot spectrograms for each class
python -m digehealth_assessment.plot_class_spectrograms

# Run exploratory data analysis
jupyter notebook notebooks/Exploratory_Data_Analysis.ipynb
```

## ⚙️ Configuration

### **Training Parameters**
All training parameters are configurable in `digehealth_assessment/training_config.py`:

## Available Models

### 1. **CNN (Convolutional Neural Network)**
- **Purpose**: Image-based classification using spectrogram representations
- **Strengths**: Excellent at capturing spatial patterns in frequency-time domain
- **Best for**: Fixed-size spectrograms, frequency-time pattern recognition

**Usage**:
```bash
# Train CNN model
python -m digehealth_assessment.train_cnn

# Train improved CNN architectures (v2/v3)
python -m digehealth_assessment.train_cnn_v2
```

**Architecture Variants**:
- **Original CNN**: Basic 3-layer convolutional network
- **CNN v2 (not recommended)**: Enhanced with Squeeze-Excitation blocks, temporal attention, residual connections
- **CNN v3 (not recommended)**: Alternative with dilated convolutions for larger receptive fields

### 2. **LSTM (Long Short-Term Memory)**
- **Purpose**: Sequential modeling of temporal patterns in audio
- **Strengths**: Captures long-term dependencies, handles variable-length sequences
- **Best for**: Time-series analysis, variable-length audio segments

**Usage**:
```bash
# Train LSTM model
python -m digehealth_assessment.train_lstm
```

**Features**:
- Bidirectional LSTM with attention mechanism
- Handles variable-length spectrograms
- Configurable hidden size, layers, and dropout

### 3. **Random Forest**
- **Purpose**: Traditional ML approach using engineered features
- **Strengths**: Interpretable, handles non-linear relationships, robust to outliers
- **Best for**: Quick baseline, feature importance analysis

**Usage**:
```bash
# Train Random Forest model
python -m digehealth_assessment.train_randomforest
```

## 🔧 Key Features

### **Audio Processing Pipeline**
- **Segmentation**: Automatic audio segmentation with configurable window sizes
- **Feature Extraction**: Log-mel spectrograms with customizable parameters
- **Normalization**: RMS-based audio normalization
- **Data Augmentation**: Overlapping segments for increased training data

### **Model Training Features**
- **Reproducible Results**: Fixed random seeds for consistent training
- **Class Balancing**: Weighted sampling to handle imbalanced datasets
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Model Checkpointing**: Automatic model saving with metadata

### **Evaluation & Visualization**
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, ROC curves
- **Confusion Matrices**: Visual class prediction analysis
- **Training Curves**: Loss and accuracy progression
- **Feature Analysis**: Random Forest feature importance plots


## 📁 Project Structure

```
digehealth-assessment/
├── digehealth_assessment/         <- Main source code
│   ├── train_cnn.py               <- CNN training script
│   ├── train_cnn.py               <- CNN training script
│   ├── train_lstm.py              <- LSTM training script
│   ├── train_randomforest.py      <- Random Forest training
│   ├── train_cnn_v2.py            <- Enhanced CNN training
│   ├── utils/                     <- Utility functions
│   └── modeling/                  <- Model architectures
│       ├── cnn.py                 <- CNN models (original, v2, v3)
│       ├── lstm.py                <- LSTM models
│       ├── data_loading.py        <- Data loading utilities
│       ├── data_loading.py        <- Data utilities for balancing, sampling, and analysis
│       ├── datasets.py            <- Pytorch Dataset configurations
│       ├── evaluation.py          <- Model evaluation tools
│       ├── preprocessing.py       <- Audio processing functions
│       └── model_training.py      <- Common training utilities
│
├── data/                          # Data directory
│   └── external/                  # Placeholder for Tech Test dataset
│   └── processed/                 # Destination of predicted annotations when running inference.py
│
├── models/                         # Trained model checkpoints
├── notebooks/                      # Jupyter notebooks
├── reports/                        # Training outputs and visualizations
│   └── figures/                   # Confusion matrices, ROC curves, etc.
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🔍 Model Performance

### **Expected Results**
- **CNN**: Typically achieves 60-80% accuracy, works better with shorter windows (<=0.3s). Data quality seems to be the limiting factor
- **LSTM**: Similar performance to CNN, works better with slightly longer window segments (~0.6s)
- **Random Forest**: 50-70% accuracy, good baseline for comparison
- **CNNv2/v3**: Similar performance to CNN, indicating the bottle neck is not the architecture, but the data quality

### **Performance Factors**
- **Audio Quality**: Clear recordings produce better results
- **Class Balance**: Imbalanced datasets require careful handling during training AND evaluating
- **Label Quality**: Accurate annotations are crucial
- **Feature Engineering**: Mel-spectrogram parameters significantly impact performance


## 🛠️ Troubleshooting

### **Common Issues**

1. **"File not found" errors**:
   - Ensure Tech Test dataset is in `data/external/Tech Test/`
   - Check file names match exactly: `23M74M.wav`, `AS_1.wav`, etc.

2. **Poor model performance**:
   - Run `plot_class_spectrograms.py` to visualize data quality
   - Check class distribution with `print_class_distribution()`
   - Adjust window size and overlap parameters

3. **Memory issues**:
   - Reduce batch size in training config
   - Use smaller spectrogram dimensions
   - Process data in smaller chunks

### **Getting Help**
- Review generated plots in `reports/figures/` for insights
- Examine training logs for convergence issues (majority of models require 10-30 epochs to stabilize)

## Advanced Usage

### **Custom Model Architectures**
Extend the existing models in `modeling/` directory:
- Add new CNN architecture in `cnn.py`
- Add new LSTM architeture in `lstm.py`
- Create custom feature extractors in `preprocessing.py`

### **Hyperparameter Tuning**
- Modify parameters in `training_config.py`
- Use cross-validation for robust evaluation 

### **Data Pipeline Customization**
- Adjust audio processing in `preprocessing.py`
- Modify feature extraction parameters
- Implement custom data augmentation

## Future Improvements

- **Improved Noise reduction**: Further pre-processing to limit noise and equalize the sound file 
- **Noise + Silence detector**: Two stage approach that first detects whether there is noise/silence reducing misclassification of the bowel movements.
- **Transfer Learning**: Use pre-trained audio models (AudioSet, VGGish) and further tune it to the use-case data
- **Real-time Processing**: Streaming audio classification
- **Multi-modal Fusion**: Combine audio with other sensor data (if available)

## License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

This is a coding assessment submission. For questions, please contact Jeroen Buil.

