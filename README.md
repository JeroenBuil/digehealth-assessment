# DigeHealth Assessment

My's submission for the coding assessment for the ML Engineer at DigeHealth, focusing on bowel movement classification using machine learning approaches.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Overview

Goal: develop a proof-of-concept ML model for identifying bowel sounds in audio data and differentiating between 3 main classes:

1.  Single burst (labelled b)
2.  Multiple burst (labelled mb)
3.  Harmonic (labelled h)

The model should identify the start time, end time and type of each bowel sound. There are some other labels for voice and noise, these can be ignored or used.

The assignment includes, explorative data analysis, and several machine learning approaches to tackle this challenge.

## ⚙️ Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd digehealth-assignment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## 💾 Data

**IMPORTANT**: All analyses in this project rely on the 'Tech Test' dataset supplied by DigHealth, which is **NOT INCLUDED** in this repository.

### Required Data Setup

To run any analysis, you must:

1. **Obtain the dataset** from DigeHealth
2. **Place the .wav and .txt file** in the `data/external/Tech Test` directory
3. **Ensure the filenames** are exactly `23M74M.wav`, `23M74M.txt`, `AS_1.wav`, and `AS_1.txt`

The dataset contains:
- 2x .wav files containing bowel sound recordings
- 2x .txt files with annotated start, end and bowel noise class of the identical named .wav file

**Note**: Without this data file, all analysis scripts will fail with a "file not found" error.

## Usage Examples

### Run Individual model

```bash
# Randomforest Model
python -m digehealth_assignment.train_randomforest.py

# CNN Model
python -m digehealth_assignment.train_cnn.py
```

### Generated Outputs

**Important**: Running each train_model scripts, automatically stores the trained model in the `\models` folder and generate performance evaluation figures that are stored in the `reports/figures/` folder. 

## 📁 Project Structure

```
├── LICENSE
├── Makefile
├── README.md                <- Project overview (this file)
├── data
│   └── external             <- Placeholder for the 'Tech Test' dataset supplied by DigeHealth
│
├── models                   <- Trained model weights
│
├── notebooks                <- Jupyter notebooks for exploration & experiments
│
├── reports
│   └── figures              <- Training curves, confusion matrices, etc.
│
├── requirements.txt         <- Python dependencies
│
└── digehealth_assessment    <- Main source code
    │
    ├── __init__.py
    ├── config.py            <- Configuration variables (paths, constants, etc.)
    ├── dataset.py           <- Dataset utilities (loading, splitting, transforms)
    ├── features.py          <- Audio feature extraction (mel-spectrograms, normalization)
    ├── ml_pipeline.py       <- Training utilities (weighted sampler, preprocessing pipeline)
    ├── train_cnn.py         <- CNN training entry point
    ├── plots.py             <- Visualization helpers
    └── evaluation.py        <- Metrics and test set evaluation
```


--------

