# Prediction Model Trainer - Stacking vs Baseline

![python-version](https://img.shields.io/badge/Python-3.12.9-blue.svg)
![last-update](https://img.shields.io/badge/Last%20Update-November%202025-green.svg)
![license](https://img.shields.io/badge/License-MIT-purple.svg)

This document outlines the implementation of a prediction model trainer that compares stacking models against baseline models using Random Forest, LSTM and GRU architectures. 

## Overview
The `PredictionModelTrainer` class is designed to train and evaluate machine learning models on time series data. It includes methods for building supervised datasets, training Random Forest models with cross-validation, and training RNN models (LSTM and GRU) with cross-validation. The data used for training comes from RNP (National Education Network) in Brazil. The chosen pairs of links were selected after the imputation process was applied, generating for each pair of links a file with the stacking and baseline methods: KNNImputer, BackwardFill, ForwardFill, Mean, and Median.

## Order of Execution

1. **prediction_all_links.py**: This script executes the training and evaluation of models on all link pairs (Total 558). Example:
```python
from trainer_model.prediction_all_links import PredictionAllLinks

if __name__ == "__main__":
    predictor = PredictionAllLinks(
        data_dir="imputed_series_selective_unfair",
        lags=6,
        cv_splits=5,
        cv_splits=5,
        min_split_size=30,
        baseline_keep="bfill,ffill,knn,mean,median",
        require_all_baselines=False,
        device="auto",
        window=24,
        same_window=False,
        epochs=25,
        patience=3,
        log_level="INFO",
        log_dir="logs",
        plots=True
    )
    predictor.run()
```

2. **improvement_filtered_links.py**: This script filters the links that showed improvement with stacking methods over baseline methods. Example:
```python

from trainer_model.improvement_filtered_links import ImprovementLinks

if __name__ == "__main__":
    improvement_links = ImprovementLinks()
    improvement_links.run()
