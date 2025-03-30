# CM3070-FP

# Credit Card Fraud Detection Project

## Overview
This project implements a comprehensive approach to detecting credit card fraud using deep learning models. It features a variety of architectures including Autoencoders, LSTM with Attention, and TabNet, and it explores ensemble methods to improve detection performance.

## Repository Structure

```plaintext
credit_card_fraud/
├── creditcard.csv                   - Contains the dataset for the project.
├── final_experiments.ipynb          - Jupyter notebook running the entire pipeline.
├── README.md                        - Documentation of the project.

├── preprocessing/
│   ├── eda.py                       - Performs exploratory data analysis.
│   ├── data_loader.py               - Prepares data for modeling.
│   └── resampling/
│       └── smote.py                 - Implements SMOTE for class imbalance handling.

├── models/
│   ├── autoencoder_model.py         - Defines the Autoencoder model.
│   ├── tabnet_model.py              - Defines the TabNet model.
│   ├── lstm_attention_model.py      - Defines the LSTM with Attention model.
│   ├── mlp_baseline.py              - Defines a baseline MLP model.
│   └── ensemble_model/
│       ├── ensemble_logic.py        - Logic for model ensembling.
│       └── tuner.py                 - Tuning of global hyperparameters.

├── training/
│   ├── autoencoder_trainer.py       - Training script for Autoencoder.
│   ├── tabnet_trainer.py            - Training script for TabNet.
│   ├── mlp_trainer.py               - Training script for baseline MLP.
│   └── lstm_attention_trainer.py    - Training script for LSTM with Attention.

├── tuning/
│   ├── autoencoder_tuning.ipynb     - Tuning notebook for Autoencoder.
│   ├── tabnet_tuning.ipynb          - Tuning notebook for TabNet.
│   ├── lstm_attention_tuning.ipynb  - Tuning notebook for LSTM with Attention.
│   └── tuning_results/              - Stores tuning results.

├── evaluation/
│   ├── metrics.py                   - Contains shared metrics for model evaluation.
│   └── plots.py                     - Contains functions for plotting results.

└── utils/
    ├── style_utils.py               - Utility functions for styling.
    └── callbacks.py                 - Callbacks for model training.
