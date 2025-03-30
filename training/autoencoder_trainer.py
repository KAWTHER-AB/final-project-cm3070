# training/autoencoder_trainer.py

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.losses import MeanSquaredError  
from models.autoencoder_model import build_autoencoder
from utils.callbacks import get_autoencoder_callbacks
from evaluation.metrics import compute_classification_metrics
from evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_loss_curve
)


def train_autoencoder(X_train_0, X_val, input_dim, encoding_dim=32, dropout=0.3,
                      noise_std=0.1, epochs=50, batch_size=256, model_path=None):
    """
    Trains an autoencoder model on class 0 (genuine) data using MSE loss safely.

    Args:
        X_train_0: Training data (class 0 only)
        X_val: Validation data
        input_dim: Input feature count
        encoding_dim: Size of encoding layer
        dropout: Dropout rate
        noise_std: Gaussian noise standard deviation (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        model_path: If set, saves trained model as .h5

    Returns:
        model: Trained autoencoder
        history: Training history object
    """
    model = build_autoencoder(input_dim, encoding_dim, dropout, noise_std)
    model.compile(optimizer='adam', loss=MeanSquaredError())  # fixed so it can be safely saved 

    callbacks = get_autoencoder_callbacks(patience=10)

    history = model.fit(
        X_train_0, X_train_0,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Save model as .h5 
    if model_path:
        model.save(model_path)

    plot_loss_curve(history, title="Autoencoder Training & Validation Loss")
    return model, history















def evaluate_autoencoder(model, X_data, y_data, threshold=None, visualize=True):
    """
    Evaluates autoencoder reconstruction error to classify anomalies.

    Returns:
        metrics: Dict of classification metrics
        threshold: Used reconstruction error threshold
    """
    reconstructions = model.predict(X_data)
    mse = np.mean(np.power(X_data - reconstructions, 2), axis=1)

    if threshold is None:
        threshold = np.percentile(mse, 95)

    y_pred = (mse > threshold).astype(int)
    metrics = compute_classification_metrics(y_true=y_data, y_pred=y_pred, y_score=mse)

    if visualize:
        plot_reconstruction_error(mse, y_data, threshold)
        plot_confusion_matrix(y_data, y_pred)
        plot_roc_and_pr_curves(y_data, mse, model_name="Autoencoder")

    return metrics, threshold

def plot_reconstruction_error(errors, y_true, threshold):
    """
    Plots reconstruction error histogram by class with decision threshold.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(errors[y_true == 0], bins=100, color='green', label='Genuine', kde=True)
    sns.histplot(errors[y_true == 1], bins=100, color='red', label='Fraudulent', kde=True)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




