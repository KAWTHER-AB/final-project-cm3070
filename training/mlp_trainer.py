# training/mlp_trainer.py

import os
from utils.style_utils import styled_print
from evaluation.metrics import compute_classification_metrics
from evaluation.plots import plot_roc_and_pr_curves, plot_confusion_matrix, plot_loss_curve

def train_baseline_mlp(model, X_train, y_train, X_val, y_val, model_name="mlp_baseline", epochs=20, batch_size=512):
    """
    Trains the baseline MLP model and returns the trained model + history.

    Args:
        model: compiled Keras model
        X_train, y_train: training data
        X_val, y_val: validation data
        model_name: name used for saving
        epochs: number of training epochs
        batch_size: batch size

    Returns:
        model: trained model
        history: training history object
    """
    styled_print(" Training Baseline MLP Model")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save model architecture and weights
    os.makedirs("artifacts/MLP", exist_ok=True)
    model.save("artifacts/MLP/mlp_model.h5")
    with open("artifacts/MLP/mlp_model_architecture.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    styled_print(" Baseline MLP Training Complete")
    return model, history


def evaluate_baseline_mlp(model, X_test, y_test):
    """
    Evaluates the baseline MLP model on test set and returns metrics.

    Args:
        model: trained model
        X_test, y_test: test data

    Returns:
        metrics (dict): classification metrics
    """
    styled_print(" Evaluating Baseline MLP on Test Set")

    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba > 0.5).astype(int)

    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_score=y_proba)

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_and_pr_curves(y_test, y_proba, model_name="Baseline MLP")

    return metrics
