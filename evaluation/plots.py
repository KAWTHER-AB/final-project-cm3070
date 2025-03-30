# evaluation/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
)


def plot_roc_and_pr_curves(y_true, y_scores, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True)
    axes[0].legend()

    # PR
    axes[1].plot(recall, precision, label=model_name)
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle(f"ROC and PR Curves for {model_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_loss_curve(history, title="Loss Curve"):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_tabnet_loss(logs, title="TabNet Training Loss"):
    """
    Plots the training and (optionally) validation loss curves from TabNet training logs.

    Parameters:
    - logs (dict): The history object returned by TabNet's .fit(), containing loss values.
                   Expected keys: 'loss' for training loss, and optionally 'val_loss'.
    - title (str): Title to display on the plot.

    The function displays the plot but does not save it to disk.
    """
        
    train_losses = logs['loss']
    
    if 'val_loss' in logs:
        val_losses = logs['val_loss']
        has_val = True
    else:
        print(" No validation loss key found in logs. Only training loss will be plotted.")
        val_losses = None
        has_val = False

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    
    if has_val:
        plt.plot(val_losses, label="Validation Loss", linewidth=2, linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_metric_bars(metrics_dict, model_name="Model"):
    """
    Bar plot for key classification metrics.

    Args:
        metrics_dict (dict): Contains precision, recall, f1_score, auc_roc, pr_auc
        model_name (str): Title for the plot
    """
    metrics_to_plot = {
        "Precision": metrics_dict.get("precision"),
        "Recall": metrics_dict.get("recall"),
        "F1-Score": metrics_dict.get("f1_score"),
        "AUC-ROC": metrics_dict.get("auc_roc"),
        "PR-AUC": metrics_dict.get("pr_auc"),
    }

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color='skyblue')
    plt.ylim(0, 1)
    plt.title(f"{model_name} — Classification Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}",
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
