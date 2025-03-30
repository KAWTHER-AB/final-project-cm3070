# evaluation/metrics.py

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report
)



def evaluate_lstm_predictions(y_true, y_pred, y_pred_prob):
    """
    Computes classification metrics for LSTM+Attention model using predicted labels and predicted probabilities.

    Args:
        y_true (np.array): Ground truth labels
        y_pred (np.array): Predicted binary class labels
        y_pred_prob (np.array): Predicted class probabilities (for AUC/PR-AUC)

    Returns:
        dict: Dictionary with precision, recall, f1, auc_roc, pr_auc, and classification report
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        classification_report
    )

    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_pred_prob),
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "report": classification_report(y_true, y_pred, digits=4)
    }

    return metrics


from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def evaluate_tabnet_predictions(y_true, y_pred, y_pred_proba):
    report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    return {
        "report": report,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }



def compute_classification_metrics(y_true, y_pred, y_score=None):
    """
    Calculates classification metrics for anomaly detection.

    Args:
        y_true (np.array): Ground truth labels
        y_pred (np.array): Predicted binary labels
        y_score (np.array): Optional anomaly score for AUC/PR-AUC

    Returns:
        dict: Precision, recall, f1-score, auc, pr_auc, and full classification report
    """
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_score is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)

    metrics["report"] = classification_report(y_true, y_pred, digits=4)
    return metrics