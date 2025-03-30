# training/tabnet_trainer.py

import numpy as np
from pytorch_tabnet.metrics import Metric

class F1Metric(Metric):
    """
    Custom metric class for computing F1 Score during TabNet training.

    Key Elements:
    - Inherits from PyTorch TabNet's `Metric` base class.
    - Implements __call__() method to calculate F1 using scikit-learn.
    - Uses `np.argmax` on prediction probabilities to derive class labels.
    - Designed to be used in eval_metric parameter during model training.
    - Sets _name = "f1_score" and _maximize = True for compatibility.
    """
    def __init__(self):
        self._name = "f1_score"
        self._maximize = True # F1 should be maximized

    def __call__(self, y_true, y_score):
        """
        Compute F1 Score using scikit-learn.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_score (np.ndarray): Predicted class probabilities.

        Returns:
            float: F1 Score of current predictions.
        """
        from sklearn.metrics import f1_score
        preds = np.argmax(y_score, axis=1)
        return f1_score(y_true, preds)


def train_tabnet_model(model, X_train, y_train, X_val, y_val, output_dir, max_epochs=50):
    """
    Train a TabNetClassifier model using AUC and custom F1 Score as metrics.

    Args:
        model: Initialized TabNetClassifier.
        X_train: Training features (DataFrame or NumPy array).
        y_train: Labels for training set.
        X_val: Validation features.
        y_val: Labels for validation set.
        output_dir: Path to save the trained model (.zip format).
        max_epochs: Maximum number of training epochs.

    Returns:
        model.history: Dictionary of training/validation metric scores per epoch.
    """

    model.fit(
        X_train=X_train.values,
        y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=["val"],
        eval_metric=["auc", F1Metric],  #  FIXED
        max_epochs=max_epochs,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    model.save_model(output_dir)
    return model.history
