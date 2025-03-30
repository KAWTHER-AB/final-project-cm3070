# models/ensemble_model/ensemble_logic.py

import numpy as np

def majority_vote(predictions_list):
    """
    Perform majority voting across predictions from multiple models.

    Args:
        predictions_list (list of np.arrays): Each array contains binary predictions (0 or 1).

    Returns:
        np.array: Final ensemble predictions (0 = genuine, 1 = fraud).
    """
    predictions_array = np.array(predictions_list)
    vote_sum = np.sum(predictions_array, axis=0)
    majority_votes = (vote_sum >= 2).astype(int)
    return majority_votes


def soft_voting(probabilities_list, threshold=0.5):
    """
    Perform soft voting by averaging predicted probabilities.

    Args:
        probabilities_list (list of np.arrays): Each array contains predicted fraud probabilities from a model.
        threshold (float): Threshold to convert average probability to class label (default=0.5).

    Returns:
        np.array: Final ensemble predictions (0 = genuine, 1 = fraud).
    """
    probabilities_array = np.array(probabilities_list)
    avg_probs = np.mean(probabilities_array, axis=0)
    return (avg_probs > threshold).astype(int)

