
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from IPython.display import display
from utils.style_utils import styled_print


def apply_smote(X, y, random_state=42, visualize=True):
    """
    Applies SMOTE to the training dataset and shows class distribution before and after.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
        random_state (int): Seed for reproducibility.
        visualize (bool): If True, show pie and bar charts.

    Returns:
        Tuple: X_resampled, y_resampled
    """
    styled_print("Applying Synthetic Minority Over-sampling Technique (SMOTE)")
    print ("")

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    # Show class distribution before and after SMOTE
    df_summary = pd.DataFrame({
        "Class": ["Non-Fraud", "Fraud"],
        "Before SMOTE": [Counter(y)[0], Counter(y)[1]],
        "After SMOTE": [Counter(y_res)[0], Counter(y_res)[1]]
    })

    styled_print("1. Class Distribution Before and After SMOTE")
    display(df_summary)

    # Show shape of resampled data
    df_shapes = pd.DataFrame({
        "Shape": ["X_resampled", "y_resampled"],
        "Rows": [X_res.shape[0], y_res.shape[0]],
        "Columns": [X_res.shape[1], 1]
    })

    styled_print("2. Resampled Dataset Shape")
    display(df_shapes)

    if visualize:
        class_counts_before = pd.Series(y).value_counts()
        class_counts_after = pd.Series(y_res).value_counts()

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        # Pie Chart Before SMOTE
        ax[0, 0].pie(class_counts_before, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%',
                     colors=['lightblue', 'salmon'])
        ax[0, 0].set_title("Before SMOTE (Pie)")

        # Bar Chart Before SMOTE
        sns.barplot(
            x=class_counts_before.index,
            y=class_counts_before.values,
            hue=class_counts_before.index,
            dodge=False,
            palette=['lightblue', 'salmon'],
            legend=False,
            ax=ax[0, 1]
        )
        ax[0, 1].set_title("Before SMOTE (Bar)")
        ax[0, 1].set_xticks([0, 1])
        ax[0, 1].set_xticklabels(['Non-Fraud', 'Fraud'])
        ax[0, 1].set_ylabel("Count")

        # Pie Chart After SMOTE
        ax[1, 0].pie(class_counts_after, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%',
                     colors=['lightgreen', 'tomato'])
        ax[1, 0].set_title("After SMOTE (Pie)")

        # Bar Chart After SMOTE
        sns.barplot(
            x=class_counts_after.index,
            y=class_counts_after.values,
            hue=class_counts_after.index,
            dodge=False,
            palette=['lightgreen', 'tomato'],
            legend=False,
            ax=ax[1, 1]
        )
        ax[1, 1].set_title("After SMOTE (Bar)")
        ax[1, 1].set_xticks([0, 1])
        ax[1, 1].set_xticklabels(['Non-Fraud', 'Fraud'])
        ax[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    return X_res, y_res

