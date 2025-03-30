#preprocessing/data_loader

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from utils.style_utils import styled_print



def load_data():
    """
    Loads the credit card fraud dataset from the project root folder.
    
    Returns:
        pd.DataFrame: Raw dataset containing features and target label.
    """
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "creditcard.csv"))
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Cleans the dataset by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): Raw input dataset.

    Returns:
        pd.DataFrame: Cleaned dataset ready for feature-label splitting.
    """
    styled_print(" Dataset Cleaning Summary")

    print(f"[1] Original shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"[2] After removing duplicates: {df.shape}")
    df.dropna(inplace=True)
    print(f"[3] After dropping missing values: {df.shape}")

    return df



def split_data(df, random_state=42):
    """
    Splits the dataset into training (70%), validation (15%), and test (15%) sets.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple of 6 elements: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    """
    Applies StandardScaler to the 'Time' and 'Amount' columns.
    Scaling is fit on training data only, then applied to validation and test sets.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.

    Returns:
        Tuple: Scaled X_train, X_val, X_test, and the fitted scaler object.
    """
    scaler = StandardScaler()
    cols_to_scale = ["Time", "Amount"]

    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

    X_val_scaled = X_val.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    X_test_scaled = X_test.copy()
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Displays a summary table showing the number of samples and fraud cases in each data split.

    Args:
        X_train, X_val, X_test: Feature datasets.
        y_train, y_val, y_test: Corresponding labels.
    """
    styled_print(" Dataset Split Summary")

    summary = pd.DataFrame({
        "Set": ["Train", "Validation", "Test"],
        "Samples": [len(X_train), len(X_val), len(X_test)],
        "Fraud Cases": [y_train.sum(), y_val.sum(), y_test.sum()]
    })

    display(summary)

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    styled_print(" Data cleaned and ready.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)

    X_train_scaled, X_val_scaled, X_test_scaled, _ = scale_data(X_train, X_val, X_test)
    styled_print(" Features scaled on 'Time' and 'Amount'")
