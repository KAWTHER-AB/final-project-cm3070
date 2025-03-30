#preprocessing/eda


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.style_utils import styled_print



def run_eda():
    """
    Perform exploratory data analysis on the credit card fraud dataset.
    Displays dataset structure, statistics, and class distribution.
    """
    # Load dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "creditcard.csv"))
    df = pd.read_csv(data_path)

    # Dataset shape
    styled_print("======================== Dataset Shape ==============================")
    print(df.shape)

    # First 5 rows
    styled_print("======================== Head (First 5 Rows) ========================")
    print(df.head())

    # Info summary
    styled_print("======================== Data Info ================================")
    print(df.info())

    # Descriptive statistics
    styled_print("======================== Statistical Summary ========================")
    print(df.describe())

    # Class distribution
    styled_print("======================== Class Distribution =========================")
    class_counts = df['Class'].value_counts()
    print(class_counts)

    return df



def visualize_data(df):
    """
    Provides comprehensive visualizations of the dataset:
    - Class distribution (Pie + Bar)
    - Transactions over time (scatterplot and distribution)
    - Histogram for each feature
    - Heatmap of correlations
    """

    import matplotlib.pyplot as plt
    import seaborn as sns


    styled_print("1. Class Distribution (Pie + Bar)")
    class_counts = df['Class'].value_counts()

    plt.figure(figsize=(12, 5))

    # Pie Chart
    plt.subplot(1, 2, 1)
    plt.pie(class_counts, labels=['Non-Fraud', 'Fraud'],
            autopct='%1.2f%%', colors=['skyblue', 'salmon'], startangle=90)
    plt.title("Class Distribution (Pie Chart)")

    # Bar Chart
    plt.subplot(1, 2, 2)
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='Blues')
    plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
    plt.title("Class Distribution (Bar Chart)")
    plt.xlabel("Class")
    plt.ylabel("Transaction Count")

    plt.tight_layout()
    plt.show()

    # --- Transactions Over Time ---
    styled_print("2. Number of Transactions Over Time")

    plt.figure(figsize=(10, 4))
    sns.scatterplot(x='Time', y='Amount', data=df, hue='Class', palette='cool', alpha=0.6)
    plt.title("Transaction Amounts Over Time")
    plt.xlabel("Time")
    plt.ylabel("Amount")
    plt.grid(True)
    plt.legend(title='Class')
    plt.tight_layout()
    plt.show()

    # 3. Distribution of Transactions Over Time
    styled_print("3.Distribution of Transaction Time")

    plt.figure(figsize=(8, 6))
    plt.title('Distribution of Transaction Time', fontsize=14)
    sns.histplot(df['Time'], bins=100)
    plt.show()

    # 🔥 NEW: Distribution of time w.r.t. transaction types (Fraud vs. Genuine)
    styled_print("4. Distribution of Time w.r.t. Transaction Types")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 4))

    sns.histplot(df[df['Class'] == 1]['Time'], bins=100, color='red', ax=axs[0])
    axs[0].set_title("Distribution of Fraud Transactions")

    sns.histplot(df[df['Class'] == 0]['Time'], bins=100, color='green', ax=axs[1])
    axs[1].set_title("Distribution of Genuine Transactions")

    plt.tight_layout()
    plt.show()

    # --- Histograms for all features ---
    styled_print("5. Histogram of All PCA Features")

    pca_features = [col for col in df.columns if col.startswith("V")]

    df[pca_features].hist(figsize=(15, 10), bins=30, color='cornflowerblue', edgecolor='black')
    plt.suptitle("Feature Distributions (Histograms)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # --- Correlation Heatmap ---
    styled_print("6. Heatmap of Correlation Matrix")

    plt.figure(figsize=(16, 12))
    corr_matrix = df.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",              # Use 2 decimal places
        cmap="coolwarm",        # Diverging color map
        linewidths=0.5,
        linecolor='gray',
        cbar=True,
        square=True,
        center=0
    )

    plt.title("Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha='right')  # Rotate x labels
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    run_eda()
    visualize_data()
