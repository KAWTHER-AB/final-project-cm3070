# models/mlp_baseline.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_baseline_mlp(input_shape):
    """
    Builds a simple baseline MLP model.

    Args:
        input_shape (tuple): Shape of the input features, e.g., (30,)

    Returns:
        model (tf.keras.Model): Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),       # 1 hidden layer
        Dense(1, activation='sigmoid')      # Output layer for binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
