# models/autoencoder_model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras import regularizers

def build_autoencoder(input_dim, encoding_dim=32, dropout_rate=0.3, noise_std=0.1):
    """
    Advanced Autoencoder: Includes GaussianNoise and Activity Regularizer.

    Architecture:
    - Input → GaussianNoise → Dense(encoding_dim, L2 regularizer) → BatchNorm → Dropout
      → Dense(encoding_dim // 2) → BatchNorm →
      Dense(encoding_dim // 2) → BatchNorm → Dropout →
      Dense(input_dim)

    Args:
        input_dim (int): Number of features in the input data (input layer size).
        encoding_dim (int): Number of neurons in the first encoder Dense layer.
        dropout_rate (float): Dropout rate applied to both encoder and decoder for regularization.
        noise_std (float): Standard deviation of the Gaussian noise applied to the input.

    Returns:
        keras.Model: A compiled Keras Model object representing the full autoencoder.
                     The model is **not trained** and must be compiled and fit externally.
    """

    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = GaussianNoise(noise_std)(input_layer)
    x = Dense(encoding_dim, activation='relu',
              activity_regularizer=regularizers.l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(encoding_dim // 2, activation='relu')(x)
    x = BatchNormalization()(x)

    # Decoder
    x = Dense(encoding_dim // 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(input_dim, activation='linear')(x)

    model_name = f"AE_dropout{dropout_rate}_enc{encoding_dim}_noise{noise_std}"
    model = Model(inputs=input_layer, outputs=output_layer, name=model_name)

    return model


