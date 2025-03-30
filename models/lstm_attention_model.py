# models/lstm_attention_model.py

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, Layer
from tensorflow.keras.models import Model

#  Custom attention layer (fixed for loading)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):  # Accept Keras standard arguments
        super(AttentionLayer, self).__init__(**kwargs)  #  Pass them to parent
        self.score_dense = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        score = self.score_dense(inputs)                               # Shape: (batch, timesteps, 1)
        weights = self.softmax(score)                                  # Shape: (batch, timesteps, 1)
        context_vector = inputs * weights                              # Shape: (batch, timesteps, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)         # Shape: (batch, features)
        return context_vector

    def get_config(self):  #  Add for .keras or .h5 support
        config = super(AttentionLayer, self).get_config()
        return config



# Build LSTM with Attention Model
def build_lstm_attention_model(input_shape, lstm_units=64, dense_units=64, dropout_rate=0.3):
    """
    Builds an LSTM with Attention architecture.

    Architecture:
        - 2 LSTM layers with return_sequences=True to preserve temporal outputs
        - Custom AttentionLayer to learn importance of each timestep
        - Dense layer for final feature mapping
        - BatchNormalization to stabilize and speed up training
        - Dropout to prevent overfitting
        - Sigmoid output layer for binary classification (fraud / no fraud)

    Args:
        input_shape (tuple): Shape of the input sequence (timesteps, features)
        lstm_units (int): Number of units in LSTM layers
        dense_units (int): Number of units in the dense layer after attention
        dropout_rate (float): Dropout rate for regularization

    Returns:
        model (tf.keras.Model): Compiled LSTM with Attention model
    """

    
    inputs = Input(shape=input_shape)
    
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = LSTM(lstm_units, return_sequences=True)(x)

    x = AttentionLayer()(x)

    x = Dense(dense_units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


