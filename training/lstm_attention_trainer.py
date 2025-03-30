# training/lstm_attention_trainer.py 

import tensorflow as tf
from models.lstm_attention_model import build_lstm_attention_model
from utils.callbacks import get_lstm_attention_callbacks
from utils.style_utils import styled_print
from evaluation.plots import plot_loss_curve


def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=256):
    styled_print("Training LSTM + Attention Model")

    model = build_lstm_attention_model(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='AUC'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision')
        ]
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=get_lstm_attention_callbacks(),
        shuffle=True,
        verbose=1
    )

    plot_loss_curve(history, title="LSTM + Attention Training & Validation Loss")
    return model, history
