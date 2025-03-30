# utils/callbacks.py

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_standard_callbacks(patience=5):
    """
    Returns standard callbacks for Autoencoder training:
    - EarlyStopping: to prevent overfitting
    - ReduceLROnPlateau: to adapt learning rate

    Args:
        patience (int): Number of epochs to wait before reducing LR or stopping.

    Returns:
        list: Keras callback objects
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, reduce_lr]





# New callbacks specifically tailored for LSTM with Attention
def get_lstm_attention_callbacks(patience_es=5, patience_rlr=3):
    """
    Custom callbacks explicitly optimized for LSTM with Attention:
    - EarlyStopping (patience = 5)
    - ReduceLROnPlateau (patience = 3)
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience_es,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_rlr,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, reduce_lr]



def get_autoencoder_callbacks(patience=10):
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5, verbose=1)
    ]

