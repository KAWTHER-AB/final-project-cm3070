# models/tabnet_model.py

from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def build_tabnet_model():
    """
    Build and return a TabNetClassifier model configured for binary classification.

    Key Components:
    - n_d, n_a: Width of decision and attention layers (both set to 32).
    - n_steps: Number of decision steps in the network (set to 5).
    - gamma: Relaxation factor controlling feature reuse between steps.
    - optimizer: Adam optimizer with a learning rate of 0.001.
    - scheduler: StepLR scheduler to decay the learning rate every 10 epochs.
    - mask_type: 'entmax' for smoother feature selection compared to 'sparsemax'.
    - verbose: Set to 0 to disable training logs (can be set to 1 for debugging).

    Returns:
        TabNetClassifier: Configured TabNet model ready for training.
    """
    model = TabNetClassifier(
        n_d=32,                      # Decision layer width
        n_a=32,                      # Attention layer width
        n_steps=5,                   # Number of decision steps
        gamma=1.5,                   # Feature reuse factor
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.001),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",          # smoother alternative to sparsemax
        verbose=0
    )
    return model
