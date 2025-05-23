import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
VOCAB_SIZE = None           # to be set by tokenizer
D_MODEL    = 512
N_LAYERS   = 6
N_HEADS    = 8
BLOCK_SIZE = 128

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
SAVE_PATH         = "data/ultrachat-small"  # where train/valid txt live
MAX_SAMPLES       = 2000                     # e.g., 50000 or None for full
VALIDATION_SPLIT  = 0.1                      # fraction of data for validation

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE      = 512
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-2
LABEL_SMOOTHING = 0.0

# -----------------------------------------------------------------------------
# Scheduler → plateau‑aware
# -----------------------------------------------------------------------------
SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'mode': 'min',
    'factor': 0.5,
    'patience': 1,
    'min_lr': 5e-5,
    'verbose': True
}

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
EPOCHS               = 3   # increase after break
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIENT_ACCUM_STEPS = 8

# -----------------------------------------------------------------------------
# Early‑stop & evaluation
# -----------------------------------------------------------------------------
EVAL_INTERVAL        = 1000
VALID_LOSS_THRESHOLD = 1.0
PATIENCE             = 6
DROPOUT = 0.1

# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"

